"""Chunks and embeddings in one SQLite file: a knowledge base that survives restarts with no server.

For apps that run on the user's machine (a desktop app, a sidecar next to a C++ or
Rust app). Search is brute-force cosine in Python over every stored vector, which
stays fast for a personal knowledge base (tens of thousands of chunks); a corpus
far larger than that belongs in ``PgVectorStore``. Uses only the standard library,
so it ports to any language that has SQLite.
"""

import asyncio
import json
import math
import sqlite3
import threading
from array import array
from pathlib import Path
from typing import Optional

from app.services.rag.contracts import (
    Chunk,
    SearchResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    url TEXT,
    metadata TEXT NOT NULL,
    lexical_tokens TEXT,
    lexical_mode TEXT,
    embedding BLOB NOT NULL,
    norm REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks (document_id);
"""

_COLUMNS = "id, document_id, chunk_index, content, title, source, url, metadata, lexical_tokens, lexical_mode"


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    tokens = row["lexical_tokens"]
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        title=row["title"],
        source=row["source"],
        url=row["url"],
        metadata=json.loads(row["metadata"]),
        lexical_tokens=json.loads(tokens) if tokens is not None else None,
        lexical_mode=row["lexical_mode"],
    )


class SQLiteVectorStore:
    """``VectorStore`` backed by one SQLite file (``":memory:"`` for tests)."""

    def __init__(self, path: str | Path, dimension: int):
        """Open (or create) the database at ``path`` for ``dimension``-sized vectors."""
        self.dimension = dimension
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection used from worker threads, one call at a time (the lock).
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    async def _run(self, fn, *args):
        def locked():
            with self._lock:
                return fn(*args)

        return await asyncio.to_thread(locked)

    def _upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dimension:
                raise ValueError(f"vector of {chunk.id} has {len(vector)} dimensions, store expects {self.dimension}")
            rows.append(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.title,
                    chunk.source,
                    chunk.url,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    json.dumps(chunk.lexical_tokens, ensure_ascii=False) if chunk.lexical_tokens is not None else None,
                    chunk.lexical_mode,
                    array("f", vector).tobytes(),
                    math.sqrt(sum(v * v for v in vector)),
                )
            )
        self._conn.executemany(
            f"INSERT OR REPLACE INTO chunks ({_COLUMNS}, embedding, norm) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        self._conn.commit()

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks by id."""
        await self._run(self._upsert, chunks, vectors)

    def _search(self, vector: list[float], top_k: int) -> list[SearchResult]:
        query_norm = math.sqrt(sum(v * v for v in vector))
        if not query_norm:
            return []
        scored: list[tuple[float, str]] = []
        for row in self._conn.execute("SELECT id, embedding, norm FROM chunks"):
            stored = array("f")
            stored.frombytes(row["embedding"])
            norm = row["norm"]
            score = sum(a * b for a, b in zip(vector, stored, strict=True)) / (query_norm * norm) if norm else 0.0
            scored.append((score, row["id"]))
        scored.sort(reverse=True)
        best = scored[:top_k]
        if not best:
            return []
        marks = ",".join("?" * len(best))
        rows = self._conn.execute(f"SELECT {_COLUMNS} FROM chunks WHERE id IN ({marks})", [i for _, i in best])
        by_id = {row["id"]: _row_to_chunk(row) for row in rows}
        return [SearchResult(chunk=by_id[i], score=max(0.0, s), method="dense") for s, i in best]

    async def search(self, vector: list[float], top_k: int) -> list[SearchResult]:
        """Return the closest chunks by cosine similarity, best first."""
        return await self._run(self._search, vector, top_k)

    def _existing_contents(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(f"SELECT id, content FROM chunks WHERE id IN ({marks})", ids)
        return {row["id"]: row["content"] for row in rows}

    async def existing_contents(self, ids: list[str]) -> dict[str, str]:
        """Map each already-stored id to its stored content."""
        return await self._run(self._existing_contents, ids)

    def _all_chunks(self) -> list[Chunk]:
        return [_row_to_chunk(row) for row in self._conn.execute(f"SELECT {_COLUMNS} FROM chunks ORDER BY id")]

    async def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk with its lexical tokens."""
        return await self._run(self._all_chunks)

    def _chunk_owners(self, document_ids: Optional[list[str]]) -> dict[str, str]:
        if document_ids is None:
            rows = self._conn.execute("SELECT id, document_id FROM chunks")
        elif not document_ids:
            return {}
        else:
            marks = ",".join("?" * len(document_ids))
            rows = self._conn.execute(
                f"SELECT id, document_id FROM chunks WHERE document_id IN ({marks})", document_ids
            )
        return {row["id"]: row["document_id"] for row in rows}

    async def chunk_owners(self, document_ids: Optional[list[str]] = None) -> dict[str, str]:
        """Map stored chunk ids to their document id, for ``document_ids`` or for every document."""
        return await self._run(self._chunk_owners, document_ids)

    def _delete(self, chunk_ids: list[str]) -> None:
        self._conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in chunk_ids])
        self._conn.commit()

    async def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id."""
        await self._run(self._delete, chunk_ids)

    def close(self) -> None:
        """Close the database file."""
        with self._lock:
            self._conn.close()
