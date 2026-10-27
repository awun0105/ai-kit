"""Vector stores.

D08 used ChromaDB. The kit already runs Postgres with pgvector for long-term
memory, so RAG chunks live there too instead of adding a second database.
``InMemoryVectorStore`` has the same interface for tests and quick demos.
"""

import json
import math
from typing import (
    Optional,
    Protocol,
)
from urllib.parse import quote_plus

from psycopg import (
    AsyncConnection,
    sql,
)
from psycopg.rows import (
    DictRow,
    dict_row,
)
from psycopg_pool import AsyncConnectionPool

from app.services.rag.contracts import (
    Chunk,
    SearchResult,
)


class VectorStore(Protocol):
    """Storage for chunks and their embeddings."""

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks by id."""
        ...

    async def search(self, vector: list[float], top_k: int) -> list[SearchResult]:
        """Return the closest chunks by cosine similarity, best first."""
        ...

    async def existing_contents(self, ids: list[str]) -> dict[str, str]:
        """Map each already-stored id to its stored content."""
        ...

    async def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk with its lexical tokens, used to build the BM25 index."""
        ...

    async def chunk_owners(self, document_ids: Optional[list[str]] = None) -> dict[str, str]:
        """Map stored chunk ids to their document id, for ``document_ids`` or for every document."""
        ...

    async def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id."""
        ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryVectorStore:
    """Brute-force cosine search over a dict. Fine for tests and a few thousand chunks."""

    def __init__(self):
        """Create an empty store."""
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks by id."""
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, vector)

    async def search(self, vector: list[float], top_k: int) -> list[SearchResult]:
        """Return the closest chunks by cosine similarity, best first."""
        scored = [
            SearchResult(chunk=chunk, score=max(0.0, _cosine(vector, stored)), method="dense")
            for chunk, stored in self._rows.values()
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def existing_contents(self, ids: list[str]) -> dict[str, str]:
        """Map each already-stored id to its stored content."""
        return {i: self._rows[i][0].content for i in ids if i in self._rows}

    async def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk."""
        return [chunk for chunk, _ in self._rows.values()]

    async def chunk_owners(self, document_ids: Optional[list[str]] = None) -> dict[str, str]:
        """Map stored chunk ids to their document id."""
        wanted = None if document_ids is None else set(document_ids)
        return {
            cid: chunk.document_id
            for cid, (chunk, _) in self._rows.items()
            if wanted is None or chunk.document_id in wanted
        }

    async def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id."""
        for cid in chunk_ids:
            self._rows.pop(cid, None)


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _row_to_chunk(row: DictRow) -> Chunk:
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        title=row["title"],
        source=row["source"],
        url=row["url"],
        metadata=row["metadata"] or {},
        lexical_tokens=row.get("lexical_tokens"),
        lexical_mode=row.get("lexical_mode"),
    )


class PgVectorStore:
    """Chunks in a Postgres table with a pgvector HNSW index (cosine distance)."""

    def __init__(self, conninfo: str, table: str, dimension: int, pool_size: int = 5):
        """Configure the store; the table and index are created on first use."""
        self._conninfo = conninfo
        self._table = sql.Identifier(table)
        self._index = sql.Identifier(f"{table}_embedding_hnsw")
        self._document_index = sql.Identifier(f"{table}_document_id")
        self._dimension = dimension
        self._pool_size = pool_size
        self._pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None

    @classmethod
    def from_settings(cls, host: str, port: int, db: str, user: str, password: str, **kwargs) -> "PgVectorStore":
        """Build the connection string from the same Postgres settings the app uses."""
        conninfo = f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"
        return cls(conninfo, **kwargs)

    async def _get_pool(self) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
        if self._pool is None:
            pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
                self._conninfo,
                open=False,
                max_size=self._pool_size,
                kwargs={"autocommit": True, "row_factory": dict_row, "connect_timeout": 5},
            )
            await pool.open()
            async with pool.connection() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {t} ("
                        " id text PRIMARY KEY, document_id text NOT NULL, chunk_index int NOT NULL,"
                        " content text NOT NULL, title text NOT NULL, source text NOT NULL, url text,"
                        " metadata jsonb NOT NULL DEFAULT '{{}}', embedding vector({d}) NOT NULL,"
                        " lexical_tokens text[], lexical_mode text)"
                    ).format(t=self._table, d=sql.Literal(self._dimension))
                )
                # Tables created before lexical tokens were stored.
                await conn.execute(
                    sql.SQL(
                        "ALTER TABLE {t} ADD COLUMN IF NOT EXISTS lexical_tokens text[],"
                        " ADD COLUMN IF NOT EXISTS lexical_mode text"
                    ).format(t=self._table)
                )
                await conn.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw (embedding vector_cosine_ops)").format(
                        i=self._index, t=self._table
                    )
                )
                await conn.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} (document_id)").format(
                        i=self._document_index, t=self._table
                    )
                )
            self._pool = pool
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks by id."""
        if not chunks:
            return
        query = sql.SQL(
            "INSERT INTO {t} (id, document_id, chunk_index, content, title, source, url, metadata, embedding,"
            " lexical_tokens, lexical_mode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, chunk_index = EXCLUDED.chunk_index,"
            " content = EXCLUDED.content, title = EXCLUDED.title, source = EXCLUDED.source, url = EXCLUDED.url,"
            " metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding,"
            " lexical_tokens = EXCLUDED.lexical_tokens, lexical_mode = EXCLUDED.lexical_mode"
        ).format(t=self._table)
        params = [
            (
                c.id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.title,
                c.source,
                c.url,
                json.dumps(c.metadata),
                _vector_literal(v),
                c.lexical_tokens,
                c.lexical_mode,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(query, params)

    async def search(self, vector: list[float], top_k: int) -> list[SearchResult]:
        """Return the closest chunks by cosine similarity, best first."""
        query = sql.SQL(
            "SELECT id, document_id, chunk_index, content, title, source, url, metadata,"
            " 1 - (embedding <=> %(v)s::vector) AS score FROM {t} ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s"
        ).format(t=self._table)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            rows = await (await conn.execute(query, {"v": _vector_literal(vector), "k": top_k})).fetchall()
        return [SearchResult(chunk=_row_to_chunk(r), score=max(0.0, float(r["score"])), method="dense") for r in rows]

    async def existing_contents(self, ids: list[str]) -> dict[str, str]:
        """Map each already-stored id to its stored content."""
        if not ids:
            return {}
        query = sql.SQL("SELECT id, content FROM {t} WHERE id = ANY(%s)").format(t=self._table)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            rows = await (await conn.execute(query, (ids,))).fetchall()
        return {r["id"]: r["content"] for r in rows}

    async def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk."""
        query = sql.SQL(
            "SELECT id, document_id, chunk_index, content, title, source, url, metadata, lexical_tokens, lexical_mode"
            " FROM {t} ORDER BY id"
        ).format(t=self._table)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            rows = await (await conn.execute(query)).fetchall()
        return [_row_to_chunk(r) for r in rows]

    async def chunk_owners(self, document_ids: Optional[list[str]] = None) -> dict[str, str]:
        """Map stored chunk ids to their document id."""
        if document_ids is None:
            query = sql.SQL("SELECT id, document_id FROM {t}").format(t=self._table)
            params: tuple = ()
        else:
            if not document_ids:
                return {}
            query = sql.SQL("SELECT id, document_id FROM {t} WHERE document_id = ANY(%s)").format(t=self._table)
            params = (document_ids,)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            rows = await (await conn.execute(query, params)).fetchall()
        return {r["id"]: r["document_id"] for r in rows}

    async def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id."""
        if not chunk_ids:
            return
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(sql.SQL("DELETE FROM {t} WHERE id = ANY(%s)").format(t=self._table), (chunk_ids,))
