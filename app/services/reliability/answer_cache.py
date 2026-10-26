"""Semantic answer cache with scope, TTL, stale-if-error and source invalidation (D25 lab, part G).

Lifetime of an entry (age = now - stored_at):

    0 ........ ttl_s ................... ttl_s + max_stale_s ........
    |  fresh   | stale: only when no provider answers | expired: dropped

- A *fresh* hit answers a first-turn knowledge question without calling a model.
- A *stale* hit is served only when every provider failed: a slightly old FAQ
  answer beats an error page.
- Entries are scoped (prompt template, model, knowledge table, cache version), so
  changing the prompt or model never serves answers produced under the old one.
- Each entry records the documents its answer was built from; re-ingesting one of
  them deletes the entry. The Postgres store lets the ingestion CLI invalidate
  entries that the API server process wrote.
"""

import hashlib
import json
import math
import re
import time
import uuid
from typing import (
    Awaitable,
    Callable,
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
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger
from app.core.prompts import _SYSTEM_PROMPT_TEMPLATE
from app.services.rag import get_rag_pipeline

# Questions about the asker's own account/order/data are never cached or served from cache.
DEFAULT_PERSONAL_PATTERN = (
    r"\b(?:my|mine|me|i'm|i am|i've|our)\b"
    r"|\b(?:của\s+(?:tôi|mình|em|anh|chị)|tài\s+khoản\s+(?:tôi|mình)|đơn\s+hàng\s+(?:tôi|mình|của))\b"
)


class CacheHit(BaseModel):
    """A usable cached answer."""

    answer: str
    similarity: float
    age_s: float
    fresh: bool
    sources: list[str]


class AnswerStore(Protocol):
    """Persistence for cached answers."""

    async def nearest(self, scope: str, embedding: list[float]) -> Optional[tuple[dict, float]]:
        """Return the most similar entry in ``scope`` and its cosine similarity."""
        ...

    async def upsert(self, scope: str, question: str, embedding: list[float], answer: str, sources: list[str]) -> None:
        """Insert an entry, replacing a near-duplicate question in the same scope."""
        ...

    async def touch(self, entry_id: str) -> None:
        """Mark an entry as used now."""
        ...

    async def delete_by_sources(self, sources: list[str]) -> int:
        """Delete entries built on any of ``sources``; return how many."""
        ...

    async def delete_older_than(self, max_age_s: float) -> int:
        """Delete expired entries; return how many."""
        ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryAnswerStore:
    """Process-local store with LRU eviction. Tests and single-process demos only."""

    def __init__(self, max_entries: int = 500, similarity_threshold: float = 0.92, clock=time.time):
        """Create an empty store."""
        self.entries: dict[str, dict] = {}
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._clock = clock

    async def nearest(self, scope: str, embedding: list[float]) -> Optional[tuple[dict, float]]:
        """Return the most similar entry in ``scope``."""
        best: Optional[tuple[dict, float]] = None
        for entry in self.entries.values():
            if entry["scope"] != scope:
                continue
            sim = _cosine(embedding, entry["embedding"])
            if best is None or sim > best[1]:
                best = (entry, sim)
        return best

    async def upsert(self, scope: str, question: str, embedding: list[float], answer: str, sources: list[str]) -> None:
        """Insert, replacing a near-duplicate question instead of piling up copies."""
        now = self._clock()
        match = await self.nearest(scope, embedding)
        entry_id = match[0]["id"] if match and match[1] >= self.similarity_threshold else uuid.uuid4().hex
        self.entries[entry_id] = {
            "id": entry_id,
            "scope": scope,
            "question": question,
            "embedding": embedding,
            "answer": answer,
            "sources": sources,
            "stored_at": now,
            "last_used": now,
        }
        if len(self.entries) > self.max_entries:
            lru = min(self.entries.values(), key=lambda e: e["last_used"])
            del self.entries[lru["id"]]

    async def touch(self, entry_id: str) -> None:
        """Mark an entry as used now."""
        if entry_id in self.entries:
            self.entries[entry_id]["last_used"] = self._clock()

    async def delete_by_sources(self, sources: list[str]) -> int:
        """Delete entries built on any of ``sources``."""
        doomed = [i for i, e in self.entries.items() if set(e["sources"]) & set(sources)]
        for i in doomed:
            del self.entries[i]
        return len(doomed)

    async def delete_older_than(self, max_age_s: float) -> int:
        """Delete expired entries."""
        cutoff = self._clock() - max_age_s
        doomed = [i for i, e in self.entries.items() if e["stored_at"] < cutoff]
        for i in doomed:
            del self.entries[i]
        return len(doomed)


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


class PgVectorAnswerStore:
    """Entries in a Postgres table with a pgvector index, shared by every process."""

    def __init__(self, conninfo: str, table: str, dimension: int, similarity_threshold: float = 0.92):
        """Configure the store; the table is created on first use."""
        self._conninfo = conninfo
        self._table = sql.Identifier(table)
        self._index = sql.Identifier(f"{table}_embedding_hnsw")
        self._dimension = dimension
        self.similarity_threshold = similarity_threshold
        self._pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None

    async def _get_pool(self) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
        if self._pool is None:
            pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
                self._conninfo,
                open=False,
                max_size=4,
                kwargs={"autocommit": True, "row_factory": dict_row, "connect_timeout": 5},
            )
            await pool.open()
            async with pool.connection() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {t} (id text PRIMARY KEY, scope text NOT NULL, question text NOT NULL,"
                        " embedding vector({d}) NOT NULL, answer text NOT NULL, sources text[] NOT NULL,"
                        " stored_at double precision NOT NULL, last_used double precision NOT NULL)"
                    ).format(t=self._table, d=sql.Literal(self._dimension))
                )
                await conn.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw (embedding vector_cosine_ops)").format(
                        i=self._index, t=self._table
                    )
                )
            self._pool = pool
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def nearest(self, scope: str, embedding: list[float]) -> Optional[tuple[dict, float]]:
        """Return the most similar entry in ``scope``."""
        query = sql.SQL(
            "SELECT id, question, answer, sources, stored_at, 1 - (embedding <=> %(v)s::vector) AS similarity"
            " FROM {t} WHERE scope = %(s)s ORDER BY embedding <=> %(v)s::vector LIMIT 1"
        ).format(t=self._table)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            row = await (await conn.execute(query, {"v": _vector_literal(embedding), "s": scope})).fetchone()
        return (dict(row), float(row["similarity"])) if row else None

    async def upsert(self, scope: str, question: str, embedding: list[float], answer: str, sources: list[str]) -> None:
        """Insert, replacing a near-duplicate question instead of piling up copies."""
        now = time.time()
        match = await self.nearest(scope, embedding)
        entry_id = match[0]["id"] if match and match[1] >= self.similarity_threshold else uuid.uuid4().hex
        query = sql.SQL(
            "INSERT INTO {t} (id, scope, question, embedding, answer, sources, stored_at, last_used)"
            " VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET"
            " question = EXCLUDED.question, embedding = EXCLUDED.embedding, answer = EXCLUDED.answer,"
            " sources = EXCLUDED.sources, stored_at = EXCLUDED.stored_at, last_used = EXCLUDED.last_used"
        ).format(t=self._table)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                query, (entry_id, scope, question, _vector_literal(embedding), answer, sources, now, now)
            )

    async def touch(self, entry_id: str) -> None:
        """Mark an entry as used now."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                sql.SQL("UPDATE {t} SET last_used = %s WHERE id = %s").format(t=self._table), (time.time(), entry_id)
            )

    async def delete_by_sources(self, sources: list[str]) -> int:
        """Delete entries built on any of ``sources``."""
        if not sources:
            return 0
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                sql.SQL("DELETE FROM {t} WHERE sources && %s::text[]").format(t=self._table), (sources,)
            )
            return cur.rowcount

    async def delete_older_than(self, max_age_s: float) -> int:
        """Delete expired entries."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                sql.SQL("DELETE FROM {t} WHERE stored_at < %s").format(t=self._table), (time.time() - max_age_s,)
            )
            return cur.rowcount


class AnswerCache:
    """Policy around an AnswerStore: eligibility, freshness and invalidation."""

    def __init__(
        self,
        store: AnswerStore,
        embed: Callable[[str], Awaitable[list[float]]],
        scope: str,
        similarity_threshold: float = 0.92,
        ttl_s: float = 3600.0,
        max_stale_s: float = 86400.0,
        personal_pattern: str = DEFAULT_PERSONAL_PATTERN,
        clock: Callable[[], float] = time.time,
    ):
        """Configure the cache.

        Args:
            store: Where entries live.
            embed: Async function returning the embedding of a question.
            scope: Version key; entries from another scope are never served.
            similarity_threshold: Minimum cosine similarity for a hit. Keep it high:
                a wrong cached answer is worse than a cache miss.
            ttl_s: Age until which a hit is fresh.
            max_stale_s: Extra age during which a hit may be served if providers fail.
            personal_pattern: Regex marking questions about the asker's own data.
            clock: Wall clock (entries are shared across processes).
        """
        self.store = store
        self._embed = embed
        self.scope = scope
        self.similarity_threshold = similarity_threshold
        self.ttl_s = ttl_s
        self.max_stale_s = max_stale_s
        self._personal = re.compile(personal_pattern, re.IGNORECASE)
        self._clock = clock

    def is_cacheable(self, question: str) -> bool:
        """Personal questions are never cached nor served from cache."""
        return bool(question.strip()) and not self._personal.search(question)

    async def lookup(self, question: str, allow_stale: bool = False) -> Optional[CacheHit]:
        """Return a hit for ``question``; stale hits only when ``allow_stale``."""
        if not self.is_cacheable(question):
            return None
        try:
            embedding = await self._embed(question)
            match = await self.store.nearest(self.scope, embedding)
        except Exception as e:
            logger.warning("answer_cache_lookup_failed", error=str(e))
            return None
        if match is None or match[1] < self.similarity_threshold:
            return None
        entry, similarity = match
        age = self._clock() - float(entry["stored_at"])
        fresh = age <= self.ttl_s
        if age > self.ttl_s + self.max_stale_s or (not fresh and not allow_stale):
            return None
        await self.store.touch(entry["id"])
        return CacheHit(
            answer=entry["answer"], similarity=similarity, age_s=age, fresh=fresh, sources=entry["sources"]
        )

    async def store_answer(self, question: str, answer: str, sources: list[str]) -> bool:
        """Remember an answer built from ``sources``; returns whether it was stored."""
        if not (self.is_cacheable(question) and answer.strip() and sources):
            return False
        try:
            embedding = await self._embed(question)
            await self.store.upsert(self.scope, question, embedding, answer, sorted(set(sources)))
            return True
        except Exception as e:
            logger.warning("answer_cache_store_failed", error=str(e))
            return False

    async def invalidate(self, sources: list[str]) -> int:
        """Drop answers built on any of ``sources`` (call after re-ingesting them)."""
        removed = await self.store.delete_by_sources(sources)
        if removed:
            logger.info("answer_cache_invalidated", sources=len(sources), removed=removed)
        return removed

    async def purge_expired(self) -> int:
        """Delete entries older than ``ttl_s + max_stale_s``."""
        return await self.store.delete_older_than(self.ttl_s + self.max_stale_s)


def scope_key(*parts: str) -> str:
    """Stable scope id from the things that change what a correct answer looks like."""
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()[:16]


_answer_cache: Optional[AnswerCache] = None
_answer_cache_built = False


def build_answer_cache_from_settings() -> Optional[AnswerCache]:
    """Return the configured answer cache, or None when disabled or RAG is off."""
    if not (settings.ANSWER_CACHE_ENABLED and settings.RAG_ENABLED):
        return None
    embedder = get_rag_pipeline().embedder
    store = (
        InMemoryAnswerStore(similarity_threshold=settings.ANSWER_CACHE_THRESHOLD)
        if settings.ANSWER_CACHE_STORE == "memory"
        else PgVectorAnswerStore(
            f"postgresql://{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}",
            table=settings.ANSWER_CACHE_TABLE,
            dimension=settings.RAG_EMBEDDING_DIM,
            similarity_threshold=settings.ANSWER_CACHE_THRESHOLD,
        )
    )
    return AnswerCache(
        store=store,
        embed=embedder.embed_query,
        # A new prompt, model, knowledge table or cache version never serves old answers.
        scope=scope_key(
            _SYSTEM_PROMPT_TEMPLATE,
            settings.DEFAULT_LLM_MODEL,
            settings.RAG_TABLE,
            settings.RAG_EMBEDDING_MODEL,
            settings.ANSWER_CACHE_VERSION,
        ),
        similarity_threshold=settings.ANSWER_CACHE_THRESHOLD,
        ttl_s=settings.ANSWER_CACHE_TTL_S,
        max_stale_s=settings.ANSWER_CACHE_MAX_STALE_S,
    )


def get_answer_cache() -> Optional[AnswerCache]:
    """Return the process-wide answer cache (None when disabled), building it on first use."""
    global _answer_cache, _answer_cache_built
    if not _answer_cache_built:
        _answer_cache = build_answer_cache_from_settings()
        _answer_cache_built = True
    return _answer_cache


def set_answer_cache(cache: Optional[AnswerCache]) -> None:
    """Replace the process-wide answer cache (used by tests)."""
    global _answer_cache, _answer_cache_built
    _answer_cache, _answer_cache_built = cache, True
