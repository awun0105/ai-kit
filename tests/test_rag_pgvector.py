"""PgVectorStore against a real Postgres with pgvector.

Skipped unless RAG_TEST_PG_URL points at a disposable database, e.g.:
    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=test pgvector/pgvector:pg16
    RAG_TEST_PG_URL=postgresql://postgres:test@localhost:55432/postgres uv run pytest tests/test_rag_pgvector.py
"""

import os
import uuid

import pytest

from app.services.rag import (
    HashingEmbedder,
    PgVectorStore,
    RAGPipeline,
)
from tests.test_rag import DOCS

PG_URL = os.getenv("RAG_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="RAG_TEST_PG_URL not set")


async def test_pgvector_round_trip():
    table = f"rag_test_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(PG_URL, table=table, dimension=64)
    pipeline = RAGPipeline(embedder=HashingEmbedder(64), store=store, score_threshold=0.2)
    try:
        first = await pipeline.ingest(DOCS)
        second = await pipeline.ingest(DOCS)
        result = await pipeline.retrieve("laptop warranty months", top_k=2)
        chunks = await store.all_chunks()
    finally:
        pool = await store._get_pool()
        async with pool.connection() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await store.close()

    assert first.embedded == 3 and second.embedded == 0
    assert len(chunks) == 3
    assert all(c.lexical_tokens and c.lexical_mode == "auto" for c in chunks)
    assert result.results[0].chunk.document_id == "policy/warranty.md"
    assert 0.0 < result.best_dense_score <= 1.0
