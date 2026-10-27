"""Running on the user's machine: SQLite knowledge base and checkpoints, in-process embeddings, local models."""

import sys
import types

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from app.core import kit
from app.core.config import settings
from app.core.langgraph.graph import LangGraphAgent
from app.services.llm.registry import parse_model_configs
from app.services.memory import memory_service
from app.services.rag import (
    FastEmbedEmbedder,
    HashingEmbedder,
    RAGPipeline,
    SQLiteVectorStore,
)
from tests.test_rag import DOCS


def _pipeline(path) -> RAGPipeline:
    return RAGPipeline(embedder=HashingEmbedder(128), store=SQLiteVectorStore(path, 128), score_threshold=0.2)


async def test_sqlite_knowledge_base_survives_a_restart(tmp_path):
    path = tmp_path / "kb.sqlite"
    first = _pipeline(path)
    await first.ingest(DOCS)
    first.store.close()  # type: ignore[attr-defined]

    reopened = _pipeline(path)
    hit = await reopened.retrieve("how many days do I have to request a refund", top_k=2)
    chunks = await reopened.store.all_chunks()

    assert hit.results[0].chunk.document_id == "policy/refund.md"
    assert all(c.lexical_tokens for c in chunks)  # BM25 tokens are stored, not recomputed
    assert next(c for c in chunks if c.document_id == "news/launch.md").url == "https://example.com/launch"


async def test_sqlite_store_skips_unchanged_chunks_and_prunes(tmp_path):
    pipeline = _pipeline(tmp_path / "kb.sqlite")
    await pipeline.ingest(DOCS)

    again = await pipeline.ingest(DOCS)
    pruned = await pipeline.ingest(DOCS[:2], prune_missing=True)

    assert again.changed_documents == []
    assert pruned.removed_documents == ["news/launch.md"]
    assert {c.document_id for c in await pipeline.store.all_chunks()} == {"policy/refund.md", "policy/warranty.md"}


async def test_sqlite_store_rejects_vectors_of_the_wrong_size(tmp_path):
    store = SQLiteVectorStore(tmp_path / "kb.sqlite", 4)
    pipeline = RAGPipeline(embedder=HashingEmbedder(8), store=store)

    with pytest.raises(ValueError, match="dimensions"):
        await pipeline.ingest(DOCS[:1])


class _FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding: no model download in tests."""

    def __init__(self, model_name, cache_dir=None):
        self.model_name, self.cache_dir = model_name, cache_dir

    def passage_embed(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def query_embed(self, text):
        return [[float(len(text)), 0.0, 1.0]]


async def test_fastembed_embedder_runs_the_model_in_process(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding))

    embedder = FastEmbedEmbedder("some/model", 3, cache_dir="/models")

    assert embedder._model.cache_dir == "/models"
    assert await embedder.embed_documents(["ab", "abcd"]) == [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]]
    assert await embedder.embed_query("abc") == [3.0, 0.0, 1.0]
    assert await embedder.embed_documents([]) == []
    with pytest.raises(ValueError, match="dimensions"):
        await FastEmbedEmbedder("some/model", 8).embed_query("x")


def test_local_models_come_from_settings():
    configs = parse_model_configs(
        '[{"name": "local", "model": "openai:qwen3", "kwargs": {"base_url": "http://127.0.0.1:8080/v1", "api_key": "none"}}]'
    )

    assert configs[0]["kwargs"]["base_url"] == "http://127.0.0.1:8080/v1"
    for bad in ("[]", "{}", '[{"name": "a"}]', '[{"name": "a", "model": "m"}, {"name": "a", "model": "n"}]'):
        with pytest.raises(ValueError):
            parse_model_configs(bad)


async def test_memory_is_skipped_when_turned_off(monkeypatch):
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_ENABLED", False)

    async def boom():
        raise AssertionError("mem0 must not be called")

    monkeypatch.setattr(memory_service, "_get_memory", boom)

    assert await memory_service.search("u1", "anything") == ""
    await memory_service.add("u1", [HumanMessage(content="remember this")])


async def test_conversations_persist_in_a_sqlite_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    monkeypatch.setattr(settings, "CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setattr(settings, "CHECKPOINT_SQLITE_PATH", str(tmp_path / "state" / "checkpoints.sqlite"))
    seen: list = []

    async def scripted(messages, *args, **kwargs):
        seen.append(messages)
        return AIMessage(content=f"reply {len(seen)}")

    config = {"configurable": {"thread_id": "local-1"}}
    agent = LangGraphAgent()
    agent.tools_by_name = {}
    monkeypatch.setattr(agent.llm_service, "call", scripted)
    graph = await agent.create_graph()
    await graph.ainvoke({"messages": [HumanMessage(content="first")]}, config)
    await graph.checkpointer.conn.close()  # type: ignore[union-attr]

    # A new process: a fresh agent and a fresh connection to the same file.
    again = LangGraphAgent()
    again.tools_by_name = {}
    monkeypatch.setattr(again.llm_service, "call", scripted)
    graph = await again.create_graph()
    await graph.ainvoke({"messages": [HumanMessage(content="second")]}, config)
    await graph.checkpointer.conn.close()  # type: ignore[union-attr]

    assert [m["content"] for m in seen[-1][1:]] == ["first", "reply 1", "second"]  # [0] is the system prompt


async def test_unknown_checkpoint_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "CHECKPOINT_BACKEND", "mongo")

    with pytest.raises(ValueError, match="CHECKPOINT_BACKEND"):
        await kit.build_checkpointer_from_settings()
