"""Answer cache: freshness windows, scoping, invalidation, and its use inside the agent (D25 part G)."""

import os
import uuid

import pytest
from langchain_core.messages import (
    AIMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.core.kit import (
    set_answer_cache,
    set_rag_pipeline,
)
from app.core.langgraph.graph import LangGraphAgent
from app.schemas import Message
from app.services.memory import memory_service
from app.services.rag import (
    Document,
    HashingEmbedder,
    InMemoryVectorStore,
    RAGPipeline,
)
from app.services.reliability import (
    AllProvidersFailedError,
    AnswerCache,
    InMemoryAnswerStore,
    PgVectorAnswerStore,
)

EMBEDDER = HashingEmbedder(128)


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def _cache(clock, scope="s1", **kw):
    store = InMemoryAnswerStore(clock=clock, **{k: v for k, v in kw.items() if k in ("max_entries",)})
    return AnswerCache(
        store=store,
        embed=EMBEDDER.embed_query,
        scope=scope,
        shared=True,
        ttl_s=60,
        max_stale_s=600,
        clock=clock,
        **{k: v for k, v in kw.items() if k not in ("max_entries",)},
    )


# ---------------------------------------------------------------- cache policy


async def test_fresh_then_stale_then_expired():
    clock = Clock()
    cache = _cache(clock)
    await cache.store_answer("How many leave days?", "15 days [1]", ["leave.md"])

    assert (await cache.lookup("How many leave days?")).fresh
    clock.now += 120
    assert await cache.lookup("How many leave days?") is None  # stale needs allow_stale
    assert not (await cache.lookup("How many leave days?", allow_stale=True)).fresh
    clock.now += 600
    assert await cache.lookup("How many leave days?", allow_stale=True) is None


async def test_scope_isolation_and_threshold():
    clock = Clock()
    store = InMemoryAnswerStore(clock=clock)
    old = AnswerCache(store=store, embed=EMBEDDER.embed_query, scope="prompt-v1", shared=True, clock=clock)
    new = AnswerCache(store=store, embed=EMBEDDER.embed_query, scope="prompt-v2", shared=True, clock=clock)
    await old.store_answer("VPN policy for remote work?", "Use WireGuard [1]", ["vpn.md"])

    assert await new.lookup("VPN policy for remote work?") is None
    assert await old.lookup("What is the refund window?") is None


async def test_personal_questions_and_unsourced_answers_are_never_cached():
    cache = _cache(Clock())

    assert not await cache.store_answer("What is my account balance?", "1000", ["x.md"])
    assert not await cache.store_answer("Số dư tài khoản của tôi là bao nhiêu?", "1000", ["x.md"])
    assert not await cache.store_answer("Generic question", "answer", [])


async def test_same_question_replaces_instead_of_piling_up_and_lru_eviction():
    clock = Clock()
    cache = _cache(clock, max_entries=2)
    await cache.store_answer("q one", "a1", ["d1"])
    await cache.store_answer("q one", "a1 updated", ["d1"])
    assert len(cache.store.entries) == 1

    clock.now += 1
    await cache.store_answer("second topic entirely", "a2", ["d2"])
    clock.now += 1
    await cache.lookup("q one")  # touch: q one is now more recent than the second
    clock.now += 1
    await cache.store_answer("third subject here", "a3", ["d3"])

    assert {e["answer"] for e in cache.store.entries.values()} == {"a1 updated", "a3"}


async def test_per_user_cache_keeps_users_apart_and_needs_the_user():
    clock = Clock()
    cache = AnswerCache(
        store=InMemoryAnswerStore(clock=clock), embed=EMBEDDER.embed_query, scope="s", shared=False, clock=clock
    )
    await cache.store_answer("leave days", "15, Quân", ["leave.md"], user_id="1")

    assert (await cache.lookup("leave days", user_id="1")).answer == "15, Quân"
    assert await cache.lookup("leave days", user_id="2") is None
    with pytest.raises(ValueError, match="user_id"):
        await cache.lookup("leave days")


async def test_invalidation_by_source():
    cache = _cache(Clock())
    await cache.store_answer("leave days", "15", ["leave.md"])
    await cache.store_answer("vpn rules", "wireguard", ["vpn.md"])

    assert await cache.invalidate(["leave.md"]) == 1
    assert await cache.lookup("leave days") is None
    assert await cache.lookup("vpn rules") is not None


async def test_ingest_reports_changed_documents_for_invalidation():
    pipeline = RAGPipeline(embedder=EMBEDDER, store=InMemoryVectorStore())
    doc = Document(id="leave.md", title="Leave", source="leave.md", content="Employees get 15 leave days.")

    first = await pipeline.ingest([doc])
    unchanged = await pipeline.ingest([doc])
    edited = await pipeline.ingest([doc.model_copy(update={"content": "Employees get 18 leave days."})])

    assert first.changed_documents == ["leave.md"]
    assert unchanged.changed_documents == []
    assert edited.changed_documents == ["leave.md"]


# ---------------------------------------------------------------- inside the agent


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def __call__(self, messages, *args, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
async def rag_and_cache(monkeypatch):
    pipeline = RAGPipeline(embedder=EMBEDDER, store=InMemoryVectorStore(), score_threshold=0.1)
    await pipeline.ingest(
        [Document(id="leave.md", title="Leave policy", source="leave.md", content="Employees get 15 leave days.")]
    )
    clock = Clock()
    cache = _cache(clock)
    set_rag_pipeline(pipeline)
    set_answer_cache(cache)

    async def no_memory(*args, **kwargs):
        return None

    monkeypatch.setattr(memory_service, "search", no_memory)
    monkeypatch.setattr(memory_service, "add", no_memory)
    yield cache, clock
    set_rag_pipeline(None)
    set_answer_cache(None)


async def _agent(monkeypatch, replies):
    from app.core.langgraph.tools.knowledge_search import knowledge_search

    agent = LangGraphAgent()
    agent.tools_by_name = {"knowledge_search": knowledge_search}
    llm = ScriptedLLM(replies)
    monkeypatch.setattr(agent.llm_service, "call", llm)
    agent._graph = await agent.create_graph(checkpointer=InMemorySaver())
    return agent, llm


def _rag_turn(answer):
    return [
        AIMessage(content="", tool_calls=[{"name": "knowledge_search", "args": {"query": "leave days"}, "id": "k1"}]),
        AIMessage(content=answer),
    ]


async def _drain_background_tasks():
    import asyncio

    for _ in range(5):
        await asyncio.sleep(0)


async def test_answer_is_cached_then_served_without_the_model(monkeypatch, rag_and_cache):
    cache, _ = rag_and_cache
    agent, llm = await _agent(monkeypatch, _rag_turn("Employees get 15 leave days [1]."))
    question = [Message(role="user", content="How many leave days do employees get?")]

    first = await agent.get_response(question, session_id="s-a")
    await _drain_background_tasks()
    second = await agent.get_response(question, session_id="s-b")

    assert first[-1].content == second[-1].content == "Employees get 15 leave days [1]."
    assert llm.calls == 2  # both from the first conversation; the second used the cache
    history = await agent.get_chat_history("s-b")
    assert [m.role for m in history] == ["user", "assistant"]


async def test_a_cached_answer_is_built_without_the_users_name_or_memory(monkeypatch, rag_and_cache):
    # The first asker's name and memories went into the prompt, so the cached answer could greet
    # them by name or repeat their facts to everyone who asked the same question afterwards.
    agent, llm = await _agent(monkeypatch, _rag_turn("Employees get 15 leave days [1]."))
    seen = []
    scripted = agent.llm_service.call

    async def recording(messages, *args, **kwargs):
        seen.append(messages)
        return await scripted(messages, *args, **kwargs)

    async def memories(user_id, query):
        return "* User has a dog called Mít"

    monkeypatch.setattr(agent.llm_service, "call", recording)
    monkeypatch.setattr(memory_service, "search", memories)

    await agent.get_response(
        [Message(role="user", content="How many leave days do employees get?")],
        session_id="s-n",
        user_id="1",
        username="Quân",
    )

    system_prompts = [m[0]["content"] for m in seen]
    assert system_prompts and all("Quân" not in s and "Mít" not in s for s in system_prompts)


async def test_turns_outside_the_cache_stay_personalised(monkeypatch, rag_and_cache):
    set_answer_cache(None)
    agent, _ = await _agent(monkeypatch, [AIMessage(content="Hi!")])
    seen = []
    scripted = agent.llm_service.call

    async def recording(messages, *args, **kwargs):
        seen.append(messages)
        return await scripted(messages, *args, **kwargs)

    monkeypatch.setattr(agent.llm_service, "call", recording)

    await agent.get_response([Message(role="user", content="hello")], session_id="s-p", user_id="1", username="Quân")

    assert "Quân" in seen[0][0]["content"]


async def test_stale_answer_is_served_when_all_providers_fail(monkeypatch, rag_and_cache):
    cache, clock = rag_and_cache
    await cache.store_answer("How many leave days do employees get?", "15 days [1].", ["leave.md"])
    clock.now += 300  # past the 60 s TTL, within the 600 s stale window
    agent, _ = await _agent(monkeypatch, [AllProvidersFailedError("down", None, None)])

    reply = await agent.get_response(
        [Message(role="user", content="How many leave days do employees get?")], session_id="s-c"
    )

    assert reply[-1].content == "15 days [1]."


async def test_failed_turn_does_not_swallow_the_next_question(monkeypatch, rag_and_cache):
    set_answer_cache(None)
    agent, llm = await _agent(
        monkeypatch, [AllProvidersFailedError("down", None, None), AIMessage(content="hello again")]
    )

    with pytest.raises(AllProvidersFailedError):
        await agent.get_response([Message(role="user", content="first try")], session_id="s-d")
    reply = await agent.get_response([Message(role="user", content="second question")], session_id="s-d")

    assert reply[-1].content == "hello again"
    history = await agent.get_chat_history("s-d")
    assert "second question" in [m.content for m in history]


# ---------------------------------------------------------------- Postgres store

PG_URL = os.getenv("RAG_TEST_PG_URL")


@pytest.mark.skipif(not PG_URL, reason="RAG_TEST_PG_URL not set")
async def test_pgvector_answer_store_round_trip():
    table = f"answer_cache_test_{uuid.uuid4().hex[:8]}"
    store = PgVectorAnswerStore(PG_URL, table=table, dimension=128)
    cache = AnswerCache(store=store, embed=EMBEDDER.embed_query, scope="s", shared=True)
    try:
        assert await cache.store_answer("leave days", "15", ["leave.md"])
        assert await cache.store_answer("leave days", "15 (updated)", ["leave.md"])
        hit = await cache.lookup("leave days")
        removed = await cache.invalidate(["leave.md"])
        after = await cache.lookup("leave days")
    finally:
        pool = await store._get_pool()
        async with pool.connection() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await store.close()

    assert hit and hit.answer == "15 (updated)" and hit.fresh
    assert removed == 1 and after is None
