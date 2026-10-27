"""Deployment concerns: cost guard, error responses that do not leak internals."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import (
    AIMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.api.v1 import chatbot
from app.api.v1.auth import get_current_session
from app.core.cache import InMemoryCacheService
from app.core.langgraph.graph import LangGraphAgent
from app.main import app
from app.schemas import Message
from app.services.agents import ResilientChatModel
from app.services.cost_guard import (
    BudgetExceededError,
    CostGuard,
    TurnUsage,
    parse_prices,
    set_cost_guard,
)
from app.services.guardrails import (
    Guardrails,
    LLMPolicyRail,
    set_guardrails,
)
from app.services.memory import memory_service
from app.utils.tasks import _background_tasks
from tests.fakes import ScriptedModel


def _reply(text, model, inp, out):
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
        response_metadata={"model_name": model},
    )


def _usage(*calls):
    usage = TurnUsage()
    for call in calls:
        usage.add(*call)
    return usage


def test_cost_prices_each_call_per_model():
    guard = CostGuard(InMemoryCacheService(), 1.0, {"big": (10.0, 30.0)}, default_price=(1.0, 2.0))
    usage = _usage(("big", 100_000, 10_000), ("small", 200_000, 50_000))  # 1.0 + 0.3, then 0.2 + 0.1 by default

    assert guard.cost_of(usage) == pytest.approx(1.6)


def test_dated_model_snapshots_use_the_longest_listed_prefix():
    # Providers answer with "gpt-5.4-mini-2026-01-01"; an exact-name lookup priced every call at $0.
    guard = CostGuard(InMemoryCacheService(), 1.0, {"gpt-5.4": (2.0, 8.0), "gpt-5.4-mini": (0.25, 2.0)})

    assert guard._price("gpt-5.4-mini-2026-01-01") == (0.25, 2.0)
    assert guard._price("gpt-5.4-2026-01-01") == (2.0, 8.0)


@pytest.mark.parametrize("nested_sees_handler", [True, False])
async def test_turn_usage_counts_a_wrapped_reply_once(nested_sees_handler):
    # Inside a graph the provider call under ResilientChatModel reports to the handler too;
    # from a plain script it does not. Either way the reply is charged exactly once.
    usage = TurnUsage()
    inner = ScriptedModel(replies=[_reply("hi", "m", 10, 5)])

    class Service:
        async def ainvoke_with_tools(self, messages, **kwargs):
            config = {"callbacks": [usage]} if nested_sees_handler else None
            return await inner.ainvoke(messages, config=config)

    await ResilientChatModel(service=Service()).ainvoke("q", config={"callbacks": [usage]})

    assert usage.calls == [("m", 10, 5)]


async def test_budget_blocks_after_it_is_spent():
    guard = CostGuard(InMemoryCacheService(), 0.5, {"m": (1.0, 1.0)})

    await guard.check("u1")
    await guard.record("u1", _usage(("m", 300_000, 300_000)))  # $0.60

    with pytest.raises(BudgetExceededError):
        await guard.check("u1")
    await guard.check("u2")  # other users are unaffected


async def test_a_turn_is_charged_for_the_policy_rail_too(monkeypatch):
    guard = CostGuard(InMemoryCacheService(), 10.0, {"rail": (1.0, 0.0), "chat": (2.0, 0.0)})
    set_cost_guard(guard)
    rail_model = ScriptedModel(
        replies=[_reply('{"allowed": true, "category": "ok", "reason": "fine"}', "rail", 100_000, 0)]
    )
    chat_model = ScriptedModel(replies=[_reply("answer", "chat", 100_000, 0)])

    async def classify(messages, schema):
        return schema.model_validate_json((await rail_model.ainvoke(messages)).content)

    async def call(messages, *args, **kwargs):
        return await chat_model.ainvoke(messages)

    async def no_memory(*args, **kwargs):
        return None

    monkeypatch.setattr(memory_service, "search", no_memory)
    monkeypatch.setattr(memory_service, "add", no_memory)
    set_guardrails(Guardrails(llm_rail=LLMPolicyRail("anything", classify)))
    agent = LangGraphAgent()
    monkeypatch.setattr(agent.llm_service, "call", call)
    agent._graph = await agent.create_graph(checkpointer=InMemorySaver())
    try:
        await agent.get_response([Message(role="user", content="hi")], session_id="cost", user_id="u")
        await asyncio.gather(*list(_background_tasks))
        spent = await guard.spent("u")
    finally:
        set_cost_guard(None)
        set_guardrails(None)

    assert spent == pytest.approx(0.1 + 0.2)  # rail call + answer, not just the answer


def test_parse_prices():
    assert parse_prices('{"a": [0.25, 2]}') == {"a": (0.25, 2.0)}
    assert parse_prices("") == {}


async def test_agent_refuses_before_calling_the_model_when_over_budget(monkeypatch):
    guard = CostGuard(InMemoryCacheService(), 0.01, {"m": (1.0, 1.0)})
    await guard.record("u", _usage(("m", 20_000, 0)))
    set_cost_guard(guard)

    async def no_memory(*args, **kwargs):
        return None

    monkeypatch.setattr(memory_service, "search", no_memory)
    agent = LangGraphAgent()
    calls = []

    async def scripted(*args, **kwargs):
        calls.append(1)
        return AIMessage(content="x")

    monkeypatch.setattr(agent.llm_service, "call", scripted)
    agent._graph = await agent.create_graph(checkpointer=InMemorySaver())
    try:
        with pytest.raises(BudgetExceededError):
            await agent.get_response([Message(role="user", content="hi")], session_id="s", user_id="u")
    finally:
        set_cost_guard(None)
    assert calls == []


class _Session:
    id = "sess-1"
    name = "named"
    user_id = 1
    username = "tester"


@pytest.fixture
def client(monkeypatch):
    app.dependency_overrides[get_current_session] = lambda: _Session()
    monkeypatch.setattr(chatbot.settings, "SESSION_NAMING_ENABLED", False)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_internal_errors_are_not_echoed_to_the_client(client, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("connection to postgres://admin:hunter2@db failed")

    monkeypatch.setattr(chatbot.agent, "get_response", boom)
    response = client.post("/api/v1/chatbot/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 500
    assert "hunter2" not in response.text and "postgres" not in response.text


def test_budget_exceeded_maps_to_402(client, monkeypatch):
    async def broke(*args, **kwargs):
        raise BudgetExceededError("monthly budget of $5.00 reached")

    monkeypatch.setattr(chatbot.agent, "get_response", broke)
    response = client.post("/api/v1/chatbot/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 402
    assert "budget" in response.json()["detail"]


def test_api_routes_survive_the_metrics_middleware():
    """Regression: an unmaintained Prometheus middleware crashed every /api route on FastAPI >= 0.16x."""
    response = TestClient(app, raise_server_exceptions=False).post("/api/v1/auth/login", data={})

    assert response.status_code == 422


def test_request_metrics_use_route_templates_not_raw_paths():
    client = TestClient(app, raise_server_exceptions=False)
    for session_id in ("abc123", "zzz999"):
        client.patch(f"/api/v1/auth/session/{session_id}/name", data={"name": "n"})
    client.get("/definitely-not-a-route")

    labels = {
        line.split('endpoint="')[1].split('"')[0]
        for line in client.get("/metrics").text.splitlines()
        if line.startswith("http_requests_total{")
    }

    assert "/api/v1/auth/session/{session_id}/name" in labels
    assert not any("abc123" in label or "zzz999" in label for label in labels)
    assert "unmatched" in labels
