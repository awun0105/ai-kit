"""Deployment concerns: cost guard, error responses that do not leak internals."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.api.v1 import chatbot
from app.api.v1.auth import get_current_session
from app.core.cache import InMemoryCacheService
from app.core.langgraph.graph import LangGraphAgent
from app.main import app
from app.schemas import Message
from app.services.cost_guard import (
    BudgetExceededError,
    CostGuard,
    parse_prices,
    set_cost_guard,
)
from app.services.memory import memory_service


def _reply(text, model, inp, out):
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
        response_metadata={"model_name": model},
    )


def test_cost_of_turn_counts_only_this_turn_and_prices_per_model():
    guard = CostGuard(InMemoryCacheService(), 1.0, {"big": (10.0, 30.0)}, default_price=(1.0, 2.0))
    messages = [
        HumanMessage(content="old question"),
        _reply("old answer", "big", 1_000_000, 0),  # previous turn: not counted
        HumanMessage(content="new question"),
        _reply("", "big", 100_000, 10_000),  # 1.0 + 0.3
        _reply("answer", "small", 200_000, 50_000),  # 0.2 + 0.1 at the default price
    ]

    assert guard.cost_of_turn(messages) == pytest.approx(1.6)


async def test_budget_blocks_after_it_is_spent():
    guard = CostGuard(InMemoryCacheService(), 0.5, {"m": (1.0, 1.0)})
    turn = [HumanMessage(content="q"), _reply("a", "m", 300_000, 300_000)]  # $0.60

    await guard.check("u1")
    await guard.record("u1", turn)

    with pytest.raises(BudgetExceededError):
        await guard.check("u1")
    await guard.check("u2")  # other users are unaffected


def test_parse_prices():
    assert parse_prices('{"a": [0.25, 2]}') == {"a": (0.25, 2.0)}
    assert parse_prices("") == {}


async def test_agent_refuses_before_calling_the_model_when_over_budget(monkeypatch):
    guard = CostGuard(InMemoryCacheService(), 0.01, {"m": (1.0, 1.0)})
    await guard.record("u", [HumanMessage(content="q"), _reply("a", "m", 20_000, 0)])
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
