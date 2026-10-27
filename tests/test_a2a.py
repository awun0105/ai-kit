"""A2A: the agent served over A2A 1.0 JSON-RPC, and the remote-agent tool, end to end in process."""

import httpx
import pytest

pytest.importorskip("a2a")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.core.langgraph.graph import LangGraphAgent  # noqa: E402
from app.services.a2a.client import (  # noqa: E402
    ask_remote_agent_raw,
    build_remote_agent_tool,
    parse_remote_agents,
)
from app.services.a2a.server import (  # noqa: E402
    agent_ask,
    build_a2a_app,
    build_agent_card,
    parse_caller_tokens,
)
from app.services.guardrails import (  # noqa: E402
    Guardrails,
    set_guardrails,
)
from app.services.memory import memory_service  # noqa: E402

BASE = "http://a2a.test"
TOKEN = "a2a-secret"


@tool
def refund(order_id: str) -> str:
    """Refund an order."""
    return "refunded"


@pytest.fixture
async def served(monkeypatch):
    async def no_memory(*args, **kwargs):
        return None

    monkeypatch.setattr(memory_service, "search", no_memory)
    monkeypatch.setattr(memory_service, "add", no_memory)
    set_guardrails(Guardrails(approval_tools=frozenset({"refund"})))

    agent = LangGraphAgent()
    agent.tools_by_name = {"refund": refund}
    replies: list[AIMessage] = []

    async def scripted(messages, *args, **kwargs):
        return replies.pop(0)

    monkeypatch.setattr(agent.llm_service, "call", scripted)
    agent._graph = await agent.create_graph(checkpointer=InMemorySaver())

    card = build_agent_card("Kit", "test agent", f"{BASE}/", [{"id": "chat", "name": "Chat", "description": "d"}])
    app = build_a2a_app(card, agent_ask(agent), token=TOKEN)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)
    yield http, replies
    await http.aclose()
    set_guardrails(None)


async def test_card_is_public_but_rpc_needs_the_token(served):
    http, _ = served

    card = await http.get("/.well-known/agent-card.json")
    rpc = await http.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {}})

    assert card.status_code == 200 and card.json()["name"] == "Kit"
    assert rpc.status_code == 401


async def test_remote_agent_answers_over_a2a(served):
    http, replies = served
    replies.append(AIMessage(content="pong"))

    answer = await ask_remote_agent_raw(BASE, "ping", token=TOKEN, http_client=http)

    assert answer == "pong"


async def test_tool_approval_surfaces_as_input_required(served):
    http, replies = served
    replies.append(AIMessage(content="", tool_calls=[{"name": "refund", "args": {"order_id": "A1"}, "id": "r1"}]))

    answer = await ask_remote_agent_raw(BASE, "refund A1", token=TOKEN, http_client=http)

    assert "Approval needed" in answer and "refund" in answer


async def test_each_token_names_its_caller():
    asked = []

    async def ask(caller, context_id, text):
        asked.append(caller)
        return "ok", False

    card = build_agent_card("Kit", "d", f"{BASE}/", [{"id": "chat", "name": "Chat", "description": "d"}])
    app = build_a2a_app(card, ask, callers={"billing": "tok-billing", "crm": "tok-crm"})
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)
    try:
        wrong = await http.post(
            "/",
            json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {}},
            headers={"Authorization": "Bearer nope"},
        )
        answer = await ask_remote_agent_raw(BASE, "hi", token="tok-crm", http_client=http)  # closes the client
    finally:
        await http.aclose()

    assert answer == "ok" and asked == ["crm"]
    assert wrong.status_code == 401


async def test_callers_reusing_a_context_id_get_separate_conversations(monkeypatch):
    # The context id is chosen by the remote side; as the whole thread id it let one caller
    # continue (and read) another caller's conversation.
    async def no_memory(*args, **kwargs):
        return None

    monkeypatch.setattr(memory_service, "search", no_memory)
    monkeypatch.setattr(memory_service, "add", no_memory)
    agent = LangGraphAgent()
    replies = [AIMessage(content="noted for billing"), AIMessage(content="noted for crm")]

    async def scripted(messages, *args, **kwargs):
        return replies.pop(0)

    monkeypatch.setattr(agent.llm_service, "call", scripted)
    agent._graph = await agent.create_graph(checkpointer=InMemorySaver())
    ask = agent_ask(agent)

    await ask("billing", "ctx-1", "invoice 42 is disputed")
    await ask("crm", "ctx-1", "hello")

    crm_history = [m.content for m in await agent.get_chat_history("a2a-crm-ctx-1")]
    assert "invoice 42 is disputed" not in crm_history
    assert crm_history == ["hello", "noted for crm"]


def test_caller_tokens_are_validated():
    assert parse_caller_tokens(["billing=t1", "crm=t2"]) == {"billing": "t1", "crm": "t2"}
    with pytest.raises(ValueError):
        parse_caller_tokens(["billing"])
    with pytest.raises(ValueError):
        parse_caller_tokens(["a=same", "b=same"])


def test_remote_agent_config_is_validated():
    assert parse_remote_agents(["billing=https://b.example/a2a/"]) == {"billing": "https://b.example/a2a"}
    with pytest.raises(ValueError):
        parse_remote_agents(["billing"])
    with pytest.raises(ValueError):
        parse_remote_agents(["billing=ftp://b.example"])


async def test_remote_agent_tool_rejects_unknown_names():
    remote = build_remote_agent_tool({"billing": "https://b.example/a2a"})

    assert "billing" in remote.description
    assert (await remote.ainvoke({"agent": "evil", "message": "hi"})).startswith("Unknown agent")


def test_a2a_server_refuses_to_start_without_a_token():
    card = build_agent_card("Kit", "d", f"{BASE}/", [])
    with pytest.raises(ValueError):
        build_a2a_app(card, agent_ask(None), token="")
