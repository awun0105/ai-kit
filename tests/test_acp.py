"""ACP: the kit's agent served to an ACP client, over real pipes and as a real subprocess."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("acp")

from acp import (  # noqa: E402
    PROTOCOL_VERSION,
    RequestError,
    connect_to_agent,
    run_agent,
    text_block,
)
from acp.schema import (  # noqa: E402
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.acp_agent import KitBackend  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.kit import set_guardrails  # noqa: E402
from app.core.langgraph.graph import LangGraphAgent  # noqa: E402
from app.core.langgraph.tools.ask_human import ask_human  # noqa: E402
from app.services.acp import (  # noqa: E402
    KitAcpAgent,
    Pending,
    TurnResult,
)
from app.services.acp.agent import FAILED_REPLY  # noqa: E402
from app.services.guardrails import Guardrails  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class FakeClient:
    """ACP client side: records messages and answers permission requests with ``answer``."""

    def __init__(self, answer: str | None = "allow"):
        self.answer = answer
        self.messages: list[str] = []
        self.permissions: list[str] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if getattr(update, "session_update", None) == "agent_message_chunk":
            self.messages.append(update.content.text)

    async def request_permission(self, session_id: str, tool_call: Any, options: list, **kwargs: Any):
        self.permissions.append(tool_call.title)
        if self.answer is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=self.answer))


class ScriptedBackend:
    """Agent backend that replays turn results and records what it was sent."""

    def __init__(self, results: list[Any]):
        self.results = list(results)
        self.sent: list[tuple[str, str]] = []

    async def turn(self, session_id: str, text: str) -> TurnResult:
        self.sent.append((session_id, text))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


async def _pipe() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    read_fd, write_fd = os.pipe()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(read_fd, "rb", 0))
    transport, protocol = await loop.connect_write_pipe(
        lambda: asyncio.StreamReaderProtocol(asyncio.StreamReader()), os.fdopen(write_fd, "wb", 0)
    )
    return reader, asyncio.StreamWriter(transport, protocol, None, loop)


async def _connect(agent: KitAcpAgent, client: FakeClient):
    to_agent_reader, to_agent_writer = await _pipe()
    to_client_reader, to_client_writer = await _pipe()
    server = asyncio.create_task(run_agent(agent, input_stream=to_client_writer, output_stream=to_agent_reader))
    conn = connect_to_agent(client, input_stream=to_agent_writer, output_stream=to_client_reader)
    await conn.initialize(protocol_version=PROTOCOL_VERSION)
    session = await conn.new_session(cwd=str(ROOT), mcp_servers=[])
    return conn, session.session_id, server


async def _close(conn, server) -> None:
    await conn.close()
    server.cancel()


async def test_a_prompt_is_answered_in_the_session():
    backend = ScriptedBackend([TurnResult(reply="Xin chào!")])
    client = FakeClient()
    conn, session_id, server = await _connect(KitAcpAgent(backend), client)

    response = await conn.prompt(session_id=session_id, prompt=[text_block("Chào bạn")])
    await _close(conn, server)

    assert response.stop_reason == "end_turn"
    assert client.messages == ["Xin chào!"]
    assert backend.sent == [(session_id, "Chào bạn")]


@pytest.mark.parametrize(("answer", "sent"), [("allow", "yes"), ("reject", "no"), (None, "no")])
async def test_an_approval_becomes_a_permission_request(answer, sent):
    backend = ScriptedBackend(
        [TurnResult(reply="", pending=Pending(kind="approval", text="Approve refund?")), TurnResult(reply="Done.")]
    )
    client = FakeClient(answer)
    conn, session_id, server = await _connect(KitAcpAgent(backend), client)

    await conn.prompt(session_id=session_id, prompt=[text_block("Refund order 7")])
    await _close(conn, server)

    assert client.permissions == ["Approve refund?"]
    assert backend.sent[1] == (session_id, sent)
    assert client.messages == ["Done."]


async def test_a_question_ends_the_turn_and_the_next_prompt_answers_it():
    backend = ScriptedBackend(
        [
            TurnResult(reply="", pending=Pending(kind="question", text="Which order?")),
            TurnResult(reply="Order 7 ships Monday."),
        ]
    )
    client = FakeClient()
    conn, session_id, server = await _connect(KitAcpAgent(backend), client)

    first = await conn.prompt(session_id=session_id, prompt=[text_block("When does my order ship?")])
    await conn.prompt(session_id=session_id, prompt=[text_block("Order 7")])
    await _close(conn, server)

    assert first.stop_reason == "end_turn"
    assert client.permissions == []
    assert client.messages == ["Which order?", "Order 7 ships Monday."]
    assert backend.sent[1] == (session_id, "Order 7")


async def test_failures_and_unknown_sessions():
    backend = ScriptedBackend([RuntimeError("provider down")])
    client = FakeClient()
    conn, session_id, server = await _connect(KitAcpAgent(backend), client)

    response = await conn.prompt(session_id=session_id, prompt=[text_block("hi")])
    with pytest.raises(RequestError):
        await conn.prompt(session_id="not-a-session", prompt=[text_block("hi")])
    await _close(conn, server)

    assert response.stop_reason == "end_turn"
    assert client.messages == [FAILED_REPLY]  # never the exception text


@tool
def refund_order(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


async def _kit_backend(monkeypatch, replies: list[AIMessage], tools: dict) -> KitBackend:
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_ENABLED", False)
    agent = LangGraphAgent()
    agent.tools_by_name = tools

    async def scripted(messages, *args, **kwargs):
        return replies.pop(0)

    monkeypatch.setattr(agent.llm_service, "call", scripted)
    await agent.create_graph(checkpointer=InMemorySaver())
    return KitBackend(agent)


async def test_the_kit_agent_tells_an_approval_from_a_question(monkeypatch):
    set_guardrails(Guardrails(approval_tools=frozenset({"refund_order"})))
    refund = AIMessage(content="", tool_calls=[{"name": "refund_order", "args": {"order_id": "7"}, "id": "r"}])
    ask = AIMessage(content="", tool_calls=[{"name": "ask_human", "args": {"question": "Which order?"}, "id": "h"}])
    try:
        backend = await _kit_backend(
            monkeypatch,
            [refund, AIMessage(content="Refunded."), ask, AIMessage(content="Thanks.")],
            {"refund_order": refund_order, "ask_human": ask_human},
        )
        approval = await backend.turn("s1", "Refund order 7")
        approved = await backend.turn("s1", "yes")
        question = await backend.turn("s2", "Help me")
        answered = await backend.turn("s2", "Order 7")
    finally:
        set_guardrails(None)

    assert approval.pending is not None and approval.pending.kind == "approval"
    assert "refund_order" in approval.pending.text
    assert approved.pending is None and approved.reply == "Refunded."
    assert question.pending == Pending(kind="question", text="Which order?")
    assert answered.reply == "Thanks."


async def test_the_entry_point_keeps_stdout_for_the_protocol(tmp_path):
    env = {
        **os.environ,
        "APP_ENV": "test",
        "LANGFUSE_TRACING_ENABLED": "false",
        "CHECKPOINT_BACKEND": "sqlite",
        "CHECKPOINT_SQLITE_PATH": str(tmp_path / "c.sqlite"),
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.acp_agent",
        cwd=ROOT,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSION}}
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write((json.dumps(request) + "\n").encode())
    await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=60)
    proc.stdin.close()
    rest, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

    reply = json.loads(line)  # the first stdout line is the reply: import-time logs went to stderr
    assert reply["id"] == 1 and reply["result"]["agentInfo"]["name"] == settings.PROJECT_NAME
    assert all(json.loads(extra) for extra in rest.decode().splitlines() if extra.strip())
    assert b"logging_initialized" in stderr
