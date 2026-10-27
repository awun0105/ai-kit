"""Serve an agent over the Agent Client Protocol (ACP), so any ACP client can embed it.

ACP clients are the apps a user works in (Zed, JetBrains IDEs, fframes Studio, or
your own app in C++ or Rust): they start the agent as a subprocess and speak
JSON-RPC over its stdin and stdout. This module is the agent side, on the official
SDK (``agent-client-protocol``, the optional ``acp`` extra; not IBM's ``acp-sdk``,
a different protocol).

The agent behind it is any ``AgentBackend``: one user message in, the reply out,
plus what the agent is waiting for. A tool that needs approval becomes an ACP
permission request answered by the client's user; a question the agent asks the
user (``ask_human``) ends the turn, and the user's next prompt answers it.
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    Optional,
    Protocol,
)

from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
    update_agent_message_text,
)
from acp.schema import (
    AgentCapabilities,
    AllowedOutcome,
    Implementation,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptCapabilities,
    PromptResponse,
    ToolCallUpdate,
)
import structlog

logger = structlog.get_logger()

APPROVE = "yes"
DECLINE = "no"
FAILED_REPLY = "The agent could not answer."


@dataclass
class Pending:
    """What the agent waits for at the end of a turn."""

    kind: Literal["approval", "question"]
    text: str


@dataclass
class TurnResult:
    """The agent's reply to one message, and what it waits for, if anything."""

    reply: str
    pending: Optional[Pending] = None


class AgentBackend(Protocol):
    """Runs one message through the agent; ``session_id`` keeps the conversation."""

    async def turn(self, session_id: str, text: str) -> TurnResult:
        """Send ``text`` (a new message, or the answer to what the agent was waiting for)."""
        ...


def _prompt_text(blocks: list[Any]) -> str:
    """Text of a prompt: text blocks and the text of embedded resources (files the user attached)."""
    parts = []
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(block.text)
        elif kind == "resource" and hasattr(block.resource, "text"):
            parts.append(f"<file uri={block.resource.uri!r}>\n{block.resource.text}\n</file>")
    return "\n\n".join(p for p in parts if p)


class KitAcpAgent:
    """ACP agent: sessions, prompts, permission requests and cancellation."""

    def __init__(self, backend: AgentBackend, name: str = "ai-kit-agent", version: str = "0.1.0"):
        """Serve ``backend`` under the given implementation ``name`` and ``version``."""
        self._backend = backend
        self._info = Implementation(name=name, version=version)
        self._conn: Optional[Client] = None
        self._sessions: dict[str, str] = {}  # ACP session id -> working directory
        self._cancelled: set[str] = set()

    def on_connect(self, conn: Client) -> None:
        """Keep the connection to send updates and permission requests to the client."""
        self._conn = conn

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        """Agree on the protocol version and say what this agent supports."""
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=False, prompt_capabilities=PromptCapabilities(embedded_context=True)
            ),
            agent_info=self._info,
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        """Start a conversation. MCP servers offered by the client are not used yet."""
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = cwd
        logger.info("acp_session_started", session_id=session_id)
        return NewSessionResponse(session_id=session_id)

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        """No authentication: the client started this process, and credentials live in its environment."""
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Stop the session's turn at its next step (a graph run already in flight finishes first)."""
        self._cancelled.add(session_id)

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        """Answer one prompt, asking the client's user for every approval the agent needs."""
        if session_id not in self._sessions:
            raise RequestError.invalid_params({"sessionId": "unknown session"})
        self._cancelled.discard(session_id)
        try:
            result = await self._backend.turn(session_id, _prompt_text(prompt))
            while result.pending is not None and result.pending.kind == "approval":
                if session_id in self._cancelled:
                    return PromptResponse(stop_reason="cancelled")
                approved = await self._ask_permission(session_id, result.pending.text)
                result = await self._backend.turn(session_id, APPROVE if approved else DECLINE)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("acp_prompt_failed", session_id=session_id, error=str(e))
            await self._say(session_id, FAILED_REPLY)
            return PromptResponse(stop_reason="end_turn")
        if session_id in self._cancelled:
            return PromptResponse(stop_reason="cancelled")
        text = result.pending.text if result.pending is not None else result.reply
        await self._say(session_id, text)
        return PromptResponse(stop_reason="end_turn")

    async def _ask_permission(self, session_id: str, question: str) -> bool:
        if self._conn is None:
            return False
        response = await self._conn.request_permission(
            session_id=session_id,
            tool_call=ToolCallUpdate(tool_call_id=uuid.uuid4().hex, title=question, kind="execute", status="pending"),
            options=[
                PermissionOption(option_id="allow", name="Allow", kind="allow_once"),
                PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
            ],
        )
        outcome = response.outcome
        approved = isinstance(outcome, AllowedOutcome) and outcome.option_id == "allow"
        logger.info("acp_permission_answered", session_id=session_id, approved=approved)
        return approved

    async def _say(self, session_id: str, text: str) -> None:
        if self._conn is not None and text:
            await self._conn.session_update(session_id=session_id, update=update_agent_message_text(text))
