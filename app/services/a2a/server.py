"""Expose the agent over the Agent2Agent (A2A) protocol, version 1.0, JSON-RPC binding.

Built on the official ``a2a-sdk``. The A2A ``context_id`` becomes the LangGraph
thread id, so a remote agent gets a multi-turn conversation with memory, and a
paused tool approval maps to the A2A ``input-required`` state: the remote side
answers "yes"/"no" in the same context to resume.
"""

import hmac
from typing import (
    Any,
    Awaitable,
    Callable,
    override,
)

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
)
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import (
    InMemoryTaskStore,
    TaskUpdater,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.types import (
    ASGIApp,
    Receive,
    Scope,
    Send,
)

from app.core.logging import logger
from app.schemas import Message

AskAgent = Callable[[str, str], Awaitable[tuple[str, bool]]]
"""(session_id, text) -> (reply, waiting_for_human)."""


class KitAgentExecutor(AgentExecutor):
    """Runs one A2A message through the agent graph."""

    def __init__(self, ask: AskAgent):
        """``ask`` sends a user message to the agent and returns its reply."""
        self._ask = ask

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Answer the message; pause as input-required when the agent waits for approval."""
        message = context.message
        if message is None:
            raise ValueError("A2A request without a message")
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        await updater.update_status(state=TaskState.TASK_STATE_WORKING)

        text = get_message_text(message) or ""
        try:
            reply, waiting = await self._ask(f"a2a-{task.context_id}", text)
        except Exception as e:
            logger.exception("a2a_execute_failed", context_id=task.context_id, error=str(e))
            await updater.update_status(
                state=TaskState.TASK_STATE_FAILED, message=new_text_message("The agent could not answer.")
            )
            return

        if waiting:
            await updater.update_status(state=TaskState.TASK_STATE_INPUT_REQUIRED, message=new_text_message(reply))
            return
        await updater.add_artifact(parts=[new_text_part(text=reply, media_type="text/plain")], name="answer")
        await updater.update_status(state=TaskState.TASK_STATE_COMPLETED)

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancellation of an in-flight graph run is not supported."""
        raise NotImplementedError("cancel is not supported")


def build_agent_card(name: str, description: str, url: str, skills: list[dict[str, Any]]) -> AgentCard:
    """Describe the agent for discovery at ``/.well-known/agent-card.json``."""
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        skills=[
            AgentSkill(
                id=s["id"],
                name=s["name"],
                description=s["description"],
                tags=s.get("tags", []),
                examples=s.get("examples", []),
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
            for s in skills
        ],
    )


def build_a2a_app(card: AgentCard, ask: AskAgent, token: str) -> Starlette:
    """Starlette app serving the card publicly and JSON-RPC behind a bearer token."""
    if not token:
        raise ValueError("A2A_SERVER_TOKEN is required: the A2A endpoint runs the agent with its tools")
    handler = DefaultRequestHandler(
        agent_executor=KitAgentExecutor(ask), task_store=InMemoryTaskStore(), agent_card=card
    )
    return Starlette(
        routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")],
        middleware=[Middleware(BearerTokenMiddleware, token=token)],
    )


class BearerTokenMiddleware:
    """Require ``Authorization: Bearer <token>`` except for the public agent card."""

    def __init__(self, app: ASGIApp, token: str):
        """Wrap ``app``."""
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject unauthenticated HTTP requests with 401."""
        if scope["type"] != "http" or scope["path"].endswith("/.well-known/agent-card.json"):
            await self.app(scope, receive, send)
            return
        supplied = dict(scope["headers"]).get(b"authorization", b"")
        if not hmac.compare_digest(supplied, self._expected):
            await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def agent_ask(agent: Any) -> AskAgent:
    """Adapt ``LangGraphAgent`` to the executor: one user message in, the reply out."""

    async def ask(session_id: str, text: str) -> tuple[str, bool]:
        replies = await agent.get_response([Message(role="user", content=text)], session_id=session_id, user_id=None)
        state = await (await agent._get_graph()).aget_state({"configurable": {"thread_id": session_id}})
        reply = replies[-1].content if replies else ""
        return reply, agent._awaiting_human(state)

    return ask
