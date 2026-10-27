"""The kit's agent as an ACP agent: ``python -m app.acp_agent`` (needs the ``acp`` extra).

An ACP client (Zed, a JetBrains IDE, fframes Studio, or your own app) starts this
process and talks to it over stdin and stdout. Each ACP session is one LangGraph
thread; approvals for risky tools are asked through the client, and everything
else (RAG, guardrails, retries, cost guard) works as in the HTTP app. With the
offline settings (docs/offline.md) it runs on the user's machine with no server.
"""

import app.services.acp.reserve_stdout  # noqa: F401, I001  (first: importing the app logs to stdout)

# isort: split

import asyncio
from typing import Any

from langchain_core.messages import AIMessage

from app.core.config import settings
from app.core.kit import (
    build_toolkit_from_settings,
    get_guardrails,
)
from app.core.langgraph.graph import LangGraphAgent
from app.core.logging import logger
from app.schemas import Message
from app.services.acp import (
    KitAcpAgent,
    Pending,
    TurnResult,
    serve_stdio,
)


class KitBackend:
    """Runs ACP prompts through ``LangGraphAgent``; one ACP session is one graph thread."""

    def __init__(self, agent: LangGraphAgent, user_id: str = "acp:local"):
        """Wrap ``agent``; ``user_id`` keys long-term memory and the cost budget."""
        self._agent = agent
        self._user_id = user_id

    async def turn(self, session_id: str, text: str) -> TurnResult:
        """Send one message (or the answer to a pending question or approval)."""
        thread_id = f"acp-{session_id}"
        replies = await self._agent.get_response(
            [Message(role="user", content=text)], session_id=thread_id, user_id=self._user_id
        )
        reply = replies[-1].content if replies else ""
        graph = await self._agent._get_graph()
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not self._agent._awaiting_human(state):
            return TurnResult(reply=reply)
        question = str(next(i.value for task in state.tasks for i in task.interrupts))
        kind = "approval" if _is_approval(state.values.get("messages", []), question) else "question"
        return TurnResult(reply=reply, pending=Pending(kind=kind, text=question))


def _is_approval(messages: list[Any], question: str) -> bool:
    """Whether a pause is the tool node asking to approve a call (not ``ask_human`` asking the user)."""
    last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last is None:
        return False
    guard = get_guardrails()
    return any(guard.approval_question(call["name"], call["args"]) == question for call in last.tool_calls)


async def main() -> None:
    """Build the agent and serve it over stdio until the client disconnects."""
    agent = LangGraphAgent()
    toolkit = build_toolkit_from_settings()
    if toolkit is not None:
        agent.register_tools(await toolkit.start())
    try:
        await serve_stdio(KitAcpAgent(KitBackend(agent), name=settings.PROJECT_NAME))
    finally:
        if toolkit is not None:
            await toolkit.stop()
        logger.info("acp_agent_stopped")


if __name__ == "__main__":
    asyncio.run(main())
