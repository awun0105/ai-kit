"""The agent driver interface (spec section 5.3), reduced from fframes Studio's ``AgentDriver``.

A driver runs one agent for one task, a turn at a time. The harness hands it a
``TurnContext`` with the task's tools and a way to ask the user; the driver
reports what the agent does through ``emit``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
)

from app.services.harness.broker import ToolResult
from app.services.harness.protocol import Operation


class DriverError(Exception):
    """The agent could not finish the turn (it exited, or its connection broke)."""


@dataclass
class TurnContext:
    """What a driver gets for each turn of a task."""

    task_id: str
    goal: str
    scope: Any
    token: str
    workdir: Optional[Path]
    list_tools: Callable[[], Awaitable[list[Operation]]]
    call_tool: Callable[[str, dict], Awaitable[ToolResult]]
    ask_permission: Callable[[str, str, Optional[dict]], Awaitable[bool]]
    emit: Callable[[str, dict], Awaitable[None]]
    mcp_server: Optional[dict] = None  # stdio MCP server spec for external agents (see ``bridge.mcp_server_spec``)


class AgentDriver(Protocol):
    """One agent working on one task."""

    needs_workdir: bool  # the agent edits files: run it in the draft's working directory
    needs_bridge: bool  # the agent reaches the tools through the MCP bridge

    async def run_turn(self, ctx: TurnContext, prompt: str) -> str:
        """Run one turn to its end and return the agent's final text; raise ``DriverError`` if it breaks."""
        ...

    async def cancel(self) -> None:
        """Ask the agent to stop the current turn."""
        ...

    async def close(self) -> None:
        """Stop the agent and release what it holds."""
        ...


DriverFactory = Callable[[], AgentDriver]
