"""Tool rails for agents that call tools themselves, outside the app graph's tool node.

The graph's tool node authorizes every call, pauses risky ones for approval and
screens results. A sub-agent (the deep-research team, any ``create_agent``
loop) runs its tools directly, so handing it raw tools would skip all of that.
``guard_tools`` wraps each tool with the same checks. There is nobody to ask
inside an autonomous run, so a tool that needs approval is refused instead.
"""

from typing import (
    Any,
    Sequence,
)

from langchain_core.messages import ToolMessage
from langchain_core.tools import (
    BaseTool,
    StructuredTool,
)
import structlog

from app.services.guardrails.policy import Guardrails

logger = structlog.get_logger()


def _text(result: Any) -> str:
    content = result.content if isinstance(result, ToolMessage) else result
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "\n".join(p for p in parts if p)
    return str(content)


def _guarded(tool: BaseTool, guard: Guardrails) -> BaseTool:
    async def run(**kwargs: Any) -> str:
        decision = guard.authorize_tool_call(tool.name, kwargs)
        if not decision.allowed:
            logger.info("guard_tool_call_blocked", tool=tool.name, reason=decision.reason)
            return f"Tool call not executed: blocked by policy: {decision.reason}."
        if decision.needs_approval:
            logger.info("guard_tool_call_needs_approval_in_autonomous_run", tool=tool.name)
            return f"Tool call not executed: {tool.name} needs human approval, which an autonomous agent cannot get."
        try:
            result = await tool.ainvoke(kwargs)
        except Exception as e:
            logger.warning("tool_call_failed", tool=tool.name, error=str(e))
            return f"Tool error: {type(e).__name__}: {e}"
        return guard.screen_tool_result(tool.name, _text(result))

    return StructuredTool.from_function(
        coroutine=run, name=tool.name, description=tool.description, args_schema=tool.args_schema
    )


def guard_tools(tools: Sequence[BaseTool], guard: Guardrails) -> list[BaseTool]:
    """Wrap ``tools`` so every call goes through ``guard``'s tool rails (same names and argument schemas)."""
    return [_guarded(tool, guard) for tool in tools]
