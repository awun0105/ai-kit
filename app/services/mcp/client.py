"""Load tools from external MCP servers into the agent.

Uses ``langchain.mcp.MCPAdapter`` (LangChain's maintained MCP integration, which
replaced the archived ``langchain-mcp-adapters``) on top of FastMCP. Servers are
declared in the standard ``mcpServers`` JSON format used by Claude Desktop,
Cursor and FastMCP, so an existing config file can be reused as is.

MCP tools run through the same tool node as built-in tools, so the guardrails
apply to them: approval for listed tools, the egress allowlist, and screening
of their output as untrusted data. A server that asks the user for input
mid-call (elicitation) pauses the graph with ``interrupt()``.
"""

import json
import os
import warnings
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import BaseTool
import structlog

logger = structlog.get_logger()


with warnings.catch_warnings():
    warnings.simplefilter("ignore", LangChainBetaWarning)
    from langchain.mcp import MCPAdapter


def load_mcp_config(path: str) -> dict[str, Any]:
    """Read an ``{"mcpServers": {...}}`` file; ``${VAR}`` in values is expanded from the environment."""
    raw = Path(path).read_text(encoding="utf-8")
    config = json.loads(os.path.expandvars(raw))
    if not isinstance(config, dict) or not isinstance(config.get("mcpServers"), dict):
        raise ValueError(f"{path} must contain an object with an 'mcpServers' mapping")
    return config


class MCPToolkit:
    """Owns the MCP connections for the lifetime of the app."""

    def __init__(self, target: Any, allowlist: Optional[set[str]] = None):
        """Wrap any ``MCPAdapter`` target: an ``mcpServers`` dict, a URL, or an in-process server.

        Args:
            target: What to connect to.
            allowlist: When given, only these tool names are exposed to the model
                (least privilege: a server may offer far more than the app needs).
        """
        self._adapter = MCPAdapter(target)
        self._allowlist = allowlist
        self._open = False
        self.tools: list[BaseTool] = []

    async def start(self) -> list[BaseTool]:
        """Connect and discover tools."""
        await self._adapter.__aenter__()
        self._open = True
        discovered = await self._adapter.list_tools()
        self.tools = [t for t in discovered if self._allowlist is None or t.name in self._allowlist]
        hidden = sorted({t.name for t in discovered} - {t.name for t in self.tools})
        logger.info("mcp_tools_loaded", tools=[t.name for t in self.tools], hidden=hidden)
        return self.tools

    async def stop(self) -> None:
        """Close the connections."""
        if self._open:
            await self._adapter.__aexit__(None, None, None)
            self._open = False
