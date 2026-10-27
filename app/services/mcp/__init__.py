"""MCP integration: load external MCP tools into the agent."""

from app.services.mcp.client import (
    MCPToolkit,
    load_mcp_config,
)

__all__ = ["MCPToolkit", "load_mcp_config"]
