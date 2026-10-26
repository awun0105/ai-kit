"""MCP integration: load external MCP tools into the agent."""

from app.services.mcp.client import (
    MCPToolkit,
    build_toolkit_from_settings,
    load_mcp_config,
)

__all__ = ["MCPToolkit", "build_toolkit_from_settings", "load_mcp_config"]
