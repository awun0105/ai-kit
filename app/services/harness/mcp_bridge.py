"""Stdio MCP server that relays an external agent's tool calls to the harness (``bridge.ToolBridge``).

Started by the agent from the ``mcp_servers`` entry of its ACP session, with
``HARNESS_BRIDGE=host:port`` and ``HARNESS_CAPABILITY=<token>`` in its environment.
"""

import asyncio
import os

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from app.services.harness.rpc import (
    LINE_LIMIT,
    JsonRpcPeer,
    RpcError,
)


async def main() -> None:
    """Connect to the harness, then serve MCP on stdio until the agent closes it."""
    host, _, port = os.environ["HARNESS_BRIDGE"].rpartition(":")
    token = os.environ["HARNESS_CAPABILITY"]
    reader, writer = await asyncio.open_connection(host, int(port), limit=LINE_LIMIT)
    peer = JsonRpcPeer(reader, writer, name="mcp-bridge").start()

    async def list_tools(ctx, params) -> types.ListToolsResult:
        result = await peer.request("tools.list", {"token": token})
        return types.ListToolsResult(
            tools=[
                types.Tool(name=tool["name"], description=tool["description"], input_schema=tool["params"])
                for tool in result["tools"]
            ]
        )

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            result = await peer.request("tools.call", {"token": token, "name": params.name, "args": params.arguments})
        except RpcError:
            result = {"text": "Tool call not executed: the harness is not reachable.", "is_error": True}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result["text"])], is_error=result["is_error"]
        )

    server = Server("app", on_list_tools=list_tools, on_call_tool=call_tool)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await peer.close()


if __name__ == "__main__":
    asyncio.run(main())
