"""The tool bridge: how an external agent (Claude Code, Codex over ACP) reaches the task's tools.

The harness listens on 127.0.0.1 and hands the agent a stdio MCP server
(``python -m app.services.harness.mcp_bridge``) that relays to it, carrying the
task's capability token. Only a holder of a live token gets tools; once the task
ends the token is revoked and every call is refused.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.services.harness.broker import ToolBroker
from app.services.harness.rpc import (
    LINE_LIMIT,
    JsonRpcPeer,
)


IMPORT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class BridgeInfo:
    """Where the bridge listens."""

    host: str
    port: int


def mcp_server_spec(info: BridgeInfo, token: str) -> dict:
    """The stdio MCP server an agent starts to reach the task's tools (an ACP ``McpServerStdio``)."""
    return {
        "name": "app",
        "command": sys.executable,
        "args": ["-m", "app.services.harness.mcp_bridge"],
        "env": {
            "HARNESS_BRIDGE": f"{info.host}:{info.port}",
            "HARNESS_CAPABILITY": token,
            # The agent starts it from the task's working directory: point it at this code.
            "PYTHONPATH": str(IMPORT_ROOT),
        },
    }


class ToolBridge:
    """Serves ``tools.list`` and ``tools.call`` to MCP bridge processes."""

    def __init__(self, broker: ToolBroker):
        """Relay calls to ``broker``."""
        self._broker = broker
        self._server: Optional[asyncio.Server] = None
        self._info: Optional[BridgeInfo] = None
        self._peers: set[JsonRpcPeer] = set()

    async def start(self) -> BridgeInfo:
        """Listen on a free local port (once); returns where."""
        if self._info is not None:
            return self._info
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0, limit=LINE_LIMIT)
        port = self._server.sockets[0].getsockname()[1]
        self._info = BridgeInfo("127.0.0.1", port)
        return self._info

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        async def list_tools(params: dict) -> dict:
            tools = await self._broker.list_tools(str(params.get("token", "")))
            return {"tools": [tool.model_dump() for tool in tools]}

        async def call_tool(params: dict) -> dict:
            result = await self._broker.call(
                str(params.get("token", "")), str(params.get("name", "")), dict(params.get("args") or {})
            )
            return {"text": result.text, "is_error": result.is_error}

        peer = JsonRpcPeer(reader, writer, {"tools.list": list_tools, "tools.call": call_tool}, name="bridge")
        self._peers.add(peer)
        peer.start()
        await peer.closed.wait()
        self._peers.discard(peer)

    async def close(self) -> None:
        """Stop listening and drop every connection."""
        for peer in list(self._peers):
            await peer.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._server, self._info = None, None
