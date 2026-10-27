"""Run an ACP agent over this process's stdin and stdout, with stdout kept for the protocol.

On stdio transports stdout carries the JSON-RPC messages, so any other byte written
there (a log line, a library's warning, a ``print``) breaks the client. The protocol
writes to a private copy of stdout (``reserve_stdout``) and file descriptor 1 points
at stderr, so everything else that writes to stdout lands in the client's log instead.
"""

import asyncio
import os
import sys
from typing import Any

from acp import run_agent

from app.services.acp.stdout_guard import reserve_stdout


async def serve_stdio(agent: Any) -> None:
    """Serve ``agent`` until the client closes stdin."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    out = os.fdopen(reserve_stdout(), "wb", buffering=0)
    transport, protocol = await loop.connect_write_pipe(
        lambda: asyncio.StreamReaderProtocol(asyncio.StreamReader()), out
    )
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    await run_agent(agent, input_stream=writer, output_stream=reader)
