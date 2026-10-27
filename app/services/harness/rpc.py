"""JSON-RPC 2.0 over a pair of byte streams, one JSON object per line, requests in both directions.

The app and the harness share one such connection: each side serves methods and
calls the other's. Kept small and dependency-free so a port can mirror it line
for line in the app's language.
"""

import asyncio
import itertools
import json
import os
import sys
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
)

import structlog

logger = structlog.get_logger()

Handler = Callable[[dict], Awaitable[Any]]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL = -32603
APP_ERROR = -32000

LINE_LIMIT = 16 * 1024 * 1024


class RpcError(Exception):
    """An error the other side sees: a code, a message meant for it, and optional data."""

    def __init__(self, code: int, message: str, data: Any = None):
        """Build the error; ``message`` goes over the wire, so never put exception text in it."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_wire(self) -> dict:
        """The ``error`` member of a response."""
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


class JsonRpcPeer:
    """One end of a bidirectional JSON-RPC connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handlers: Optional[dict[str, Handler]] = None,
        name: str = "peer",
    ):
        """Serve ``handlers`` (method name -> async function of the params) to the other side."""
        self._reader = reader
        self._writer = writer
        self._handlers: dict[str, Handler] = dict(handlers or {})
        self._name = name
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._tasks: set[asyncio.Task] = set()
        self._loop_task: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()
        self.closed = asyncio.Event()

    def on(self, method: str, handler: Handler) -> None:
        """Serve one more method."""
        self._handlers[method] = handler

    def start(self) -> "JsonRpcPeer":
        """Start reading messages; returns ``self``."""
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._read_loop())
        return self

    async def request(self, method: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> Any:
        """Call ``method`` on the other side and return its result; raises ``RpcError`` on an error reply."""
        if self.closed.is_set():
            raise RpcError(INTERNAL, "connection closed")
        request_id = next(self._ids)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a notification (no reply expected)."""
        if self.closed.is_set():
            return
        await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    async def close(self) -> None:
        """Stop reading, fail pending calls and close the writer."""
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
        for task in list(self._tasks):
            task.cancel()
        self._finish()
        try:
            self._writer.close()
        except Exception:  # noqa: BLE001  (already closed by the other side)
            pass

    async def _send(self, message: dict) -> None:
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode()
        async with self._write_lock:
            try:
                self._writer.write(data)
                await self._writer.drain()
            except (ConnectionError, RuntimeError):
                self._finish()

    async def _read_loop(self) -> None:
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    await self._send(
                        {"jsonrpc": "2.0", "id": None, "error": RpcError(PARSE_ERROR, "parse error").to_wire()}
                    )
                    continue
                self._dispatch(message)
        except (asyncio.CancelledError, ConnectionError, ValueError):
            pass
        finally:
            self._finish()

    def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        if "method" in message:
            task = asyncio.create_task(self._serve(message))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return
        future = self._pending.get(message.get("id"))  # type: ignore[arg-type]
        if future is None or future.done():
            return
        if "error" in message:
            error = message["error"] or {}
            future.set_exception(RpcError(error.get("code", INTERNAL), error.get("message", ""), error.get("data")))
        else:
            future.set_result(message.get("result"))

    async def _serve(self, message: dict) -> None:
        method = message["method"]
        request_id = message.get("id")
        params = message.get("params") or {}
        handler = self._handlers.get(method)
        try:
            if handler is None:
                raise RpcError(METHOD_NOT_FOUND, f"method not found: {method}")
            if not isinstance(params, dict):
                raise RpcError(INVALID_PARAMS, "params must be an object")
            result = await handler(params)
            reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except RpcError as error:
            reply = {"jsonrpc": "2.0", "id": request_id, "error": error.to_wire()}
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("rpc_handler_failed", peer=self._name, method=method)
            reply = {"jsonrpc": "2.0", "id": request_id, "error": RpcError(INTERNAL, "internal error").to_wire()}
        if request_id is not None:
            await self._send(reply)

    def _finish(self) -> None:
        if self.closed.is_set():
            return
        self.closed.set()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RpcError(INTERNAL, "connection closed"))


_protocol_fd: Optional[int] = None


def reserve_stdout() -> int:
    """Keep stdout for the protocol: returns a private copy of fd 1 and points fd 1 at stderr.

    Call it before importing anything that may print or log to stdout. Calling it
    again returns the same fd.
    """
    global _protocol_fd
    if _protocol_fd is None:
        sys.stdout.flush()
        _protocol_fd = os.dup(1)
        os.dup2(2, 1)
    return _protocol_fd


async def _writer_for_fd(fd: int) -> asyncio.StreamWriter:
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.connect_write_pipe(
        lambda: asyncio.StreamReaderProtocol(asyncio.StreamReader()), os.fdopen(fd, "wb", buffering=0)
    )
    return asyncio.StreamWriter(transport, protocol, None, loop)


async def open_stdio_peer(handlers: dict[str, Handler], out_fd: int, name: str = "stdio") -> JsonRpcPeer:
    """A started peer on this process's stdin and ``out_fd`` (a reserved copy of stdout)."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=LINE_LIMIT)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    return JsonRpcPeer(reader, await _writer_for_fd(out_fd), handlers, name=name).start()


async def spawn_peer(
    command: list[str],
    handlers: dict[str, Handler],
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    name: str = "child",
) -> tuple[JsonRpcPeer, asyncio.subprocess.Process]:
    """Start ``command`` and return a started peer on its stdin and stdout (its stderr is inherited)."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        limit=LINE_LIMIT,
    )
    assert process.stdout is not None and process.stdin is not None
    return JsonRpcPeer(process.stdout, process.stdin, handlers, name=name).start(), process
