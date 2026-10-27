"""Drive an external agent (Claude Code, Codex, Gemini CLI... through their ACP adapters) for a task.

Needs the ``acp`` extra. The agent runs with the task's working directory as
``cwd``, reaches the app's tools through the MCP bridge, and asks permission
through the harness. Its file requests (``fs/*``) are confined to the working
directory; the terminal capability is not offered. An agent's own shell tool
can still write elsewhere: that is the agent's sandbox to enforce, not this
driver's (see ``docs/harness.md``).
"""

import asyncio
import contextlib
import os
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from acp import (
    PROTOCOL_VERSION,
    DeclineElicitationResponse,
    RequestError,
    spawn_agent_process,
    text_block,
)
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    EnvVariable,
    FileSystemCapabilities,
    Implementation,
    McpServerStdio,
    ReadTextFileResponse,
    RequestPermissionResponse,
    WriteTextFileResponse,
)
import structlog

from app.services.harness.driver import (
    DriverError,
    TurnContext,
)

logger = structlog.get_logger()

ALLOW_KINDS = ("allow_once", "allow_always")
REJECT_KINDS = ("reject_once", "reject_always")


class _HarnessClient:
    """The ACP client side the agent talks to."""

    def __init__(self, driver: "AcpDriver"):
        self._driver = driver

    def on_connect(self, conn: Any) -> None:
        pass

    @property
    def _ctx(self) -> TurnContext:
        assert self._driver.ctx is not None
        return self._driver.ctx

    def _inside(self, path: str) -> Path:
        workdir = self._ctx.workdir
        if workdir is None:
            raise RequestError.invalid_params({"path": "no working directory"})
        resolved = (workdir / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not resolved.is_relative_to(workdir.resolve()):
            raise RequestError.invalid_params({"path": "outside the task's working directory"})
        return resolved

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        kind = getattr(update, "session_update", None)
        if kind == "agent_message_chunk":
            text = getattr(update.content, "text", "") or ""
            self._driver.text.append(text)
            await self._ctx.emit("message.delta", {"task_id": self._ctx.task_id, "text": text})
        elif kind == "tool_call":
            await self._ctx.emit(
                "tool.call",
                {"task_id": self._ctx.task_id, "name": update.title, "kind": update.kind, "level": "agent"},
            )

    async def request_permission(self, session_id: str, tool_call: Any, options: list, **kwargs: Any):
        title = getattr(tool_call, "title", None) or "agent action"
        allowed = await self._ctx.ask_permission(title, "agent", {"tool_call_id": tool_call.tool_call_id})
        wanted = ALLOW_KINDS if allowed else REJECT_KINDS
        option = next((o for kind in wanted for o in options if o.kind == kind), None)
        if option is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=option.option_id))

    async def read_text_file(
        self, session_id: str, path: str, line: Optional[int] = None, limit: Optional[int] = None, **kwargs: Any
    ) -> ReadTextFileResponse:
        target = self._inside(path)
        if not target.is_file():
            raise RequestError.resource_not_found(path)
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        start = max((line or 1) - 1, 0)
        end = start + limit if limit else None
        return ReadTextFileResponse(content="".join(lines[start:end]))

    async def write_text_file(self, session_id: str, path: str, content: str, **kwargs: Any) -> WriteTextFileResponse:
        target = self._inside(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return WriteTextFileResponse()

    async def create_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/kill")

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        # The agent asks the user for input through the task instead (an ask or a turn's answer).
        return DeclineElicitationResponse(action="decline")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        pass

    async def ext_method(self, method: str, params: dict) -> dict:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        pass


def _mcp_server(spec: dict) -> McpServerStdio:
    return McpServerStdio(
        name=spec["name"],
        command=spec["command"],
        args=list(spec["args"]),
        env=[EnvVariable(name=key, value=value) for key, value in spec["env"].items()],
    )


class AcpDriver:
    """One external ACP agent process per task, kept across the task's turns."""

    needs_workdir = True
    needs_bridge = True

    def __init__(self, command: list[str], env: Optional[dict[str, str]] = None, client_name: str = "ai-kit-harness"):
        """Start ``command`` (e.g. ``["npx", "@zed-industries/claude-code-acp"]``) with extra ``env``."""
        if not command:
            raise ValueError("the agent command is empty")
        self._command = command
        self._env = env or {}
        self._client_name = client_name
        self._stack: Optional[contextlib.AsyncExitStack] = None
        self._conn: Any = None
        self._process: Any = None
        self._session_id: Optional[str] = None
        self.ctx: Optional[TurnContext] = None
        self.text: list[str] = []

    async def _start(self, ctx: TurnContext) -> None:
        stack = contextlib.AsyncExitStack()
        try:
            self._conn, self._process = await stack.enter_async_context(
                spawn_agent_process(
                    _HarnessClient(self),
                    self._command[0],
                    *self._command[1:],
                    env={**os.environ, **self._env},
                    cwd=str(ctx.workdir) if ctx.workdir else None,
                )
            )
            await self._conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(
                    fs=FileSystemCapabilities(read_text_file=True, write_text_file=True), terminal=False
                ),
                client_info=Implementation(name=self._client_name, version="0.1"),
            )
            servers = [_mcp_server(ctx.mcp_server)] if ctx.mcp_server else []
            session = await self._conn.new_session(cwd=str(ctx.workdir or Path.cwd()), mcp_servers=servers)
        except Exception as error:
            await stack.aclose()
            raise DriverError("the agent did not start") from error
        self._session_id = session.session_id
        self._stack = stack

    async def run_turn(self, ctx: TurnContext, prompt: str) -> str:
        """Send ``prompt`` and wait for the end of the agent's turn."""
        self.ctx = ctx
        self.text = []
        if self._conn is None:
            await self._start(ctx)
        assert self._conn is not None and self._process is not None
        prompt_task = asyncio.create_task(self._conn.prompt(session_id=self._session_id, prompt=[text_block(prompt)]))
        exit_task = asyncio.create_task(self._process.wait())
        try:
            done, _ = await asyncio.wait({prompt_task, exit_task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            prompt_task.cancel()
            exit_task.cancel()
            raise
        exit_task.cancel()
        if prompt_task not in done:
            prompt_task.cancel()
            raise DriverError("the agent exited during the turn")
        try:
            response = prompt_task.result()
        except Exception as error:
            raise DriverError("the agent's turn failed") from error
        logger.info("harness_acp_turn_finished", task_id=ctx.task_id, stop_reason=response.stop_reason)
        return "".join(self.text)

    async def cancel(self) -> None:
        """Ask the agent to stop its turn."""
        if self._conn is not None and self._session_id is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._conn.cancel(session_id=self._session_id), timeout=2)

    async def close(self) -> None:
        """End the agent process."""
        stack, self._stack = self._stack, None
        process = self._process
        self._conn = None
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
        if stack is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stack.aclose(), timeout=5)
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
