"""A scripted ACP agent for the harness suite: ``python tests/conformance/scripted_agent.py SCRIPT LOG``.

``SCRIPT`` is JSON ``{"turns": [[step, ...], ...]}``; turn *n* of the session runs
``turns[n]``. Steps:

- ``{"tool": name, "args": {...}}``: call a tool on the session's MCP server
- ``{"write_file": path, "content": text}`` / ``{"read_file": path}``: ACP ``fs/*`` requests
- ``{"ask": title}``: ACP ``session/request_permission``
- ``{"say": text}``, ``{"sleep": seconds}``, ``{"exit": code}``

Every step's outcome, and the session's ``mcp_servers`` on ``session/new``, is appended
to ``LOG`` as one JSON line, so tests can check what the agent saw.
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    RequestError,
    update_agent_message_text,
)
from acp.schema import (
    AgentCapabilities,
    AllowedOutcome,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    ToolCallUpdate,
)
from mcp.client.session import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)

from app.services.acp import serve_stdio


class ScriptedAgent:
    def __init__(self, script: dict, log: Path):
        self.turns = list(script["turns"])
        self.log = log
        self.conn: Any = None
        self.servers: list = []
        self.turn = 0

    def write(self, entry: dict) -> None:
        with self.log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def on_connect(self, conn: Any) -> None:
        self.conn = conn

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocol_version=PROTOCOL_VERSION, agent_capabilities=AgentCapabilities())

    async def new_session(self, cwd: str, mcp_servers: list | None = None, **kwargs: Any) -> NewSessionResponse:
        self.servers = list(mcp_servers or [])
        self.write({"session": {"cwd": cwd, "mcp_servers": [s.model_dump() for s in self.servers]}})
        return NewSessionResponse(session_id=uuid.uuid4().hex)

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self.write({"cancelled": True})

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        steps = self.turns[self.turn] if self.turn < len(self.turns) else []
        self.turn += 1
        mcp = [step for step in steps if "tool" in step]
        if mcp:
            server = self.servers[0]
            params = StdioServerParameters(
                command=server.command, args=list(server.args), env={e.name: e.value for e in server.env}
            )
            async with stdio_client(params) as streams:
                async with ClientSession(*streams[:2]) as session:
                    await session.initialize()
                    await self._run(session_id, steps, session)
        else:
            await self._run(session_id, steps, None)
        return PromptResponse(stop_reason="end_turn")

    async def _run(self, session_id: str, steps: list[dict], mcp: ClientSession | None) -> None:
        for index, step in enumerate(steps):
            entry: dict[str, Any] = {"turn": self.turn - 1, "step": index}
            if "tool" in step:
                assert mcp is not None
                result = await mcp.call_tool(step["tool"], step.get("args") or {})
                entry["result"] = {"text": result.content[0].text, "is_error": result.is_error}  # type: ignore[union-attr]
            elif "write_file" in step:
                try:
                    await self.conn.write_text_file(
                        session_id=session_id, path=step["write_file"], content=step["content"]
                    )
                    entry["result"] = {"ok": True}
                except RequestError:
                    entry["result"] = {"ok": False}
            elif "read_file" in step:
                try:
                    response = await self.conn.read_text_file(session_id=session_id, path=step["read_file"])
                    entry["result"] = {"ok": True, "content": response.content}
                except RequestError:
                    entry["result"] = {"ok": False}
            elif "ask" in step:
                response = await self.conn.request_permission(
                    session_id=session_id,
                    tool_call=ToolCallUpdate(tool_call_id=uuid.uuid4().hex, title=step["ask"], kind="execute"),
                    options=[
                        PermissionOption(option_id="yes", name="Allow", kind="allow_once"),
                        PermissionOption(option_id="no", name="Reject", kind="reject_once"),
                    ],
                )
                outcome = response.outcome
                entry["result"] = {"allowed": isinstance(outcome, AllowedOutcome) and outcome.option_id == "yes"}
            elif "say" in step:
                await self.conn.session_update(session_id=session_id, update=update_agent_message_text(step["say"]))
                entry["result"] = {"ok": True}
            elif "sleep" in step:
                await asyncio.sleep(step["sleep"])
                entry["result"] = {"ok": True}
            elif "exit" in step:
                self.write({**entry, "result": {"exit": step["exit"]}})
                os._exit(step["exit"])
            self.write(entry)


if __name__ == "__main__":
    script_path, log_path = sys.argv[1], sys.argv[2]
    agent = ScriptedAgent(json.loads(Path(script_path).read_text(encoding="utf-8")), Path(log_path))
    asyncio.run(serve_stdio(agent))
