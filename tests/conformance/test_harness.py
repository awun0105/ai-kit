"""Harness conformance: a harness (the Python sidecar or a port) keeps the flow of the spec.

The test plays the app: it starts the harness with the CLI flags of ``docs/harness.md``,
serves the sample notes adapter on the harness's stdio, and drives tasks through the
app-facing API. The agent is ``scripted_agent.py``, so no model is needed::

    HARNESS_CMD='["./my-harness"]' uv run pytest tests/conformance/test_harness.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("acp")

from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.stdio import (  # noqa: E402
    StdioServerParameters,
    stdio_client,
)

from app.services.harness import (  # noqa: E402
    RpcError,
    adapter_handlers,
    spawn_peer,
)
from examples.harness_notes import NotesAdapter  # noqa: E402
from tests.conformance.conftest import (  # noqa: E402
    ROOT,
    _json_env,
)

AGENT = ROOT / "tests" / "conformance" / "scripted_agent.py"
EVENTS = [
    "task.started",
    "task.state",
    "task.candidate",
    "task.validation",
    "task.finished",
    "permission.requested",
    "permission.closed",
    "message.delta",
    "message.final",
    "tool.call",
    "effect.done",
    "effect.failed",
    "revision.undone",
]
TIMEOUT = 30


def write(note_id: str, title: str, body: str = "") -> dict:
    return {"tool": "write_note", "args": {"id": note_id, "title": title, "body": body}}


class App:
    """The app side of the connection: the notes adapter plus an event log."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.notes = NotesAdapter(tmp_path / "data")
        self.events: list[tuple[str, dict]] = []
        self.changed = asyncio.Event()
        self.log = tmp_path / "agent.log"
        self.peer: Any = None
        self.process: Any = None

    async def start(self, turns: list[list[dict]], policy: str = "allow_in_draft", review: str = "manual") -> None:
        script = self.tmp_path / "script.json"
        script.write_text(json.dumps({"turns": turns}), encoding="utf-8")
        agent = [sys.executable, str(AGENT), str(script), str(self.log)]
        command = _json_env("HARNESS_CMD", [sys.executable, "-m", "app.harness_sidecar"]) + [
            "--agent-cmd",
            json.dumps(agent),
            "--policy",
            policy,
            "--review",
            review,
        ]
        handlers = adapter_handlers(self.notes)
        for name in EVENTS:
            handlers[name] = self._collector(name)
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        self.peer, self.process = await spawn_peer(command, handlers, cwd=str(ROOT), env=env, name="harness")

    def _collector(self, name: str):
        async def collect(params: dict) -> None:
            self.events.append((name, params))
            self.changed.set()

        return collect

    async def call(self, method: str, **params: Any) -> Any:
        return await self.peer.request(method, params, timeout=TIMEOUT)

    def of(self, name: str) -> list[dict]:
        return [data for event, data in self.events if event == name]

    async def until(self, check: Callable[[], Any]) -> Any:
        async def wait() -> Any:
            while not (found := check()):
                self.changed.clear()
                await self.changed.wait()
            return found

        return await asyncio.wait_for(wait(), TIMEOUT)

    async def state(self, task_id: str, state: str) -> None:
        await self.until(lambda: any(d["task_id"] == task_id and d["state"] == state for d in self.of("task.state")))

    async def finished(self, task_id: str) -> dict:
        return await self.until(lambda: next((d for d in self.of("task.finished") if d["task_id"] == task_id), None))

    async def candidate(self, task_id: str) -> dict:
        return await self.until(lambda: next((d for d in self.of("task.candidate") if d["task_id"] == task_id), None))

    async def permission(self, index: int = 0) -> dict:
        return await self.until(
            lambda: len(self.of("permission.requested")) > index and self.of("permission.requested")[index]
        )

    def agent_log(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def step_results(self) -> list[dict]:
        return [entry["result"] for entry in self.agent_log() if "result" in entry]

    async def stop(self) -> None:
        if self.peer is not None:
            await self.peer.close()
        if self.process is not None:
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()


@pytest.fixture
async def app(tmp_path):
    app = App(tmp_path)
    yield app
    await app.stop()


async def test_writes_stay_in_draft_until_accept(app):
    await app.start([[write("a", "Alpha", "first"), {"tool": "validate_draft"}]])
    task_id = (await app.call("task.start", goal="Add a note"))["task_id"]
    candidate = await app.candidate(task_id)

    assert app.notes.context().state == {"notes": {}}
    assert candidate["diff"]["summary"] == "1 added, 0 changed, 0 removed"
    assert json.loads(app.step_results()[1]["text"])["ok"] is True

    revision = (await app.call("candidate.accept", task_id=task_id))["revision"]
    assert (await app.finished(task_id))["state"] == "accepted"
    assert app.notes.context().state == {"notes": {"a": "Alpha"}} and revision == "1"


async def test_dangerous_asks_queues_and_runs_after_accept(app):
    await app.start([[write("a", "Alpha"), {"tool": "publish_note", "args": {"id": "a"}}]])
    task_id = (await app.call("task.start", goal="Write and publish"))["task_id"]
    request = await app.permission()
    await app.call("permission.reply", id=request["id"], choice="allow")
    candidate = await app.candidate(task_id)
    outbox = app.notes.data_dir / "outbox"

    assert request["kind"] == "dangerous"
    assert [effect["op"] for effect in candidate["effects"]] == ["publish_note"]
    assert not any(outbox.iterdir())

    await app.call("candidate.accept", task_id=task_id)
    await app.until(lambda: app.of("effect.done"))
    assert (outbox / "a.md").exists()


async def test_dangerous_dropped_on_reject(app):
    await app.start([[write("a", "Alpha"), {"tool": "publish_note", "args": {"id": "a"}}]])
    task_id = (await app.call("task.start", goal="Write and publish"))["task_id"]
    await app.call("permission.reply", id=(await app.permission())["id"], choice="allow")
    await app.candidate(task_id)
    await app.call("candidate.reject", task_id=task_id)

    assert (await app.finished(task_id))["state"] == "rejected"
    assert not any((app.notes.data_dir / "outbox").iterdir())
    assert app.notes.context().revision == "0"


async def test_permission_reply_twice_or_unknown_is_refused(app):
    await app.start([[write("a", "Alpha")]], policy="ask")
    task_id = (await app.call("task.start", goal="Ask first"))["task_id"]
    request = await app.permission()
    await app.call("permission.reply", id=request["id"], choice="reject")
    await app.candidate(task_id)

    with pytest.raises(RpcError):
        await app.call("permission.reply", id=request["id"], choice="allow")
    with pytest.raises(RpcError):
        await app.call("permission.reply", id="no-such-request", choice="allow")
    assert app.step_results()[0]["is_error"] is True


async def test_validation_failure_gets_one_repair(app):
    await app.start([[write("a", "")], [write("a", "Fixed")]])
    task_id = (await app.call("task.start", goal="Add a note"))["task_id"]
    await app.candidate(task_id)
    states = [d["state"] for d in app.of("task.state") if d["task_id"] == task_id]

    assert states == ["running", "validating", "repair_needed", "running", "validating", "candidate_ready"]


async def test_validation_fails_after_the_repair(app):
    await app.start([[write("a", "")], [write("a", "")]])
    task_id = (await app.call("task.start", goal="Add a note"))["task_id"]

    assert (await app.finished(task_id))["state"] == "failed"
    assert app.notes.context().revision == "0"


async def test_malformed_file_is_a_validation_failure(app):
    await app.start(
        [[{"write_file": "e.md", "content": "no heading"}], [{"write_file": "e.md", "content": "# Echo\n"}]]
    )
    task_id = (await app.call("task.start", goal="Add a note as a file"))["task_id"]
    candidate = await app.candidate(task_id)

    assert "repair_needed" in [d["state"] for d in app.of("task.state")]
    assert candidate["diff"]["detail"]["added"] == ["e"]


async def test_conflict_when_base_changed(app):
    await app.start([[write("a", "Alpha")]])
    task_id = (await app.call("task.start", goal="Add a note"))["task_id"]
    await app.candidate(task_id)
    other = app.notes.open_draft("0")
    app.notes.call(other, "write_note", {"id": "z", "title": "Zed", "body": ""})
    app.notes.commit(other, "0")

    with pytest.raises(RpcError):
        await app.call("candidate.accept", task_id=task_id)
    assert (await app.finished(task_id))["state"] == "conflict"
    assert app.notes.context().state == {"notes": {"z": "Zed"}}


async def test_undo(app):
    await app.start([[write("a", "Alpha")]], review="auto")
    task_id = (await app.call("task.start", goal="Add a note"))["task_id"]
    assert (await app.finished(task_id))["state"] == "accepted"

    with pytest.raises(RpcError):
        await app.call("revision.undo", revision="0")
    assert (await app.call("revision.undo", revision="1"))["revision"] == "2"
    assert app.notes.context().state == {"notes": {}}


async def test_fs_outside_workdir_is_refused(app, tmp_path):
    outside = tmp_path / "escape.md"
    await app.start(
        [
            [
                {"write_file": "../escape.md", "content": "# Out\n"},
                {"write_file": str(outside), "content": "# Out\n"},
                {"write_file": "d.md", "content": "# Delta\n"},
                {"read_file": "d.md"},
            ]
        ]
    )
    task_id = (await app.call("task.start", goal="Add a note as a file"))["task_id"]
    candidate = await app.candidate(task_id)
    results = app.step_results()

    assert [r["ok"] for r in results] == [False, False, True, True]
    assert not outside.exists()
    assert candidate["diff"]["detail"]["added"] == ["d"]


async def test_capability_dead_after_task(app):
    await app.start([[{"tool": "task_context"}]])
    task_id = (await app.call("task.start", goal="Look around"))["task_id"]
    await app.candidate(task_id)
    await app.call("candidate.reject", task_id=task_id)
    await app.finished(task_id)
    server = next(entry for entry in app.agent_log() if "session" in entry)["session"]["mcp_servers"][0]

    assert json.loads(app.step_results()[0]["text"])["goal"] == "Look around"
    params = StdioServerParameters(
        command=server["command"], args=server["args"], env={e["name"]: e["value"] for e in server["env"]}
    )
    async with stdio_client(params) as streams:
        async with ClientSession(*streams[:2]) as session:
            await session.initialize()
            late = await session.call_tool("task_context", {})
            assert late.is_error


async def test_cancel_running(app):
    await app.start([[{"sleep": 30}]])
    task_id = (await app.call("task.start", goal="Take your time"))["task_id"]
    await app.state(task_id, "running")
    await app.call("task.cancel", task_id=task_id)

    assert (await app.finished(task_id))["state"] == "cancelled"


async def test_cancel_waiting_permission(app):
    await app.start([[write("a", "Alpha")]], policy="ask")
    task_id = (await app.call("task.start", goal="Ask first"))["task_id"]
    await app.permission()
    await app.call("task.cancel", task_id=task_id)

    assert (await app.finished(task_id))["state"] == "cancelled"
    assert app.of("permission.closed")[0]["outcome"] == "cancelled"
    assert app.notes.context().revision == "0"


async def test_agent_permission_request_goes_to_the_app(app):
    await app.start([[{"ask": "Run the build"}, {"ask": "Delete the cache"}]])
    task_id = (await app.call("task.start", goal="Build"))["task_id"]
    await app.call("permission.reply", id=(await app.permission(0))["id"], choice="allow")
    await app.call("permission.reply", id=(await app.permission(1))["id"], choice="reject")
    await app.candidate(task_id)

    assert [r["allowed"] for r in app.step_results()] == [True, False]
    assert app.of("permission.requested")[0]["kind"] == "agent"


async def test_agent_exit_fails_task(app):
    await app.start([[{"exit": 3}]])
    task_id = (await app.call("task.start", goal="Crash"))["task_id"]

    assert (await app.finished(task_id))["state"] == "failed"
    assert app.notes.context().revision == "0"


async def test_single_writer(app):
    await app.start([[{"sleep": 30}]])
    task_id = (await app.call("task.start", goal="First"))["task_id"]
    with pytest.raises(RpcError):
        await app.call("task.start", goal="Second")
    await app.call("task.cancel", task_id=task_id)


async def test_deny_writes(app):
    await app.start([[write("a", "Alpha"), {"tool": "publish_note", "args": {"id": "a"}}]], policy="deny_writes")
    task_id = (await app.call("task.start", goal="Try to write"))["task_id"]
    await app.candidate(task_id)

    assert all(r["is_error"] for r in app.step_results())
    assert app.of("permission.requested") == []
