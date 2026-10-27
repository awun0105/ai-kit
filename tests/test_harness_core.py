"""Harness core: RPC peer, states, capabilities, permissions, broker and task orchestration (in process)."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from app.services.harness import (
    AdapterClient,
    CapabilityError,
    CapabilityStore,
    DriverError,
    Harness,
    JsonRpcPeer,
    PermissionPolicy,
    RpcError,
    TaskState,
    TaskStateMachine,
    Transcript,
    adapter_handlers,
    decide,
)
from app.services.harness.protocol import protocol_schema
from app.services.harness.state import (
    ACTIVE,
    TERMINAL,
    InvalidTransition,
)
from examples.harness_notes import NotesAdapter

ROOT = Path(__file__).resolve().parent.parent


async def _pipe() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    read_fd, write_fd = os.pipe()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(read_fd, "rb", 0))
    transport, protocol = await loop.connect_write_pipe(
        lambda: asyncio.StreamReaderProtocol(asyncio.StreamReader()), os.fdopen(write_fd, "wb", 0)
    )
    return reader, asyncio.StreamWriter(transport, protocol, None, loop)


async def _peers(a_handlers: dict, b_handlers: dict) -> tuple[JsonRpcPeer, JsonRpcPeer]:
    a_in, b_out = await _pipe()
    b_in, a_out = await _pipe()
    return JsonRpcPeer(a_in, a_out, a_handlers).start(), JsonRpcPeer(b_in, b_out, b_handlers).start()


# RPC and protocol ---------------------------------------------------------------------------------


async def test_rpc_requests_both_ways_and_notifications():
    seen = []

    async def echo(params):
        return params

    async def add(params):
        return params["a"] + params["b"]

    async def leak(params):
        raise ValueError("secret-detail")

    async def note(params):
        seen.append(params)

    a, b = await _peers({"echo": echo, "note": note}, {"add": add, "leak": leak})

    assert await a.request("add", {"a": 1, "b": 2}) == 3
    assert await b.request("echo", {"x": 1}) == {"x": 1}
    await b.notify("note", {"n": 1})
    with pytest.raises(RpcError) as missing:
        await a.request("nope")
    with pytest.raises(RpcError) as leaked:
        await a.request("leak")
    await asyncio.sleep(0.05)
    await a.close()
    await b.close()

    assert seen == [{"n": 1}]
    assert missing.value.code == -32601
    assert leaked.value.message == "internal error"


def test_protocol_schema_is_published():
    published = json.loads((ROOT / "docs" / "harness" / "protocol.schema.json").read_text(encoding="utf-8"))
    assert published == protocol_schema()  # regenerate with: uv run python -m scripts.harness_schema


# States, capabilities, transcript ----------------------------------------------------------------


def test_state_machine_follows_the_spec():
    machine = TaskStateMachine()
    for state in (TaskState.running, TaskState.validating, TaskState.repair_needed, TaskState.running):
        machine.to(state)
    machine.to(TaskState.validating)
    machine.to(TaskState.candidate_ready)
    with pytest.raises(InvalidTransition):
        machine.to(TaskState.running)
    machine.to(TaskState.accepted)
    assert not machine.active
    with pytest.raises(InvalidTransition):
        machine.to(TaskState.cancelled)


@pytest.mark.parametrize("state", sorted(ACTIVE))
def test_every_active_state_can_be_cancelled(state):
    from app.services.harness.state import TRANSITIONS

    assert TaskState.cancelled in TRANSITIONS[state]
    assert all(not TRANSITIONS[end] for end in TERMINAL)


def test_capability_expires_and_is_revoked():
    now = [0.0]
    store = CapabilityStore(ttl_s=10, clock=lambda: now[0])
    token = store.issue("t1")
    assert store.resolve(token) == "t1"
    now[0] = 11
    with pytest.raises(CapabilityError):
        store.resolve(token)

    other = store.issue("t2")
    store.revoke("t2")
    with pytest.raises(CapabilityError):
        store.resolve(other)


def test_transcript_redacts(tmp_path):
    transcript = Transcript(tmp_path / "t.jsonl", redact=lambda text: text.replace("sk-live-123", "[REDACTED]"))
    transcript.write("message.delta", {"task_id": "t", "text": "key sk-live-123", "nested": ["sk-live-123"]})

    line = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8"))
    assert "sk-live-123" not in json.dumps(line)
    assert line["event"] == "message.delta"


@pytest.mark.parametrize(
    ("level", "policy", "verdict"),
    [
        ("read", PermissionPolicy.deny_writes, "allow"),
        ("write", PermissionPolicy.ask, "ask"),
        ("write", PermissionPolicy.allow_in_draft, "allow"),
        ("write", PermissionPolicy.deny_writes, "deny"),
        ("dangerous", PermissionPolicy.allow_in_draft, "ask"),
        ("dangerous", PermissionPolicy.ask, "ask"),
        ("dangerous", PermissionPolicy.deny_writes, "deny"),
    ],
)
def test_permission_policy(level, policy, verdict):
    assert decide(level, policy) == verdict


# Harness with the notes app in process -----------------------------------------------------------


class ScriptedDriver:
    """Driver whose turns are async functions of the turn context."""

    needs_workdir = False
    needs_bridge = False

    def __init__(self, turns, needs_workdir=False):
        self.turns = list(turns)
        self.needs_workdir = needs_workdir
        self.prompts: list[str] = []
        self.closed = False
        self.results: list = []

    async def run_turn(self, ctx, prompt):
        self.prompts.append(prompt)
        turn = self.turns.pop(0)
        result = await turn(ctx)
        self.results.append(result)
        return "done"

    async def cancel(self):
        pass

    async def close(self):
        self.closed = True


def call(name, **args):
    async def turn(ctx):
        return await ctx.call_tool(name, args)

    return turn


def calls(*steps):
    async def turn(ctx):
        return [await ctx.call_tool(name, args) for name, args in steps]

    return turn


class Rig:
    """The harness wired to an in-process notes app over a real JSON-RPC pipe."""

    def __init__(self, tmp_path: Path):
        self.notes = NotesAdapter(tmp_path / "data")
        self.events: list[tuple[str, dict]] = []
        self.driver: ScriptedDriver | None = None
        self.tmp_path = tmp_path

    async def start(self, driver: ScriptedDriver, **options) -> Harness:
        harness_side, app_side = await _peers({}, adapter_handlers(self.notes))
        self.app_side = app_side
        self.driver = driver

        async def emit(event, data):
            self.events.append((event, data))

        self.harness = Harness(AdapterClient(harness_side), lambda: driver, emit, **options)
        return self.harness

    def of(self, event: str) -> list[dict]:
        return [data for name, data in self.events if name == event]

    async def run(self, goal="Write a note", **kw) -> tuple[str, TaskState]:
        task_id = await self.harness.start_task(goal, **kw)
        return task_id, await self._settle(task_id)

    async def _settle(self, task_id: str) -> TaskState:
        for _ in range(200):
            state = self.harness.state(task_id)
            if state in TERMINAL:
                await self.harness.wait(task_id)
            if state in TERMINAL or state == TaskState.candidate_ready:
                await self.harness.flush()
                return state
            await asyncio.sleep(0.01)
        raise AssertionError(f"task stuck in {self.harness.state(task_id)}")

    async def answer_permissions(self, choice: str, count: int = 1) -> None:
        for _ in range(200):
            await self.harness.flush()
            asked = self.of("permission.requested")
            if len(asked) >= count:
                self.harness.reply_permission(asked[count - 1]["id"], choice)
                return
            await asyncio.sleep(0.01)
        raise AssertionError("no permission request")


@pytest.fixture
async def rig(tmp_path):
    rig = Rig(tmp_path)
    yield rig
    await rig.harness.close()


WRITE = ("write_note", {"id": "a", "title": "Alpha", "body": "first"})


async def test_task_runs_to_candidate_without_touching_data(rig):
    harness = await rig.start(ScriptedDriver([calls(WRITE, ("read_note", {"id": "a"}))]))
    task_id, state = await rig.run()

    assert state == TaskState.candidate_ready
    assert rig.notes.context().state == {"notes": {}}
    assert json.loads(rig.driver.results[0][1].text)["title"] == "Alpha"  # reads see the draft
    assert rig.of("task.candidate")[0]["diff"]["summary"] == "1 added, 0 changed, 0 removed"

    revision = await harness.accept(task_id)
    assert rig.notes.context().state == {"notes": {"a": "Alpha"}}
    assert revision == "1" and harness.state(task_id) == TaskState.accepted
    assert rig.driver.closed


async def test_auto_review_commits(rig):
    await rig.start(ScriptedDriver([calls(WRITE)]), review="auto")
    _, state = await rig.run()
    assert state == TaskState.accepted
    assert rig.notes.context().revision == "1"


async def test_one_repair_then_failed(rig):
    bad = ("write_note", {"id": "a", "title": "", "body": ""})
    await rig.start(ScriptedDriver([calls(bad), calls(bad)]))
    _, state = await rig.run()

    assert state == TaskState.failed
    assert len(rig.driver.prompts) == 2 and "note a has no title" in rig.driver.prompts[1]
    assert rig.notes.context().revision == "0"


async def test_repair_succeeds(rig):
    bad = ("write_note", {"id": "a", "title": "", "body": ""})
    await rig.start(ScriptedDriver([calls(bad), calls(WRITE)]))
    _, state = await rig.run()
    states = [data["state"] for data in rig.of("task.state")]

    assert state == TaskState.candidate_ready
    assert states == ["running", "validating", "repair_needed", "running", "validating", "candidate_ready"]


async def test_effects_run_after_commit_in_order(rig):
    publish = ("publish_note", {"id": "a"})
    harness = await rig.start(ScriptedDriver([calls(WRITE, publish)]))
    task_id = await harness.start_task("Write and publish")
    await rig.answer_permissions("allow")
    state = await rig._settle(task_id)
    outbox = rig.notes.data_dir / "outbox"

    assert state == TaskState.candidate_ready
    assert "Queued" in rig.driver.results[0][1].text
    assert not any(outbox.iterdir())  # approved, but not run before the commit
    assert [e["op"] for e in rig.of("task.candidate")[0]["effects"]] == ["publish_note"]

    await harness.accept(task_id)
    await harness.flush()
    assert (outbox / "a.md").exists()
    assert len(rig.of("effect.done")) == 1


async def test_effect_failure_keeps_revision_and_runs_the_rest(rig):
    steps = calls(WRITE, ("publish_note", {"id": "missing"}), ("publish_note", {"id": "a"}))
    harness = await rig.start(ScriptedDriver([steps]))
    task_id = await harness.start_task("Publish")
    await rig.answer_permissions("allow", 1)
    await rig.answer_permissions("allow", 2)
    await rig._settle(task_id)
    revision = await harness.accept(task_id)
    await harness.flush()

    assert rig.notes.context().revision == revision
    assert len(rig.of("effect.failed")) == 1 and len(rig.of("effect.done")) == 1
    assert (rig.notes.data_dir / "outbox" / "a.md").exists()


async def test_effects_dropped_on_reject(rig):
    harness = await rig.start(ScriptedDriver([calls(WRITE, ("publish_note", {"id": "a"}))]))
    task_id = await harness.start_task("Publish")
    await rig.answer_permissions("allow")
    await rig._settle(task_id)
    await harness.reject(task_id)

    assert harness.state(task_id) == TaskState.rejected
    assert not any((rig.notes.data_dir / "outbox").iterdir())
    assert rig.notes.context().revision == "0"


async def test_conflict_when_base_changed(rig):
    harness = await rig.start(ScriptedDriver([calls(WRITE)]))
    task_id, _ = await rig.run()
    other = rig.notes.open_draft("0")
    rig.notes.call(other, "write_note", {"id": "z", "title": "Zed", "body": ""})
    rig.notes.commit(other, "0")

    with pytest.raises(RpcError):
        await harness.accept(task_id)
    assert harness.state(task_id) == TaskState.conflict
    assert rig.notes.context().state == {"notes": {"z": "Zed"}}


async def test_single_writer(rig):
    gate = asyncio.Event()

    async def wait(ctx):
        await gate.wait()

    harness = await rig.start(ScriptedDriver([wait]))
    await harness.start_task("first")
    with pytest.raises(RpcError):
        await harness.start_task("second")
    gate.set()


async def test_cancel_while_running(rig):
    async def forever(ctx):
        await asyncio.sleep(60)

    harness = await rig.start(ScriptedDriver([forever]))
    task_id = await harness.start_task("slow")
    await asyncio.sleep(0.05)
    await harness.cancel(task_id)

    assert harness.state(task_id) == TaskState.cancelled
    assert rig.driver.closed


async def test_cancel_while_waiting_permission_rejects_the_request(rig):
    harness = await rig.start(ScriptedDriver([calls(WRITE)]), policy=PermissionPolicy.ask)
    task_id = await harness.start_task("ask me")
    for _ in range(100):
        if harness.state(task_id) == TaskState.waiting_permission:
            break
        await asyncio.sleep(0.01)
    request = rig.of("permission.requested") or [None]
    await harness.cancel(task_id)
    await harness.flush()

    assert harness.state(task_id) == TaskState.cancelled
    assert rig.of("permission.closed")[0]["outcome"] == "cancelled"
    with pytest.raises(RpcError):  # a late answer is refused
        harness.reply_permission(request[0]["id"], "allow")


async def test_permission_replies_are_taken_once(rig):
    harness = await rig.start(ScriptedDriver([calls(WRITE)]), policy=PermissionPolicy.ask)
    task_id = await harness.start_task("ask me")
    await rig.answer_permissions("reject")
    await rig._settle(task_id)
    request_id = rig.of("permission.requested")[0]["id"]

    with pytest.raises(RpcError):
        harness.reply_permission(request_id, "allow")
    with pytest.raises(RpcError):
        harness.reply_permission("no-such-id", "allow")
    assert rig.driver.results[0][0].is_error  # the rejected write did not run
    assert rig.of("task.candidate")[0]["diff"]["summary"].startswith("0 added")


async def test_deny_writes(rig):
    await rig.start(ScriptedDriver([calls(WRITE, ("publish_note", {"id": "a"}))]), policy=PermissionPolicy.deny_writes)
    await rig.run()

    assert all(result.is_error for result in rig.driver.results[0])
    assert rig.of("permission.requested") == []


async def test_capability_revoked_after_task(rig):
    seen = {}

    async def remember(ctx):
        seen["ctx"] = ctx
        return await ctx.call_tool(*WRITE)

    harness = await rig.start(ScriptedDriver([remember]))
    task_id, _ = await rig.run()
    await harness.reject(task_id)
    late = await seen["ctx"].call_tool("list_notes", {})

    assert late.is_error and "capability" in late.text
    assert await seen["ctx"].list_tools() == []


async def test_calls_outside_running_are_refused(rig):
    seen = {}

    async def remember(ctx):
        seen["ctx"] = ctx

    await rig.start(ScriptedDriver([remember]))
    await rig.run()  # candidate_ready: the agent's turn is over
    late = await seen["ctx"].call_tool(*WRITE)
    assert late.is_error and "not running" in late.text


async def test_undo_latest_only(rig):
    harness = await rig.start(ScriptedDriver([calls(WRITE)]), review="auto")
    await rig.run()
    with pytest.raises(RpcError):
        await harness.undo("0")
    revision = await harness.undo("1")

    assert revision == "2"
    assert rig.notes.context().state == {"notes": {}}


async def test_driver_crash_fails_task_and_cleans_up(rig):
    async def crash(ctx):
        raise DriverError("agent exited")

    harness = await rig.start(ScriptedDriver([crash], needs_workdir=True))
    task_id, state = await rig.run()
    task = harness._tasks[task_id]

    assert state == TaskState.failed
    assert task.workdir is not None and not task.workdir.exists()
    assert (await harness.broker.list_tools("anything")) == []


async def test_workdir_files_become_the_draft(rig):
    async def edit(ctx):
        (ctx.workdir / "b.md").write_text("# Beta\n\nfrom a file\n", encoding="utf-8")

    harness = await rig.start(ScriptedDriver([edit], needs_workdir=True))
    task_id, state = await rig.run()
    await harness.accept(task_id)

    assert state == TaskState.candidate_ready
    assert rig.notes.context().state == {"notes": {"b": "Beta"}}


async def test_malformed_file_counts_as_failed_validation(rig):
    async def bad(ctx):
        (ctx.workdir / "c.md").write_text("no heading", encoding="utf-8")

    async def fix(ctx):
        (ctx.workdir / "c.md").write_text("# Gamma\n", encoding="utf-8")

    await rig.start(ScriptedDriver([bad, fix], needs_workdir=True))
    _, state = await rig.run()

    assert state == TaskState.candidate_ready
    assert "c.md must start with" in rig.driver.prompts[1]


async def test_events_are_redacted(rig):
    rig_harness = await rig.start(
        ScriptedDriver([calls(("write_note", {"id": "k", "title": "key sk-live-123", "body": ""}))]),
        redact=lambda text: text.replace("sk-live-123", "[REDACTED]"),
        transcript_dir=rig.tmp_path / "transcripts",
    )
    task_id, _ = await rig.run(policy=PermissionPolicy.allow_in_draft)
    await rig_harness.flush()

    assert "sk-live-123" not in json.dumps(rig.events)
    assert "sk-live-123" not in (rig.tmp_path / "transcripts" / f"{task_id}.jsonl").read_text(encoding="utf-8")
