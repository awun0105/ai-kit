"""The harness: runs tasks through draft, agent turns, validation, review and revision.

One writing task at a time. The task's tools are served by the broker behind a
capability token; external agents work in a working directory made from the
draft. Approved ``dangerous`` operations run only after the commit. Every event
goes out redacted, in order, through ``emit`` (the app's event stream) and to
the task's transcript.
"""

import asyncio
import contextlib
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    Optional,
)

import structlog

from app.services.harness.adapter import AdapterClient
from app.services.harness.bridge import (
    ToolBridge,
    mcp_server_spec,
)
from app.services.harness.broker import (
    TaskRuntime,
    ToolBroker,
)
from app.services.harness.capability import CapabilityStore
from app.services.harness.driver import (
    AgentDriver,
    DriverFactory,
    TurnContext,
)
from app.services.harness.permissions import (
    PermissionDesk,
    PermissionPolicy,
)
from app.services.harness.protocol import (
    AppDescription,
    ValidationReport,
)
from app.services.harness.rpc import (
    APP_ERROR,
    RpcError,
)
from app.services.harness.state import (
    TaskState,
    TaskStateMachine,
)
from app.services.harness.transcript import (
    Transcript,
    redact_values,
)

logger = structlog.get_logger()

Review = Literal["auto", "manual"]

FIRST_PROMPT = """{goal}

Scope: {scope}

Work only through the app's tools{files}. Your changes go to a draft that the user reviews.
Call validate_draft before you finish and fix what it reports."""

FILES_NOTE = " and the files in your working directory"

REPAIR_PROMPT = "Validation failed. Fix these problems, then finish:\n- {issues}"


def _identity(text: str) -> str:
    return text


class Harness:
    """Runs in-app agent tasks against one app adapter."""

    def __init__(
        self,
        adapter: AdapterClient,
        driver_factory: DriverFactory,
        emit: Callable[[str, dict], Awaitable[None]],
        *,
        policy: PermissionPolicy = PermissionPolicy.allow_in_draft,
        review: Review = "manual",
        max_repairs: int = 1,
        capability_ttl_s: float = 3600,
        redact: Callable[[str], str] = _identity,
        transcript_dir: Optional[Path] = None,
    ):
        """Wire the harness.

        Args:
            adapter: Client for the app's adapter.
            driver_factory: Makes one agent driver per task.
            emit: Sends an event to the app (``peer.notify`` on the sidecar connection).
            policy: Default permission policy; a task may pick another.
            review: ``manual`` waits for ``candidate.accept``; ``auto`` commits a candidate that validates.
            max_repairs: Repair turns after a failed validation before the task fails.
            capability_ttl_s: Lifetime of a task's capability at most.
            redact: Applied to every string in events and transcripts (``guardrails.redact`` in the app).
            transcript_dir: Where to write one JSONL transcript per task; ``None`` writes none.
        """
        self.adapter = adapter
        self._driver_factory = driver_factory
        self._emit_out = emit
        self._policy = policy
        self._review: Review = review
        self._max_repairs = max_repairs
        self._redact = redact
        self._transcript_dir = transcript_dir
        self._caps = CapabilityStore(ttl_s=capability_ttl_s)
        self._desk = PermissionDesk(self._emit)
        self.broker = ToolBroker(self._tasks_get, self._caps, self._desk, self._emit)
        self._bridge = ToolBridge(self.broker)
        self._tasks: dict[str, TaskRuntime] = {}
        self._drivers: dict[str, AgentDriver] = {}
        self._runs: dict[str, asyncio.Task] = {}
        self._done: dict[str, asyncio.Event] = {}
        self._transcripts: dict[str, Transcript] = {}
        self._writer: Optional[str] = None
        self._latest_revision: Optional[str] = None
        self._description: Optional[AppDescription] = None
        self._events: asyncio.Queue = asyncio.Queue()
        self._pump: Optional[asyncio.Task] = None

    # Events -------------------------------------------------------------------------------------

    def _tasks_get(self, task_id: str) -> Optional[TaskRuntime]:
        return self._tasks.get(task_id)

    def _post(self, event: str, data: dict) -> None:
        """Queue an event; one pump delivers them in order."""
        data = redact_values(data, self._redact)
        transcript = self._transcripts.get(str(data.get("task_id", "")))
        if transcript is not None:
            transcript.write(event, data)
        if self._pump is None or self._pump.done():
            self._pump = asyncio.create_task(self._deliver())
        self._events.put_nowait((event, data))

    async def _emit(self, event: str, data: dict) -> None:
        self._post(event, data)

    async def _deliver(self) -> None:
        while True:
            event, data = await self._events.get()
            try:
                await self._emit_out(event, data)
            except Exception:
                logger.exception("harness_event_delivery_failed", harness_event=event)
            finally:
                self._events.task_done()

    async def flush(self) -> None:
        """Wait until every queued event has been delivered."""
        await self._events.join()

    def _move(self, task: TaskRuntime, state: TaskState) -> None:
        task.machine.to(state)

    # App-facing API -----------------------------------------------------------------------------

    async def describe(self) -> AppDescription:
        """The app's description (asked once)."""
        if self._description is None:
            self._description = await self.adapter.describe()
        return self._description

    async def start_task(
        self,
        goal: str,
        scope: Any = None,
        policy: Optional[PermissionPolicy] = None,
        review: Optional[Review] = None,
    ) -> str:
        """Start a task and return its id; refused while another task is writing."""
        if self._writer is not None:
            raise RpcError(APP_ERROR, "another task is writing")
        description = await self.describe()
        task_id = uuid.uuid4().hex[:12]

        def on_change(old: TaskState, new: TaskState) -> None:
            self._post("task.state", {"task_id": task_id, "state": new.value})

        task = TaskRuntime(
            id=task_id,
            goal=goal,
            scope=scope,
            policy=PermissionPolicy(policy or self._policy),
            adapter=self.adapter,
            operations={op.name: op for op in description.operations},
            machine=TaskStateMachine(on_change),
        )
        self._writer = task_id
        self._tasks[task_id] = task
        self._done[task_id] = asyncio.Event()
        if self._transcript_dir is not None:
            self._transcripts[task_id] = Transcript(self._transcript_dir / f"{task_id}.jsonl", self._redact)
        self._post("task.started", {"task_id": task_id, "goal": goal, "scope": scope})
        self._runs[task_id] = asyncio.create_task(self._run(task, review or self._review))
        return task_id

    def reply_permission(self, request_id: str, choice: str) -> None:
        """Answer a ``permission.requested`` event."""
        self._desk.reply(request_id, choice)

    async def cancel(self, task_id: str) -> None:
        """Cancel a task that has not ended."""
        task = self._active(task_id)
        driver = self._drivers.get(task_id)
        if driver is not None:
            with contextlib.suppress(Exception):
                await driver.cancel()
        run = self._runs.get(task_id)
        if run is not None and not run.done():
            run.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await run
        if task.machine.active:
            self._move(task, TaskState.cancelled)
            await self._finish(task)

    async def accept(self, task_id: str) -> str:
        """Commit a candidate (``manual`` review); returns the new revision."""
        task = self._active(task_id)
        if task.machine.state != TaskState.candidate_ready:
            raise RpcError(APP_ERROR, "the task has no candidate to accept")
        return await self._accept(task)

    async def reject(self, task_id: str) -> None:
        """Drop a candidate."""
        task = self._active(task_id)
        if task.machine.state != TaskState.candidate_ready:
            raise RpcError(APP_ERROR, "the task has no candidate to reject")
        self._move(task, TaskState.rejected)
        await self._finish(task)

    async def undo(self, revision: str) -> str:
        """Undo the latest revision this harness committed; returns the revision that undoes it."""
        if self._writer is not None:
            raise RpcError(APP_ERROR, "another task is writing")
        if revision != self._latest_revision:
            raise RpcError(APP_ERROR, "only the latest revision can be undone")
        draft_id = await self.adapter.undo(revision)
        report = await self.adapter.validate(draft_id)
        if not report.ok:
            await self.adapter.discard(draft_id)
            raise RpcError(APP_ERROR, "the undo does not pass validation", report.model_dump())
        result = await self.adapter.commit(draft_id, revision)
        if result.conflict or result.revision is None:
            await self.adapter.discard(draft_id)
            raise RpcError(APP_ERROR, "the app changed since that revision")
        self._latest_revision = result.revision
        self._post("revision.undone", {"revision": revision, "new_revision": result.revision})
        return result.revision

    def state(self, task_id: str) -> TaskState:
        """The task's current state."""
        task = self._tasks.get(task_id)
        if task is None:
            raise RpcError(APP_ERROR, "unknown task")
        return task.machine.state

    async def wait(self, task_id: str) -> TaskState:
        """Wait until the task ends; returns its final state."""
        if task_id not in self._done:
            raise RpcError(APP_ERROR, "unknown task")
        await self._done[task_id].wait()
        return self._tasks[task_id].machine.state

    async def close(self) -> None:
        """Cancel the running task, stop the bridge and deliver the last events."""
        if self._writer is not None:
            with contextlib.suppress(RpcError):
                await self.cancel(self._writer)
        await self._bridge.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.flush(), timeout=5)
        if self._pump is not None:
            self._pump.cancel()

    # Task run -----------------------------------------------------------------------------------

    def _active(self, task_id: str) -> TaskRuntime:
        task = self._tasks.get(task_id)
        if task is None or not task.machine.active:
            raise RpcError(APP_ERROR, "unknown or finished task")
        return task

    async def _run(self, task: TaskRuntime, review: Review) -> None:
        try:
            await self._prepare_and_work(task)
            if not task.machine.active:
                return
            if review == "auto":
                await self._accept(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("harness_task_failed", task_id=task.id)
            if task.machine.active:
                self._move(task, TaskState.failed)
                await self._finish(task)

    async def _prepare_and_work(self, task: TaskRuntime) -> None:
        context = await self.adapter.context()
        task.base_revision = context.revision
        task.draft_id = await self.adapter.open_draft(context.revision)
        driver = self._driver_factory()
        self._drivers[task.id] = driver
        if driver.needs_workdir:
            if "materialize" not in (await self.describe()).supports:
                raise RuntimeError("the app does not support working directories for external agents")
            task.workdir = Path(tempfile.mkdtemp(prefix=f"harness-{task.id}-"))
            await self.adapter.materialize(task.draft_id, str(task.workdir))
        token = self._caps.issue(task.id)
        mcp_server = mcp_server_spec(await self._bridge.start(), token) if driver.needs_bridge else None

        async def ask(title: str, kind: str, detail: Optional[dict]) -> bool:
            return await self.broker.ask(task, title, kind, detail)

        async def list_tools() -> list:
            return await self.broker.list_tools(token)

        async def call_tool(name: str, args: dict) -> Any:
            return await self.broker.call(token, name, args)

        ctx = TurnContext(
            task_id=task.id,
            goal=task.goal,
            scope=task.scope,
            token=token,
            workdir=task.workdir,
            list_tools=list_tools,
            call_tool=call_tool,
            ask_permission=ask,
            emit=self._emit,
            mcp_server=mcp_server,
        )
        prompt = FIRST_PROMPT.format(
            goal=task.goal,
            scope=task.scope if task.scope is not None else "the whole app",
            files=FILES_NOTE if task.workdir else "",
        )
        repairs = 0
        while True:
            self._move(task, TaskState.running)
            text = await driver.run_turn(ctx, prompt)
            if text:
                self._post("message.final", {"task_id": task.id, "text": text})
            self._move(task, TaskState.validating)
            report = await self._check(task)
            if report.ok:
                break
            if repairs >= self._max_repairs:
                self._post("task.validation", {"task_id": task.id, "report": report.model_dump()})
                self._move(task, TaskState.failed)
                await self._finish(task)
                return
            repairs += 1
            self._move(task, TaskState.repair_needed)
            prompt = REPAIR_PROMPT.format(issues="\n- ".join(issue.message for issue in report.issues))

        await self._close_driver(task)
        diff = await self.adapter.diff(task.draft_id)
        self._move(task, TaskState.candidate_ready)
        self._post(
            "task.candidate",
            {"task_id": task.id, "diff": diff.model_dump(), "effects": [e.model_dump() for e in task.effects]},
        )

    async def _check(self, task: TaskRuntime) -> ValidationReport:
        if task.workdir is not None:
            synced = await self.adapter.sync(task.draft_id)
            if not synced.ok:
                return synced
        return await self.adapter.validate(task.draft_id)

    async def _accept(self, task: TaskRuntime) -> str:
        result = await self.adapter.commit(task.draft_id, task.base_revision)
        if result.conflict or result.revision is None:
            self._move(task, TaskState.conflict)
            await self._finish(task)
            raise RpcError(APP_ERROR, "the app changed since the task started")
        self._move(task, TaskState.accepted)
        self._latest_revision = result.revision
        for effect in task.effects:
            try:
                await self.adapter.run_effect(result.revision, effect)
                self._post("effect.done", {"task_id": task.id, "revision": result.revision, "effect_id": effect.id})
            except Exception:
                logger.exception("harness_effect_failed", task_id=task.id, op=effect.op)
                self._post("effect.failed", {"task_id": task.id, "revision": result.revision, "effect_id": effect.id})
        await self._finish(task, result.revision)
        return result.revision

    async def _close_driver(self, task: TaskRuntime) -> None:
        driver = self._drivers.pop(task.id, None)
        if driver is not None:
            try:
                await driver.close()
            except Exception:
                logger.exception("harness_driver_close_failed", task_id=task.id)

    async def _finish(self, task: TaskRuntime, revision: Optional[str] = None) -> None:
        """Release everything the task held, however it ended."""
        self._caps.revoke(task.id)
        self._desk.close_task(task.id)
        await self._close_driver(task)
        if task.machine.state != TaskState.accepted and task.draft_id:
            try:
                await self.adapter.discard(task.draft_id)
            except Exception:
                logger.exception("harness_discard_failed", task_id=task.id)
        if task.workdir is not None:
            shutil.rmtree(task.workdir, ignore_errors=True)
        if self._writer == task.id:
            self._writer = None
        self._post("task.finished", {"task_id": task.id, "state": task.machine.state.value, "revision": revision})
        self._done[task.id].set()
