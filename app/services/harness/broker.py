"""The tool broker: the task's operations as tools, behind its capability and permission policy.

The agent never names a task or a draft: the capability token picks the task,
and the broker fills in its draft. Writes land in the draft; approved
``dangerous`` operations are queued as effects that run after the revision is
committed. Rewritten from fframes Studio's ``agent_tools/broker.rs``.
"""

import json
import uuid
from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
)

import structlog

from app.services.harness.adapter import AdapterClient
from app.services.harness.capability import (
    CapabilityError,
    CapabilityStore,
)
from app.services.harness.permissions import (
    Emit,
    PermissionDesk,
    PermissionPolicy,
    decide,
)
from app.services.harness.protocol import (
    Effect,
    Operation,
)
from app.services.harness.rpc import RpcError
from app.services.harness.state import (
    TaskState,
    TaskStateMachine,
)

logger = structlog.get_logger()

TASK_CONTEXT = Operation(
    name="task_context",
    description="The task's goal, scope and base revision. Call it first.",
    level="read",
)
VALIDATE_DRAFT = Operation(
    name="validate_draft",
    description="Check the draft the way the app will before accepting it; fix what it reports before finishing.",
    level="read",
)


@dataclass
class ToolResult:
    """What a tool call returns to the agent."""

    text: str
    is_error: bool = False


@dataclass
class TaskRuntime:
    """A running task as the broker and the harness see it."""

    id: str
    goal: str
    scope: Any
    policy: PermissionPolicy
    adapter: AdapterClient
    operations: dict[str, Operation]
    machine: TaskStateMachine
    base_revision: str = ""
    draft_id: str = ""
    workdir: Optional[Path] = None
    effects: list[Effect] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    asking: int = 0


def _refuse(reason: str) -> ToolResult:
    return ToolResult(f"Tool call not executed: {reason}", is_error=True)


class ToolBroker:
    """Runs tool calls for whichever task a capability token names."""

    def __init__(
        self,
        lookup: Callable[[str], Optional[TaskRuntime]],
        caps: CapabilityStore,
        desk: PermissionDesk,
        emit: Emit,
    ):
        """``lookup(task_id)`` returns the task (``None`` once it is gone); ``emit`` sends ``tool.call`` events."""
        self._lookup = lookup
        self._caps = caps
        self._desk = desk
        self._emit = emit

    def _task(self, token: str) -> Optional[TaskRuntime]:
        try:
            return self._lookup(self._caps.resolve(token))
        except CapabilityError:
            return None

    async def list_tools(self, token: str) -> list[Operation]:
        """The tools of the token's task; empty when the token is not valid."""
        task = self._task(token)
        if task is None:
            return []
        return [TASK_CONTEXT, VALIDATE_DRAFT, *task.operations.values()]

    async def ask(self, task: TaskRuntime, title: str, kind: str, detail: Optional[dict] = None) -> bool:
        """Ask the user on behalf of ``task``, moving it to ``waiting_permission`` while the question is open."""
        if task.machine.state not in (TaskState.running, TaskState.waiting_permission):
            return False
        if task.asking == 0:
            task.machine.to(TaskState.waiting_permission)
        task.asking += 1
        try:
            allowed = await self._desk.ask(task.id, title, kind, detail)
        finally:
            task.asking -= 1
            if task.asking == 0 and task.machine.state == TaskState.waiting_permission:
                task.machine.to(TaskState.running)
        return allowed and task.machine.state == TaskState.running

    async def call(self, token: str, name: str, args: dict) -> ToolResult:
        """Run one tool call for the token's task."""
        task = self._task(token)
        if task is None:
            return _refuse("the capability is not valid (the task has ended or expired).")
        if task.machine.state not in (TaskState.running, TaskState.waiting_permission):
            return _refuse("the task is not running.")
        if name == TASK_CONTEXT.name:
            return ToolResult(
                json.dumps({"goal": task.goal, "scope": task.scope, "base_revision": task.base_revision})
            )
        if name == VALIDATE_DRAFT.name:
            return await self._validate(task)
        op = task.operations.get(name)
        if op is None:
            return _refuse(f"unknown tool {name}.")

        task.tools_called.append(name)
        await self._emit("tool.call", {"task_id": task.id, "name": name, "args": args, "level": op.level})
        verdict = decide(op.level, task.policy)
        if verdict == "deny":
            return _refuse("the task's permission policy does not allow changes.")
        if verdict == "ask":
            title = f"{name} {json.dumps(args, ensure_ascii=False)}"
            if not await self.ask(task, title, op.level, {"op": name, "args": args}):
                return _refuse("the user did not allow it.")
        if op.level == "dangerous":
            task.effects.append(Effect(id=uuid.uuid4().hex, op=name, args=args))
            return ToolResult(f"Queued: {name} runs after the user accepts the changes. Do not call it again.")
        return await self._run(task, name, args)

    async def _run(self, task: TaskRuntime, name: str, args: dict) -> ToolResult:
        try:
            if task.workdir is not None:
                # The agent edits files too: take its file edits into the draft before the tool sees it.
                synced = await task.adapter.sync(task.draft_id)
                if not synced.ok:
                    return ToolResult("Fix the files first: " + synced.model_dump_json(), is_error=True)
            # Reads go to the draft too, so the agent sees its own edits.
            result = await task.adapter.call(task.draft_id or None, name, args)
            if task.workdir is not None and task.operations[name].level == "write":
                await task.adapter.materialize(task.draft_id, str(task.workdir))
        except RpcError as error:
            return ToolResult(f"The app refused: {error.message}", is_error=True)
        except Exception:
            logger.exception("harness_tool_failed", task_id=task.id, tool=name)
            return ToolResult("The app could not run this tool.", is_error=True)
        return ToolResult(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False))

    async def _validate(self, task: TaskRuntime) -> ToolResult:
        try:
            if task.workdir is not None:
                synced = await task.adapter.sync(task.draft_id)
                if not synced.ok:
                    return ToolResult(synced.model_dump_json())
            report = await task.adapter.validate(task.draft_id)
        except RpcError as error:
            return ToolResult(f"The app refused: {error.message}", is_error=True)
        return ToolResult(report.model_dump_json())
