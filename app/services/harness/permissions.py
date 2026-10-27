"""Permission policy (spec section 5.5) and the desk that asks the user and takes the answer once.

Every request, from a harness tool or from the agent itself (an ACP permission
request), goes out as one ``permission.requested`` event and is answered with
``permission.reply``. A reply to an unknown, answered or closed request is
refused, as fframes Studio's broker does.
"""

import asyncio
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    Optional,
)

from app.services.harness.protocol import OpLevel
from app.services.harness.rpc import (
    APP_ERROR,
    RpcError,
)

Emit = Callable[[str, dict], Awaitable[None]]
Verdict = Literal["allow", "ask", "deny"]


class PermissionPolicy(StrEnum):
    """How much a task may do without asking."""

    ask = "ask"  # ask for every write
    allow_in_draft = "allow_in_draft"  # writes land in the draft without asking
    deny_writes = "deny_writes"  # read only


def decide(level: OpLevel, policy: PermissionPolicy) -> Verdict:
    """Whether an operation of ``level`` runs, needs the user, or is refused under ``policy``.

    ``dangerous`` always needs the user; no policy turns that off.
    """
    if level == "read":
        return "allow"
    if policy == PermissionPolicy.deny_writes:
        return "deny"
    if level == "dangerous" or policy == PermissionPolicy.ask:
        return "ask"
    return "allow"


@dataclass
class _Request:
    task_id: str
    future: asyncio.Future


class PermissionDesk:
    """Pending permission requests, each answered at most once."""

    def __init__(self, emit: Emit):
        """Send ``permission.*`` events through ``emit(event, data)``."""
        self._emit = emit
        self._pending: dict[str, _Request] = {}

    async def ask(self, task_id: str, title: str, kind: str, detail: Optional[dict[str, Any]] = None) -> bool:
        """Ask the user; ``True`` when allowed. A task that ends meanwhile resolves it as rejected."""
        request_id = uuid.uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = _Request(task_id, future)
        await self._emit(
            "permission.requested",
            {"id": request_id, "task_id": task_id, "title": title, "kind": kind, "detail": detail or {}},
        )
        try:
            outcome = await future
        except asyncio.CancelledError:
            await self._emit("permission.closed", {"id": request_id, "outcome": "cancelled"})
            raise
        finally:
            self._pending.pop(request_id, None)
        await self._emit("permission.closed", {"id": request_id, "outcome": outcome})
        return outcome == "allow"

    def reply(self, request_id: str, choice: str) -> None:
        """Answer a request with ``allow`` or ``reject``."""
        if choice not in ("allow", "reject"):
            raise RpcError(APP_ERROR, "choice must be allow or reject")
        request = self._pending.get(request_id)
        if request is None or request.future.done():
            raise RpcError(APP_ERROR, "unknown or closed permission request")
        request.future.set_result(choice)

    def close_task(self, task_id: str) -> None:
        """Resolve the task's pending requests as cancelled (which counts as rejected)."""
        for request in list(self._pending.values()):
            if request.task_id == task_id and not request.future.done():
                request.future.set_result("cancelled")
