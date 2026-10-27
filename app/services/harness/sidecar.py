"""The harness as a sidecar: the app starts it and talks to it over one stdio JSON-RPC connection.

On that connection the harness calls the app's adapter (``app.*``, ``draft.*``,
``op.call``, ``effect.run``), the app calls the harness (the methods below) and
receives events as notifications (``task.*``, ``permission.*``, ``message.*``,
``tool.call``, ``effect.*``).
"""

from pathlib import Path
from typing import (
    Callable,
    Optional,
)

from app.services.harness.adapter import AdapterClient
from app.services.harness.driver import DriverFactory
from app.services.harness.harness import (
    Harness,
    Review,
)
from app.services.harness.permissions import PermissionPolicy
from app.services.harness.rpc import (
    INVALID_PARAMS,
    Handler,
    RpcError,
    open_stdio_peer,
)


def _need(params: dict, key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise RpcError(INVALID_PARAMS, f"missing param: {key}")
    return value


def harness_handlers(harness: Harness) -> dict[str, Handler]:
    """The app-facing methods of ``harness``."""

    async def start(params: dict) -> dict:
        policy = params.get("policy")
        review = params.get("review")
        if policy is not None and policy not in PermissionPolicy.__members__:
            raise RpcError(INVALID_PARAMS, "unknown policy")
        if review is not None and review not in ("auto", "manual"):
            raise RpcError(INVALID_PARAMS, "review must be auto or manual")
        task_id = await harness.start_task(
            _need(params, "goal"),
            params.get("scope"),
            PermissionPolicy(policy) if policy else None,
            review,
        )
        return {"task_id": task_id}

    async def cancel(params: dict) -> dict:
        await harness.cancel(_need(params, "task_id"))
        return {}

    async def reply(params: dict) -> dict:
        harness.reply_permission(_need(params, "id"), _need(params, "choice"))
        return {}

    async def accept(params: dict) -> dict:
        return {"revision": await harness.accept(_need(params, "task_id"))}

    async def reject(params: dict) -> dict:
        await harness.reject(_need(params, "task_id"))
        return {}

    async def undo(params: dict) -> dict:
        return {"revision": await harness.undo(_need(params, "revision"))}

    async def state(params: dict) -> dict:
        return {"state": harness.state(_need(params, "task_id")).value}

    return {
        "task.start": start,
        "task.cancel": cancel,
        "task.state": state,
        "permission.reply": reply,
        "candidate.accept": accept,
        "candidate.reject": reject,
        "revision.undo": undo,
    }


async def run_sidecar(
    driver_factory: DriverFactory,
    out_fd: int,
    *,
    policy: PermissionPolicy = PermissionPolicy.allow_in_draft,
    review: Review = "manual",
    max_repairs: int = 1,
    transcript_dir: Optional[Path] = None,
    redact: Callable[[str], str] = lambda text: text,
) -> None:
    """Serve the harness on stdin and ``out_fd`` until the app closes the connection."""
    peer = await open_stdio_peer({}, out_fd, name="app")
    harness = Harness(
        AdapterClient(peer),
        driver_factory,
        peer.notify,
        policy=policy,
        review=review,
        max_repairs=max_repairs,
        redact=redact,
        transcript_dir=transcript_dir,
    )
    for method, handler in harness_handlers(harness).items():
        peer.on(method, handler)
    try:
        await peer.closed.wait()
    finally:
        await harness.close()
        await peer.close()
