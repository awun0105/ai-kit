"""The adapter protocol (spec section 5.1): the harness calls the app through these methods.

``AdapterClient`` is the harness side. ``AppAdapter`` and ``adapter_handlers``
let a Python app serve its adapter on the shared connection; an app in another
language implements the same wire methods itself.
"""

import inspect
from typing import (
    Any,
    Optional,
    Protocol,
)

from app.services.harness.protocol import (
    AppContext,
    AppDescription,
    CommitResult,
    DraftDiff,
    Effect,
    ValidationReport,
)
from app.services.harness.rpc import (
    INVALID_PARAMS,
    Handler,
    JsonRpcPeer,
    RpcError,
)


class AdapterClient:
    """Typed calls to the app's adapter over the connection."""

    def __init__(self, peer: JsonRpcPeer, timeout_s: float = 60):
        """Call the adapter through ``peer``; each call waits at most ``timeout_s``."""
        self.peer = peer
        self._timeout_s = timeout_s

    async def _call(self, method: str, params: Optional[dict] = None) -> Any:
        return await self.peer.request(method, params, timeout=self._timeout_s)

    async def describe(self) -> AppDescription:
        """The operations and features of the app (``app.describe``)."""
        return AppDescription.model_validate(await self._call("app.describe"))

    async def context(self) -> AppContext:
        """The current revision and a bounded view of the state (``app.context``)."""
        return AppContext.model_validate(await self._call("app.context"))

    async def open_draft(self, base_revision: str) -> str:
        """Open a draft from ``base_revision``; returns its id (``draft.open``)."""
        return (await self._call("draft.open", {"base_revision": base_revision}))["draft_id"]

    async def call(self, draft_id: Optional[str], op: str, args: dict) -> Any:
        """Run one operation, on the draft when ``draft_id`` is given (``op.call``)."""
        return await self._call("op.call", {"draft_id": draft_id, "op": op, "args": args})

    async def validate(self, draft_id: str) -> ValidationReport:
        """Check the draft (``draft.validate``)."""
        return ValidationReport.model_validate(await self._call("draft.validate", {"draft_id": draft_id}))

    async def diff(self, draft_id: str) -> DraftDiff:
        """What the draft changes compared with its base (``draft.diff``)."""
        return DraftDiff.model_validate(await self._call("draft.diff", {"draft_id": draft_id}))

    async def commit(self, draft_id: str, base_revision: str) -> CommitResult:
        """Make the draft the next revision, or report a conflict (``draft.commit``)."""
        result = await self._call("draft.commit", {"draft_id": draft_id, "base_revision": base_revision})
        return CommitResult.model_validate(result)

    async def discard(self, draft_id: str) -> None:
        """Drop the draft (``draft.discard``)."""
        await self._call("draft.discard", {"draft_id": draft_id})

    async def undo(self, revision: str) -> str:
        """Open a draft that reverses ``revision``, the latest one (``revision.undo``)."""
        return (await self._call("revision.undo", {"revision": revision}))["draft_id"]

    async def materialize(self, draft_id: str, path: str) -> None:
        """Write the draft as files into ``path`` (``draft.materialize``)."""
        await self._call("draft.materialize", {"draft_id": draft_id, "path": path})

    async def sync(self, draft_id: str) -> ValidationReport:
        """Read the working directory back into the draft (``draft.sync``)."""
        return ValidationReport.model_validate(await self._call("draft.sync", {"draft_id": draft_id}))

    async def run_effect(self, revision: str, effect: Effect) -> Any:
        """Run an approved dangerous operation after its revision was committed (``effect.run``)."""
        return await self._call("effect.run", {"revision": revision, "effect": effect.model_dump()})


class AppAdapter(Protocol):
    """What a Python app implements to serve its adapter (methods may be sync or async).

    Raise ``RpcError(APP_ERROR, ...)`` for an expected failure (unknown note,
    write without a draft); the message goes to the harness as is. Each method
    is the wire method of the same meaning on ``AdapterClient``.
    """

    # ruff: noqa: D102  (documented on AdapterClient)

    def describe(self) -> AppDescription: ...
    def context(self) -> AppContext: ...
    def open_draft(self, base_revision: str) -> str: ...
    def call(self, draft_id: Optional[str], op: str, args: dict) -> Any: ...
    def validate(self, draft_id: str) -> ValidationReport: ...
    def diff(self, draft_id: str) -> DraftDiff: ...
    def commit(self, draft_id: str, base_revision: str) -> CommitResult: ...
    def discard(self, draft_id: str) -> None: ...
    def undo(self, revision: str) -> str: ...
    def materialize(self, draft_id: str, path: str) -> None: ...
    def sync(self, draft_id: str) -> ValidationReport: ...
    def run_effect(self, revision: str, effect: Effect) -> Any: ...


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def adapter_handlers(adapter: Any) -> dict[str, Handler]:
    """Wire methods serving ``adapter`` (an ``AppAdapter``) on a ``JsonRpcPeer``."""

    def need(params: dict, *keys: str) -> list[Any]:
        missing = [key for key in keys if key not in params]
        if missing:
            raise RpcError(INVALID_PARAMS, f"missing params: {', '.join(missing)}")
        return [params[key] for key in keys]

    async def describe(params: dict) -> Any:
        return _dump(await _maybe_await(adapter.describe()))

    async def context(params: dict) -> Any:
        """The current revision and a bounded view of the state (``app.context``)."""
        return _dump(await _maybe_await(adapter.context()))

    async def open_draft(params: dict) -> Any:
        """Open a draft from ``base_revision``; returns its id (``draft.open``)."""
        (base,) = need(params, "base_revision")
        return {"draft_id": await _maybe_await(adapter.open_draft(base))}

    async def call(params: dict) -> Any:
        """Run one operation, on the draft when ``draft_id`` is given (``op.call``)."""
        op, args = need(params, "op", "args")
        return _dump(await _maybe_await(adapter.call(params.get("draft_id"), op, args)))

    async def validate(params: dict) -> Any:
        """Check the draft (``draft.validate``)."""
        (draft_id,) = need(params, "draft_id")
        return _dump(await _maybe_await(adapter.validate(draft_id)))

    async def diff(params: dict) -> Any:
        """What the draft changes compared with its base (``draft.diff``)."""
        (draft_id,) = need(params, "draft_id")
        return _dump(await _maybe_await(adapter.diff(draft_id)))

    async def commit(params: dict) -> Any:
        """Make the draft the next revision, or report a conflict (``draft.commit``)."""
        draft_id, base = need(params, "draft_id", "base_revision")
        return _dump(await _maybe_await(adapter.commit(draft_id, base)))

    async def discard(params: dict) -> Any:
        """Drop the draft (``draft.discard``)."""
        (draft_id,) = need(params, "draft_id")
        await _maybe_await(adapter.discard(draft_id))
        return {}

    async def undo(params: dict) -> Any:
        """Open a draft that reverses ``revision``, the latest one (``revision.undo``)."""
        (revision,) = need(params, "revision")
        return {"draft_id": await _maybe_await(adapter.undo(revision))}

    async def materialize(params: dict) -> Any:
        """Write the draft as files into ``path`` (``draft.materialize``)."""
        draft_id, path = need(params, "draft_id", "path")
        await _maybe_await(adapter.materialize(draft_id, path))
        return {}

    async def sync(params: dict) -> Any:
        """Read the working directory back into the draft (``draft.sync``)."""
        (draft_id,) = need(params, "draft_id")
        return _dump(await _maybe_await(adapter.sync(draft_id)))

    async def run_effect(params: dict) -> Any:
        """Run an approved dangerous operation after its revision was committed (``effect.run``)."""
        revision, effect = need(params, "revision", "effect")
        return _dump(await _maybe_await(adapter.run_effect(revision, Effect.model_validate(effect))))

    return {
        "app.describe": describe,
        "app.context": context,
        "draft.open": open_draft,
        "op.call": call,
        "draft.validate": validate,
        "draft.diff": diff,
        "draft.commit": commit,
        "draft.discard": discard,
        "revision.undo": undo,
        "draft.materialize": materialize,
        "draft.sync": sync,
        "effect.run": run_effect,
    }
