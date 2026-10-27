"""Adapter conformance: an app's adapter keeps the contract of spec section 5.1.

Runs against any adapter through the protocol only (see ``conftest.py``)::

    HARNESS_ADAPTER_CMD='["./my-app", "--adapter", "--data", "{data}"]' \\
    HARNESS_ADAPTER_FIXTURE=my_app_fixture.json uv run pytest tests/conformance/test_adapter.py
"""

import uuid

import pytest

from app.services.harness import (
    PROTOCOL_VERSION,
    AdapterClient,
    Effect,
    RpcError,
)


async def _write(adapter: AdapterClient, draft_id, sample: dict):
    return await adapter.call(draft_id, sample["op"], sample["args"])


async def _commit_sample(adapter: AdapterClient, ops: dict) -> tuple[str, str]:
    base = (await adapter.context()).revision
    draft = await adapter.open_draft(base)
    await _write(adapter, draft, ops["write"])
    result = await adapter.commit(draft, base)
    assert not result.conflict and result.revision
    return base, result.revision


async def _supports(adapter: AdapterClient, feature: str) -> None:
    if feature not in (await adapter.describe()).supports:
        pytest.skip(f"the adapter does not support {feature}")


async def test_describe(adapter, ops):
    description = await adapter.describe()
    names = {op.name: op.level for op in description.operations}

    assert description.protocol_version == PROTOCOL_VERSION
    assert names[ops["write"]["op"]] == "write"
    assert names[ops["dangerous"]["op"]] == "dangerous"


async def test_write_on_draft_leaves_data_alone(adapter, ops):
    before = await adapter.context()
    draft = await adapter.open_draft(before.revision)
    await _write(adapter, draft, ops["write"])

    assert await adapter.context() == before


async def test_commit_creates_a_revision(adapter, ops):
    base = (await adapter.context()).revision
    draft = await adapter.open_draft(base)
    await _write(adapter, draft, ops["write"])

    assert (await adapter.validate(draft)).ok
    assert (await adapter.diff(draft)).summary
    result = await adapter.commit(draft, base)
    assert not result.conflict and result.revision not in (None, base)
    assert (await adapter.context()).revision == result.revision


async def test_stale_base_conflicts(adapter, ops):
    base = (await adapter.context()).revision
    first, second = await adapter.open_draft(base), await adapter.open_draft(base)
    await _write(adapter, first, ops["write"])
    await _write(adapter, second, ops["write"])

    assert not (await adapter.commit(first, base)).conflict
    assert (await adapter.commit(second, base)).conflict


async def test_invalid_write_fails_validation(adapter, ops):
    draft = await adapter.open_draft((await adapter.context()).revision)
    await _write(adapter, draft, ops["invalid_write"])
    report = await adapter.validate(draft)

    assert not report.ok and report.issues


async def test_discard(adapter, ops):
    before = await adapter.context()
    draft = await adapter.open_draft(before.revision)
    await _write(adapter, draft, ops["write"])
    await adapter.discard(draft)

    assert await adapter.context() == before


async def test_write_without_draft_is_refused(adapter, ops):
    before = await adapter.context()
    with pytest.raises(RpcError):
        await _write(adapter, None, ops["write"])
    assert await adapter.context() == before


async def test_undo_restores_previous(adapter, ops):
    await _supports(adapter, "undo")
    before = await adapter.context()
    _, revision = await _commit_sample(adapter, ops)

    draft = await adapter.undo(revision)
    assert (await adapter.validate(draft)).ok
    result = await adapter.commit(draft, revision)

    after = await adapter.context()
    assert not result.conflict and after.revision == result.revision
    assert after.state == before.state


async def test_materialize_and_sync_round_trip(adapter, ops, tmp_path):
    await _supports(adapter, "materialize")
    draft = await adapter.open_draft((await adapter.context()).revision)
    await _write(adapter, draft, ops["write"])
    workdir = tmp_path / "work"
    workdir.mkdir()
    await adapter.materialize(draft, str(workdir))
    diff_before = await adapter.diff(draft)

    assert any(workdir.iterdir())
    assert (await adapter.sync(draft)).ok
    assert (await adapter.validate(draft)).ok
    assert (await adapter.diff(draft)) == diff_before


async def test_effect_runs_after_commit(adapter, ops):
    await _supports(adapter, "effects")
    _, revision = await _commit_sample(adapter, ops)
    sample = ops["dangerous"]

    await adapter.run_effect(revision, Effect(id=uuid.uuid4().hex, op=sample["op"], args=sample["args"]))
