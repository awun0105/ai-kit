"""How the conformance suites find what they test.

- ``HARNESS_ADAPTER_CMD``: JSON list starting an app's adapter on stdio; ``{data}`` is replaced by
  an empty temp folder. Default: the sample notes app.
- ``HARNESS_ADAPTER_FIXTURE``: JSON file naming a valid write, a write that fails validation and a
  dangerous operation of that app (see ``tests/fixtures/harness_notes_fixture.json``).
- ``HARNESS_CMD``: JSON list starting a harness sidecar; the suite appends the CLI flags of
  ``docs/harness.md``. Default: ``python -m app.harness_sidecar``.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from app.services.harness import (
    AdapterClient,
    spawn_peer,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_CMD = [sys.executable, "-m", "examples.harness_notes", "--data", "{data}"]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "harness_notes_fixture.json"


def _json_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    return json.loads(raw) if raw else list(default)


@pytest.fixture
def ops() -> dict:
    """The app's sample operations: ``write``, ``invalid_write``, ``dangerous`` (each ``{op, args}``)."""
    path = Path(os.environ.get("HARNESS_ADAPTER_FIXTURE") or DEFAULT_FIXTURE)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
async def adapter(tmp_path):
    """An ``AdapterClient`` talking to a freshly started adapter process."""
    data = tmp_path / "data"
    command = [part.replace("{data}", str(data)) for part in _json_env("HARNESS_ADAPTER_CMD", DEFAULT_ADAPTER_CMD)]
    peer, process = await spawn_peer(command, {}, cwd=str(ROOT), name="adapter")
    yield AdapterClient(peer, timeout_s=30)
    await peer.close()
    if process.returncode is None:
        process.terminate()
    await process.wait()
