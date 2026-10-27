"""Serve the notes app's adapter on stdio, alone: ``python -m examples.harness_notes --data DIR``.

This is what the adapter conformance suite starts (``HARNESS_ADAPTER_CMD``). In
the real app the adapter shares the connection to the harness sidecar instead;
see ``examples/harness_notes/demo.py``.
"""

import app.services.harness.reserve_stdout  # noqa: F401, I001  (first: keep stdout for the protocol)

# isort: split

import argparse
import asyncio
from pathlib import Path

from app.services.harness import adapter_handlers
from app.services.harness.rpc import (
    open_stdio_peer,
    reserve_stdout,
)
from examples.harness_notes.notes import NotesAdapter


async def main() -> None:
    """Serve until the other side closes stdin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Data folder (created if missing)")
    args = parser.parse_args()
    peer = await open_stdio_peer(adapter_handlers(NotesAdapter(args.data)), reserve_stdout(), name="harness")
    await peer.closed.wait()


if __name__ == "__main__":
    asyncio.run(main())
