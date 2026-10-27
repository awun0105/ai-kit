"""The in-app agent harness as a sidecar for an external ACP agent (needs the ``acp`` extra).

    python -m app.harness_sidecar --agent-cmd '["npx", "@zed-industries/claude-code-acp"]'

The app starts this process and shares its stdio as one JSON-RPC connection: the
harness calls the app's adapter, the app calls ``task.*`` and gets events (see
``docs/harness.md``). Logs go to stderr. A harness port takes the same flags, so
the conformance suite can start either.
"""

import app.services.harness.reserve_stdout  # noqa: F401, I001  (first: keep stdout for the protocol)

# isort: split

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from app.services.guardrails.detectors import (
    find_sensitive,
    redact,
)
from app.services.harness import PermissionPolicy
from app.services.harness.acp_driver import AcpDriver
from app.services.harness.driver import DriverFactory
from app.services.harness.rpc import reserve_stdout
from app.services.harness.sidecar import run_sidecar


def redact_secrets(text: str) -> str:
    """Mask keys, cards and personal numbers before text reaches events or transcripts."""
    findings = find_sensitive(text)
    return redact(text, findings) if findings else text


def parser() -> argparse.ArgumentParser:
    """The sidecar CLI, shared with ``app.harness_kit_driver``."""
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--agent-cmd", help="JSON list: the ACP agent command")
    cli.add_argument("--policy", choices=[p.value for p in PermissionPolicy], default="allow_in_draft")
    cli.add_argument("--review", choices=["auto", "manual"], default="manual")
    cli.add_argument("--max-repairs", type=int, default=1)
    cli.add_argument("--transcripts", type=Path, help="Folder for one JSONL transcript per task")
    return cli


async def serve(args: argparse.Namespace, driver_factory: DriverFactory, transcripts: Optional[Path]) -> None:
    """Run the sidecar with the parsed flags."""
    await run_sidecar(
        driver_factory,
        reserve_stdout(),
        policy=PermissionPolicy(args.policy),
        review=args.review,
        max_repairs=args.max_repairs,
        transcript_dir=transcripts,
        redact=redact_secrets,
    )


def main() -> None:
    """Parse the flags and serve until the app closes the connection."""
    args = parser().parse_args()
    if not args.agent_cmd:
        raise SystemExit("--agent-cmd is required")
    command = json.loads(args.agent_cmd)
    asyncio.run(serve(args, lambda: AcpDriver(command), args.transcripts))


if __name__ == "__main__":
    main()
