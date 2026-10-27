"""Flags and startup shared by the harness sidecars (``app.harness_sidecar``, ``app.harness_kit_driver``).

Import ``app.services.harness.reserve_stdout`` before this module in an entry point.
"""

import argparse
from pathlib import Path

from app.services.guardrails.detectors import (
    find_sensitive,
    redact,
)
from app.services.harness import PermissionPolicy
from app.services.harness.driver import DriverFactory
from app.services.harness.rpc import reserve_stdout
from app.services.harness.sidecar import run_sidecar


def redact_secrets(text: str) -> str:
    """Mask keys, cards and personal numbers before text reaches events or transcripts."""
    findings = find_sensitive(text)
    return redact(text, findings) if findings else text


def parser(description: str) -> argparse.ArgumentParser:
    """The sidecar CLI of ``docs/harness.md`` (a port takes the same flags)."""
    cli = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--agent-cmd", help="JSON list: the ACP agent command")
    cli.add_argument("--policy", choices=[p.value for p in PermissionPolicy], default="allow_in_draft")
    cli.add_argument("--review", choices=["auto", "manual"], default="manual")
    cli.add_argument("--max-repairs", type=int, default=1)
    cli.add_argument("--transcripts", type=Path, help="Folder for one JSONL transcript per task")
    return cli


async def serve(args: argparse.Namespace, driver_factory: DriverFactory) -> None:
    """Run the sidecar with the parsed flags until the app closes the connection."""
    await run_sidecar(
        driver_factory,
        reserve_stdout(),
        policy=PermissionPolicy(args.policy),
        review=args.review,
        max_repairs=args.max_repairs,
        transcript_dir=args.transcripts,
        redact=redact_secrets,
    )
