"""The in-app agent harness as a sidecar for an external ACP agent (needs the ``acp`` extra).

    python -m app.harness_sidecar --agent-cmd '["npx", "@zed-industries/claude-code-acp"]'

The app starts this process and shares its stdio as one JSON-RPC connection: the
harness calls the app's adapter, the app calls ``task.*`` and gets events (see
``docs/harness.md``). Logs go to stderr. A harness port takes the same flags, so
the conformance suite can start either.
"""

import app.services.harness.reserve_stdout  # noqa: F401, I001  (first: keep stdout for the protocol)

# isort: split

import asyncio
import json

from app.harness_cli import (
    parser,
    serve,
)
from app.services.harness.acp_driver import AcpDriver


def main() -> None:
    """Parse the flags and serve until the app closes the connection."""
    args = parser(__doc__ or "").parse_args()
    if not args.agent_cmd:
        raise SystemExit("--agent-cmd is required")
    command = json.loads(args.agent_cmd)
    asyncio.run(serve(args, lambda: AcpDriver(command)))


if __name__ == "__main__":
    main()
