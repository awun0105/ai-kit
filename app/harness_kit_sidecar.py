"""The in-app agent harness as a sidecar running the kit's own agent: ``python -m app.harness_kit_sidecar``.

Takes the sidecar flags of ``docs/harness.md`` except ``--agent-cmd``. The model and
the guardrails come from the app's settings (``.env.<APP_ENV>``); with the offline
settings of ``docs/offline.md`` it needs no server.
"""

import app.services.harness.reserve_stdout  # noqa: F401, I001  (first: importing the app logs to stdout)

# isort: split

import asyncio

from app.core.kit import get_guardrails
from app.harness_cli import (
    parser,
    serve,
)
from app.harness_kit_driver import KitDriver
from app.services.agents import ResilientChatModel
from app.services.llm import llm_service


def main() -> None:
    """Serve the harness with the kit's agent until the app closes the connection."""
    args = parser(__doc__ or "").parse_args()
    asyncio.run(serve(args, lambda: KitDriver(ResilientChatModel(service=llm_service), get_guardrails())))


if __name__ == "__main__":
    main()
