"""Import first in an ACP entry point: reserves stdout for the protocol at import time."""

from app.services.acp.stdout_guard import reserve_stdout

reserve_stdout()
