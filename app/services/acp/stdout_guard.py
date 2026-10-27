"""Keep stdout for the ACP protocol on stdio transports.

``reserve_stdout`` duplicates file descriptor 1 for the protocol and then points fd 1
at stderr, so log lines and warnings printed later never reach the client as broken
JSON-RPC. An entry point must reserve before it imports the rest of the app (whose
imports already log): import ``app.services.acp.reserve_stdout`` first.
"""

import os
import sys

_protocol_fd: int | None = None


def reserve_stdout() -> int:
    """Move the protocol to a private copy of stdout; returns its fd. Calling it again returns the same fd."""
    global _protocol_fd
    if _protocol_fd is None:
        sys.stdout.flush()
        _protocol_fd = os.dup(1)
        os.dup2(2, 1)
    return _protocol_fd
