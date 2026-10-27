"""Capability tokens: the right to use a task's tools, for that task only, until it ends or expires.

From fframes Studio's per-task capability files (``agent_tools/broker.rs``): the
agent never names a task or a draft; the token says which one it works on.
"""

import secrets
import time
from typing import Callable


class CapabilityError(Exception):
    """The token is unknown, expired or revoked."""


class CapabilityStore:
    """Issues, resolves and revokes capability tokens."""

    def __init__(self, ttl_s: float = 3600, clock: Callable[[], float] = time.monotonic):
        """Tokens live ``ttl_s`` seconds at most; ``clock`` is injectable for tests."""
        self._ttl_s = ttl_s
        self._clock = clock
        self._tokens: dict[str, tuple[str, float]] = {}

    def issue(self, task_id: str) -> str:
        """A new token for ``task_id``."""
        token = secrets.token_urlsafe(24)
        self._tokens[token] = (task_id, self._clock() + self._ttl_s)
        return token

    def resolve(self, token: str) -> str:
        """The task the token belongs to; raises ``CapabilityError`` if it is not valid any more."""
        entry = self._tokens.get(token)
        if entry is None:
            raise CapabilityError("unknown or revoked capability")
        task_id, expires = entry
        if self._clock() >= expires:
            self._tokens.pop(token, None)
            raise CapabilityError("expired capability")
        return task_id

    def revoke(self, task_id: str) -> None:
        """Invalidate every token of ``task_id``."""
        for token in [t for t, (owner, _) in self._tokens.items() if owner == task_id]:
            del self._tokens[token]
