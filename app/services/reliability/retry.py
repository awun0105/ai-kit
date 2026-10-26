"""Backoff with full jitter and a token-bucket retry budget (D25 lab, ``retry.py``)."""

import random
import time
from typing import Optional


class FullJitterBackoff:
    """Exponential backoff with full jitter (Brooker, AWS Architecture Blog 2015).

    ``delay(n) = uniform(0, min(cap, base * 2**(n-1)))`` for the n-th retry.
    A provider ``retry_after`` is a floor: ``retry_after + uniform(0, base)``.
    """

    def __init__(self, base_s: float = 0.5, cap_s: float = 8.0, rng: Optional[random.Random] = None):
        """Configure the delays; pass a seeded ``rng`` in tests."""
        if base_s <= 0 or cap_s <= 0:
            raise ValueError("base_s and cap_s must be positive")
        self.base_s = base_s
        self.cap_s = cap_s
        self._rng = rng or random.Random()

    def delay(self, retry_number: int, retry_after: Optional[float] = None) -> float:
        """Seconds to wait before retry number ``retry_number`` (1-based)."""
        if retry_number < 1:
            raise ValueError("retry_number starts at 1")
        if retry_after is not None:
            return retry_after + self._rng.uniform(0, self.base_s)
        return self._rng.uniform(0, min(self.cap_s, self.base_s * 2 ** (retry_number - 1)))


class RetryBudget:
    """Token bucket that caps retries at a fraction of traffic (Finagle / gRPC retry throttling).

    Each first attempt deposits ``ratio`` tokens, tokens also refill at
    ``min_per_s``, and a retry needs one whole token. With ``ratio=0.1``
    retries stay under ~10% of requests however bad the incident, which is
    what stops a retry storm from amplifying an outage.
    """

    def __init__(self, ratio: float = 0.1, min_per_s: float = 0.5, max_tokens: float = 20.0, clock=time.monotonic):
        """Start with a full bucket."""
        self.ratio = ratio
        self.min_per_s = min_per_s
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self._clock = clock
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.max_tokens, self.tokens + (now - self._last) * self.min_per_s)
        self._last = now

    def record_request(self) -> None:
        """Call once per first attempt."""
        self._refill()
        self.tokens = min(self.max_tokens, self.tokens + self.ratio)

    def try_acquire_retry(self) -> bool:
        """Take a token for a retry; False means the budget is spent, do not retry."""
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False
