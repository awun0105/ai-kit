"""Sliding-window circuit breaker (D25 lab, ``breaker.py``; Resilience4j semantics).

Unlike "open after N consecutive failures":

- trips on the failure *rate* over the last ``window_size`` results, once at
  least ``min_calls`` are in;
- counts slow successes as unhealthy, catching degradation before timeouts;
- HALF_OPEN admits at most ``half_open_max_probes`` requests in flight;
- every admitted request gets a ``Permit``; results arriving after the state
  changed (older ``generation``) are ignored.

All methods are synchronous and do no I/O, so they are safe to call from
concurrent asyncio tasks on one event loop without a lock.
"""

import time
from collections import deque
from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
from typing import (
    Callable,
    Optional,
)


class BreakerState(str, Enum):
    """Breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class Permit:
    """Ticket for one attempt; report its outcome exactly once."""

    breaker: str
    generation: int
    probe: bool


@dataclass
class SlidingWindowBreaker:
    """Circuit breaker for one dependency (one model deployment)."""

    name: str
    window_size: int = 20
    min_calls: int = 10
    failure_rate_threshold: float = 0.5
    slow_call_s: float = 20.0
    slow_rate_threshold: float = 0.6
    open_duration_s: float = 30.0
    half_open_max_probes: int = 2
    half_open_successes: int = 2
    clock: Callable[[], float] = time.monotonic
    on_transition: Optional[Callable[[str, BreakerState, BreakerState, str], None]] = None

    state: BreakerState = BreakerState.CLOSED
    generation: int = 0
    opened_at: float = 0.0
    probes_in_flight: int = 0
    probe_successes: int = 0
    _window: deque[tuple[bool, bool]] = field(default_factory=deque)  # (failed, slow)

    def _move(self, to: BreakerState, reason: str) -> None:
        frm = self.state
        self.state = to
        self.generation += 1
        self.probes_in_flight = 0
        self.probe_successes = 0
        if to is BreakerState.OPEN:
            self.opened_at = self.clock()
        if to is BreakerState.CLOSED:
            self._window.clear()
        if self.on_transition:
            self.on_transition(self.name, frm, to, reason)

    def _record(self, failed: bool, slow: bool) -> None:
        self._window.append((failed, slow))
        while len(self._window) > self.window_size:
            self._window.popleft()
        if len(self._window) < self.min_calls:
            return
        n = len(self._window)
        if sum(f for f, _ in self._window) / n >= self.failure_rate_threshold:
            self._move(BreakerState.OPEN, "failure_rate")
        elif sum(s and not f for f, s in self._window) / n >= self.slow_rate_threshold:
            self._move(BreakerState.OPEN, "slow_call_rate")

    def allow(self) -> Optional[Permit]:
        """Return a permit if a request may be sent now, else None (fail fast)."""
        if self.state is BreakerState.OPEN:
            if self.clock() - self.opened_at < self.open_duration_s:
                return None
            self._move(BreakerState.HALF_OPEN, "open_timeout_elapsed")
        if self.state is BreakerState.HALF_OPEN:
            if self.probes_in_flight >= self.half_open_max_probes:
                return None
            self.probes_in_flight += 1
            return Permit(self.name, self.generation, probe=True)
        return Permit(self.name, self.generation, probe=False)

    def on_success(self, permit: Permit, latency_s: float) -> None:
        """Record a successful attempt."""
        if permit.generation != self.generation:
            return
        slow = latency_s >= self.slow_call_s
        if self.state is BreakerState.HALF_OPEN:
            self.probes_in_flight = max(0, self.probes_in_flight - 1)
            if slow:
                self._move(BreakerState.OPEN, "probe_slow")
                return
            self.probe_successes += 1
            if self.probe_successes >= self.half_open_successes:
                self._move(BreakerState.CLOSED, "probes_succeeded")
        else:
            self._record(failed=False, slow=slow)

    def on_failure(self, permit: Permit) -> None:
        """Record a failure that reflects provider health."""
        if permit.generation != self.generation:
            return
        if self.state is BreakerState.HALF_OPEN:
            self._move(BreakerState.OPEN, "probe_failed")
        else:
            self._record(failed=True, slow=False)

    def release(self, permit: Permit) -> None:
        """End an attempt that says nothing about health (client error, rate limit, cancelled)."""
        if permit.generation != self.generation:
            return
        if self.state is BreakerState.HALF_OPEN and permit.probe:
            self.probes_in_flight = max(0, self.probes_in_flight - 1)
