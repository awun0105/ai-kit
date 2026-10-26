"""Run one logical call over a fallback ladder of providers.

Per provider: circuit breaker, cooldown after rate-limit/quota errors, a
per-attempt timeout, and retries only for transient errors, spaced with full
jitter and capped by a shared retry budget. This is the single retry layer:
SDK-level retries must be off (the LLM registry sets ``max_retries=0``).
"""

import asyncio
import time
from typing import (
    Awaitable,
    Callable,
    Optional,
    Sequence,
    TypeVar,
)

from app.core.logging import logger
from app.core.metrics import (
    llm_attempts_total,
    llm_breaker_state,
)
from app.services.reliability.breaker import (
    BreakerState,
    SlidingWindowBreaker,
)
from app.services.reliability.errors import (
    ErrorKind,
    classify,
    counts_toward_breaker,
    failover_helps,
    is_retryable,
    retry_after_seconds,
)
from app.services.reliability.retry import (
    FullJitterBackoff,
    RetryBudget,
)

T = TypeVar("T")

_STATE_VALUE = {BreakerState.CLOSED: 0, BreakerState.HALF_OPEN: 1, BreakerState.OPEN: 2}


class AllProvidersFailedError(RuntimeError):
    """Every rung of the ladder failed, was cooling down, or had an open breaker."""

    def __init__(self, message: str, last_error: Optional[BaseException], last_kind: Optional[ErrorKind]):
        """Keep the last underlying error for callers and logs."""
        super().__init__(message)
        self.last_error = last_error
        self.last_kind = last_kind


class ResilientInvoker:
    """Breakers, cooldowns, retries and failover for a set of named providers."""

    def __init__(
        self,
        max_attempts: int = 3,
        attempt_timeout_s: float = 30.0,
        max_retry_after_s: float = 10.0,
        rate_limit_cooldown_s: float = 5.0,
        quota_cooldown_s: float = 300.0,
        backoff: Optional[FullJitterBackoff] = None,
        budget: Optional[RetryBudget] = None,
        breaker_factory: Optional[Callable[[str], SlidingWindowBreaker]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        """Configure the policy.

        Args:
            max_attempts: Attempts per provider, first try included.
            attempt_timeout_s: Upper bound for one attempt.
            max_retry_after_s: A longer ``retry-after`` means fail over now instead of waiting.
            rate_limit_cooldown_s: Skip a rate-limited provider this long when it gave no hint.
            quota_cooldown_s: Skip a provider whose quota is exhausted this long.
            backoff: Delay policy between retries.
            budget: Shared retry budget.
            breaker_factory: Builds the breaker for a provider name.
            clock: Monotonic clock (injectable for tests).
            sleep: Async sleep (injectable for tests).
        """
        self.max_attempts = max(1, max_attempts)
        self.attempt_timeout_s = attempt_timeout_s
        self.max_retry_after_s = max_retry_after_s
        self.rate_limit_cooldown_s = rate_limit_cooldown_s
        self.quota_cooldown_s = quota_cooldown_s
        self.backoff = backoff or FullJitterBackoff()
        self.budget = budget or RetryBudget()
        self._breaker_factory = breaker_factory or (lambda name: SlidingWindowBreaker(name=name))
        self._breakers: dict[str, SlidingWindowBreaker] = {}
        self._cooldown_until: dict[str, float] = {}
        self._clock = clock
        self._sleep = sleep

    def breaker(self, name: str) -> SlidingWindowBreaker:
        """Return (creating on first use) the breaker for ``name``."""
        if name not in self._breakers:
            breaker = self._breaker_factory(name)
            breaker.on_transition = self._log_transition
            self._breakers[name] = breaker
            llm_breaker_state.labels(model=name).set(0)
        return self._breakers[name]

    @staticmethod
    def _log_transition(name: str, frm: BreakerState, to: BreakerState, reason: str) -> None:
        llm_breaker_state.labels(model=name).set(_STATE_VALUE[to])
        logger.warning("llm_breaker_transition", model=name, frm=frm.value, to=to.value, reason=reason)

    def _cooling(self, name: str) -> bool:
        return self._clock() < self._cooldown_until.get(name, 0.0)

    def _cool(self, name: str, seconds: float) -> None:
        self._cooldown_until[name] = max(self._cooldown_until.get(name, 0.0), self._clock() + seconds)

    async def invoke(self, ladder: Sequence[tuple[str, Callable[[], Awaitable[T]]]]) -> T:
        """Return the first successful result, trying providers in ladder order."""
        self.budget.record_request()
        last_error: Optional[BaseException] = None
        last_kind: Optional[ErrorKind] = None

        for name, attempt_fn in ladder:
            if self._cooling(name):
                llm_attempts_total.labels(model=name, outcome="skipped_cooldown").inc()
                continue
            breaker = self.breaker(name)
            for attempt in range(1, self.max_attempts + 1):
                permit = breaker.allow()
                if permit is None:
                    llm_attempts_total.labels(model=name, outcome="skipped_open").inc()
                    break
                started = self._clock()
                try:
                    result = await asyncio.wait_for(attempt_fn(), timeout=self.attempt_timeout_s)
                except asyncio.CancelledError:
                    breaker.release(permit)
                    raise
                except (TypeError, AttributeError, KeyError):
                    # Programming errors fail identically everywhere: surface them as-is.
                    breaker.release(permit)
                    raise
                except Exception as e:
                    kind = classify(e)
                    last_error, last_kind = e, kind
                    (breaker.on_failure if counts_toward_breaker(kind) else breaker.release)(permit)
                    llm_attempts_total.labels(model=name, outcome=kind.value).inc()
                    logger.warning(
                        "llm_attempt_failed", model=name, attempt=attempt, kind=kind.value, error=str(e)[:300]
                    )
                    if not failover_helps(kind):
                        raise
                    hint = retry_after_seconds(e)
                    if kind is ErrorKind.QUOTA_EXHAUSTED:
                        self._cool(name, self.quota_cooldown_s)
                        break
                    if kind is ErrorKind.RATE_LIMITED and (hint is None or hint > self.max_retry_after_s):
                        self._cool(name, hint if hint is not None else self.rate_limit_cooldown_s)
                        break
                    if is_retryable(kind) and attempt < self.max_attempts and self.budget.try_acquire_retry():
                        await self._sleep(self.backoff.delay(attempt, hint))
                        continue
                    break  # fail over to the next rung
                else:
                    breaker.on_success(permit, self._clock() - started)
                    llm_attempts_total.labels(model=name, outcome="success").inc()
                    return result

        tried = ", ".join(name for name, _ in ladder)
        raise AllProvidersFailedError(
            f"no provider answered (ladder: {tried}); last error: {last_kind.value if last_kind else 'none'}: {last_error}",
            last_error,
            last_kind,
        )
