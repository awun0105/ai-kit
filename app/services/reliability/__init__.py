"""Reliability primitives: error classification, retries, circuit breakers, resilient invocation, answer cache."""

from app.services.reliability.answer_cache import (
    AnswerCache,
    CacheHit,
    InMemoryAnswerStore,
    PgVectorAnswerStore,
    build_answer_cache_from_settings,
    get_answer_cache,
    scope_key,
    set_answer_cache,
)
from app.services.reliability.breaker import (
    BreakerState,
    Permit,
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
from app.services.reliability.invoker import (
    AllProvidersFailedError,
    ResilientInvoker,
    check_timeout_budget,
)
from app.services.reliability.retry import (
    FullJitterBackoff,
    RetryBudget,
)

__all__ = [
    "AllProvidersFailedError",
    "AnswerCache",
    "CacheHit",
    "InMemoryAnswerStore",
    "PgVectorAnswerStore",
    "build_answer_cache_from_settings",
    "get_answer_cache",
    "scope_key",
    "set_answer_cache",
    "BreakerState",
    "ErrorKind",
    "FullJitterBackoff",
    "Permit",
    "ResilientInvoker",
    "RetryBudget",
    "SlidingWindowBreaker",
    "check_timeout_budget",
    "classify",
    "counts_toward_breaker",
    "failover_helps",
    "is_retryable",
    "retry_after_seconds",
]
