"""Classify provider failures; each kind needs a different reaction.

The rules follow the D25 Reliability lab (``errors.py``): rate limits and quota
exhaustion are about *our* tenant and never open a breaker; client errors are
our bug and are never retried or failed over; overload, server errors and
timeouts say the provider is unhealthy.
"""

import asyncio
from enum import Enum
from typing import Optional

import httpx
import openai


class ErrorKind(str, Enum):
    """What kind of failure a provider call ended with."""

    RATE_LIMITED = "rate_limited"  # 429 with retry-after: over our tenant limit
    QUOTA_EXHAUSTED = "quota_exhausted"  # 429 insufficient_quota: billing, retrying is pointless
    OVERLOADED = "overloaded"  # 503 / 529: provider shedding load or down
    SERVER_ERROR = "server_error"  # other 5xx
    TIMEOUT = "timeout"  # we stopped waiting
    CLIENT_ERROR = "client_error"  # 4xx: malformed request, auth, unknown model


def _status(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and isinstance(inner.get("code"), str):
            return inner["code"]
    return ""


def classify(exc: BaseException) -> ErrorKind:
    """Map an exception raised by a model call to an ErrorKind.

    Works for the OpenAI SDK (and OpenAI-compatible endpoints) and for any
    provider exception that exposes an HTTP status. Unknown exceptions count
    as server errors so failover still happens for other providers.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, openai.APITimeoutError, httpx.TimeoutException)):
        return ErrorKind.TIMEOUT
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError)):
        return ErrorKind.SERVER_ERROR
    status = _status(exc)
    if status is None:
        return ErrorKind.SERVER_ERROR
    if status == 408:
        return ErrorKind.TIMEOUT
    if status == 429:
        return ErrorKind.QUOTA_EXHAUSTED if _code(exc) == "insufficient_quota" else ErrorKind.RATE_LIMITED
    if status in (503, 529):
        return ErrorKind.OVERLOADED
    if status >= 500:
        return ErrorKind.SERVER_ERROR
    return ErrorKind.CLIENT_ERROR


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Return the provider's ``retry-after`` hint in seconds, if it sent one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return max(0.0, float(value) * scale)
        except (TypeError, ValueError):
            continue  # an HTTP date; ignore and fall back to backoff
    return None


def is_retryable(kind: ErrorKind) -> bool:
    """May the same provider be tried again after a delay?"""
    return kind in (ErrorKind.RATE_LIMITED, ErrorKind.OVERLOADED, ErrorKind.SERVER_ERROR, ErrorKind.TIMEOUT)


def counts_toward_breaker(kind: ErrorKind) -> bool:
    """Does the error say something about the provider's health?"""
    return kind in (ErrorKind.OVERLOADED, ErrorKind.SERVER_ERROR, ErrorKind.TIMEOUT)


def failover_helps(kind: ErrorKind) -> bool:
    """Would another provider succeed? Not for a request that is malformed everywhere."""
    return kind is not ErrorKind.CLIENT_ERROR
