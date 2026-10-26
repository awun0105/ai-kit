"""Reliability tests modelled on the D25 game-day incidents, with real OpenAI SDK exceptions and a fake clock."""

import asyncio
import random

import httpx
import openai
import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.core.langgraph.graph import LangGraphAgent
from app.services.llm import (
    LLMRegistry,
    LLMService,
)
from app.services.reliability import (
    AllProvidersFailedError,
    BreakerState,
    ErrorKind,
    FullJitterBackoff,
    ResilientInvoker,
    RetryBudget,
    SlidingWindowBreaker,
    classify,
    retry_after_seconds,
)

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _err(cls, status, code=None, headers=None):
    body = {"error": {"code": code, "message": "x"}} if code else None
    return cls("x", response=httpx.Response(status, request=REQUEST, headers=headers or {}), body=body)


RATE_LIMIT = lambda after=None: _err(  # noqa: E731
    openai.RateLimitError, 429, headers={"retry-after": str(after)} if after is not None else None
)
QUOTA = lambda: _err(openai.RateLimitError, 429, code="insufficient_quota")  # noqa: E731
OVERLOADED = lambda: _err(openai.InternalServerError, 529)  # noqa: E731
SERVER = lambda: _err(openai.InternalServerError, 500)  # noqa: E731
BAD_REQUEST = lambda: _err(openai.BadRequestError, 400)  # noqa: E731


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# ---------------------------------------------------------------- classification


def test_classify_real_sdk_errors():
    assert classify(RATE_LIMIT(2)) is ErrorKind.RATE_LIMITED
    assert classify(QUOTA()) is ErrorKind.QUOTA_EXHAUSTED
    assert classify(OVERLOADED()) is ErrorKind.OVERLOADED
    assert classify(_err(openai.InternalServerError, 503)) is ErrorKind.OVERLOADED
    assert classify(SERVER()) is ErrorKind.SERVER_ERROR
    assert classify(BAD_REQUEST()) is ErrorKind.CLIENT_ERROR
    assert classify(_err(openai.AuthenticationError, 401)) is ErrorKind.CLIENT_ERROR
    assert classify(openai.APITimeoutError(request=REQUEST)) is ErrorKind.TIMEOUT
    assert classify(asyncio.TimeoutError()) is ErrorKind.TIMEOUT
    assert classify(openai.APIConnectionError(request=REQUEST)) is ErrorKind.SERVER_ERROR


def test_retry_after_header():
    assert retry_after_seconds(RATE_LIMIT(3)) == 3.0
    assert retry_after_seconds(_err(openai.RateLimitError, 429, headers={"retry-after-ms": "1500"})) == 1.5
    assert retry_after_seconds(SERVER()) is None


# ---------------------------------------------------------------- backoff and budget


def test_full_jitter_bounds_and_retry_after_floor():
    backoff = FullJitterBackoff(base_s=0.5, cap_s=4.0, rng=random.Random(7))
    for n in range(1, 8):
        assert 0 <= backoff.delay(n) <= min(4.0, 0.5 * 2 ** (n - 1))
    assert 3.0 <= backoff.delay(1, retry_after=3.0) <= 3.5
    with pytest.raises(ValueError):
        backoff.delay(0)


def test_retry_budget_caps_retries_at_a_fraction_of_traffic():
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, min_per_s=0.0, max_tokens=2.0, clock=clock)
    granted = 0
    for _ in range(100):
        budget.record_request()
        granted += budget.try_acquire_retry()
    assert granted <= 2 + 10  # the initial bucket plus 10% of 100 requests


# ---------------------------------------------------------------- circuit breaker


def _breaker(clock, **kw):
    defaults = dict(name="m", window_size=10, min_calls=4, failure_rate_threshold=0.5, open_duration_s=30)
    return SlidingWindowBreaker(clock=clock, **{**defaults, **kw})


def test_breaker_needs_min_calls_then_trips_on_failure_rate():
    clock = FakeClock()
    b = _breaker(clock)
    for _ in range(3):
        b.on_failure(b.allow())
    assert b.state is BreakerState.CLOSED  # 3 < min_calls
    b.on_failure(b.allow())
    assert b.state is BreakerState.OPEN
    assert b.allow() is None  # fail fast while open


def test_breaker_trips_on_slow_successes():
    clock = FakeClock()
    b = _breaker(clock, slow_call_s=5.0, slow_rate_threshold=0.5)
    for _ in range(4):
        b.on_success(b.allow(), latency_s=9.0)
    assert b.state is BreakerState.OPEN


def test_half_open_limits_probes_and_closes_after_successes():
    clock = FakeClock()
    b = _breaker(clock, half_open_max_probes=2, half_open_successes=2)
    for _ in range(4):
        b.on_failure(b.allow())
    clock.now += 31
    p1, p2 = b.allow(), b.allow()
    assert b.state is BreakerState.HALF_OPEN and p1 and p2 and p1.probe
    assert b.allow() is None  # no thundering herd on a recovering provider
    b.on_success(p1, 0.1)
    b.on_success(p2, 0.1)
    assert b.state is BreakerState.CLOSED


def test_failed_probe_reopens_and_stale_permits_are_ignored():
    clock = FakeClock()
    b = _breaker(clock)
    old = b.allow()
    for _ in range(4):
        b.on_failure(b.allow())
    clock.now += 31
    probe = b.allow()
    b.on_success(old, 0.1)  # issued before the breaker opened: ignored
    assert b.state is BreakerState.HALF_OPEN
    b.on_failure(probe)
    assert b.state is BreakerState.OPEN


# ---------------------------------------------------------------- invoker


class Provider:
    """Scripted provider: raises the queued errors, then answers."""

    def __init__(self, name, errors=(), delay=0.0):
        self.name = name
        self.errors = list(errors)
        self.calls = 0
        self.delay = delay

    async def __call__(self):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.errors:
            err = self.errors.pop(0)
            raise err() if callable(err) else err
        return f"answer from {self.name}"


def _invoker(clock, **kw):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        clock.now += s

    inv = ResilientInvoker(
        clock=clock,
        sleep=fake_sleep,
        budget=RetryBudget(clock=clock),
        backoff=FullJitterBackoff(rng=random.Random(1)),
        breaker_factory=lambda n: SlidingWindowBreaker(name=n, min_calls=4, window_size=10, clock=clock),
        **kw,
    )
    return inv, sleeps


def _ladder(*providers):
    return [(p.name, p) for p in providers]


async def test_outage_is_bounded_not_amplified():
    clock = FakeClock()
    inv, _ = _invoker(clock, max_attempts=3)
    primary, backup = Provider("a", [OVERLOADED] * 100), Provider("b", [OVERLOADED] * 100)

    with pytest.raises(AllProvidersFailedError) as info:
        await inv.invoke(_ladder(primary, backup))

    assert primary.calls == 3 and backup.calls == 3  # one retry layer: never 36 attempts
    assert info.value.last_kind is ErrorKind.OVERLOADED


async def test_transient_error_retries_same_provider_then_succeeds():
    clock = FakeClock()
    inv, sleeps = _invoker(clock)
    primary = Provider("a", [SERVER])

    assert await inv.invoke(_ladder(primary, Provider("b"))) == "answer from a"
    assert primary.calls == 2 and len(sleeps) == 1


async def test_client_error_is_raised_without_retry_or_failover():
    clock = FakeClock()
    inv, _ = _invoker(clock)
    primary, backup = Provider("a", [BAD_REQUEST]), Provider("b")

    with pytest.raises(openai.BadRequestError):
        await inv.invoke(_ladder(primary, backup))
    assert primary.calls == 1 and backup.calls == 0
    assert inv.breaker("a").state is BreakerState.CLOSED  # our bug never opens a breaker


async def test_quota_exhausted_fails_over_and_cools_down():
    clock = FakeClock()
    inv, _ = _invoker(clock, quota_cooldown_s=300)
    primary, backup = Provider("a", [QUOTA]), Provider("b")

    assert await inv.invoke(_ladder(primary, backup)) == "answer from b"
    assert await inv.invoke(_ladder(primary, backup)) == "answer from b"
    assert primary.calls == 1  # skipped during the cooldown


async def test_short_retry_after_is_honoured_long_one_fails_over():
    clock = FakeClock()
    inv, sleeps = _invoker(clock, max_retry_after_s=10)
    short = Provider("a", [lambda: RATE_LIMIT(2)])
    assert await inv.invoke(_ladder(short, Provider("b"))) == "answer from a"
    assert sleeps and sleeps[0] >= 2

    inv2, sleeps2 = _invoker(clock, max_retry_after_s=10)
    long_wait, backup = Provider("a", [lambda: RATE_LIMIT(60)]), Provider("b")
    assert await inv2.invoke(_ladder(long_wait, backup)) == "answer from b"
    assert sleeps2 == []


async def test_open_breaker_skips_provider_without_calling_it():
    clock = FakeClock()
    inv, _ = _invoker(clock, max_attempts=1)
    primary, backup = Provider("a", [SERVER] * 4), Provider("b")
    for _ in range(4):
        await inv.invoke(_ladder(primary, backup))
    assert inv.breaker("a").state is BreakerState.OPEN

    calls_before = primary.calls
    assert await inv.invoke(_ladder(primary, backup)) == "answer from b"
    assert primary.calls == calls_before


async def test_attempt_timeout_fails_over():
    clock = FakeClock()
    inv, _ = _invoker(clock, attempt_timeout_s=0.05, max_attempts=1)
    slow, backup = Provider("a", delay=1.0), Provider("b")

    assert await inv.invoke(_ladder(slow, backup)) == "answer from b"


# ---------------------------------------------------------------- LLM service and registry


def test_registry_disables_sdk_retries():
    for entry in LLMRegistry.LLMS:
        assert entry["llm"].max_retries == 0


async def test_service_ladder_starts_at_primary_and_skips_models_without_tools(monkeypatch):
    names = LLMRegistry.get_all_names()
    monkeypatch.setattr(LLMRegistry, "supports_tools", classmethod(lambda cls, name: name != names[1]))
    service = LLMService()
    service.bind_tools([])
    service._bound_tools = ["placeholder"]

    first = [name for name, _ in service._ladder([], None, None, {})]

    assert first[0] == service._primary
    assert names[1] not in first


# ---------------------------------------------------------------- tool node


side_effects: list[str] = []


@tool
def charge_card(amount: int) -> str:
    """Charge the customer."""
    side_effects.append("charged")
    return "charged"


@tool
def flaky_lookup(q: str) -> str:
    """Look something up."""
    raise RuntimeError("upstream 500")


async def test_failing_tool_does_not_repeat_other_side_effects(monkeypatch):
    side_effects.clear()
    agent = LangGraphAgent()
    agent.tools_by_name = {"charge_card": charge_card, "flaky_lookup": flaky_lookup}
    replies = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "charge_card", "args": {"amount": 5}, "id": "c1"},
                {"name": "flaky_lookup", "args": {"q": "x"}, "id": "c2"},
            ],
        ),
        AIMessage(content="done"),
    ]

    async def scripted(messages, *args, **kwargs):
        return replies.pop(0)

    monkeypatch.setattr(agent.llm_service, "call", scripted)
    graph = await agent.create_graph(checkpointer=InMemorySaver())
    state = await graph.ainvoke({"messages": [HumanMessage(content="pay")]}, {"configurable": {"thread_id": "r"}})

    assert side_effects == ["charged"]
    errors = [m.content for m in state["messages"] if isinstance(m, ToolMessage) and m.name == "flaky_lookup"]
    assert errors and errors[0].startswith("Tool error: RuntimeError")
