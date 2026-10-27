"""LLM service: one resilient call path over the model registry, with optional structured output.

Every call walks a fallback ladder that starts at the requested (or default)
model and continues through the registry order. ``ResilientInvoker`` handles
breakers, cooldowns, jittered retries under a retry budget and failover; see
``docs/reliability.md``. Unlike the original circular fallback, a failure never
switches the process-wide default: the next call starts from the preferred
model again, and an open breaker makes skipping it cheap.
"""

import asyncio
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import llm_calls_total
from app.services.llm.registry import LLMRegistry
from app.services.reliability import (
    AllProvidersFailedError,
    FullJitterBackoff,
    ResilientInvoker,
    RetryBudget,
    SlidingWindowBreaker,
    check_timeout_budget,
)

T = TypeVar("T", bound=BaseModel)


def build_invoker_from_settings() -> ResilientInvoker:
    """Assemble the reliability policy from the LLM_* settings."""
    check_timeout_budget(settings.MAX_LLM_CALL_RETRIES, settings.LLM_ATTEMPT_TIMEOUT, settings.LLM_TOTAL_TIMEOUT)
    return ResilientInvoker(
        max_attempts=settings.MAX_LLM_CALL_RETRIES,
        attempt_timeout_s=settings.LLM_ATTEMPT_TIMEOUT,
        max_retry_after_s=settings.LLM_MAX_RETRY_AFTER_S,
        quota_cooldown_s=settings.LLM_QUOTA_COOLDOWN_S,
        backoff=FullJitterBackoff(settings.LLM_RETRY_BASE_S, settings.LLM_RETRY_CAP_S),
        budget=RetryBudget(ratio=settings.LLM_RETRY_BUDGET_RATIO),
        breaker_factory=lambda name: SlidingWindowBreaker(
            name=name,
            window_size=settings.LLM_BREAKER_WINDOW,
            min_calls=settings.LLM_BREAKER_MIN_CALLS,
            failure_rate_threshold=settings.LLM_BREAKER_FAILURE_RATE,
            slow_call_s=settings.LLM_BREAKER_SLOW_CALL_S,
            open_duration_s=settings.LLM_BREAKER_OPEN_S,
            half_open_max_probes=settings.LLM_BREAKER_HALF_OPEN_PROBES,
        ),
    )


class LLMService:
    """Calls models from the registry through one resilient ladder.

    Two kinds of call:

    - **Default path** (no model_name / response_format / model_kwargs): the
      tool-bound agent models. Models without tool support are skipped.
    - **One-off path** (any override provided): a fresh runnable per model,
      optionally with structured output, never touching the bound instances.
    """

    def __init__(self, invoker: Optional[ResilientInvoker] = None):
        """Initialize the service with the configured default model."""
        self.invoker = invoker or build_invoker_from_settings()
        self._bound_tools: List = []
        self._bound: dict[str, Any] = {}

        all_names = LLMRegistry.get_all_names()
        if settings.DEFAULT_LLM_MODEL in all_names:
            self._primary = settings.DEFAULT_LLM_MODEL
        else:
            self._primary = all_names[0]
            logger.warning(
                "default_model_not_found_using_first", requested=settings.DEFAULT_LLM_MODEL, using=self._primary
            )
        self._llm: Any = LLMRegistry.get(self._primary)
        logger.info(
            "llm_service_initialized",
            default_model=self._primary,
            total_models=len(all_names),
            environment=settings.ENVIRONMENT.value,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str] = ...,
        response_format: None = ...,
        **model_kwargs: Any,
    ) -> BaseMessage: ...

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str] = ...,
        *,
        response_format: Type[T],
        **model_kwargs: Any,
    ) -> T: ...

    async def call(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **model_kwargs: Any,
    ) -> Union[BaseMessage, BaseModel]:
        """Call the LLM through the fallback ladder.

        Args:
            messages: Conversation messages to send.
            model_name: Start the ladder at this model. ``None`` uses the default.
            response_format: Pydantic schema for structured output; the call
                returns a validated instance instead of a ``BaseMessage``.
            **model_kwargs: Extra kwargs forwarded to ``LLMRegistry.get`` when
                building one-off model instances (e.g. ``temperature``).

        Returns:
            ``BaseMessage`` when ``response_format`` is ``None``, otherwise an
            instance of ``response_format``.

        Raises:
            ValueError: ``model_name`` is not in the registry.
            RuntimeError: No model answered, or the total timeout budget ran out.
            Exception: A client error (bad request, auth) from the provider,
                re-raised unchanged because no other model would accept it either.
        """
        all_names = LLMRegistry.get_all_names()
        if model_name and model_name not in all_names:
            logger.error("requested_model_not_found", model_name=model_name)
            raise ValueError(f"model '{model_name}' not found in registry. available models: {', '.join(all_names)}")
        try:
            result = await asyncio.wait_for(
                self.invoker.invoke(self._ladder(messages, model_name, response_format, model_kwargs)),
                timeout=settings.LLM_TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            llm_calls_total.labels(outcome="failed").inc()
            logger.exception("llm_total_timeout_exceeded", timeout_seconds=settings.LLM_TOTAL_TIMEOUT)
            raise RuntimeError(f"llm call timed out after {settings.LLM_TOTAL_TIMEOUT}s total budget")
        except AllProvidersFailedError:
            llm_calls_total.labels(outcome="failed").inc()
            raise
        except Exception:
            # Client errors and programming errors are our fault, not the provider's;
            # they are excluded from the availability SLI.
            llm_calls_total.labels(outcome="client_error").inc()
            raise
        llm_calls_total.labels(outcome="success").inc()
        return result

    def get_llm(self) -> Any:
        """Return the tool-bound default model instance (used for metric labels)."""
        return self._llm

    def bind_tools(self, tools: List) -> "LLMService":
        """Bind tools to every registry model that supports tool calling.

        Args:
            tools: List of tools to bind.

        Returns:
            Self for method chaining.
        """
        self._bound_tools = tools
        self._bound = {
            entry["name"]: entry["llm"].bind_tools(tools)
            for entry in LLMRegistry.LLMS
            if LLMRegistry.supports_tools(entry["name"])
        }
        self._llm = self._bound.get(self._primary, self._llm)
        logger.debug("tools_bound_to_llm", tool_count=len(tools), models=list(self._bound))
        return self

    async def ainvoke_with_tools(
        self,
        messages: LanguageModelInput,
        tools: Sequence[Any] = (),
        tool_kwargs: Optional[dict[str, Any]] = None,
        model_name: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
    ) -> BaseMessage:
        """Call the ladder with an arbitrary tool set (used by ``ResilientChatModel``).

        Each rung binds ``tools`` to its registry model; models without tool
        support are skipped when tools are given. ``config`` (callbacks, tags)
        is passed to the provider call so traces nest under the caller.
        """
        names = LLMRegistry.get_all_names()
        start = names.index(model_name or self._primary)
        rungs: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
        for name in names[start:] + names[:start]:
            if tools and not LLMRegistry.supports_tools(name):
                continue

            def attempt(name: str = name) -> Awaitable[Any]:
                model: Any = LLMRegistry.get(name)
                if tools:
                    model = model.bind_tools(list(tools), **(tool_kwargs or {}))
                return model.ainvoke(messages, config=config)

            rungs.append((name, attempt))
        try:
            result = await asyncio.wait_for(self.invoker.invoke(rungs), timeout=settings.LLM_TOTAL_TIMEOUT)
        except asyncio.TimeoutError:
            llm_calls_total.labels(outcome="failed").inc()
            raise RuntimeError(f"llm call timed out after {settings.LLM_TOTAL_TIMEOUT}s total budget")
        except AllProvidersFailedError:
            llm_calls_total.labels(outcome="failed").inc()
            raise
        except Exception:
            llm_calls_total.labels(outcome="client_error").inc()
            raise
        llm_calls_total.labels(outcome="success").inc()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ladder(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str],
        response_format: Optional[Type[BaseModel]],
        model_kwargs: dict,
    ) -> list[tuple[str, Callable[[], Awaitable[Any]]]]:
        names = LLMRegistry.get_all_names()
        start = names.index(model_name or self._primary)
        ordered = names[start:] + names[:start]
        one_off = bool(model_name or response_format or model_kwargs)

        rungs: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
        for name in ordered:
            if one_off:

                def one_off_attempt(name: str = name) -> Awaitable[Any]:
                    base = LLMRegistry.get(name, **model_kwargs)
                    runnable = base.with_structured_output(response_format) if response_format else base
                    return runnable.ainvoke(messages)

                rungs.append((name, one_off_attempt))
            elif not self._bound_tools or name in self._bound:  # skip models that cannot call tools
                runnable = self._bound.get(name) or LLMRegistry.get(name)

                def bound_attempt(runnable: Any = runnable) -> Awaitable[Any]:
                    return runnable.ainvoke(messages)

                rungs.append((name, bound_attempt))
        return rungs


llm_service = LLMService()
