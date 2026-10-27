# LLM Service

## Overview

The LLM service (`app/services/llm/`) handles all language model calls through one resilient path: a fallback ladder over the registry models with per-model circuit breakers, classified errors, jittered retries under a retry budget, and a total timeout budget. Your agent code calls `llm_service.call(messages)` (or uses `ResilientChatModel`, see [agents](agents.md)) and the service handles everything else. The full reliability design is in [reliability.md](reliability.md).

The package is split into two modules:

- `app/services/llm/registry.py` — `LLMRegistry`: defines available models
- `app/services/llm/service.py` — `LLMService`: ladder construction, structured output, tool binding
- `app/services/reliability/` — `ResilientInvoker`: error classification, retries, breakers, cooldowns

## Model registry

Models are defined in `LLMRegistry.LLM_CONFIGS` in order of preference:

| Name           | Model        | Notes                              |
| -------------- | ------------ | ----------------------------------- |
| `gpt-5.6-luna` | gpt-5.6-luna | Default. Medium reasoning effort.  |
| `gpt-5.4`      | gpt-5.4      | Medium reasoning effort.           |
| `gpt-5.4-mini` | gpt-5.4-mini | Low reasoning effort.              |
| `gpt-5.4-nano` | gpt-5.4-nano | Fast, low reasoning effort.        |

Set `DEFAULT_LLM_MODEL` in your `.env` to choose the starting model.

Every entry is built through LangChain's `init_chat_model`. A bare `model` name (like the ones above) is inferred as OpenAI, so existing entries and env vars keep working unchanged. To add a model from another provider, prefix `model` with `provider:`, e.g.:

```python
{"name": "claude-opus", "model": "anthropic:claude-opus-4-6", "kwargs": {"temperature": 0.2}},
```

That provider's LangChain integration package must be installed (`uv sync --extra anthropic`, `--extra ollama`, or `--extra google-genai`) and its API key set via the provider's own env var (e.g. `ANTHROPIC_API_KEY`) — the registry does not manage non-OpenAI credentials itself.

To add or change models, edit `LLMRegistry.LLM_CONFIGS` in `app/services/llm/registry.py`.

## Retry and fallback behaviour

```mermaid
flowchart TD
    Call["llm_service.call(messages)"]
    Ladder["Ladder: requested or default model first,\nthen the rest of the registry in order"]
    Skip{"Breaker open or\nmodel cooling down?"}
    Try["Attempt (LLM_ATTEMPT_TIMEOUT)"]
    Kind{"Error kind"}
    Retry{"Attempts left and\nretry budget?"}
    Wait["Full-jitter backoff\n(retry-after is a floor)"]
    Next["Next model"]
    Raise["Raise the client error"]
    Fail["AllProvidersFailedError"]
    Success["Return response"]

    Call --> Ladder --> Skip
    Skip -->|yes| Next
    Skip -->|no| Try
    Try -->|success| Success
    Try --> Kind
    Kind -->|400/401/403/404/422| Raise
    Kind -->|quota exhausted, long rate limit| Next
    Kind -->|5xx, 529, timeout, short rate limit| Retry
    Retry -->|yes| Wait --> Try
    Retry -->|no| Next
    Next -->|more models| Skip
    Next -->|none left| Fail
```

- **One retry layer.** SDK retries are off (`max_retries=0` in the registry); `MAX_LLM_CALL_RETRIES`
  is the number of attempts per model.
- **Classified errors.** Client errors are raised at once and never fail over or open a breaker;
  quota exhaustion cools the model down; overload, server errors and timeouts are retried and
  count toward the model's breaker. This works for any provider whose exceptions carry an HTTP status.
- **No global switching.** Every call starts from the preferred model again. A model that keeps
  failing has an open breaker, so skipping it costs nothing.
- **Total timeout**: `LLM_TOTAL_TIMEOUT` (default 90 s) caps the whole ladder. Startup fails if it cannot fit `MAX_LLM_CALL_RETRIES` x `LLM_ATTEMPT_TIMEOUT` plus one attempt on a fallback model, because the ladder would then never fail over from a hung provider.

Settings and the reasoning behind each rule: [reliability.md](reliability.md).

## Tools

Tools are bound to the LLM at startup:

```python
llm_service.bind_tools(tools)
```

Tools are bound to every registry model that supports tool calling. Models marked `"supports_tools": False` in `LLM_CONFIGS` are skipped for tool-bound calls.

## Structured output

Pass a Pydantic model as `response_format` to get a validated instance back instead of a raw `BaseMessage`:

```python
from app.schemas.my_schema import MySchema

result: MySchema = await llm_service.call(
    messages,
    model_name="gpt-5.4-nano",   # optional — uses current default if omitted
    response_format=MySchema,
    temperature=0.2,
)
```

The service chains `.with_structured_output(schema)` on each model of the ladder, so retries and failover work the same way.

## Adding a new model

```python
# app/services/llm/registry.py — LLMRegistry.LLM_CONFIGS
{
    "name": "gpt-5.4",
    "model": "gpt-5.4",  # or "provider:model", e.g. "anthropic:claude-opus-4-6"
    "kwargs": {
        "max_completion_tokens": settings.MAX_TOKENS,
        "reasoning": {"effort": "medium"},  # OpenAI-only; drop for other providers
    },
},
```

Add it at any position in the list. The fallback order follows the list order.
