# Reliability

Every model call goes through one resilient path; knowledge answers can be cached
and served stale when every provider is down; an error-budget alert pages before
users notice.

## What changed from the base template, and why

The template retried with tenacity (3 attempts, fixed 2–10 s waits, retrying any
`APIError` including HTTP 400) on top of the OpenAI SDK's own 2 retries, then cycled
through 4 models: up to **36 attempts per call**. That is the "retry storm" the D25
game-day lab opens with, where a 3-minute provider incident became 5 minutes of
total outage. It also switched the process-wide default model on any failure, and
retried the whole tool node on a tool error, re-running side effects.

| Problem | Now |
|---|---|
| Stacked retries (app × SDK × models) | One retry layer; SDK `max_retries=0`; `MAX_LLM_CALL_RETRIES` attempts per model |
| Retrying 400/401 and failing over on them | Errors are classified; client errors raise immediately, never fail over, never open a breaker |
| Fixed backoff, no jitter | Full-jitter exponential backoff; `retry-after` honoured as a floor |
| Retries scale with the outage | Token-bucket retry budget: retries stay under ~10 % of traffic |
| A dead model is called on every request | Per-model sliding-window circuit breaker (failure rate and slow calls, limited half-open probes) |
| Rate limit / quota treated like an outage | Cooldown for that model, failover to the next; quota exhaustion cools down for 5 minutes |
| Any failure moves the global default model | Each call walks the ladder from the preferred model; open breakers make skipping it free |
| Models without tool calling get tool requests | `"supports_tools": False` in the registry removes a model from the tool-bound ladder |
| No per-request timeout | `LLM_ATTEMPT_TIMEOUT` per attempt inside the `LLM_TOTAL_TIMEOUT` budget |
| Tool node retried as a whole | A failing tool returns `Tool error: ...` to the model; other tools are not re-run |
| A failed turn left `next=chat`, so the user's next message was taken as an interrupt resume and lost | Resume only when a task is really waiting on `interrupt()` |

## Error kinds

| Kind | Example | Retry same model | Counts toward breaker | Fail over |
|---|---|---|---|---|
| `rate_limited` | 429 | yes, if `retry-after` ≤ `LLM_MAX_RETRY_AFTER_S`; otherwise cooldown | no | yes |
| `quota_exhausted` | 429 `insufficient_quota` | no (cooldown `LLM_QUOTA_COOLDOWN_S`) | no | yes |
| `overloaded` | 503, 529 | yes | yes | yes |
| `server_error` | 500, 502, connection error | yes | yes | yes |
| `timeout` | attempt timeout, 408 | yes | yes | yes |
| `client_error` | 400, 401, 403, 404, 422 | no | no | **no** (raised) |

## Answer cache (stale-if-error)

Off by default (`ANSWER_CACHE_ENABLED=false`); needs `RAG_ENABLED=true`.

- Only the **first question of a new conversation** is looked up or stored: later turns
  depend on context.
- Never for **personal questions** ("my balance", "đơn hàng của tôi"), nor when the input
  rails found PII.
- Only answers that used `knowledge_search` are stored, with the ids of the documents
  they came from. Re-ingesting a changed document deletes every answer built on it
  (`scripts/ingest.py` does this automatically).
- **Fresh** (younger than `ANSWER_CACHE_TTL_S`): answered without calling a model.
  **Stale** (up to `ANSWER_CACHE_MAX_STALE_S` more): served only when every provider failed.
  Streaming serves a stale answer only if nothing had been streamed yet.
- Entries are **scoped** by a hash of the system prompt template, default model, RAG table,
  embedding model and `ANSWER_CACHE_VERSION`: changing any of them never serves old answers.
- Similarity threshold 0.92 by default. A wrong cached answer is worse than a miss; lower it
  only after checking near-duplicate questions in your golden set.
- The store is a Postgres/pgvector table, so the ingestion CLI can invalidate entries
  written by the API processes. Cached turns are written into the conversation history.

## SLO and alerts

`llm_calls_total{outcome}` counts logical calls (after retries and failover).
`prometheus/alerts.yml` defines a 99 % availability SLO with multi-window burn-rate
alerts (Google SRE Workbook):

| Alert | Condition | Severity |
|---|---|---|
| `LLMErrorBudgetFastBurn` | burn rate > 14.4 over 1 h and 5 min, or > 6 over 6 h and 30 min | page |
| `LLMErrorBudgetSlowBurn` | burn rate > 1 over 3 d and 6 h | ticket |
| `LLMCircuitBreakerOpen` | a model's breaker open for 5 min | ticket |

Client errors are excluded from the SLI. `prometheus/alerts_test.yml` checks the rules
with `promtool` (20 % errors pages; 0.5 % does not; client errors do not), and CI runs it.

Other metrics: `llm_attempts_total{model, outcome}`, `llm_breaker_state{model}`
(0 closed, 1 half-open, 2 open), `answer_cache_events_total{outcome}`.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `MAX_LLM_CALL_RETRIES` | `3` | Attempts per model, first try included |
| `LLM_ATTEMPT_TIMEOUT` | `30` | Seconds per attempt (also the SDK timeout) |
| `LLM_TOTAL_TIMEOUT` | `60` | Seconds for the whole call across the ladder |
| `LLM_RETRY_BASE_S` / `LLM_RETRY_CAP_S` | `0.5` / `8` | Full-jitter backoff |
| `LLM_RETRY_BUDGET_RATIO` | `0.1` | Retries allowed per first attempt, on average |
| `LLM_MAX_RETRY_AFTER_S` | `10` | Longer `retry-after` hints fail over instead of waiting |
| `LLM_QUOTA_COOLDOWN_S` | `300` | Skip a model with exhausted quota this long |
| `LLM_BREAKER_WINDOW` / `LLM_BREAKER_MIN_CALLS` | `20` / `10` | Sliding window size and minimum results before tripping |
| `LLM_BREAKER_FAILURE_RATE` | `0.5` | Failure share that opens the breaker |
| `LLM_BREAKER_SLOW_CALL_S` | `20` | A success slower than this counts as slow (60 % slow also opens) |
| `LLM_BREAKER_OPEN_S` | `30` | Open duration before half-open probes |
| `LLM_BREAKER_HALF_OPEN_PROBES` | `2` | Concurrent probes; that many successes close it |
| `ANSWER_CACHE_ENABLED` | `false` | Turn the answer cache on |
| `ANSWER_CACHE_STORE` / `ANSWER_CACHE_TABLE` | `pgvector` / `rag_answer_cache` | `memory` only for single-process demos |
| `ANSWER_CACHE_THRESHOLD` | `0.92` | Minimum cosine similarity for a hit |
| `ANSWER_CACHE_TTL_S` / `ANSWER_CACHE_MAX_STALE_S` | `3600` / `86400` | Fresh window, then extra stale-if-error window |
| `ANSWER_CACHE_VERSION` | `1` | Bump to discard every cached answer |

## Limits

- Breakers, cooldowns and the retry budget live in each process. With several workers
  each learns about an outage on its own (it takes `LLM_BREAKER_MIN_CALLS` failures per
  worker), which is acceptable for a handful of workers.
- Idempotency keys for side-effecting tools across crashes (D25 part D) are not built in;
  pass the `tool_call_id` to your downstream API as the idempotency key when it supports one.
- Expired cache entries are skipped at lookup; call `AnswerCache.purge_expired()` from a
  periodic job to delete them.

## Code

| File | Responsibility |
|---|---|
| `app/services/reliability/errors.py` | `classify`, `retry_after_seconds`, retry/breaker/failover rules |
| `app/services/reliability/retry.py` | `FullJitterBackoff`, `RetryBudget` |
| `app/services/reliability/breaker.py` | `SlidingWindowBreaker` |
| `app/services/reliability/invoker.py` | `ResilientInvoker` (ladder, cooldowns, metrics) |
| `app/services/reliability/answer_cache.py` | `AnswerCache`, in-memory and pgvector stores |
| `app/services/llm/service.py`, `registry.py` | Ladder construction, SDK retries off, tool capability |
| `prometheus/alerts.yml`, `alerts_test.yml` | SLO recording rules, burn-rate alerts, promtool tests |

## Provenance

Specification: AI20K D25 lab "Game Day: Retry Storm, Circuit Breaker, Cache, Idempotency &
Error Budget" (`VinUni-AI20k/K4-Track03-Circuit-Breakers-Caching-Reliability`: `errors.py`,
`retry.py`, `breaker.py`, `cache.py`, `slo/alerting.py`). The lab ships TODO stubs; the
implementations here are new. Design references: Brooker, "Exponential Backoff and Jitter"
(AWS Architecture Blog, 2015); gRPC retry throttling; Resilience4j CircuitBreaker; Google
SRE Workbook, "Alerting on SLOs". Python circuit-breaker libraries were reviewed
(pybreaker: consecutive-failure count only, no failure-rate window, slow-call or probe
limits; others unmaintained), so the breaker is implemented here.
