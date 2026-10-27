# Deployment and CI/CD

## Pipeline

```
pull request ──> CI: ruff, pyright, pytest (+ pgvector service, offline retrieval gate), promtool alert tests
                 eval.yml (label "run-eval"): golden-set evaluation with real models, report in the run summary
merge to master ──> CI ──success──> Release image: build, push ghcr.io/<repo>:sha-<commit> and :production,
                                    SBOM + provenance, Trivy scan (fails on fixable critical CVEs)
nightly ──> eval.yml: full golden-set evaluation, fails when a gate fails
```

Deploy and roll back by the immutable `sha-<commit>` tag; `production` only points at the latest
release. Optional features are baked in with the repository variable `IMAGE_EXTRAS`
(e.g. `rerank a2a deep`). Secrets used by `eval.yml`: `OPENAI_API_KEY`, optionally
`EVALUATION_API_KEY` and the variable `EVALUATION_LLM`.

## Image

- Two stages: the builder has compilers and uv; the runtime image only holds the virtualenv and
  the code. Dev tools (ruff, pyright, pre-commit) are not installed.
- Runs as uid 10001, not root.
- Listens on `$PORT` (default 8000), which Cloud Run, Render and Railway set.
- `HEALTHCHECK` calls `/health` with Python (the slim image has no curl). `/health` returns 503
  when the database is unreachable, so the platform takes the instance out of rotation.
- On SIGTERM uvicorn stops accepting connections and gives in-flight requests 30 s
  (`--timeout-graceful-shutdown`), so deploys do not cut requests off.
- `.env*` files are never copied into the image; pass configuration as environment variables.

```bash
docker build --build-arg EXTRAS="rerank a2a" -t ai-kit .
docker run -p 8000:8000 --env-file .env.production ai-kit
```

`docker compose` (`make docker-up`) runs the app with Postgres/pgvector; `make stack-up` adds
Valkey, Prometheus (with the SLO alert rules) and Grafana.

## Cost guard

Rate limits cap requests; the cost guard caps money. Each user gets a monthly USD budget.
A `TurnUsage` callback handler rides along with each turn and collects the token usage of every
chat model call in it: the answer, the LLM policy rail, and every call a deep-research sub-agent
makes (a reply reported by both `ResilientChatModel` and the provider model under it is counted
once, by message id). The guard prices the calls with your table and adds them to the user's
total for the month (UTC), kept in Valkey when `VALKEY_HOST` is set, otherwise in process memory.
Once the budget is spent the API answers **402** before any model is called.

Prices are looked up by the exact model name, then by the longest listed prefix, because
providers report dated snapshots (`gpt-5.4-mini-2026-01-01` is priced as `gpt-5.4-mini`).
Long-term memory (mem0) calls OpenAI with its own client and is not counted.

```bash
COST_MONTHLY_BUDGET_USD=5
# USD per million tokens [input, output] for each registry model, from your provider's price list
COST_PRICES_JSON='{"gpt-5.4-mini": [0.25, 2.0], "gpt-5.4-nano": [0.05, 0.4]}'
COST_DEFAULT_INPUT_PER_1M=1.0      # for models missing from the table (logged once)
COST_DEFAULT_OUTPUT_PER_1M=4.0
```

The prices above are placeholders; copy the real ones. Totals are read-then-write, so concurrent
turns of one user can undercount slightly: it is a budget brake, not an invoice.

## Errors returned to clients

Unexpected exceptions are logged with their details and the client gets a generic message:
the template used to return `str(exception)`, which can include provider responses, file paths,
connection strings or SQL. Budget exhaustion is the one error whose message is shown (402).

## Request metrics

`MetricsMiddleware` labels `http_requests_total` and `http_request_duration_seconds` with the
route template (`/api/v1/auth/session/{session_id}/name`), not the raw path, so the number of
series stays bounded. Unknown paths are labelled `unmatched`.

`starlette_prometheus` was removed: it was unmaintained, duplicated these metrics, and on
FastAPI 0.16x / Starlette 1.x it raised `'_IncludedRouter' object has no attribute 'path'` on
every `/api` request.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Listening port inside the container |
| `COST_MONTHLY_BUDGET_USD` | `0` (off) | Monthly budget per user |
| `COST_PRICES_JSON` | – | `{"model": [input_per_1M, output_per_1M]}` |
| `COST_DEFAULT_INPUT_PER_1M` / `COST_DEFAULT_OUTPUT_PER_1M` | `0` | Price for models missing from the table |

## Provenance

| Part | Source |
|---|---|
| Multi-stage non-root image, Python healthcheck, `$PORT`, graceful shutdown, cost guard | Author's solution to the AI20K D12 Cloud-Service-Deployment lab (`Dockerfile`, `app/cost_guard.py`, `app/lifecycle.py`) |
| Quality gate before release, scheduled re-evaluation | AI20K D21 CI/CD for AI Systems lab (test → train → quality gate → release) |
| Actions | `docker/build-push-action`, `docker/metadata-action`, `aquasecurity/trivy-action`, `astral-sh/setup-uv` |
