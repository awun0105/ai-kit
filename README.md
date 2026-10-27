# AI Kit — a production starter for LLM and agent apps

A FastAPI + LangGraph application you start new agent apps from. It ships the parts every
LLM product ends up needing, each one tested offline, documented, and off by default when it
costs money or changes behaviour:

| Layer | What you get | Doc |
|---|---|---|
| Agent core | LangGraph agent with Postgres checkpointing, tool calling, human-in-the-loop, mem0 long-term memory, JWT sessions, rate limiting | [architecture](docs/architecture.md) |
| RAG | Ingest md/txt/pdf/docx/html, hybrid retrieval (pgvector + BM25 with Vietnamese word segmentation + RRF), optional reranker, cited answers via `knowledge_search` | [rag](docs/rag.md) |
| Guardrails | Prompt-injection and PII rails (EN + VI), optional LLM policy rail, human approval for risky tools, egress allowlist, output redaction while streaming | [guardrails](docs/guardrails.md) |
| Reliability | One retry layer with classified errors, jitter and a retry budget, per-model circuit breakers, fallback ladder, answer cache with stale-if-error, SLO burn-rate alerts | [reliability](docs/reliability.md) |
| Tools | Load MCP server tools, serve the knowledge base over MCP, serve the agent and call other agents over A2A | [tools](docs/tools.md) |
| Agents | `ResilientChatModel` for any LangChain agent, a deep-research team (lead + researchers) on deepagents, citation checks | [agents](docs/agents.md) |
| Evaluation | Golden-set evaluation: retrieval metrics, the real agent graded by a separate judge, tool trajectories, repeated runs, simulated multi-turn users, judge-bias controls, CI gates | [evaluation](docs/evaluation.md) |
| History | Per-call history budget that never drops the current turn; optional rolling summary of the oldest turns | [compaction](docs/compaction.md) |
| Delivery | Gated release to GHCR with SBOM and Trivy, nightly evals, non-root multi-stage image, per-user cost guard | [deployment](docs/deployment.md) |

## Start a new app

```bash
# generate a project and choose its features
uvx copier copy --trust <kit-repo-or-path> my-app
cd my-app && git init

# or clone the kit and set it up in place
python3 scripts/bootstrap.py --features rag,answer-cache
```

Either way you get `.env.development` with a random JWT secret and the chosen features on,
plus the next commands to run:

```bash
uv sync                          # add --extra rerank / a2a / deep for those features
make test                        # offline test suite, no API key needed
# put OPENAI_API_KEY in .env.development, then
make docker-up                   # API + Postgres/pgvector: http://localhost:8000/docs
make ingest CORPUS=path/to/docs  # load documents into the knowledge base
```

Update a generated app with later kit improvements: `uvx copier update --trust`.

### Take one component into another project

Each layer under `app/services/` (rag, guardrails, reliability, evaluation, mcp, a2a, agents,
cost_guard, compaction) imports on its own: no settings, database or model registry, which
`tests/test_toolbox_imports.py` checks. Copy the package with its tests and follow the
**Copy this module** and **When you plug it in** sections of its doc; `app/core/kit.py` shows
how the app builds each one from settings.

### First request

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/register -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "Secret123!", "username": "you"}' | jq -r .token.access_token)  # pragma: allowlist secret
SESSION=$(curl -s -X POST localhost:8000/api/v1/auth/session -H "Authorization: Bearer $TOKEN" | jq -r .token.access_token)
curl -X POST localhost:8000/api/v1/chatbot/chat -H "Authorization: Bearer $SESSION" \
  -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

See [getting started](docs/getting-started.md) for local development without Docker.

## Everyday commands

| Command | Does |
|---|---|
| `make test` | Run the test suite (offline; integration tests skip unless `RAG_TEST_PG_URL` is set) |
| `make check` | ruff + pyright |
| `make dev` | Run the API locally with reload |
| `make ingest CORPUS=dir` | Ingest documents (re-runs skip unchanged chunks, drop chunks a document no longer has, and invalidate stale cached answers; `--prune` via `scripts.ingest` also removes deleted documents) |
| `make eval-rag-offline` | Retrieval regression check on the sample golden set (the CI gate) |
| `make eval-rag` | Full golden-set evaluation with a judge model |
| `python -m app.mcp_server` | Serve the knowledge base to other agents over MCP (stdio) |
| `python -m scripts.research "topic"` | Run the deep-research team from the command line |
| `make stack-up` | API, Postgres, Valkey, Prometheus (with SLO alerts), Grafana |

## Project structure

```
app/
  api/v1/                 route handlers (auth, chat, streaming)
  core/langgraph/         agent graph: input guard -> chat -> tools (approval, screening) -> output rail
  core/                   settings, logging, metrics, middleware, cache, rate limits, prompts
  core/kit.py             wiring: builds each toolbox component from settings (get_*/set_*)
  services/
    llm/                  model registry and the resilient call path
    reliability/          error classes, retry budget, circuit breakers, invoker, answer cache
    rag/                  ingestion, chunking, embeddings, stores, ranking, citations
    guardrails/           detectors, LLM policy rail, tool policy, stream redaction
    mcp/  a2a/            MCP toolkit; A2A server and remote-agent tool
    agents/               ResilientChatModel, deep-research team
    evaluation/           golden sets, retrieval metrics, judge, reports
    cost_guard.py         per-user monthly budget
    compaction.py         summarize the oldest turns of long conversations
  mcp_server.py           the knowledge base as an MCP server
scripts/                  bootstrap, ingest, eval_rag, research
evals/                    Langfuse trace evaluator (base template) and datasets/ (golden sets)
prometheus/               scrape config, SLO alert rules and their promtool tests
tests/                    offline tests for every layer
docs/                     one page per layer; ROADMAP.md lists them
```

## Documentation

| Guide | What it covers |
|---|---|
| [Roadmap](docs/ROADMAP.md) | The layers, their status and where each came from |
| [Getting started](docs/getting-started.md) | Prerequisites, local setup, first API call |
| [Architecture](docs/architecture.md) | System design, request flow |
| [Configuration](docs/configuration.md) | Base environment variables (each layer's page lists its own) |
| [Authentication](docs/authentication.md) | JWT flow, sessions |
| [Database](docs/database.md) | Schema, Alembic migrations, pgvector |
| [LLM service](docs/llm-service.md) | Model registry, fallback ladder, structured output |
| [Memory](docs/memory.md) | mem0 long-term memory |
| [RAG](docs/rag.md), [Guardrails](docs/guardrails.md), [Reliability](docs/reliability.md), [Tools](docs/tools.md), [Agents](docs/agents.md), [Evaluation](docs/evaluation.md), [Deployment](docs/deployment.md) | The kit layers |
| [Observability](docs/observability.md) | Langfuse, structured logs, Prometheus, profiling |
| [Docker](docs/docker.md) | Compose and the monitoring stack |

Coding conventions for people and coding agents: [AGENTS.md](AGENTS.md).

## Where it comes from

- Base: [fastapi-langgraph-agent-production-ready-template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template) (MIT).
- Layers ported or specified from the AI20K (VinUniversity) labs, mostly from the author's own
  solutions: D08 RAG, D11 guardrails, D12 deployment, D14 evaluation, D09 MCP/A2A; and from the
  lab specifications of D16, D18, D23, D24, D25. Each layer's doc names the files and commits.
- Libraries chosen per concern: LangGraph, LangChain (`langchain.mcp`), deepagents, FastMCP,
  a2a-sdk, MarkItDown, underthesea, bm25s, rerankers, pgvector.

Security issues: see [SECURITY.md](SECURITY.md). License: [LICENSE](LICENSE) (MIT, from the base template).
