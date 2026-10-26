# AI Kit roadmap

Goal: one agent-development kit that a new LLM/agent app starts from. Each layer
fills one slot of the app, uses one library per concern, ships with offline tests,
and records where its code came from.

Base: [fastapi-langgraph-agent-production-ready-template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template)
(MIT) — FastAPI, LangGraph agent with Postgres checkpointer, LLM retries and model
fallback, mem0 long-term memory, JWT auth, rate limiting, Langfuse, Prometheus/Grafana,
LLM-as-judge evals, Docker Compose.

## Layers

| # | Layer | Status | Sources |
|---|---|---|---|
| 0 | Test harness (scripted LLM, in-memory checkpointer, pytest in CI) | done | – |
| 1 | RAG: ingestion, hybrid retrieval, Vietnamese BM25, rerank, `knowledge_search` tool — [docs/rag.md](rag.md) | done | D08 (author's solution), D18; MarkItDown, underthesea, bm25s, rerankers |
| 2 | Guardrails: injection and PII rails, LLM policy rail, tool approval (HITL), egress allowlist, stream redaction — [docs/guardrails.md](guardrails.md) | done | D11 (author's solution), D24 |
| 3 | Reliability: error classes, one retry layer with budget, circuit breakers, answer cache with stale-if-error, burn-rate alerts — [docs/reliability.md](reliability.md) | done | D25 (spec) |
| 4 | Tools: MCP client and server, A2A server and client — [docs/tools.md](tools.md) | done | D09 (author's solution); langchain.mcp, FastMCP, a2a-sdk |
| 5 | Agents: `ResilientChatModel`, deep-research team on deepagents, citation checks — [docs/agents.md](agents.md) | done | D16, D23; deepagents |
| 6 | Evaluation: golden sets, retrieval metrics, judged agent answers, judge-bias controls, CI gate — [docs/evaluation.md](evaluation.md) | done | D14 (author's solution), D24 |
| 7 | CI/CD and deploy: gated release to GHCR with SBOM and Trivy, nightly eval, multi-stage non-root image, cost guard, safe errors, bounded metrics — [docs/deployment.md](deployment.md) | done | D12 (author's solution), D21 |
| 8 | Packaging: `copier` template + `scripts/bootstrap.py`, kit README, `ai-kit` agent skill (in `~/.agents/skills/ai-kit`) | done | – |

## Rules for every layer

1. One library per concern; adapters behind a small protocol so it can be swapped.
2. Off by default when it needs a model download, a paid API, or changes behaviour.
3. Tests run offline (`uv run pytest`); integration tests skip unless their service is configured.
4. Each layer documents its settings, limits and provenance in `docs/<layer>.md`.
