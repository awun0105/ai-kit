<a id="top"></a>

![AI Kit — agent development kit](assets/banner.png)

<div align="center">

[Tiếng Việt: README.md](README.md)

**A toolbox of tested parts for LLM and agent apps, assembled into a runnable FastAPI + LangGraph app.**

[![CI](https://github.com/awun0105/ai-kit/actions/workflows/ci.yaml/badge.svg)](https://github.com/awun0105/ai-kit/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Advanced Usage](#advanced-usage)
- [Documentation](#documentation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Overview

AI Kit collects the parts every LLM product ends up needing: retrieval, guardrails, retries
across models, evaluation and a release pipeline. Each part lives in its own package under
`app/services/`, has offline tests and a doc page, and stays off by default when it costs money
or changes behaviour. The parts are wired together into a working agent app, so a new project
can start from the whole app or copy a single part into an existing codebase.

## Key Features

- **Agent core:** LangGraph agent with Postgres checkpoints, tool calling, human approval, mem0 memory, JWT sessions, rate limits and a history budget with optional summarization of old turns.
- **RAG:** ingest md, txt, pdf, docx and html; hybrid search (pgvector + BM25 with Vietnamese word segmentation); optional reranker; answers with checked citations.
- **Guardrails:** prompt-injection and PII rails in English and Vietnamese, an optional LLM policy rail, approval for risky tools, an egress allowlist, redaction while streaming.
- **Reliability:** one retry layer with a retry budget, per-model circuit breakers, a fallback ladder across models, an answer cache that serves stale answers when providers are down.
- **Tools and agents:** MCP client and server, A2A server and client, a deep-research team, an ACP agent (embed the kit's agent in Zed, JetBrains or your own app), `ResilientChatModel` for any LangChain agent.
- **Offline:** runs on the user's machine with no server: SQLite checkpoints and knowledge base, in-process embeddings (FastEmbed), local models through Ollama or llama.cpp.
- **Evaluation:** golden sets with retrieval metrics, a separate judge model, tool-call checks, repeated runs, simulated multi-turn users and CI gates.
- **Delivery:** non-root Docker image, release to GHCR after a Trivy scan, Prometheus alerts, a per-user monthly cost limit.

## Repository Structure

| Path | Description |
|---|---|
| `app/services/` | The toolbox: `rag/`, `guardrails/`, `reliability/`, `evaluation/`, `mcp/`, `a2a/`, `agents/`, `cost_guard.py`, `compaction.py` |
| `app/core/kit.py` | Builds each toolbox part from settings; the one place that reads them |
| `app/core/langgraph/` | The agent graph: input guard, chat, tools, output rail |
| `app/api/v1/` | Auth, chat and streaming routes |
| `scripts/` | `bootstrap`, `ingest`, `eval_rag`, `research` |
| `evals/datasets/` | Sample corpus, golden set and conversation scenarios |
| `tests/` | Offline tests for every part |
| `docs/` | One page per part; `ROADMAP.md` lists them |

## Getting Started

Needs Python 3.13+, [uv](https://docs.astral.sh/uv/) and Docker.

1. Generate a project and choose its features (RAG, answer cache, LLM rail, reranker, MCP, A2A, deep research):

   ```bash
   uvx copier copy --trust gh:awun0105/ai-kit my-app
   cd my-app && git init
   ```

   This writes `.env.development` with a random JWT secret and the chosen features turned on.

2. Install and run the offline tests (no API key needed):

   ```bash
   uv sync        # add --extra rerank / a2a / deep for those features
   make test
   ```

3. Put `OPENAI_API_KEY` in `.env.development`, then start the API with Postgres and pgvector:

   ```bash
   make docker-up
   ```

4. Load documents into the knowledge base and open `http://localhost:8000/docs`:

   ```bash
   make ingest CORPUS=path/to/docs
   ```

<details>
<summary>First request from the command line</summary>

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/register -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "Secret123!", "username": "you"}' | jq -r .token.access_token)  # pragma: allowlist secret
SESSION=$(curl -s -X POST localhost:8000/api/v1/auth/session -H "Authorization: Bearer $TOKEN" | jq -r .token.access_token)
curl -X POST localhost:8000/api/v1/chatbot/chat -H "Authorization: Bearer $SESSION" \
  -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

</details>

Local development without Docker: [docs/getting-started.md](docs/getting-started.md).

## Advanced Usage

**Take one part into another project.** Each package under `app/services/` imports on its own,
without the app's settings, database or model registry (`tests/test_toolbox_imports.py` checks
it). Copy the package with its tests and follow the **Copy this module** and **When you plug it
in** sections of its doc. `app/core/kit.py` shows how the app builds it from settings.

**Pull later kit improvements** into a generated project with `uvx copier update --trust`.

**Evaluate before shipping:** `make eval-rag-offline` checks retrieval without an API key (the
CI gate); `make eval-rag` runs the real agent against the golden set with a judge model. Options
such as `--runs 3` and `--scenarios` are in [docs/evaluation.md](docs/evaluation.md).

## Documentation

Start with [docs/ROADMAP.md](docs/ROADMAP.md): every part, its status and where it came from.
Each part has its own page: [RAG](docs/rag.md), [guardrails](docs/guardrails.md),
[reliability](docs/reliability.md), [tools](docs/tools.md), [agents](docs/agents.md),
[evaluation](docs/evaluation.md), [conversation history](docs/compaction.md), [offline](docs/offline.md),
[deployment](docs/deployment.md). Settings are in [docs/configuration.md](docs/configuration.md);
coding rules for people and coding agents are in [AGENTS.md](AGENTS.md).

## Acknowledgement

- [fastapi-langgraph-agent-production-ready-template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template) (MIT): the base app.
- AI20K (VinUniversity) labs D08, D09, D11, D12, D14 (the author's solutions) and D16, D18, D23, D24, D25 (specifications).
- [Google ADK](https://github.com/google/adk-python): ideas for history compaction and agent evaluation, rewritten for LangGraph.
- LangGraph, LangChain, deepagents, FastMCP, a2a-sdk, MarkItDown, underthesea, bm25s, rerankers, pgvector.

Each part's doc names the exact files and commits it came from.

## License

MIT, inherited from the base template: see [LICENSE](LICENSE). To report a security issue, see
[SECURITY.md](SECURITY.md).

---

<div align="center">

[Back to Top](#top)

</div>
