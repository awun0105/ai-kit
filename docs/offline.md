# Offline and on-device

Run the kit on the user's machine with no server and, once models are downloaded, no network:
a desktop app, or a sidecar process next to an app written in another language (start it with
the ACP adapter, [tools](tools.md#acp-agent)).

Install the extra: `uv sync --extra offline` (FastEmbed and the LangGraph SQLite checkpointer).

## What changes

| Part | Server default | Offline | Setting |
|---|---|---|---|
| Conversation checkpoints | Postgres | One SQLite file | `CHECKPOINT_BACKEND=sqlite`, `CHECKPOINT_SQLITE_PATH` |
| Knowledge base | pgvector | One SQLite file (`SQLiteVectorStore`) | `RAG_VECTOR_STORE=sqlite`, `RAG_SQLITE_PATH` |
| Embeddings | OpenAI API | In-process on the CPU (`FastEmbedEmbedder`), or a local Ollama through `RAG_EMBEDDING_BASE_URL` | `RAG_EMBEDDING_PROVIDER=fastembed`, `RAG_EMBEDDING_MODEL`, `RAG_EMBEDDING_DIM`, `RAG_EMBEDDING_CACHE_DIR` |
| Chat models | The built-in list in `registry.py` | Any list from settings, e.g. Ollama or llama.cpp | `LLM_MODELS_JSON`, `DEFAULT_LLM_MODEL` |
| Long-term memory (mem0) | On | Off (it calls an LLM and an embedding API every turn) | `LONG_TERM_MEMORY_ENABLED=false` |

Example `.env` for a laptop:

```bash
CHECKPOINT_BACKEND=sqlite
CHECKPOINT_SQLITE_PATH=data/checkpoints.sqlite
RAG_ENABLED=true
RAG_VECTOR_STORE=sqlite
RAG_SQLITE_PATH=data/knowledge.sqlite
RAG_EMBEDDING_PROVIDER=fastembed
RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_EMBEDDING_DIM=384
RAG_EMBEDDING_CACHE_DIR=data/models
LLM_MODELS_JSON='[{"name": "local", "model": "ollama:qwen3:8b"}]'   # needs --extra ollama
DEFAULT_LLM_MODEL=local
LONG_TERM_MEMORY_ENABLED=false
LANGFUSE_TRACING_ENABLED=false
```

For llama.cpp (or any OpenAI-compatible local server), give the endpoint and a key of its own:
`[{"name": "local", "model": "openai:qwen3", "kwargs": {"base_url": "http://127.0.0.1:8080/v1", "api_key": "none"}}]`.

## Embedding models

FastEmbed downloads the model into `RAG_EMBEDDING_CACHE_DIR` on first use; ship that folder with
the app, or run once online, to be fully offline. Models that cover Vietnamese:

| Model | Dimensions | Size |
|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 0.22 GB |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | 1.0 GB |
| `intfloat/multilingual-e5-large` | 1024 | 2.24 GB |

`RAG_EMBEDDING_DIM` must match the model; a mismatch fails at the first embedding instead of
storing unusable vectors. Changing the model means re-ingesting (new vectors, new dimension).
Checked on 2026-10-27: with the MiniLM model, "máy tính xách tay bảo hành bao lâu?" scores 0.60
against "Laptop được bảo hành 24 tháng." and 0.12 against an unrelated sentence.

## When you plug it in

- SQLite has one writer at a time: right for one user on one machine, wrong for a server. Keep
  Postgres there.
- `SQLiteVectorStore` searches by brute force in Python: fine for tens of thousands of chunks; a
  bigger corpus belongs in pgvector.
- Small local chat models call tools less reliably than hosted ones. Run the golden-set
  evaluation (`--runs 3`, `answers.trajectory`) against the local model before shipping.
- The HTTP API still needs Postgres for users and sessions. Offline use goes through the agent
  directly (the ACP adapter, or `LangGraphAgent` in your own code), not through the API.

## Code

| File | Responsibility |
|---|---|
| `app/services/rag/sqlite_store.py` | `SQLiteVectorStore` (standard library only) |
| `app/services/rag/embeddings.py` | `FastEmbedEmbedder` |
| `app/services/llm/registry.py` | `parse_model_configs`, `LLM_MODELS_JSON` |
| `app/core/kit.py` | `build_checkpointer_from_settings`, the store and embedder choices |
| `tests/test_offline.py` | Restart survival, prune, dimension checks, model list, memory off, SQLite checkpoints |

## Copy this module

`sqlite_store.py` needs only `app/services/rag/contracts.py` and the standard library;
`FastEmbedEmbedder` needs `fastembed`. Both implement the `VectorStore` / `Embedder` protocols in
`app/services/rag/`, so they drop into any `RAGPipeline`.

## Provenance

Written for the kit. FastEmbed (Apache-2.0), `langgraph-checkpoint-sqlite` (MIT).
