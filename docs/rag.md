# RAG (Retrieval-Augmented Generation)

The agent gets a `knowledge_search` tool when `RAG_ENABLED=true`. The tool runs
hybrid retrieval over documents you ingested and returns numbered sources that
the model must cite as `[1]`, `[2]`, ...

## Flow

```
ingest:   .md .txt .pdf .docx .html
            -> MarkItDown (non-text formats) -> chunk (recursive, stable ids)
            -> BM25 tokens (underthesea for Vietnamese) + embeddings -> upsert into pgvector

retrieve: query -> dense (cosine, pgvector) ─┐
                -> BM25 (bm25s, stored tokens) ─┴> RRF -> [optional reranker] -> top_k + confidence flag
                -> numbered context with citation rules
```

- **Stable chunk ids** (`<relative path>::chunk-<n>`) make re-ingestion an upsert.
  Chunks whose content did not change are not re-embedded or re-tokenized.
- **Vietnamese word segmentation.** "học phí" is one token (`học_phí`), so a query for
  tuition does not match a chunk that only has "học" and "phí" in different places.
  `auto` segments any text containing Vietnamese letters. Tokens are computed at
  ingestion and stored with the chunk, so a new server process builds the BM25
  index without re-segmenting the corpus (9k chunks: 27 s to segment, 0.4 s to load).
- **RRF** fuses by rank only, so cosine and BM25 scores are never added together.
- **Confidence gate** uses the raw top cosine score, not the RRF or rerank score. Below
  `RAG_SCORE_THRESHOLD` the context carries a low-confidence warning, and with no
  results the model is told to answer that the sources are insufficient.
- **Failure handling.** Embedding provider down: BM25 alone. Reranker error: fused order.
  A file that fails to convert: skipped with a warning.

## Usage

```bash
# 1. enable it in .env.development
RAG_ENABLED=true

# 2. ingest a folder (re-run any time; unchanged chunks are skipped)
APP_ENV=development uv run python -m scripts.ingest path/to/corpus --query "a test question"
```

| Variable | Default | Meaning |
|---|---|---|
| `RAG_ENABLED` | `false` | Register the `knowledge_search` tool |
| `RAG_VECTOR_STORE` | `pgvector` | `pgvector` (table `RAG_TABLE` in the app database) or `memory` |
| `RAG_EMBEDDING_PROVIDER` | `openai` | `openai` = any OpenAI-compatible endpoint; `hashing` = offline, no semantics, tests only |
| `RAG_EMBEDDING_MODEL` / `RAG_EMBEDDING_DIM` | `text-embedding-3-small` / `1536` | Changing either needs a new `RAG_TABLE` |
| `RAG_EMBEDDING_BASE_URL` / `RAG_EMBEDDING_API_KEY` | – / `OPENAI_API_KEY` | Non-OpenAI embedding endpoint |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `500` / `50` | Characters |
| `RAG_TOP_K` | `5` | Sources returned to the model |
| `RAG_SCORE_THRESHOLD` | `0.3` | Calibrate per embedding model on a golden set |
| `RAG_LEXICAL_TOKENIZER` | `auto` | `auto`, `vi` (always segment) or `simple`. Changing it requires re-ingesting |
| `RAG_RERANKER` | empty (off) | A [rerankers](https://github.com/AnswerDotAI/rerankers) model name; needs `uv sync --extra rerank` |
| `RAG_RERANKER_TYPE` | empty | e.g. `cross-encoder`, `flashrank`, `jina`, `cohere`; empty lets rerankers infer it |
| `RAG_RERANK_CANDIDATES` | `20` | Fused candidates the reranker re-scores |

### Choosing a reranker

Measured on Vietnamese questions against the AI20K lab corpus:

| Model | Size | Vietnamese |
|---|---|---|
| `flashrank` (default `ms-marco-TinyBERT-L-2-v2`) | 3 MB | Puts the right chunk first but scores unrelated chunks 0.66–0.99; not reliable |
| `ms-marco-MultiBERT-L-12` (`RAG_RERANKER_TYPE=flashrank`) | 99 MB | Same problem: an unrelated store-opening chunk scored 0.997 |
| `BAAI/bge-reranker-v2-m3` (`RAG_RERANKER_TYPE=cross-encoder`) | ~2 GB, needs torch | Multilingual; the recommended local option (not benchmarked here) |
| Jina / Cohere multilingual rerank APIs | – | Hosted alternative through the same adapter |

Leave reranking off unless a golden-set evaluation shows it helps.

## Code

| File | Responsibility |
|---|---|
| `app/services/rag/contracts.py` | `Document`, `Chunk`, `SearchResult`, `RetrievalResult` |
| `app/services/rag/chunking.py` | Load files (MarkItDown for PDF/DOCX/HTML), extract title/url, chunk |
| `app/services/rag/embeddings.py` | `Embedder` protocol, `OpenAIEmbedder`, `HashingEmbedder` |
| `app/services/rag/store.py` | `VectorStore` protocol, `PgVectorStore`, `InMemoryVectorStore` |
| `app/services/rag/ranking.py` | Tokenizer, BM25 index, RRF, `Reranker` protocol and `rerankers` adapter |
| `app/services/rag/pipeline.py` | `RAGPipeline.ingest/retrieve`, `format_context` |
| `app/core/langgraph/tools/knowledge_search.py` | Agent tool |
| `scripts/ingest.py` | CLI |

## Limits and next steps

- BM25 runs in process over all stored chunks (index reloaded at most every 5 minutes
  or after an ingest). Fine for tens of thousands of chunks; beyond that move the
  lexical side into Postgres full-text search, as [R2R](https://github.com/SciPhi-AI/R2R) does.
- Not implemented yet: query expansion / HyDE, parent-child chunks, contextual
  enrichment (D18 lab, module 5), GraphRAG (D19 lab; see LightRAG).
- Scanned PDFs need OCR; MarkItDown only extracts embedded text. Use
  [Docling](https://github.com/docling-project/docling) for scanned or table-heavy PDFs.

## Provenance

| Part | Source |
|---|---|
| Contracts, chunking, BM25, RRF, pipeline, citation prompt | Author's solution to the AI20K D08 RAG-Pipeline lab (`VinUni-AI20k/K4-L3B-RAG-Pipeline`, branch `solution` at `d3e6e6d`: `src/contracts.py`, `task3`–`task7`, `task9`, `task10`) |
| Vietnamese segmentation, reranking idea | AI20K D18 Production-RAG lab (`m2_search.segment_vietnamese`, `m3_rerank`) |
| Libraries | [MarkItDown](https://github.com/microsoft/markitdown) (MIT), [underthesea](https://github.com/undertheseanlp/underthesea) (Apache-2.0), [bm25s](https://github.com/xhluca/bm25s) (MIT), [rerankers](https://github.com/AnswerDotAI/rerankers) (Apache-2.0), pgvector |

Changes from the labs: Pydantic contracts, async API, pgvector instead of ChromaDB,
one OpenAI-compatible embedder instead of four provider branches, a confidence flag
instead of the PageIndex fallback, tokens stored at ingestion.
