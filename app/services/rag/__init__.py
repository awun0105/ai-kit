"""Retrieval-augmented generation: ingestion, hybrid retrieval and citation formatting."""

from typing import (
    Optional,
    cast,
    get_args,
)

from app.core.config import settings
from app.services.rag.contracts import (
    Chunk,
    Document,
    RetrievalResult,
    SearchResult,
)
from app.services.rag.embeddings import (
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
)
from app.services.rag.pipeline import (
    RAGPipeline,
    format_context,
)
from app.services.rag.ranking import (
    Reranker,
    RerankersReranker,
    TokenizerMode,
)
from app.services.rag.store import (
    InMemoryVectorStore,
    PgVectorStore,
    VectorStore,
)

_pipeline: Optional[RAGPipeline] = None


def build_pipeline_from_settings() -> RAGPipeline:
    """Assemble a pipeline from the RAG_* settings."""
    embedder: Embedder
    if settings.RAG_EMBEDDING_PROVIDER == "hashing":
        embedder = HashingEmbedder(settings.RAG_EMBEDDING_DIM)
    else:
        embedder = OpenAIEmbedder(
            model=settings.RAG_EMBEDDING_MODEL,
            dimension=settings.RAG_EMBEDDING_DIM,
            api_key=settings.RAG_EMBEDDING_API_KEY,
            base_url=settings.RAG_EMBEDDING_BASE_URL or None,
        )

    store: VectorStore
    if settings.RAG_VECTOR_STORE == "memory":
        store = InMemoryVectorStore()
    else:
        store = PgVectorStore.from_settings(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            db=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            table=settings.RAG_TABLE,
            dimension=settings.RAG_EMBEDDING_DIM,
        )

    reranker: Optional[Reranker] = None
    if settings.RAG_RERANKER:
        reranker = RerankersReranker(settings.RAG_RERANKER, settings.RAG_RERANKER_TYPE or None)

    tokenizer = cast(TokenizerMode, settings.RAG_LEXICAL_TOKENIZER)
    if tokenizer not in get_args(TokenizerMode):
        raise ValueError(f"RAG_LEXICAL_TOKENIZER must be one of {get_args(TokenizerMode)}, got {tokenizer!r}")

    return RAGPipeline(
        embedder=embedder,
        store=store,
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        top_k=settings.RAG_TOP_K,
        score_threshold=settings.RAG_SCORE_THRESHOLD,
        tokenizer=tokenizer,
        reranker=reranker,
        rerank_candidates=settings.RAG_RERANK_CANDIDATES,
    )


def get_rag_pipeline() -> RAGPipeline:
    """Return the process-wide pipeline, building it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline_from_settings()
    return _pipeline


def set_rag_pipeline(pipeline: Optional[RAGPipeline]) -> None:
    """Replace the process-wide pipeline (used by tests)."""
    global _pipeline
    _pipeline = pipeline


__all__ = [
    "Chunk",
    "Document",
    "Embedder",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "OpenAIEmbedder",
    "PgVectorStore",
    "RAGPipeline",
    "Reranker",
    "RerankersReranker",
    "RetrievalResult",
    "SearchResult",
    "VectorStore",
    "build_pipeline_from_settings",
    "format_context",
    "get_rag_pipeline",
    "set_rag_pipeline",
]
