"""Retrieval-augmented generation: ingestion, hybrid retrieval and citation formatting."""

from app.services.rag.contracts import (
    Chunk,
    Document,
    RetrievalResult,
    SearchResult,
)
from app.services.rag.embeddings import (
    Embedder,
    FastEmbedEmbedder,
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
from app.services.rag.sqlite_store import SQLiteVectorStore

__all__ = [
    "Chunk",
    "Document",
    "Embedder",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "OpenAIEmbedder",
    "PgVectorStore",
    "SQLiteVectorStore",
    "RAGPipeline",
    "Reranker",
    "RerankersReranker",
    "RetrievalResult",
    "SearchResult",
    "TokenizerMode",
    "VectorStore",
    "format_context",
]
