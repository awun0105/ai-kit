"""Retrieval-augmented generation: ingestion, hybrid retrieval and citation formatting."""

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
    "TokenizerMode",
    "VectorStore",
    "format_context",
]
