"""Data contracts shared by every RAG stage.

Ported from the D08 RAG-Pipeline lab (``src/contracts.py``), switched from
TypedDict to Pydantic so invalid records fail at construction time.
"""

from typing import (
    Any,
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
)

RetrievalMethod = Literal["dense", "bm25", "hybrid", "rerank"]


class Document(BaseModel):
    """A source document before chunking."""

    id: str = Field(min_length=1, description="Stable id, e.g. the file path relative to the corpus root")
    content: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1, description="Human-readable origin, e.g. a file name")
    url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable piece of a document."""

    id: str = Field(min_length=1, description="'<document id>::chunk-<n>', stable across re-ingestion")
    document_id: str
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    title: str
    source: str
    url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    lexical_tokens: Optional[list[str]] = Field(
        default=None, description="BM25 tokens computed at ingestion, so queries never re-tokenize the corpus"
    )
    lexical_mode: Optional[str] = Field(default=None, description="Tokenizer mode that produced lexical_tokens")


class SearchResult(BaseModel):
    """A chunk returned by a retriever, with the score on that retriever's own scale."""

    chunk: Chunk
    score: float
    method: RetrievalMethod


class RetrievalResult(BaseModel):
    """Final output of the retrieval pipeline."""

    query: str
    results: list[SearchResult]
    best_dense_score: float = Field(
        description="Top cosine similarity from dense search, used for the confidence gate"
    )
    low_confidence: bool = Field(description="True when best_dense_score is below the configured threshold")
