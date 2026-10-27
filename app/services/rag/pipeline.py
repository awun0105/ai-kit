"""Hybrid retrieval pipeline: dense + BM25, fused with RRF, gated on dense confidence.

Ported from D08 ``task9_retrieval_pipeline.py`` and ``task10_generation.py``.
Differences from the lab:

- D08 fell back to a PageIndex search when confidence was low. Here the result
  is flagged ``low_confidence`` and the agent is told to say the sources are
  insufficient, which keeps the pipeline free of a second LLM-driven retriever.
- Re-ingestion skips chunks whose stored content is unchanged (D08's resume logic).
- An optional reranker (D08's cross-encoder bonus, here via the ``rerankers``
  library) re-scores the fused candidates.
"""

import asyncio
import time

from pydantic import BaseModel

from app.core.logging import logger
from app.services.rag.chunking import chunk_documents
from app.services.rag.contracts import (
    Chunk,
    Document,
    RetrievalResult,
    SearchResult,
)
from app.services.rag.embeddings import Embedder
from app.services.rag.ranking import (
    BM25Index,
    Reranker,
    TokenizerMode,
    reciprocal_rank_fusion,
    tokenize,
)
from app.services.rag.store import VectorStore

INSUFFICIENT_EVIDENCE = "The available sources do not contain enough information to answer this."


class IngestReport(BaseModel):
    """What an ingestion run did."""

    documents: int
    chunks: int
    embedded: int
    reused: int
    removed_chunks: int = 0
    changed_documents: list[str] = []  # new, edited, shortened or removed: invalidate answers built on them
    removed_documents: list[str] = []


class RAGPipeline:
    """Ingest documents and retrieve the most relevant chunks for a query."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        score_threshold: float = 0.3,
        lexical_ttl_seconds: float = 300.0,
        embed_batch_size: int = 64,
        tokenizer: TokenizerMode = "auto",
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
    ):
        """Configure the pipeline.

        Args:
            embedder: Produces vectors for chunks and queries.
            store: Holds chunks and answers dense search.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters shared by neighbouring chunks.
            top_k: Results returned by ``retrieve`` when the caller does not say.
            score_threshold: Minimum top cosine similarity before results count as confident.
            lexical_ttl_seconds: How long the BM25 index is reused before reloading the corpus,
                so ingestion by another process is eventually picked up.
            embed_batch_size: Chunks sent to the embedder per request.
            tokenizer: BM25 tokenization: ``auto`` segments Vietnamese text with underthesea,
                ``vi`` always does, ``simple`` only splits on non-word characters.
            reranker: Optional reranker applied to the fused candidates.
            rerank_candidates: How many fused candidates the reranker sees.
        """
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._lexical_ttl = lexical_ttl_seconds
        self._embed_batch_size = embed_batch_size
        self._tokenizer: TokenizerMode = tokenizer
        self.reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._bm25: BM25Index | None = None
        self._bm25_stale = True
        self._bm25_built_at = 0.0
        self._bm25_lock = asyncio.Lock()

    async def ingest(self, documents: list[Document], prune_missing: bool = False) -> IngestReport:
        """Chunk, embed and upsert documents. Unchanged chunks are not re-embedded.

        Chunks a re-ingested document no longer produces (it got shorter) are deleted.
        With ``prune_missing`` the batch is the whole corpus: documents absent from it
        are deleted too. Deletion runs after the upsert, so a failed embedding call
        leaves the old chunks in place.
        """
        chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        stored = await self.store.existing_contents([c.id for c in chunks])
        pending = [c for c in chunks if stored.get(c.id) != c.content]

        for start in range(0, len(pending), self._embed_batch_size):
            batch = await asyncio.to_thread(self._with_lexical_tokens, pending[start : start + self._embed_batch_size])
            vectors = await self.embedder.embed_documents([c.content for c in batch])
            await self.store.upsert(batch, vectors)

        batch_documents = {d.id for d in documents}
        owners = await self.store.chunk_owners(None if prune_missing else sorted(batch_documents))
        current = {c.id for c in chunks}
        stale = sorted(cid for cid in owners if cid not in current)
        await self.store.delete(stale)

        self._bm25_stale = True  # corpus changed; rebuild lazily on next query, reusing cached tokens
        report = IngestReport(
            documents=len(documents),
            chunks=len(chunks),
            embedded=len(pending),
            reused=len(chunks) - len(pending),
            removed_chunks=len(stale),
            changed_documents=sorted({c.document_id for c in pending} | {owners[cid] for cid in stale}),
            removed_documents=sorted({owners[cid] for cid in stale} - batch_documents),
        )
        logger.info("rag_ingest_completed", **report.model_dump(exclude={"changed_documents", "removed_documents"}))
        return report

    def _with_lexical_tokens(self, chunks: list[Chunk]) -> list[Chunk]:
        """Attach BM25 tokens so the query path never segments the corpus (CPU-bound, run in a thread)."""
        return [
            c.model_copy(
                update={"lexical_tokens": tokenize(c.content, self._tokenizer), "lexical_mode": self._tokenizer}
            )
            for c in chunks
        ]

    def _lexical_outdated(self) -> bool:
        expired = time.monotonic() - self._bm25_built_at > self._lexical_ttl
        return self._bm25 is None or self._bm25_stale or expired

    async def _lexical_index(self) -> BM25Index:
        """Return the BM25 index, rebuilding it once however many queries arrive meanwhile.

        Building tokenizes and indexes the whole corpus (CPU-bound), so it runs in a thread.
        """
        if self._lexical_outdated():
            async with self._bm25_lock:
                if self._lexical_outdated():  # another query may have rebuilt it while this one waited
                    self._bm25_stale = False  # before loading, so an ingest during the load marks it again
                    chunks = await self.store.all_chunks()
                    self._bm25 = await asyncio.to_thread(BM25Index, chunks, self._tokenizer, self._bm25)
                    self._bm25_built_at = time.monotonic()
        assert self._bm25 is not None
        return self._bm25

    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Run dense and BM25 search, fuse them once with RRF, and flag low confidence."""
        k = top_k or self.top_k
        if not query.strip():
            return RetrievalResult(query=query, results=[], best_dense_score=0.0, low_confidence=True)

        # Wider pools give RRF candidates from both retrievers, and the reranker more to choose from.
        pool_k = max(k * 2, self._rerank_candidates) if self.reranker else k * 2
        dense: list[SearchResult] = []
        try:
            dense = await self.store.search(await self.embedder.embed_query(query), pool_k)
        except Exception as e:
            # Keep answering from BM25 when the embedding provider is down.
            logger.warning("rag_dense_search_failed", error=str(e))
        lexical = (await self._lexical_index()).search(query, pool_k)

        fused = reciprocal_rank_fusion([r for r in (dense, lexical) if r], top_k=pool_k if self.reranker else k)
        if self.reranker and fused:
            try:
                fused = await self.reranker.rerank(query, fused, k)
            except Exception as e:
                logger.warning("rag_rerank_failed", error=str(e))
                fused = fused[:k]
        # The gate uses the raw cosine score: RRF scores are rank-based and not comparable.
        best_dense = dense[0].score if dense else 0.0
        result = RetrievalResult(
            query=query, results=fused, best_dense_score=best_dense, low_confidence=best_dense < self.score_threshold
        )
        logger.info(
            "rag_retrieve_completed",
            dense=len(dense),
            lexical=len(lexical),
            returned=len(fused),
            best_dense_score=round(best_dense, 4),
            low_confidence=result.low_confidence,
        )
        return result


def format_context(result: RetrievalResult) -> str:
    """Render results as numbered sources for the LLM, with citation instructions.

    Labels follow retrieval rank, so ``[1]`` is always the best-ranked chunk.
    """
    if not result.results:
        return f"No matching sources. Reply: {INSUFFICIENT_EVIDENCE}"

    blocks = []
    for n, item in enumerate(result.results, start=1):
        origin = item.chunk.url or item.chunk.source
        blocks.append(f"[{n}] {item.chunk.title} ({origin})\n{item.chunk.content}")
    header = (
        "Answer only from the sources below. Cite every claim with its label, e.g. [1] or [1][2]. "
        f"Do not invent sources. If they do not answer the question, reply: {INSUFFICIENT_EVIDENCE}"
    )
    if result.low_confidence:
        header += "\nWarning: retrieval confidence is low; these sources may be off-topic."
    return header + "\n\n" + "\n\n".join(blocks)
