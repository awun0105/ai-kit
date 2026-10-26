"""Lexical search, rank fusion and optional reranking.

BM25 and RRF are ported from D08 ``task6_lexical_search.py`` and
``task7_reranking.py``. Scoring uses bm25s; Vietnamese word segmentation uses
underthesea, so compounds like "học phí" become one token instead of two.
Reranking wraps the ``rerankers`` library (cross-encoders, FlashRank, Cohere,
Jina, ...) behind one interface; it is an optional extra.
"""

import asyncio
import re
import unicodedata
from typing import (
    Literal,
    Protocol,
)

import bm25s
from underthesea import word_tokenize

from app.services.rag.contracts import (
    Chunk,
    SearchResult,
)

TokenizerMode = Literal["auto", "vi", "simple"]

_TOKEN = re.compile(r"\w+", re.UNICODE)
# Letters that only occur in Vietnamese among Latin-script languages.
_VIETNAMESE_CHARS = set("ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")


def _looks_vietnamese(text: str) -> bool:
    return any(ch in _VIETNAMESE_CHARS for ch in text.lower())


def tokenize(text: str, mode: TokenizerMode = "auto") -> list[str]:
    """Lower-case word tokens; the same mode must be used for corpus and query.

    In ``vi`` mode (or ``auto`` when the text contains Vietnamese letters),
    underthesea joins multi-syllable words with ``_`` before splitting.
    """
    text = unicodedata.normalize("NFC", text)
    if mode == "vi" or (mode == "auto" and _looks_vietnamese(text)):
        text = str(word_tokenize(text, format="text"))
    return _TOKEN.findall(text.lower())


class BM25Index:
    """BM25 over a fixed chunk list. Rebuild it when the corpus changes.

    Tokens come, in order of preference, from ``chunk.lexical_tokens`` (stored
    at ingestion), from the previous index passed as ``reuse``, or from
    tokenizing now. Vietnamese segmentation costs a few milliseconds per chunk,
    so tokenizing a large corpus at query time is what this avoids.
    """

    def __init__(self, chunks: list[Chunk], mode: TokenizerMode = "auto", reuse: "BM25Index | None" = None):
        """Tokenize ``chunks`` (reusing tokens from ``reuse`` when content is unchanged) and build the index."""
        self._chunks = chunks
        self._mode: TokenizerMode = mode
        previous = reuse._tokens if reuse is not None and reuse._mode == mode else {}
        self._tokens: dict[str, tuple[str, list[str]]] = {}
        for chunk in chunks:
            cached = previous.get(chunk.id)
            if chunk.lexical_tokens is not None and chunk.lexical_mode == mode:
                tokens = chunk.lexical_tokens
            elif cached and cached[0] == chunk.content:
                tokens = cached[1]
            else:
                tokens = tokenize(chunk.content, mode)
            self._tokens[chunk.id] = (chunk.content, tokens)

        self._bm25: bm25s.BM25 | None = None
        if chunks:
            self._bm25 = bm25s.BM25()
            self._bm25.index([self._tokens[c.id][1] for c in chunks], show_progress=False)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return chunks with a positive BM25 score, best first."""
        tokens = tokenize(query, self._mode)
        if self._bm25 is None or not tokens or top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(self._chunks)), key=lambda i: -float(scores[i]))
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]), method="bm25")
            for i in order[:top_k]
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(ranked_lists: list[list[SearchResult]], top_k: int, k: int = 60) -> list[SearchResult]:
    """Fuse rankings with RRF(d) = sum(1 / (k + rank)), rank starting at 1.

    Only ranks are used, so cosine and BM25 scores never get added together.
    The fused score reflects rank agreement, not relevance; do not threshold on it.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            scores[result.chunk.id] = scores.get(result.chunk.id, 0.0) + 1.0 / (max(1, k) + rank)
            first_seen.setdefault(result.chunk.id, result)
    best = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    return [SearchResult(chunk=first_seen[cid].chunk, score=scores[cid], method="hybrid") for cid in best]


class Reranker(Protocol):
    """Re-scores candidates against the query, best first."""

    async def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Return at most ``top_k`` candidates in the new order."""
        ...


class RerankersReranker:
    """Adapter for the ``rerankers`` library (install the ``rerank`` extra).

    ``model`` is passed straight to ``rerankers.Reranker``, e.g. ``"flashrank"``
    (small, CPU, English-centric) or ``"BAAI/bge-reranker-v2-m3"`` with
    ``model_type="cross-encoder"`` (multilingual, ~2 GB).
    """

    def __init__(self, model: str, model_type: str | None = None):
        """Load the model once; the first use may download weights."""
        from rerankers import Reranker as _Reranker  # optional "rerank" extra

        kwargs = {"model_type": model_type} if model_type else {}
        ranker = _Reranker(model, verbose=0, **kwargs)
        if ranker is None:  # rerankers returns None when the backend's dependencies are missing
            raise RuntimeError(f"rerankers could not load {model!r}; install the matching extra")
        self._ranker = ranker

    def _rank(self, query: str, candidates: list[SearchResult]) -> list[tuple[int, float]]:
        ranked = self._ranker.rank(
            query=query, docs=[c.chunk.content for c in candidates], doc_ids=list(range(len(candidates)))
        )
        return [
            (int(r.document.doc_id), float(r.score))
            for r in ranked.results
            if r.document.doc_id is not None and r.score is not None
        ]

    async def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Return at most ``top_k`` candidates in the new order."""
        if not candidates:
            return []
        # Model inference is CPU-bound and synchronous; keep it off the event loop.
        ranked = await asyncio.to_thread(self._rank, query, candidates)
        return [SearchResult(chunk=candidates[i].chunk, score=score, method="rerank") for i, score in ranked[:top_k]]
