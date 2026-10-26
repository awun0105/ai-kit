"""Retrieval metrics against labelled source documents. Deterministic: no model needed.

Chunks are mapped to their document id, duplicates keep their best rank, so a
document counts once however many of its chunks were returned.
"""

import math
from collections import defaultdict
from typing import Sequence

from pydantic import BaseModel


class RetrievalScores(BaseModel):
    """Per-question retrieval scores."""

    hit: float  # 1 if any expected document is in the top k
    recall: float  # share of expected documents in the top k
    reciprocal_rank: float  # 1 / rank of the first expected document, 0 if none
    ndcg: float  # binary-relevance nDCG@k


def ranked_documents(document_ids: Sequence[str]) -> list[str]:
    """Collapse chunk-level results to documents in first-seen order."""
    seen: list[str] = []
    for doc in document_ids:
        if doc not in seen:
            seen.append(doc)
    return seen


def score_retrieval(retrieved: Sequence[str], expected: Sequence[str], k: int) -> RetrievalScores:
    """Score one ranked list of document ids against the expected ones."""
    ranked = ranked_documents(retrieved)[:k]
    relevant = set(expected)
    if not relevant:
        return RetrievalScores(hit=1.0, recall=1.0, reciprocal_rank=1.0, ndcg=1.0)
    hits = [i for i, doc in enumerate(ranked) if doc in relevant]
    dcg = sum(1.0 / math.log2(i + 2) for i in hits)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return RetrievalScores(
        hit=1.0 if hits else 0.0,
        recall=len(hits) / len(relevant),
        reciprocal_rank=1.0 / (hits[0] + 1) if hits else 0.0,
        ndcg=dcg / ideal if ideal else 0.0,
    )


def aggregate(scores: dict[str, RetrievalScores], groups: dict[str, str]) -> dict[str, dict[str, float]]:
    """Mean of every metric overall and per group (e.g. difficulty)."""
    buckets: dict[str, list[RetrievalScores]] = defaultdict(list)
    for item_id, s in scores.items():
        buckets["all"].append(s)
        buckets[groups.get(item_id, "unspecified")].append(s)
    return {
        name: {
            metric: round(sum(getattr(s, metric) for s in values) / len(values), 4)
            for metric in ("hit", "recall", "reciprocal_rank", "ndcg")
        }
        | {"n": len(values)}
        for name, values in buckets.items()
    }
