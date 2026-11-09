"""Deterministic information-retrieval ranking metrics.

Precision@K uses K as its denominator even when fewer than K results are returned.
Duplicate retrieved IDs count only once, at their first rank.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

METRIC_NAMES = ("hit_at_k", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")


def _normalise_ids(values: Iterable[object], *, field: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be an iterable of IDs, not a string")
    try:
        result = []
        for value in values:
            item = str(value).strip()
            if not item:
                raise ValueError(f"{field} contains an empty ID")
            result.append(item)
        return result
    except TypeError as exc:
        raise ValueError(f"{field} must be iterable") from exc


def ranking_metrics(
    ranked_ids: Iterable[object],
    relevant_ids: Iterable[object],
    k: int = 5,
) -> dict[str, float]:
    """Calculate Hit@K, Precision@K, Recall@K, MRR@K and NDCG@K for one query."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > 100:
        raise ValueError("k must be an integer between 1 and 100")

    ranked = _normalise_ids(ranked_ids, field="ranked_ids")
    relevant_list = _normalise_ids(relevant_ids, field="relevant_ids")
    relevant = set(relevant_list)
    if not relevant:
        raise ValueError("each query needs at least one relevant ID")

    unique_ranked = []
    seen = set()
    for item in ranked:
        if item not in seen:
            unique_ranked.append(item)
            seen.add(item)

    top_k = unique_ranked[:k]
    hits = [item in relevant for item in top_k]
    hit_count = sum(hits)
    reciprocal_rank = next(
        (1.0 / rank for rank, hit in enumerate(hits, start=1) if hit),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
        if hit
    )
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return {
        "hit_at_k": 1.0 if hit_count else 0.0,
        "precision_at_k": hit_count / k,
        "recall_at_k": hit_count / len(relevant),
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average query-level metrics; reject empty benchmark runs."""
    if not rows:
        raise ValueError("cannot aggregate an empty benchmark")
    return {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in METRIC_NAMES
    }
