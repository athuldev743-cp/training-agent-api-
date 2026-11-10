"""Nutrition-only hybrid retrieval: pgvector + PostgreSQL FTS + RRF + optional remote reranking.

The nutrition corpus is physically separated from fitness_knowledge. The cross-encoder
runs in a separate service so its model memory is not loaded into the main API process.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

import httpx
from fastembed import TextEmbedding

from rag.neon import get_connection

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
RRF_K = 60
MAX_LIMIT = 20
MAX_CANDIDATES = 100
DEFAULT_RERANKER_TIMEOUT_SECONDS = 30.0

_embedding_model: TextEmbedding | None = None

NUTRITION_HYBRID_SQL = """
WITH semantic AS (
    SELECT chunk_id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
    FROM nutrition_knowledge
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT %s
),
keyword AS (
    SELECT chunk_id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) DESC
           ) AS rank
    FROM nutrition_knowledge
    WHERE content_tsv @@ plainto_tsquery('english', %s)
    ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) DESC
    LIMIT %s
),
fused AS (
    SELECT chunk_id, 1.0 / (%s + rank) AS score FROM semantic
    UNION ALL
    SELECT chunk_id, 1.0 / (%s + rank) AS score FROM keyword
),
combined AS (
    SELECT chunk_id, SUM(score) AS rrf_score
    FROM fused
    GROUP BY chunk_id
)
SELECT n.chunk_id, n.content, n.source_id, n.source, n.title,
       n.page, n.chunk_number, c.rrf_score
FROM combined c
JOIN nutrition_knowledge n ON n.chunk_id = c.chunk_id
ORDER BY c.rrf_score DESC
LIMIT %s
"""


def _reranker_timeout_from_env() -> float:
    raw = os.getenv("NUTRITION_RERANKER_TIMEOUT", str(DEFAULT_RERANKER_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid nutrition reranker timeout; using default")
        return DEFAULT_RERANKER_TIMEOUT_SECONDS
    if not math.isfinite(value) or not 0.1 <= value <= 120.0:
        logger.warning("Out-of-range nutrition reranker timeout; using default")
        return DEFAULT_RERANKER_TIMEOUT_SECONDS
    return value


def _get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedding_model


def _vector_literal(values: Any) -> str:
    vector = [float(value) for value in values]
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"embedding must contain {EMBEDDING_DIMENSIONS} dimensions")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    return "[" + ",".join(format(value, ".8g") for value in vector) + "]"


def _remote_rerank_scores(query: str, documents: list[str]) -> list[float] | None:
    """Return remote cross-encoder scores, or None when reranking is unavailable."""
    base_url = os.getenv("NUTRITION_RERANKER_URL", "").strip().rstrip("/")
    api_key = os.getenv("NUTRITION_RERANKER_API_KEY", "").strip()
    if not base_url or not api_key:
        return None

    try:
        response = httpx.post(
            f"{base_url}/rerank",
            json={"query": query, "documents": documents},
            headers={"X-API-Key": api_key},
            timeout=_reranker_timeout_from_env(),
        )
        response.raise_for_status()
        payload = response.json()
        raw_scores = payload.get("scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != len(documents):
            raise ValueError("reranker returned an unexpected number of scores")
        scores = [float(score) for score in raw_scores]
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("reranker returned a non-finite score")
        return scores
    except Exception as exc:
        # RRF remains a valid ranking; an optional remote service must not take
        # nutrition retrieval offline or expose request text in logs.
        logger.warning("Nutrition reranker unavailable; using RRF ranking (%s)", type(exc).__name__)
        return None


def _local_rerank_scores(query: str, documents: list[str]) -> list[float] | None:
    """Score with the fitness cross-encoder already loaded in this process (shared, lazy)."""
    try:
        from rag.hybrid import _get_reranker_model
        scores = [float(score) for score in _get_reranker_model().rerank(query, documents)]
        if len(scores) != len(documents) or any(not math.isfinite(s) for s in scores):
            raise ValueError("local reranker returned invalid scores")
        return scores
    except Exception as exc:
        logger.warning("Local nutrition reranker unavailable; using RRF ranking (%s)", type(exc).__name__)
        return None


def _rerank_scores(query: str, documents: list[str]) -> list[float] | None:
    """NUTRITION_RERANKER_MODE=local uses the in-process model; anything else keeps the remote service."""
    if os.getenv("NUTRITION_RERANKER_MODE", "remote").strip().lower() == "local":
        return _local_rerank_scores(query, documents)
    return _remote_rerank_scores(query, documents)


def search_nutrition(
    query: str,
    *,
    limit: int = 5,
    candidate_limit: int = 30,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """Search nutrition_knowledge and return source-attributed chunks."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or not limit <= candidate_limit <= MAX_CANDIDATES:
        raise ValueError(f"candidate_limit must be between limit and {MAX_CANDIDATES}")

    clean_query = query.strip()
    vector = _vector_literal(next(_get_embedding_model().embed([clean_query])))
    params = (
        vector, vector, candidate_limit,
        clean_query, clean_query, clean_query, candidate_limit,
        RRF_K, RRF_K, candidate_limit,
    )
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(NUTRITION_HYBRID_SQL, params)
            rows = cursor.fetchall()

    results = [
        {
            "chunk_id": row[0],
            "content": row[1],
            "source_id": row[2],
            "source": row[3],
            "title": row[4],
            "page": row[5],
            "chunk_number": row[6],
            "rrf_score": float(row[7]),
            "rerank_status": "disabled",
        }
        for row in rows
    ]

    if rerank and results:
        scores = _rerank_scores(clean_query, [item["content"] for item in results])
        if scores is None:
            for item in results:
                item["rerank_status"] = "fallback_rrf"
        else:
            for item, score in zip(results, scores):
                item["rerank_score"] = score
                item["rerank_status"] = "applied"
            results.sort(key=lambda item: item["rerank_score"], reverse=True)

    return results[:limit]
