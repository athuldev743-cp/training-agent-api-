"""Nutrition-only hybrid retrieval: pgvector + PostgreSQL FTS + RRF + reranking.

The nutrition corpus is physically separated from fitness_knowledge.
"""
from __future__ import annotations

from typing import Any

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.neon import get_connection

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIMENSIONS = 384
RRF_K = 60
MAX_LIMIT = 20
MAX_CANDIDATES = 100

_embedding_model: TextEmbedding | None = None
_reranker_model: TextCrossEncoder | None = None

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


def _get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedding_model


def _get_reranker_model() -> TextCrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = TextCrossEncoder(model_name=RERANKER_MODEL)
    return _reranker_model


def _vector_literal(values: Any) -> str:
    vector = [float(value) for value in values]
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"embedding must contain {EMBEDDING_DIMENSIONS} dimensions")
    return "[" + ",".join(format(value, ".8g") for value in vector) + "]"


def search_nutrition(
    query: str,
    *,
    limit: int = 5,
    candidate_limit: int = 30,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """Search only nutrition_knowledge and return ranked source-attributed chunks."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    if not limit <= candidate_limit <= MAX_CANDIDATES:
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
        }
        for row in rows
    ]

    if rerank and results:
        scores = list(_get_reranker_model().rerank(
            clean_query, [item["content"] for item in results]
        ))
        if len(scores) != len(results):
            raise RuntimeError("Reranker returned an unexpected number of scores")
        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)
        results.sort(key=lambda item: item["rerank_score"], reverse=True)

    return results[:limit]
