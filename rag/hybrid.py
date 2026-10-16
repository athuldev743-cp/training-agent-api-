"""Hybrid fitness retrieval with RRF fusion and cross-encoder reranking."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.neon import get_connection

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

_embedding_model = None
_reranker_model = None

HYBRID_SQL = """
WITH semantic AS (
    SELECT id,
           ROW_NUMBER() OVER (
               ORDER BY embedding <=> %s::vector
           ) AS rank
    FROM fitness_knowledge
    ORDER BY embedding <=> %s::vector
    LIMIT %s
),
keyword AS (
    SELECT id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(
                   content_tsv, plainto_tsquery('english', %s)
               ) DESC
           ) AS rank
    FROM fitness_knowledge
    WHERE content_tsv @@ plainto_tsquery('english', %s)
    ORDER BY ts_rank_cd(
        content_tsv, plainto_tsquery('english', %s)
    ) DESC
    LIMIT %s
),
fused AS (
    SELECT id, 1.0 / (%s + rank) AS score FROM semantic
    UNION ALL
    SELECT id, 1.0 / (%s + rank) AS score FROM keyword
),
combined AS (
    SELECT id, SUM(score) AS rrf_score
    FROM fused
    GROUP BY id
)
SELECT k.id, k.content, k.source, k.page, k.chunk_number,
       c.rrf_score
FROM combined c
JOIN fitness_knowledge k ON k.id = c.id
ORDER BY c.rrf_score DESC
LIMIT %s
"""


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedding_model


def _get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = TextCrossEncoder(model_name=RERANKER_MODEL)
    return _reranker_model


def search_knowledge(
    query: str,
    limit: int = 5,
    candidate_limit: int = 20,
    rerank: bool = True,
):
    """Retrieve with vector search + FTS, fuse with RRF, optionally rerank."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if limit < 1 or candidate_limit < 1:
        raise ValueError("limit and candidate_limit must be positive")

    query = query.strip()
    vector = next(_get_embedding_model().embed([query]))
    vector_text = "[" + ",".join(
        format(float(value), ".8g") for value in vector
    ) + "]"

    params = (
        vector_text, vector_text, candidate_limit,
        query, query, query, candidate_limit,
        RRF_K, RRF_K, candidate_limit,
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(HYBRID_SQL, params)
            rows = cursor.fetchall()

    results = [
        {
            "id": row[0],
            "content": row[1],
            "source": row[2],
            "page": row[3],
            "chunk_number": row[4],
            "rrf_score": float(row[5]),
        }
        for row in rows
    ]

    if rerank and results:
        documents = [result["content"] for result in results]
        scores = list(
            _get_reranker_model().rerank(query, documents)
        )

        if len(scores) != len(results):
            raise RuntimeError("Reranker returned an unexpected number of scores")

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        results.sort(key=lambda item: item["rerank_score"], reverse=True)

    return results[:limit]

