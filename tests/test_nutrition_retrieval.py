from unittest.mock import MagicMock, patch

import pytest

from rag import nutrition_retrieval as retrieval


def test_query_validation_rejects_blank():
    with pytest.raises(ValueError, match="non-empty"):
        retrieval.search_nutrition("   ")


@pytest.mark.parametrize("limit,candidates", [(0, 5), (21, 30), (5, 4), (5, 101)])
def test_limits_are_bounded(limit, candidates):
    with pytest.raises(ValueError):
        retrieval.search_nutrition("protein", limit=limit, candidate_limit=candidates)


def test_vector_dimension_is_enforced():
    with pytest.raises(ValueError, match="384 dimensions"):
        retrieval._vector_literal([0.1, 0.2])


def test_vector_literal_is_serialized():
    assert retrieval._vector_literal([0.0] * 384).startswith("[0,0,0")
    assert retrieval._vector_literal([0.0] * 384).endswith("]")


def test_hybrid_sql_is_scoped_to_nutrition_table():
    assert "FROM nutrition_knowledge" in retrieval.NUTRITION_HYBRID_SQL
    assert "JOIN nutrition_knowledge" in retrieval.NUTRITION_HYBRID_SQL
    assert "fitness_knowledge" not in retrieval.NUTRITION_HYBRID_SQL
    assert "UNION ALL" in retrieval.NUTRITION_HYBRID_SQL


def test_rrf_sql_combines_semantic_and_keyword_rankings():
    sql = retrieval.NUTRITION_HYBRID_SQL
    assert "embedding <=> %s::vector" in sql
    assert "content_tsv @@ plainto_tsquery" in sql
    assert "1.0 / (%s + rank)" in sql
    assert "SUM(score)" in sql


def _retrieval_fixtures():
    embedding = MagicMock()
    embedding.embed.return_value = iter([[0.0] * 384])
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        ("a" * 64, "Candidate A", "s1", "a.pdf", "A", 1, 0, 0.02),
        ("b" * 64, "Candidate B", "s2", "b.pdf", "B", 1, 0, 0.01),
    ]
    return embedding, connection, cursor


def test_search_returns_source_attributed_results_without_reranking():
    embedding, connection, cursor = _retrieval_fixtures()
    with patch.object(retrieval, "_get_embedding_model", return_value=embedding), \
         patch.object(retrieval, "get_connection", return_value=connection):
        results = retrieval.search_nutrition("protein", rerank=False)

    assert len(results) == 2
    assert results[0]["source"] == "a.pdf"
    assert results[0]["page"] == 1
    assert results[0]["rrf_score"] == pytest.approx(0.02)
    cursor.execute.assert_called_once()


def test_remote_reranker_scores_and_orders_results():
    embedding, connection, _ = _retrieval_fixtures()
    with patch.object(retrieval, "_get_embedding_model", return_value=embedding), \
         patch.object(retrieval, "get_connection", return_value=connection), \
         patch.object(retrieval, "_remote_rerank_scores", return_value=[0.1, 0.9]) as rerank:
        results = retrieval.search_nutrition("query", rerank=True)

    assert [item["chunk_id"] for item in results] == ["b" * 64, "a" * 64]
    assert results[0]["rerank_score"] == pytest.approx(0.9)
    rerank.assert_called_once_with("query", ["Candidate A", "Candidate B"])


def test_unavailable_remote_reranker_falls_back_to_rrf_order():
    embedding, connection, _ = _retrieval_fixtures()
    with patch.object(retrieval, "_get_embedding_model", return_value=embedding), \
         patch.object(retrieval, "get_connection", return_value=connection), \
         patch.object(retrieval, "_remote_rerank_scores", return_value=None):
        results = retrieval.search_nutrition("query", rerank=True)

    assert [item["chunk_id"] for item in results] == ["a" * 64, "b" * 64]
    assert all("rerank_score" not in item for item in results)


def test_remote_reranker_is_not_called_when_disabled():
    embedding, connection, _ = _retrieval_fixtures()
    with patch.object(retrieval, "_get_embedding_model", return_value=embedding), \
         patch.object(retrieval, "get_connection", return_value=connection), \
         patch.object(retrieval, "_remote_rerank_scores") as rerank:
        retrieval.search_nutrition("query", rerank=False)
    rerank.assert_not_called()


def test_remote_reranker_requires_url_and_key(monkeypatch):
    monkeypatch.delenv("NUTRITION_RERANKER_URL", raising=False)
    monkeypatch.delenv("NUTRITION_RERANKER_API_KEY", raising=False)
    with patch.object(retrieval.httpx, "post") as post:
        assert retrieval._remote_rerank_scores("query", ["text"]) is None
    post.assert_not_called()
