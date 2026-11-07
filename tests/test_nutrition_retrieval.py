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


def test_search_returns_source_attributed_results_without_reranking():
    embedding = MagicMock()
    embedding.embed.return_value = iter([[0.0] * 384])
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        ("a" * 64, "Protein supports tissue repair.", "src-1",
         "guide.pdf", "Guide", 2, 0, 0.03)
    ]
    with patch.object(retrieval, "_get_embedding_model", return_value=embedding), \
         patch.object(retrieval, "get_connection", return_value=connection):
        results = retrieval.search_nutrition("protein", rerank=False)

    assert len(results) == 1
    assert results[0]["source"] == "guide.pdf"
    assert results[0]["page"] == 2
    assert results[0]["rrf_score"] == pytest.approx(0.03)
    cursor.execute.assert_called_once()


def test_reranker_scores_and_orders_results():
    embedding = MagicMock()
    embedding.embed.return_value = iter([[0.0] * 384])
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        ("a" * 64, "Candidate A", "s1", "a.pdf", "A", 1, 0, 0.02),
        ("b" * 64, "Candidate B", "s2", "b.pdf", "B", 1, 0, 0.01),
    ]
    reranker = MagicMock()
    reranker.rerank.return_value = [0.1, 0.9]
    with patch.object(retrieval, "_get_embedding_model", return_value=embedding), \
         patch.object(retrieval, "_get_reranker_model", return_value=reranker), \
         patch.object(retrieval, "get_connection", return_value=connection):
        results = retrieval.search_nutrition("query", rerank=True)

    assert [item["chunk_id"] for item in results] == ["b" * 64, "a" * 64]
    assert results[0]["rerank_score"] == pytest.approx(0.9)


def test_reranker_score_count_mismatch_fails_closed():
    embedding = MagicMock()
    embedding.embed.return_value = iter([[0.0] * 384])
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        ("a" * 64, "Candidate A", "s1", "a.pdf", "A", 1, 0, 0.02)
    ]
    reranker = MagicMock()
    reranker.rerank.return_value = []
    with patch.object(retrieval, "_get_embedding_model", return_value=embedding), \
         patch.object(retrieval, "_get_reranker_model", return_value=reranker), \
         patch.object(retrieval, "get_connection", return_value=connection):
        with pytest.raises(RuntimeError, match="unexpected number"):
            retrieval.search_nutrition("query", rerank=True)
