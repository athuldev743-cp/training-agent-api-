import pytest
from scripts.benchmark_nutrition_retrieval import ranking_metrics

def test_ranking_metrics_perfect_order():
    metrics = ranking_metrics(["a", "b", "c"], {"a", "b"}, 3)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["mrr_at_k"] == pytest.approx(1.0)
    assert metrics["ndcg_at_k"] == pytest.approx(1.0)

def test_ranking_metrics_partial_recall_and_rank():
    metrics = ranking_metrics(["x", "b", "a"], {"a", "b"}, 3)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["mrr_at_k"] == pytest.approx(0.5)
    assert 0.0 < metrics["ndcg_at_k"] < 1.0

def test_ranking_metrics_rejects_empty_relevance_set():
    with pytest.raises(ValueError, match="at least one"):
        ranking_metrics(["a"], set(), 1)
