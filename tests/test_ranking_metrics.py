import pytest

from evaluation.ranking import mean_metrics, ranking_metrics


def test_perfect_ranking_has_all_metrics_at_one():
    metrics = ranking_metrics(["a", "b", "c"], ["a", "b"], k=3)
    assert metrics == {
        "hit_at_k": pytest.approx(1.0),
            "precision_at_k": pytest.approx(2 / 3),
        "recall_at_k": pytest.approx(1.0),
        "mrr_at_k": pytest.approx(1.0),
        "ndcg_at_k": pytest.approx(1.0),
    }


def test_metrics_penalize_relevant_items_below_irrelevant_items():
    metrics = ranking_metrics(["x", "b", "a"], ["a", "b"], k=3)
    assert metrics["precision_at_k"] == pytest.approx(2 / 3)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["mrr_at_k"] == pytest.approx(0.5)
    assert 0.0 < metrics["ndcg_at_k"] < 1.0


def test_precision_uses_k_denominator_when_fewer_results_returned():
    metrics = ranking_metrics(["a"], ["a", "b"], k=5)
    assert metrics["precision_at_k"] == pytest.approx(0.2)
    assert metrics["recall_at_k"] == pytest.approx(0.5)


def test_duplicate_retrieved_ids_count_once_at_first_rank():
    metrics = ranking_metrics(["x", "a", "a", "b"], ["a", "b"], k=3)
    assert metrics["precision_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_k"] == pytest.approx(0.5)


@pytest.mark.parametrize("k", [0, -1, 101, True, 1.5])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="k must"):
        ranking_metrics(["a"], ["a"], k=k)


def test_empty_relevance_set_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ranking_metrics(["a"], [], k=5)


def test_mean_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        mean_metrics([])

def test_hit_at_k_is_binary_per_query():
    from evaluation.ranking import ranking_metrics
    assert ranking_metrics(["a", "b", "c"], ["c"], k=3)["hit_at_k"] == 1.0
    assert ranking_metrics(["a", "b", "c"], ["c"], k=2)["hit_at_k"] == 0.0
    assert ranking_metrics([], ["x"], k=5)["hit_at_k"] == 0.0
