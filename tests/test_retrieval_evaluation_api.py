import pytest
from fastapi.testclient import TestClient

from api.app import app
import api.retrieval_evaluation as retrieval_evaluation

PATH = "/internal/evaluation/retrieval"
SECRET = "test-evaluation-secret-that-is-at-least-32-chars"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _payload(**overrides):
    value = {
        "domain": "fitness",
        "case_id": "fitness-001",
        "query": "How do I improve my squat?",
        "relevant_chunk_ids": ["7"],
        "candidate_counts": [10, 20],
    }
    value.update(overrides)
    return value


def test_evaluation_route_is_hidden_from_openapi():
    assert PATH not in app.openapi()["paths"]


def test_evaluation_route_is_disabled_without_secret(client, monkeypatch):
    monkeypatch.delenv("EVALUATION_API_KEY", raising=False)

    response = client.post(
        PATH,
        json=_payload(),
        headers={"X-Evaluation-Key": SECRET},
    )

    assert response.status_code == 404


def test_evaluation_route_rejects_invalid_secret(client, monkeypatch):
    monkeypatch.setenv("EVALUATION_API_KEY", SECRET)

    response = client.post(
        PATH,
        json=_payload(),
        headers={"X-Evaluation-Key": "wrong-secret"},
    )

    assert response.status_code == 401


def test_evaluation_route_rejects_weak_configured_secret(client, monkeypatch):
    monkeypatch.setenv("EVALUATION_API_KEY", "too-short")

    response = client.post(
        PATH,
        json=_payload(),
        headers={"X-Evaluation-Key": "too-short"},
    )

    assert response.status_code == 503


def test_evaluation_route_returns_metrics_without_exposing_chunk_content(
    client, monkeypatch
):
    monkeypatch.setenv("EVALUATION_API_KEY", SECRET)

    def fake_vector(domain, query, *, limit):
        return [{"id": 7, "content": "private chunk content"}]

    def fake_search(domain, query, *, limit, candidate_limit, rerank):
        if rerank:
            return [{
                "id": 7,
                "content": "private chunk content",
                "rerank_score": 0.9,
                "rerank_status": "applied",
            }]
        return [{"id": 7, "content": "private chunk content"}]

    monkeypatch.setattr(retrieval_evaluation, "_search_vector_only", fake_vector)
    monkeypatch.setattr(retrieval_evaluation, "_search", fake_search)

    response = client.post(
        PATH,
        json=_payload(),
        headers={"X-Evaluation-Key": SECRET},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["evaluation"] == "retrieval-candidate-sweep"
    assert result["vector_only"]["metrics"]["mrr_at_k"] == 1.0
    assert [item["candidate_limit"] for item in result["candidate_results"]] == [10, 20]
    for item in result["candidate_results"]:
        assert item["hybrid_rrf"]["metrics"]["mrr_at_k"] == 1.0
        assert item["hybrid_reranked"]["metrics"]["mrr_at_k"] == 1.0
        assert item["hybrid_reranked"]["reranking_status"] == "applied"
        assert isinstance(item["paired_overhead_ms"], float)
    assert "private chunk content" not in response.text


def test_fallback_does_not_report_reranked_quality_or_overhead(client, monkeypatch):
    monkeypatch.setenv("EVALUATION_API_KEY", SECRET)

    monkeypatch.setattr(
        retrieval_evaluation,
        "_search_vector_only",
        lambda domain, query, *, limit: [{"id": 7}],
    )

    def fake_search(domain, query, *, limit, candidate_limit, rerank):
        if rerank:
            return [{"id": 7, "rerank_status": "fallback_rrf"}]
        return [{"id": 7}]

    monkeypatch.setattr(retrieval_evaluation, "_search", fake_search)

    response = client.post(
        PATH,
        json=_payload(candidate_counts=[10]),
        headers={"X-Evaluation-Key": SECRET},
    )

    assert response.status_code == 200
    item = response.json()["candidate_results"][0]
    assert item["hybrid_reranked"]["reranking_status"] == "fallback_rrf"
    assert item["hybrid_reranked"]["metrics"] is None
    assert item["paired_overhead_ms"] is None


def test_evaluation_route_rejects_invalid_candidate_limits(client, monkeypatch):
    monkeypatch.setenv("EVALUATION_API_KEY", SECRET)

    response = client.post(
        PATH,
        json=_payload(candidate_counts=[4]),
        headers={"X-Evaluation-Key": SECRET},
    )

    assert response.status_code == 422
