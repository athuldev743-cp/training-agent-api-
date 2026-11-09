from fastapi.testclient import TestClient

from services.nutrition_reranker import app as service


def test_health_does_not_load_model():
    client = TestClient(service.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert service._model is None


def test_rerank_requires_api_key(monkeypatch):
    monkeypatch.setenv("RERANKER_API_KEY", "expected-secret")
    client = TestClient(service.app)
    response = client.post(
        "/rerank",
        json={"query": "protein", "documents": ["source text"]},
        headers={"X-API-Key": "wrong"},
    )
    assert response.status_code == 401


def test_rerank_returns_scores_in_input_order(monkeypatch):
    monkeypatch.setenv("RERANKER_API_KEY", "expected-secret")

    class FakeModel:
        def rerank(self, query, documents):
            assert query == "protein"
            assert documents == ["first", "second"]
            return [0.2, 0.8]

    monkeypatch.setattr(service, "_get_model", lambda: FakeModel())
    client = TestClient(service.app)
    response = client.post(
        "/rerank",
        json={"query": "protein", "documents": ["first", "second"]},
        headers={"X-API-Key": "expected-secret"},
    )
    assert response.status_code == 200
    assert response.json() == {"scores": [0.2, 0.8]}
