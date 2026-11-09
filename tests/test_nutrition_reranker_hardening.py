import math

import pytest

from rag import nutrition_retrieval as retrieval


@pytest.mark.parametrize("raw", ["not-a-number", "nan", "inf", "0", "-1", "9999"])
def test_invalid_reranker_timeout_uses_safe_default(monkeypatch, raw):
    monkeypatch.setenv("NUTRITION_RERANKER_TIMEOUT", raw)
    assert retrieval._reranker_timeout_from_env() == 30.0


def test_valid_reranker_timeout_is_accepted(monkeypatch):
    monkeypatch.setenv("NUTRITION_RERANKER_TIMEOUT", "12.5")
    assert retrieval._reranker_timeout_from_env() == 12.5


def test_non_finite_reranker_score_is_rejected_and_falls_back(monkeypatch):
    monkeypatch.setenv("NUTRITION_RERANKER_URL", "https://reranker.example")
    monkeypatch.setenv("NUTRITION_RERANKER_API_KEY", "test-key")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"scores": [math.nan]}

    monkeypatch.setattr(retrieval.httpx, "post", lambda *a, **k: Response())
    assert retrieval._remote_rerank_scores("query", ["document"]) is None


def test_missing_reranker_configuration_is_reported_as_rrf_fallback(monkeypatch):
    monkeypatch.delenv("NUTRITION_RERANKER_URL", raising=False)
    monkeypatch.delenv("NUTRITION_RERANKER_API_KEY", raising=False)
    monkeypatch.setattr(retrieval, "_get_embedding_model", lambda: type(
        "Embedding", (), {"embed": lambda self, texts: iter([[0.0] * 384])}
    )())

    class Cursor:
        def execute(self, *args):
            pass

        def fetchall(self):
            return [("chunk-1", "content", "source-id", "guide.pdf", "Guide", 1, 1, 0.02)]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Connection:
        def cursor(self):
            return Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(retrieval, "get_connection", lambda: Connection())
    results = retrieval.search_nutrition("protein", rerank=True)
    assert results[0]["rerank_status"] == "fallback_rrf"
    assert "rerank_score" not in results[0]
