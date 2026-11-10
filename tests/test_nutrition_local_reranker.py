import pytest

import rag.hybrid as hybrid
import rag.nutrition_retrieval as nr


class FakeModel:
    def rerank(self, query, documents):
        return [float(len(d)) for d in documents]


def test_default_mode_uses_remote(monkeypatch):
    monkeypatch.delenv("NUTRITION_RERANKER_MODE", raising=False)
    monkeypatch.setattr(nr, "_remote_rerank_scores", lambda q, d: [9.0])
    assert nr._rerank_scores("q", ["a"]) == [9.0]


def test_local_mode_uses_shared_fitness_model(monkeypatch):
    monkeypatch.setenv("NUTRITION_RERANKER_MODE", "local")
    monkeypatch.setattr(hybrid, "_get_reranker_model", lambda: FakeModel())
    monkeypatch.setattr(nr, "_remote_rerank_scores", lambda q, d: pytest.fail("remote must not be called"))
    assert nr._rerank_scores("q", ["a", "abc"]) == [1.0, 3.0]


def test_local_mode_failure_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("NUTRITION_RERANKER_MODE", "local")
    def boom():
        raise RuntimeError("model missing")
    monkeypatch.setattr(hybrid, "_get_reranker_model", boom)
    assert nr._rerank_scores("q", ["a"]) is None
