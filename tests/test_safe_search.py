import time

from rag.hybrid import RerankerUnavailableError
from rag.safe_search import retrieve_evidence


def test_safe_search_retries_without_reranker_only_for_reranker_failure():
    calls = []

    def search(query, *, limit, rerank):
        calls.append(rerank)
        if rerank:
            raise RerankerUnavailableError("private model path")
        return [{"id": "chunk-1", "content": "Evidence", "source": "guide.pdf", "page": 2}]

    result = retrieve_evidence("protein", search=search)

    assert result.status == "ok"
    assert result.degraded == ["reranker_unavailable"]
    assert calls == [True, False]
    assert result.evidence[0].source == "guide.pdf"


def test_database_failure_does_not_get_mislabeled_as_reranker_failure():
    calls = []

    def search(query, *, limit, rerank):
        calls.append(rerank)
        raise ConnectionError("private database URL")

    result = retrieve_evidence("protein", search=search)

    assert result.status == "unavailable"
    assert result.reason == "ConnectionError"
    assert result.degraded == []
    assert calls == [True]


def test_safe_search_timeout_returns_controlled_unavailable():
    def slow_search(query, *, limit, rerank):
        time.sleep(0.15)
        return [{"id": 1, "content": "late"}]

    result = retrieve_evidence("protein", timeout_s=0.01, search=slow_search)

    assert result.status == "unavailable"
    assert result.reason == "timeout"


def test_safe_search_rejects_invalid_limit_and_timeout():
    assert retrieve_evidence("protein", limit=0, search=lambda *a, **k: []).status == "invalid_query"
    assert retrieve_evidence("protein", timeout_s=0, search=lambda *a, **k: []).status == "invalid_query"


def test_safe_search_deduplicates_evidence_and_does_not_leak_exception_text():
    def search(query, *, limit, rerank):
        return [
            {"id": 1, "content": "Same evidence", "source": "one.pdf"},
            {"id": 2, "content": " same   evidence ", "source": "two.pdf"},
        ]

    result = retrieve_evidence("protein", search=search)

    assert result.status == "ok"
    assert len(result.evidence) == 1
