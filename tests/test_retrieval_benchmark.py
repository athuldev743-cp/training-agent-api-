import json

import pytest

from evaluation import benchmark


def _fake_vector_only(domain, query, *, limit):
    key = "chunk_id" if domain == "nutrition" else "id"
    return [{key: "chunk-a" if domain == "nutrition" else 7}]


def test_dataset_requires_human_reviewed_relevant_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"query": "protein", "relevant_chunk_ids": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed relevant_chunk_ids"):
        benchmark.load_dataset(path)


def test_dataset_rejects_invalid_json_with_line_number(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        benchmark.load_dataset(path)


def test_nutrition_benchmark_reports_all_three_retrieval_modes(monkeypatch, tmp_path):
    dataset = tmp_path / "nutrition.jsonl"
    dataset.write_text(json.dumps({
        "case_id": "n1", "query": "protein recovery",
        "relevant_chunk_ids": ["chunk-a"],
    }) + "\n", encoding="utf-8")
    calls = []

    def fake_search(query, *, limit, candidate_limit, rerank):
        calls.append(rerank)
        status = "fallback_rrf" if rerank else "disabled"
        return [{"chunk_id": "chunk-a", "rerank_status": status}]

    monkeypatch.setattr(benchmark, "_search_vector_only", _fake_vector_only)
    monkeypatch.setattr("rag.nutrition_retrieval.search_nutrition", fake_search)
    report = benchmark.run_benchmark(
        "nutrition", dataset, tmp_path / "report.json", k=1
    )

    assert calls == [False, True]
    assert set(report["metrics"]) == {"vector_only", "hybrid_rrf", "hybrid_reranked"}
    assert report["metrics"]["hybrid_reranked"]["reranking_status_counts"] == {
        "fallback_rrf": 1
    }
    assert report["metrics"]["hybrid_reranked"]["precision_at_k"] == 1.0
    assert report["metrics"]["vector_only"]["precision_at_k"] == 1.0
    assert (tmp_path / "report.json").exists()


def test_fitness_benchmark_retries_only_cross_encoder_failure(monkeypatch, tmp_path):
    from rag.hybrid import RerankerUnavailableError

    dataset = tmp_path / "fitness.jsonl"
    dataset.write_text(json.dumps({
        "case_id": "f1", "query": "squat",
        "relevant_chunk_ids": ["7"],
    }) + "\n", encoding="utf-8")
    calls = []

    def fake_search(query, *, limit, candidate_limit, rerank):
        calls.append(rerank)
        if rerank:
            raise RerankerUnavailableError("model unavailable")
        return [{"id": 7}]

    monkeypatch.setattr(benchmark, "_search_vector_only", _fake_vector_only)
    monkeypatch.setattr("rag.hybrid.search_knowledge", fake_search)
    report = benchmark.run_benchmark(
        "fitness", dataset, tmp_path / "fitness-report.json", k=1
    )

    assert calls == [False, True, False]
    assert report["metrics"]["hybrid_reranked"]["reranking_status_counts"] == {
        "fallback_rrf": 1
    }


def test_fitness_database_error_is_not_silently_retried(monkeypatch, tmp_path):
    dataset = tmp_path / "fitness.jsonl"
    dataset.write_text(json.dumps({
        "query": "squat", "relevant_chunk_ids": ["7"],
    }) + "\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise ConnectionError("database unavailable")

    monkeypatch.setattr(benchmark, "_search_vector_only", _fake_vector_only)
    monkeypatch.setattr("rag.hybrid.search_knowledge", fail)
    with pytest.raises(ConnectionError):
        benchmark.run_benchmark("fitness", dataset, tmp_path / "out.json", k=5)
