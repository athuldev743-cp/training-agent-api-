import json

import pytest

from evaluation.routing import load_routing_dataset, run_routing_benchmark


def test_routing_benchmark_reports_accuracy_and_confusion_matrix(tmp_path):
    dataset = tmp_path / "routing.jsonl"
    dataset.write_text("\n".join([
        json.dumps({"case_id": "f1", "message": "Build a workout plan", "expected_route": "fitness"}),
        json.dumps({"case_id": "n1", "message": "Explain protein in meals", "expected_route": "nutrition"}),
        json.dumps({"case_id": "m1", "message": "Workout and meal plan", "expected_route": "mixed"}),
        json.dumps({"case_id": "c1", "message": "Hello there", "expected_route": "clarify"}),
    ]) + "\n", encoding="utf-8")

    report = run_routing_benchmark(dataset, tmp_path / "report.json")

    assert report["query_count"] == 4
    assert report["accuracy"] == 1.0
    assert report["confusion_matrix"]["fitness"]["fitness"] == 1
    assert report["per_route"]["mixed"]["recall"] == 1.0
    assert (tmp_path / "report.json").exists()


def test_routing_dataset_rejects_unknown_label(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(json.dumps({
        "message": "Build a workout", "expected_route": "unknown"
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected_route"):
        load_routing_dataset(dataset)


def test_routing_dataset_requires_non_empty_messages(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(json.dumps({
        "message": " ", "expected_route": "fitness"
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty message"):
        load_routing_dataset(dataset)
