"""Evaluate deterministic fitness/nutrition request classification on labelled JSONL."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.orchestrator import classify_request

ROUTES = ("fitness", "nutrition", "mixed", "clarify")


def _route_name(agents) -> str:
    values = {item.value for item in agents}
    if values == {"fitness"}:
        return "fitness"
    if values == {"nutrition"}:
        return "nutrition"
    if values == {"fitness", "nutrition"}:
        return "mixed"
    return "clarify"


def load_routing_dataset(path: Path) -> list[dict[str, str]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on dataset line {line_number}") from exc
        message = item.get("message")
        expected = item.get("expected_route")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"dataset line {line_number} needs a non-empty message")
        if expected not in ROUTES:
            raise ValueError(
                f"dataset line {line_number} expected_route must be one of: {', '.join(ROUTES)}"
            )
        cases.append({
            "case_id": str(item.get("case_id") or f"line-{line_number}"),
            "message": message.strip(),
            "expected_route": expected,
        })
    if not cases:
        raise ValueError("routing dataset is empty")
    return cases


def run_routing_benchmark(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    cases = load_routing_dataset(dataset_path)
    matrix = {actual: {expected: 0 for expected in ROUTES} for actual in ROUTES}
    per_route = {
        route: {"support": 0, "correct": 0, "predicted": 0}
        for route in ROUTES
    }
    predictions = []
    correct = 0

    for case in cases:
        predicted = _route_name(classify_request(case["message"]))
        expected = case["expected_route"]
        matrix[expected][predicted] += 1
        per_route[expected]["support"] += 1
        per_route[predicted]["predicted"] += 1
        if predicted == expected:
            correct += 1
            per_route[expected]["correct"] += 1
        predictions.append({
            **case,
            "predicted_route": predicted,
            "correct": predicted == expected,
        })

    route_metrics = {}
    for route, values in per_route.items():
        tp = values["correct"]
        precision = tp / values["predicted"] if values["predicted"] else 0.0
        recall = tp / values["support"] if values["support"] else 0.0
        route_metrics[route] = {
            "support": values["support"],
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall))
            if precision + recall else 0.0,
        }

    report = {
        "evaluation": "request-routing",
        "dataset": str(dataset_path),
        "query_count": len(cases),
        "accuracy": correct / len(cases),
        "confusion_matrix": matrix,
        "per_route": route_metrics,
        "per_query": predictions,
        "label_note": "Results are only as reliable as the human-reviewed expected_route labels.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
