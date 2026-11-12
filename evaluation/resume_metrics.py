"""Resume-metric evaluations composed from existing deterministic evaluators.

No user stores are opened or mutated. Nutrition generation uses synthetic evaluation
queries and the configured application LLM/retriever; it never loads a real profile.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from engine.models import SetResult, Target
from engine.progression import next_target

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evaluation" / "datasets"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"evaluation dataset is empty: {path.name}")
    return rows


def evaluate_progression(dataset_path: Path | None = None) -> dict[str, Any]:
    """Evaluate deterministic progression decisions against labelled cases."""
    path = dataset_path or DATASETS / "progression_cases.jsonl"
    cases = _load_jsonl(path)
    results = []
    for case in cases:
        planned = Target(
            exercise=case["case_id"],
            sets=case["planned_sets"],
            reps=case["planned_reps"],
            weight_kg=case["planned_weight_kg"],
        )
        completed = [SetResult(**item) for item in case["results"]]
        decision = next_target(planned, completed)
        expected = (int(case["expected_reps"]), float(case["expected_weight_kg"]))
        actual = (decision.target.reps, float(decision.target.weight_kg))
        results.append({
            "case_id": case["case_id"],
            "passed": actual == expected,
            "expected": {"reps": expected[0], "weight_kg": expected[1]},
            "actual": {"reps": actual[0], "weight_kg": actual[1]},
        })
    passed = sum(item["passed"] for item in results)
    return {
        "evaluation": "progression-rules",
        "dataset": path.name,
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / len(results),
        "failures": [item for item in results if not item["passed"]],
        "scope": "Deterministic progression engine decisions against explicit expected outputs; not a clinical or coaching-outcome assessment.",
    }


def run_static_resume_evaluations() -> dict[str, Any]:
    """Run routing, workout-constraint, progression and deterministic safety suites."""
    from evaluation.routing import run_routing_benchmark
    from evaluation.safety import run_safety_benchmark
    from evaluation.workout_constraints import run_workout_benchmark

    with tempfile.TemporaryDirectory(prefix="resume-eval-") as temp:
        temp_path = Path(temp)
        routing = run_routing_benchmark(
            DATASETS / "routing_heldout.jsonl", temp_path / "routing.json"
        )
        workout = run_workout_benchmark(
            DATASETS / "workout_profiles.jsonl", temp_path / "workout.json"
        )
        safety = run_safety_benchmark(
            DATASETS / "safety_red_team.jsonl", temp_path / "safety.json"
        )
        progression = evaluate_progression()

    return {
        "evaluation": "live-resume-metrics-static",
        "routing": {
            "query_count": routing["query_count"],
            "correct": round(routing["accuracy"] * routing["query_count"]),
            "accuracy": routing["accuracy"],
            "confusion_matrix": routing["confusion_matrix"],
            "per_route": routing["per_route"],
            "dataset": "routing_heldout.jsonl",
        },
        "workout_constraints": {
            "profiles_total": workout["profiles_total"],
            "plans_generated": workout["plans_generated"],
            "profiles_with_zero_hard_violations": workout["profiles_with_zero_hard_violations"],
            "zero_violation_rate": workout["zero_violation_rate_of_generated"],
            "per_category_pass": workout["per_category_pass"],
            "guardrail_cases": workout["guardrail_cases"],
            "dataset": "workout_profiles.jsonl",
        },
        "progression": progression,
        "safety": {
            "total": safety["total"],
            "passed": safety["passed"],
            "pass_rate": safety["pass_rate"],
            "critical_failures": safety["critical_failures"],
            "per_category": safety["per_category"],
            "dataset": "safety_red_team.jsonl",
            "scope": safety["scope_note"],
        },
        "user_data_mutated": False,
        "notes": [
            "Routing labels and evaluation datasets must be human-reviewed before publishing scores.",
            "Workout and safety results evaluate deterministic backend rules on synthetic cases.",
            "Safety results do not measure LLM refusal quality.",
        ],
    }


def evaluate_nutrition_case(case_index: int) -> dict[str, Any]:
    """Run one synthetic nutrition query through the deployed NutritionAgent."""
    cases = _load_jsonl(DATASETS / "nutrition.jsonl")
    if isinstance(case_index, bool) or not isinstance(case_index, int) or not 0 <= case_index < len(cases):
        raise ValueError(f"case_index must be between 0 and {len(cases) - 1}")
    case = cases[case_index]

    # Import lazily to avoid coupling the deterministic evaluator to app startup.
    from agent.contracts import AgentRequest, AgentStatus
    from agent.nutrition import NutritionAgent
    from api.app import model

    result = NutritionAgent(model()).run(
        AgentRequest(user_id="evaluation-only", message=case["query"], profile=None)
    )
    validation = result.metadata.get("citation_validation")
    citation_contract_valid = validation == "passed" and result.status is AgentStatus.OK
    insufficient_evidence_refusal = (
        result.status is AgentStatus.REFUSED and validation == "insufficient_evidence"
    )
    return {
        "case_id": case["case_id"],
        "case_index": case_index,
        "status": result.status.value,
        "citation_validation": validation,
        "citation_attempts": result.metadata.get("citation_attempts"),
        "citation_contract_valid": citation_contract_valid,
        "grounded_answer_success": citation_contract_valid,
        "insufficient_evidence_refusal": insufficient_evidence_refusal,
        "evidence_count": len(result.evidence),
        "failure_type": result.metadata.get("failure_type"),
        "citation_sources": result.metadata.get("citation_sources", []),
        "answer": result.message,
        "manual_factual_support_review_required": True,
        "scope": "Automated citation-contract and fail-closed outcome check. It does not establish that cited sources semantically support every factual claim.",
    }


def aggregate_nutrition_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case NutritionAgent outputs without treating fallbacks as successes."""
    if not rows:
        raise ValueError("nutrition evaluation needs at least one result")
    valid = sum(bool(row["citation_contract_valid"]) for row in rows)
    refused = sum(bool(row["insufficient_evidence_refusal"]) for row in rows)
    return {
        "evaluation": "nutrition-grounding-live",
        "total_requests": len(rows),
        "grounded_answer_successes": valid,
        "grounded_answer_success_rate": valid / len(rows),
        "citation_contract_validity_rate": valid / len(rows),
        "insufficient_evidence_refusals": refused,
        "unavailable_or_invalid": len(rows) - valid - refused,
        "per_case": rows,
        "manual_factual_support_review_required": True,
        "hand_checked_citation_validity": None,
        "note": "The automated rate measures successful generation with valid in-scope citation labels. A human must inspect the cited evidence and each material claim before describing this as factual grounding or reporting a hand-checked citation-validity rate.",
    }
