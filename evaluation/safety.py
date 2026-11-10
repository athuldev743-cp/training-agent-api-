"""Evaluate the deterministic safety layer (engine/safety.py) on reviewed adversarial cases."""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

from engine.safety import assess_policy, assess_symptoms

CRITICAL_CATEGORIES = {"urgent_symptom", "medical_condition", "safety_bypass", "prompt_injection", "diagnosis_request"}


def load_safety_dataset(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not cases:
        raise ValueError("safety dataset is empty")
    return cases


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    level = assess_symptoms(case["message"]).level
    kind, _ = assess_policy(case["message"])
    problems = []
    if "expect_symptom" in case and level != case["expect_symptom"]:
        problems.append(f"symptom level {level!r}, expected {case['expect_symptom']!r}")
    if "expect_policy" in case and kind != case["expect_policy"]:
        problems.append(f"policy {kind!r}, expected {case['expect_policy']!r}")
    if "expect_handled" in case and (level != "none" or kind is not None) != case["expect_handled"]:
        problems.append(f"handled={(level != 'none' or kind is not None)}, expected {case['expect_handled']}")
    return {"case_id": case["case_id"], "category": case["category"], "message": case["message"],
            "symptom_level": level, "policy_kind": kind, "passed": not problems, "problems": problems,
            "critical": case["category"] in CRITICAL_CATEGORIES}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_safety_benchmark(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    results = [evaluate_case(c) for c in load_safety_dataset(dataset_path)]
    per_cat: dict[str, dict[str, int]] = {}
    for r in results:
        c = per_cat.setdefault(r["category"], {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += r["passed"]
    passed = sum(r["passed"] for r in results)
    report = {
        "evaluation": "safety-deterministic-layer",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset": str(dataset_path),
        "total": len(results), "passed": passed, "pass_rate": passed / len(results),
        "critical_failures": [r for r in results if r["critical"] and not r["passed"]],
        "per_category": per_cat,
        "per_case": results,
        "scope_note": "Backend deterministic rules only. LLM refusals (nutrition agent) are evaluated separately on the live server.",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
