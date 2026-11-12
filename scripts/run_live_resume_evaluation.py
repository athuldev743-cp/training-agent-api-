"""Run the complete resume-metric suite against the deployed API.

The client only reads committed evaluation datasets and writes the final report.
Retrieval and NutritionAgent work execute on the deployed server.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://training-agent-api.fastapicloud.dev"


def post(client: httpx.Client, url: str, *, retries: int = 4) -> dict:
    for attempt in range(retries):
        response = client.post(url)
        if response.status_code == 429 and attempt + 1 < retries:
            try:
                delay = float(response.headers.get("Retry-After", "2"))
            except ValueError:
                delay = 2.0
            time.sleep(min(max(delay, 1.0), 30.0))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"rate limited: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation/live-resume-metrics.json")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()

    key = os.getenv("EVALUATION_API_KEY", "").strip()
    if len(key) < 32:
        parser.error("set EVALUATION_API_KEY to the strong secret configured on the deployed API")

    args.output = args.output.expanduser().resolve()
    base = args.base_url.rstrip("/")
    headers = {"X-Evaluation-Key": key}
    timeout = httpx.Timeout(args.timeout, connect=20.0)
    report = {
        "evaluation": "live-resume-metrics",
        "base_url": base,
        "environment": "deployed API",
        "user_data_mutated": False,
    }

    with httpx.Client(headers=headers, timeout=timeout) as client:
        print("Running routing, workout, progression and safety evaluations...")
        report["static_evaluations"] = post(
            client, base + "/internal/evaluation/resume-metrics/static"
        )
        total_nutrition = int(
            report["static_evaluations"]["nutrition_grounding_dataset_cases"]
        )

        nutrition_rows = []
        print(f"Running {total_nutrition} live NutritionAgent evaluation cases...")
        for index in range(total_nutrition):
            try:
                row = post(
                    client,
                    base + f"/internal/evaluation/resume-metrics/nutrition/{index}",
                )
            except Exception as exc:
                row = {
                    "case_index": index,
                    "status": "evaluation_error",
                    "citation_contract_valid": False,
                    "grounded_answer_success": False,
                    "insufficient_evidence_refusal": False,
                    "error_type": type(exc).__name__,
                }
            nutrition_rows.append(row)
            print(f"  [{index + 1}/{total_nutrition}] {row.get('case_id', index)}: {row.get('status')}")

    valid = sum(bool(row.get("citation_contract_valid")) for row in nutrition_rows)
    refused = sum(bool(row.get("insufficient_evidence_refusal")) for row in nutrition_rows)
    report["nutrition_grounding"] = {
        "total_requests": len(nutrition_rows),
        "grounded_answer_successes": valid,
        "grounded_answer_success_rate": valid / len(nutrition_rows) if nutrition_rows else None,
        "citation_contract_validity_rate": valid / len(nutrition_rows) if nutrition_rows else None,
        "insufficient_evidence_refusals": refused,
        "unavailable_or_invalid": len(nutrition_rows) - valid - refused,
        "per_case": nutrition_rows,
        "manual_factual_support_review_required": True,
        "hand_checked_citation_validity": None,
        "note": (
            "Automated success means the NutritionAgent returned OK with valid in-scope "
            "citation labels. This does not prove each factual claim is supported. "
            "Hand-checked citation validity requires a human reviewer."
        ),
    }

    if not args.skip_retrieval:
        report["retrieval"] = {}
        for domain in ("fitness", "nutrition"):
            dataset = ROOT / "evaluation" / "datasets" / f"{domain}.jsonl"
            output = args.output.parent / f"live-{domain}-candidate-sweep.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_live_retrieval_sweep.py"),
                "--base-url", base,
                "--domain", domain,
                "--dataset", str(dataset),
                "--candidate-counts", "10", "20", "30", "50", "75",
                "--output", str(output),
                "--timeout", str(args.timeout),
            ]
            print(f"Running live {domain} retrieval candidate sweep...")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"{domain} retrieval sweep failed with exit code {completed.returncode}")
            report["retrieval"][domain] = json.loads(output.read_text(encoding="utf-8"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    static = report["static_evaluations"]
    print("\n========== LIVE RESUME METRICS ==========")
    print(f"Routing: {static['routing']['correct']}/{static['routing']['query_count']} ({static['routing']['accuracy']:.1%})")
    w = static["workout_constraints"]
    print(f"Workout constraints: {w['profiles_with_zero_hard_violations']}/{w['plans_generated']} zero-violation plans")
    p = static["progression"]
    print(f"Progression: {p['passed']}/{p['total']} ({p['pass_rate']:.1%})")
    s = static["safety"]
    print(f"Deterministic safety: {s['passed']}/{s['total']} ({s['pass_rate']:.1%})")
    n = report["nutrition_grounding"]
    print(f"Nutrition citation-contract success: {n['grounded_answer_successes']}/{n['total_requests']}")
    print("Nutrition factual support: HUMAN REVIEW REQUIRED")
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
