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
    parser.add_argument("--max-nutrition-cases", type=int, default=5, help="maximum new nutrition cases per run")
    parser.add_argument("--max-retrieval-cases", type=int, default=5, help="maximum retrieval queries per domain")
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=[10], help="reranker candidate limits; default: 10")
    parser.add_argument("--pause-seconds", type=float, default=3.0, help="pause between live requests")
    parser.add_argument("--resume", action="store_true", help="resume from the existing output checkpoint")
    parser.add_argument("--interactive-review", action="store_true", help="hand-review nutrition citations and factual support")
    args = parser.parse_args()
    if args.max_nutrition_cases < 1 or args.max_retrieval_cases < 1:
        parser.error("case limits must be positive")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds cannot be negative")

    key = os.getenv("EVALUATION_API_KEY", "").strip()
    if len(key) < 32:
        parser.error("set EVALUATION_API_KEY to the strong secret configured on the deployed API")

    args.output = args.output.expanduser().resolve()
    base = args.base_url.rstrip("/")
    headers = {"X-Evaluation-Key": key}
    timeout = httpx.Timeout(args.timeout, connect=20.0)
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report.get("base_url") != base:
            parser.error("checkpoint base_url differs; choose another --output")
        print(f"Resuming checkpoint: {args.output}")
    else:
        report = {
            "evaluation": "live-resume-metrics",
            "base_url": base,
            "environment": "deployed API",
            "user_data_mutated": False,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=headers, timeout=timeout) as client:
        if "static_evaluations" not in report:
            print("Running routing, workout, progression and safety evaluations...")
            report["static_evaluations"] = post(
                client, base + "/internal/evaluation/resume-metrics/static"
            )
            args.output.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
        total_nutrition = int(
            report["static_evaluations"]["nutrition_grounding_dataset_cases"]
        )

        nutrition_rows = report.get("nutrition_grounding", {}).get("per_case", [])
        completed_indices = {int(row["case_index"]) for row in nutrition_rows if "case_index" in row and row.get("status") != "evaluation_error"}
        pending_indices = [index for index in range(total_nutrition) if index not in completed_indices][:args.max_nutrition_cases]
        print(f"Nutrition cases: {len(completed_indices)}/{total_nutrition} complete; running up to {len(pending_indices)} this session...")
        stopped_on_error = False
        for index in pending_indices:
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
            nutrition_rows = [item for item in nutrition_rows if int(item.get("case_index", -1)) != index]
            nutrition_rows.append(row)
            report["nutrition_grounding"] = {"per_case": nutrition_rows}
            args.output.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
            print(f"  [{index + 1}/{total_nutrition}] {row.get('case_id', index)}: {row.get('status')}")
            if row.get("status") == "evaluation_error":
                stopped_on_error = True
                break
            if args.pause_seconds:
                time.sleep(args.pause_seconds)

    nutrition_rows.sort(key=lambda item: int(item.get("case_index", 0)))
    successful_indices = {int(row["case_index"]) for row in nutrition_rows if "case_index" in row and row.get("status") != "evaluation_error"}
    if stopped_on_error or len(successful_indices) < total_nutrition:
        report["nutrition_grounding"] = {"total_requests": len(nutrition_rows), "per_case": nutrition_rows}
        report["run_status"] = "partial_nutrition"
        args.output.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
        print(f"Checkpoint saved: {args.output}")
        print("Run again with --resume after the service is healthy to continue the remaining cases.")
        return 0

    if args.interactive_review:
        for row in nutrition_rows:
            if row.get("status") != "ok":
                continue
            print("\n" + "=" * 72)
            print(f"Human review: {row.get('case_id')}")
            print(row.get("answer", ""))
            for evidence in row.get("evidence_excerpts", []):
                print(
                    f"\n{evidence.get('source')} | page={evidence.get('page')} "
                    f"| chunk={evidence.get('chunk_number')}\n{evidence.get('quote', '')}"
                )

            def ask_review(label: str):
                while True:
                    answer = input(f"{label} [y/n/s=skip]: ").strip().lower()
                    if answer in ("y", "yes"):
                        return True
                    if answer in ("n", "no"):
                        return False
                    if answer in ("s", "skip", ""):
                        return None

            row["human_citation_valid"] = ask_review(
                "Does every factual claim have a valid citation to one of the displayed sources?"
            )
            row["human_factual_support"] = ask_review(
                "Do the cited excerpts actually support every material factual claim?"
            )

    valid = sum(bool(row.get("citation_contract_valid")) for row in nutrition_rows)
    refused = sum(bool(row.get("insufficient_evidence_refusal")) for row in nutrition_rows)
    reviewed_citations = [row["human_citation_valid"] for row in nutrition_rows if row.get("human_citation_valid") is not None]
    reviewed_support = [row["human_factual_support"] for row in nutrition_rows if row.get("human_factual_support") is not None]
    report["nutrition_grounding"] = {
        "total_requests": len(nutrition_rows),
        "grounded_answer_successes": valid,
        "grounded_answer_success_rate": valid / len(nutrition_rows) if nutrition_rows else None,
        "citation_contract_validity_rate": valid / len(nutrition_rows) if nutrition_rows else None,
        "insufficient_evidence_refusals": refused,
        "unavailable_or_invalid": len(nutrition_rows) - valid - refused,
        "per_case": nutrition_rows,
        "manual_factual_support_review_required": True,
        "human_review": {
            "citation_cases_reviewed": len(reviewed_citations),
            "citation_cases_passed": sum(reviewed_citations),
            "citation_validity_rate": (
                sum(reviewed_citations) / len(reviewed_citations) if reviewed_citations else None
            ),
            "factual_support_cases_reviewed": len(reviewed_support),
            "factual_support_cases_passed": sum(reviewed_support),
            "factual_support_rate": (
                sum(reviewed_support) / len(reviewed_support) if reviewed_support else None
            ),
        },
        "hand_checked_citation_validity": (
            sum(reviewed_citations) / len(reviewed_citations) if reviewed_citations else None
        ),
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
                "--candidate-counts", *[str(value) for value in args.candidate_counts],
                "--max-cases", str(args.max_retrieval_cases),
                "--pause-seconds", str(args.pause_seconds),
                "--output", str(output),
                "--timeout", str(args.timeout),
            ]
            print(f"Running live {domain} retrieval candidate sweep...")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"{domain} retrieval sweep failed with exit code {completed.returncode}")
            report["retrieval"][domain] = json.loads(output.read_text(encoding="utf-8"))

    report["run_status"] = "completed"
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
