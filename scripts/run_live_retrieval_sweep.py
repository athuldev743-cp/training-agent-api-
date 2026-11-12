"""Run a candidate-count sweep against the deployed retrieval evaluation endpoint.

Example:
  $env:EVALUATION_API_KEY = "<deployment evaluation secret>"
  python scripts/run_live_retrieval_sweep.py --domain fitness --dataset evaluation/datasets/fitness.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import load_dataset

DEFAULT_BASE_URL = "https://training-agent-api.fastapicloud.dev"
ENDPOINT_PATH = "/internal/evaluation/retrieval"
METRIC_NAMES = (
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
)


def percentile(values: list[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 3)


def average_metrics(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return {
        name: round(mean(float(row[name]) for row in rows), 6)
        for name in METRIC_NAMES
    }


def post_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict,
) -> dict:
    for attempt in range(4):
        response = client.post(url, json=payload)
        if response.status_code == 429 and attempt < 3:
            try:
                delay = float(response.headers.get("Retry-After", "2"))
            except ValueError:
                delay = 2.0
            time.sleep(min(max(delay, 1.0), 30.0))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("evaluation endpoint remained rate-limited after retries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("fitness", "nutrition"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--candidate-counts",
        nargs="+",
        type=int,
        default=[10],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-cases", type=int, default=5, help="maximum queries per run (default: 5)")
    parser.add_argument("--pause-seconds", type=float, default=2.0, help="pause between requests")
    args = parser.parse_args()

    api_key = os.getenv("EVALUATION_API_KEY", "").strip()
    if len(api_key) < 32:
        parser.error("set EVALUATION_API_KEY to the evaluation secret configured on the deployed API")
    if not 1 <= len(args.candidate_counts) <= 5:
        parser.error("provide between 1 and 5 candidate counts")
    if any(value < 5 or value > 100 for value in args.candidate_counts):
        parser.error("candidate counts must be between 5 and 100")
    if len(set(args.candidate_counts)) != len(args.candidate_counts):
        parser.error("candidate counts must be unique")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds cannot be negative")

    cases = load_dataset(args.dataset)
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be positive")
        cases = cases[: args.max_cases]
    if not cases:
        parser.error("dataset contains no cases")

    base_url = args.base_url.rstrip("/")
    url = base_url + ENDPOINT_PATH
    headers = {"X-Evaluation-Key": api_key}
    timeout = httpx.Timeout(args.timeout, connect=15.0)

    with httpx.Client(headers=headers, timeout=timeout) as client:
        warmup = post_with_retry(
            client,
            url,
            {
                "domain": args.domain,
                "case_id": cases[0]["case_id"],
                "query": cases[0]["query"],
                "relevant_chunk_ids": cases[0]["relevant_chunk_ids"],
                "candidate_counts": [30],
                "warmup_only": True,
            },
        )
        warmup_status = warmup.get("reranking_status")
        print(f"Warm-up: {warmup_status}")
        if warmup_status != "applied":
            print(
                f"Warning: reranker warm-up status is {warmup_status!r}; "
                "the sweep will continue and report reranking as unavailable/fallback."
            )
        if args.pause_seconds:
            time.sleep(args.pause_seconds)

        vector_rows = []
        vector_latencies = []
        by_candidate = {
            candidate: {
                "rrf_metrics": [],
                "rrf_latencies": [],
                "reranked_metrics": [],
                "reranked_latencies": [],
                "paired_overheads": [],
                "statuses": Counter(),
                "per_query": [],
            }
            for candidate in args.candidate_counts
        }

        for index, case in enumerate(cases, start=1):
            result = post_with_retry(
                client,
                url,
                {
                    "domain": args.domain,
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "relevant_chunk_ids": case["relevant_chunk_ids"],
                    "candidate_counts": args.candidate_counts,
                },
            )
            if result.get("evaluation") != "retrieval-candidate-sweep":
                raise RuntimeError(f"unexpected evaluation response for {case['case_id']}")

            vector_rows.append(result["vector_only"]["metrics"])
            vector_latencies.append(float(result["vector_only"]["latency_ms"]))

            for item in result["candidate_results"]:
                candidate = int(item["candidate_limit"])
                target = by_candidate[candidate]
                rrf = item["hybrid_rrf"]
                reranked = item["hybrid_reranked"]
                status = str(reranked["reranking_status"])

                target["rrf_metrics"].append(rrf["metrics"])
                target["rrf_latencies"].append(float(rrf["latency_ms"]))
                target["statuses"][status] += 1
                if reranked["metrics"] is not None and status == "applied":
                    target["reranked_metrics"].append(reranked["metrics"])
                    target["reranked_latencies"].append(float(reranked["latency_ms"]))
                    target["paired_overheads"].append(float(item["paired_overhead_ms"]))

                target["per_query"].append(
                    {
                        "case_id": case["case_id"],
                        "reranking_status": status,
                        "rrf_metrics": rrf["metrics"],
                        "rrf_latency_ms": rrf["latency_ms"],
                        "reranked_metrics": reranked["metrics"],
                        "reranked_latency_ms": reranked["latency_ms"],
                        "paired_overhead_ms": item["paired_overhead_ms"],
                    }
                )

            print(f"[{index}/{len(cases)}] {case['case_id']} complete")
            if args.pause_seconds:
                time.sleep(args.pause_seconds)

    candidate_report = {}
    for candidate, values in by_candidate.items():
        complete = values["statuses"].get("applied", 0) == len(cases)
        candidate_report[str(candidate)] = {
            "query_count": len(cases),
            "reranking_status_counts": dict(values["statuses"]),
            "hybrid_rrf": {
                "metrics": average_metrics(values["rrf_metrics"]),
                "latency_ms_p50": percentile(values["rrf_latencies"], 0.50),
                "latency_ms_p95": percentile(values["rrf_latencies"], 0.95),
            },
            "hybrid_reranked": {
                "metrics": average_metrics(values["reranked_metrics"]) if complete else None,
                "latency_ms_p50": percentile(values["reranked_latencies"], 0.50) if complete else None,
                "latency_ms_p95": percentile(values["reranked_latencies"], 0.95) if complete else None,
                "quality_and_latency_complete": complete,
            },
            "paired_reranking_overhead_ms": {
                "p50": percentile(values["paired_overheads"], 0.50) if complete else None,
                "p95": percentile(values["paired_overheads"], 0.95) if complete else None,
                "sample_count": len(values["paired_overheads"]),
            },
            "per_query": values["per_query"],
        }

    report = {
        "evaluation": "live-retrieval-candidate-sweep",
        "base_url": base_url,
        "domain": args.domain,
        "dataset": str(args.dataset.resolve()),
        "dataset_query_count": len(load_dataset(args.dataset)),
        "executed_query_count": len(cases),
        "candidate_counts": args.candidate_counts,
        "k": 5,
        "warmup_reranking_status": warmup_status,
        "vector_only": {
            "metrics": average_metrics(vector_rows),
            "latency_ms_p50": percentile(vector_latencies, 0.50),
            "latency_ms_p95": percentile(vector_latencies, 0.95),
        },
        "candidate_results": candidate_report,
        "measurement_scope": (
            "Server-side warmed retrieval-call timings returned by the deployed API; "
            "paired overhead is reranked-call latency minus RRF-call latency for each query. "
            "This is not end-to-end chat latency."
        ),
    }

    output = args.output or Path("artifacts") / "evaluation" / f"live-{args.domain}-candidate-sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nCandidate sweep summary")
    print("N | RRF MRR@5 | Reranked MRR@5 | Paired p95 overhead (ms) | Applied")
    for candidate, item in candidate_report.items():
        rrf_mrr = item["hybrid_rrf"]["metrics"]["mrr_at_k"]
        rerank_metrics = item["hybrid_reranked"]["metrics"]
        rerank_mrr = rerank_metrics["mrr_at_k"] if rerank_metrics else "INCOMPLETE"
        overhead = item["paired_reranking_overhead_ms"]["p95"]
        overhead_text = f"{overhead:.3f}" if overhead is not None else "INCOMPLETE"
        applied = item["reranking_status_counts"].get("applied", 0)
        print(f"{candidate} | {rrf_mrr} | {rerank_mrr} | {overhead_text} | {applied}/{len(cases)}")

    print(f"\nReport saved to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
