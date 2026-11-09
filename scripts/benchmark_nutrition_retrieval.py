"""Compatibility wrapper for the shared retrieval benchmark."""
import argparse
import json
from pathlib import Path

from evaluation.benchmark import run_benchmark as _run_benchmark


def run_benchmark(dataset_path: Path, output_path: Path, k: int = 5) -> dict:
    report = _run_benchmark("nutrition", dataset_path, output_path, k=k)
    results = {}
    for old_name, new_name in (
        ("vector_only", "vector_only"),
        ("hybrid", "hybrid_rrf"),
        ("hybrid_reranked", "hybrid_reranked"),
    ):
        mode = report["metrics"][new_name]
        results[old_name] = {
            "queries": mode["query_count"],
            "mean_precision_at_k": mode["precision_at_k"],
            "mean_recall_at_k": mode["recall_at_k"],
            "mean_mrr_at_k": mode["mrr_at_k"],
            "mean_ndcg_at_k": mode["ndcg_at_k"],
            "latency_ms_p50": mode["latency_ms_p50"],
            "latency_ms_p95": mode["latency_ms_p95"],
            "reranking_status_counts": mode["reranking_status_counts"],
            "per_query": mode["per_query"],
        }
    return {
        "module": "nutrition-retrieval-benchmark",
        "k": k,
        "dataset": str(dataset_path),
        "evaluation_scope": "live Neon retrieval; quality depends on reviewed relevant_chunk_ids",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark live nutrition retrieval strategies.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL with query and reviewed relevant_chunk_ids")
    parser.add_argument("--output", type=Path, default=Path("artifacts/module-7/nutrition.json"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.dataset, args.output, args.k), indent=2))


if __name__ == "__main__":
    main()
