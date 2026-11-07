"""Evaluate real nutrition retrieval rankings against reviewed relevant chunk IDs."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from rag.nutrition_retrieval import search_nutrition


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def ranking_metrics(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    if not relevant_ids:
        raise ValueError("each benchmark query needs at least one relevant chunk ID")
    top_k = ranked_ids[:k]
    hits = [1 if item in relevant_ids else 0 for item in top_k]
    recall = sum(hits) / len(relevant_ids)
    reciprocal_rank = next((1.0 / i for i, hit in enumerate(hits, start=1) if hit), 0.0)
    dcg = sum(hit / math.log2(i + 1) for i, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return {
        "recall_at_k": recall,
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def run_benchmark(dataset_path: Path, output_path: Path, k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")
    cases = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("benchmark dataset is empty")
    modes = {"hybrid": False, "hybrid_reranked": True}
    results = {}
    for mode, rerank in modes.items():
        per_query = []
        latencies = []
        for case in cases:
            query = case.get("query")
            relevant = set(case.get("relevant_chunk_ids", []))
            if not isinstance(query, str) or not query.strip() or not relevant:
                raise ValueError("each case requires a non-empty query and relevant_chunk_ids")
            started = time.perf_counter()
            hits = search_nutrition(query, limit=k, candidate_limit=max(k, 30), rerank=rerank)
            latencies.append((time.perf_counter() - started) * 1000)
            metrics = ranking_metrics([item["chunk_id"] for item in hits], relevant, k)
            per_query.append({"query": query, **metrics, "returned": len(hits)})
        results[mode] = {
            "queries": len(per_query),
            "mean_recall_at_k": statistics.mean(item["recall_at_k"] for item in per_query),
            "mean_mrr_at_k": statistics.mean(item["mrr_at_k"] for item in per_query),
            "mean_ndcg_at_k": statistics.mean(item["ndcg_at_k"] for item in per_query),
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "per_query": per_query,
        }
    report = {
        "module": "nutrition-retrieval-benchmark",
        "k": k,
        "dataset": str(dataset_path),
        "evaluation_scope": "live Neon retrieval; quality depends on reviewed relevant_chunk_ids",
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark live nutrition hybrid retrieval.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL with query and relevant_chunk_ids")
    parser.add_argument("--output", type=Path, default=Path("artifacts/module-3/retrieval_benchmark.json"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.dataset, args.output, args.k), indent=2))


if __name__ == "__main__":
    main()
