"""Run reviewed relevance datasets against live fitness and nutrition retrievers."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from evaluation.ranking import mean_metrics, ranking_metrics


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on dataset line {line_number}") from exc
        query = case.get("query")
        relevant = case.get("relevant_chunk_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"dataset line {line_number} needs a non-empty query")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"dataset line {line_number} needs reviewed relevant_chunk_ids")
        if any(value is None or not str(value).strip() for value in relevant):
            raise ValueError(f"dataset line {line_number} contains an invalid relevant ID")
        cases.append({
            "query": query.strip(),
            "relevant_chunk_ids": list(dict.fromkeys(str(value).strip() for value in relevant)),
            "case_id": str(case.get("case_id") or f"line-{line_number}"),
        })
    if not cases:
        raise ValueError("benchmark dataset is empty")
    return cases


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _search(domain: str, query: str, *, limit: int, candidate_limit: int, rerank: bool):
    if domain == "nutrition":
        from rag.nutrition_retrieval import search_nutrition
        return search_nutrition(
            query, limit=limit, candidate_limit=candidate_limit, rerank=rerank
        )
    if domain == "fitness":
        from rag.hybrid import search_knowledge
        return search_knowledge(
            query, limit=limit, candidate_limit=candidate_limit, rerank=rerank
        )
    raise ValueError("domain must be fitness or nutrition")


def _search_vector_only(domain: str, query: str, *, limit: int) -> list[dict[str, Any]]:
    """Semantic-only baseline using the same embedding model and database corpus."""
    from rag.neon import get_connection

    if domain == "fitness":
        from rag.hybrid import _get_embedding_model
        vector = next(_get_embedding_model().embed([query]))
        vector_text = "[" + ",".join(format(float(value), ".8g") for value in vector) + "]"
        sql = """
            SELECT id, content, source, page, chunk_number
            FROM fitness_knowledge
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (vector_text, limit))
                rows = cursor.fetchall()
        return [
            {"id": row[0], "content": row[1], "source": row[2],
             "page": row[3], "chunk_number": row[4], "rerank_status": "not_applicable"}
            for row in rows
        ]

    if domain == "nutrition":
        from rag.nutrition_retrieval import _get_embedding_model, _vector_literal
        vector = _vector_literal(next(_get_embedding_model().embed([query])))
        sql = """
            SELECT chunk_id, content, source_id, source, title, page, chunk_number
            FROM nutrition_knowledge
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (vector, limit))
                rows = cursor.fetchall()
        return [
            {"chunk_id": row[0], "content": row[1], "source_id": row[2],
             "source": row[3], "title": row[4], "page": row[5],
             "chunk_number": row[6], "rerank_status": "not_applicable"}
            for row in rows
        ]

    raise ValueError("domain must be fitness or nutrition")


def _run_one(
    domain: str,
    query: str,
    *,
    k: int,
    rerank: bool,
    strategy: str = "hybrid",
):
    started = time.perf_counter()
    status = "disabled" if not rerank else "unknown"
    if strategy == "vector_only":
        rows = _search_vector_only(domain, query, limit=k)
        status = "not_applicable"
    else:
        try:
            rows = _search(
                domain, query, limit=k, candidate_limit=max(k, 30), rerank=rerank
            )
        except Exception as exc:
            if domain != "fitness" or not rerank:
                raise
            from rag.hybrid import RerankerUnavailableError
            if not isinstance(exc, RerankerUnavailableError):
                raise
            rows = _search(
                domain, query, limit=k, candidate_limit=max(k, 30), rerank=False
            )
            status = "fallback_rrf"

    elapsed_ms = (time.perf_counter() - started) * 1000
    if not rows:
        status = "no_results"
    elif strategy != "vector_only" and rerank and status == "unknown":
        if domain == "nutrition":
            status = str(rows[0].get("rerank_status") or (
                "applied" if "rerank_score" in rows[0] else "fallback_rrf"
            ))
        else:
            status = "applied" if "rerank_score" in rows[0] else "fallback_rrf"

    id_key = "chunk_id" if domain == "nutrition" else "id"
    ranked_ids = [str(row[id_key]) for row in rows if row.get(id_key) is not None]
    return ranked_ids, elapsed_ms, status


def run_benchmark(
    domain: str,
    dataset_path: Path,
    output_path: Path,
    *,
    k: int = 5,
) -> dict[str, Any]:
    if domain not in {"fitness", "nutrition"}:
        raise ValueError("domain must be fitness or nutrition")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")

    cases = load_dataset(dataset_path)
    modes = {
        "vector_only": {"strategy": "vector_only", "rerank": False},
        "hybrid_rrf": {"strategy": "hybrid", "rerank": False},
        "hybrid_reranked": {"strategy": "hybrid", "rerank": True},
    }
    report_modes: dict[str, Any] = {}

    for mode, config in modes.items():
        metrics_rows = []
        latencies = []
        statuses: dict[str, int] = {}
        per_query = []
        for case in cases:
            ranked_ids, elapsed_ms, rerank_status = _run_one(
                domain, case["query"], k=k, **config
            )
            metrics = ranking_metrics(ranked_ids, case["relevant_chunk_ids"], k=k)
            metrics_rows.append(metrics)
            latencies.append(elapsed_ms)
            statuses[rerank_status] = statuses.get(rerank_status, 0) + 1
            per_query.append({
                "case_id": case["case_id"],
                "query": case["query"],
                "relevant_count": len(case["relevant_chunk_ids"]),
                "retrieved_ids": ranked_ids,
                "reranking_status": rerank_status,
                **metrics,
                "latency_ms": round(elapsed_ms, 3),
            })

        report_modes[mode] = {
            "query_count": len(per_query),
            **mean_metrics(metrics_rows),
            "latency_ms_p50": round(_percentile(latencies, 0.50), 3),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
            "reranking_status_counts": statuses,
            "per_query": per_query,
        }

    report = {
        "evaluation": "retrieval-ranking",
        "domain": domain,
        "k": k,
        "dataset": str(dataset_path),
        "query_count": len(cases),
        "relevance_labels": "human-reviewed chunk IDs supplied by the dataset",
        "comparison": ["vector_only", "hybrid_rrf", "hybrid_reranked"],
        "metrics": report_modes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
