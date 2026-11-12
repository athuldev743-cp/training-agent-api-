"""Secret-gated, read-only retrieval evaluation endpoint.

The endpoint evaluates one labelled query per request so long-running sweeps can
be orchestrated by a client without holding an API request open for an entire
dataset. It never reads or writes user workout/profile/history state.
"""
from __future__ import annotations

import hmac
import os
import time
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from evaluation.benchmark import _search, _search_vector_only
from evaluation.ranking import ranking_metrics

router = APIRouter()
DEFAULT_CANDIDATE_COUNTS = [10, 20, 30, 50, 75]
EVALUATION_PATH = "/internal/evaluation/retrieval"


class RetrievalEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: Literal["fitness", "nutrition"]
    case_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=1000)
    relevant_chunk_ids: list[str] = Field(min_length=1, max_length=100)
    candidate_counts: list[int] = Field(
        default_factory=lambda: list(DEFAULT_CANDIDATE_COUNTS),
        min_length=1,
        max_length=5,
    )
    warmup_only: bool = False

    @field_validator("case_id", "query")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("relevant_chunk_ids")
    @classmethod
    def validate_relevant_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if isinstance(value, str)]
        if len(cleaned) != len(values) or any(not value for value in cleaned):
            raise ValueError("relevant_chunk_ids must contain non-empty strings")
        return list(dict.fromkeys(cleaned))

    @field_validator("candidate_counts")
    @classmethod
    def validate_candidate_counts(cls, values: list[int]) -> list[int]:
        if any(type(value) is not int or not 5 <= value <= 100 for value in values):
            raise ValueError("candidate_counts must contain integers between 5 and 100")
        if len(set(values)) != len(values):
            raise ValueError("candidate_counts must be unique")
        return values


def require_evaluation_key(
    evaluation_key: str | None = Header(default=None, alias="X-Evaluation-Key"),
) -> None:
    """Fail closed; the endpoint is unavailable unless a strong secret is configured."""
    expected = os.getenv("EVALUATION_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=404, detail="not found")
    if len(expected) < 32:
        raise HTTPException(status_code=503, detail="evaluation authentication is not configured securely")
    if not evaluation_key or not hmac.compare_digest(evaluation_key, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


def _reranking_status(domain: str, rows: list[dict]) -> str:
    if not rows:
        return "no_results"
    if domain == "nutrition":
        return str(rows[0].get("rerank_status") or "unknown")
    return "applied" if "rerank_score" in rows[0] else "fallback_rrf"


def _ranked_ids(domain: str, rows: list[dict]) -> list[str]:
    key = "chunk_id" if domain == "nutrition" else "id"
    return [str(row[key]) for row in rows if row.get(key) is not None]


def _timed_search(domain: str, query: str, *, candidate_limit: int, rerank: bool):
    started = time.perf_counter()
    rows = _search(
        domain,
        query,
        limit=5,
        candidate_limit=candidate_limit,
        rerank=rerank,
    )
    return rows, round((time.perf_counter() - started) * 1000, 3)


@router.post(EVALUATION_PATH, include_in_schema=False)
def evaluate_retrieval(
    body: RetrievalEvaluationRequest,
    _authorized: None = Depends(require_evaluation_key),
):
    """Evaluate one labelled query across candidate limits; all retrieval is read-only."""
    if body.warmup_only:
        _search_vector_only(body.domain, body.query, limit=5)
        candidate_limit = body.candidate_counts[0]
        _search(
            body.domain,
            body.query,
            limit=5,
            candidate_limit=candidate_limit,
            rerank=False,
        )
        try:
            reranked_rows = _search(
                body.domain,
                body.query,
                limit=5,
                candidate_limit=candidate_limit,
                rerank=True,
            )
            status = _reranking_status(body.domain, reranked_rows)
        except Exception as exc:
            from rag.hybrid import RerankerUnavailableError

            if body.domain != "fitness" or not isinstance(exc, RerankerUnavailableError):
                raise
            status = "unavailable"
        return {
            "evaluation": "retrieval-candidate-sweep",
            "domain": body.domain,
            "warmed": True,
            "reranking_status": status,
        }

    vector_started = time.perf_counter()
    vector_rows = _search_vector_only(body.domain, body.query, limit=5)
    vector_latency_ms = round((time.perf_counter() - vector_started) * 1000, 3)
    vector_metrics = ranking_metrics(
        _ranked_ids(body.domain, vector_rows),
        body.relevant_chunk_ids,
        k=5,
    )

    candidate_results = []
    for candidate_limit in body.candidate_counts:
        rrf_rows, rrf_latency_ms = _timed_search(
            body.domain,
            body.query,
            candidate_limit=candidate_limit,
            rerank=False,
        )
        rrf_metrics = ranking_metrics(
            _ranked_ids(body.domain, rrf_rows),
            body.relevant_chunk_ids,
            k=5,
        )

        rerank_started = time.perf_counter()
        reranking_status = "unknown"
        reranked_rows: list[dict] = []
        try:
            reranked_rows = _search(
                body.domain,
                body.query,
                limit=5,
                candidate_limit=candidate_limit,
                rerank=True,
            )
            reranking_status = _reranking_status(body.domain, reranked_rows)
        except Exception as exc:
            from rag.hybrid import RerankerUnavailableError

            if body.domain != "fitness" or not isinstance(exc, RerankerUnavailableError):
                raise
            reranking_status = "unavailable"
        rerank_latency_ms = round((time.perf_counter() - rerank_started) * 1000, 3)

        reranker_applied = reranking_status == "applied"
        reranked_metrics = (
            ranking_metrics(
                _ranked_ids(body.domain, reranked_rows),
                body.relevant_chunk_ids,
                k=5,
            )
            if reranker_applied
            else None
        )
        candidate_results.append(
            {
                "candidate_limit": candidate_limit,
                "hybrid_rrf": {
                    "metrics": rrf_metrics,
                    "latency_ms": rrf_latency_ms,
                },
                "hybrid_reranked": {
                    "metrics": reranked_metrics,
                    "latency_ms": rerank_latency_ms,
                    "reranking_status": reranking_status,
                },
                "paired_overhead_ms": (
                    round(rerank_latency_ms - rrf_latency_ms, 3)
                    if reranker_applied
                    else None
                ),
            }
        )

    return {
        "evaluation": "retrieval-candidate-sweep",
        "domain": body.domain,
        "case_id": body.case_id,
        "k": 5,
        "vector_only": {
            "metrics": vector_metrics,
            "latency_ms": vector_latency_ms,
        },
        "candidate_results": candidate_results,
    }



# Secret-gated comprehensive resume-metric evaluation. Kept on the existing
# internal router so api/app.py and normal application route registration stay untouched.
@router.post("/internal/evaluation/resume-metrics/static", include_in_schema=False)
def evaluate_static_resume_metrics(
    _authorized: None = Depends(require_evaluation_key),
):
    """Run deterministic routing, workout, progression and safety evaluations."""
    from evaluation.resume_metrics import run_static_resume_evaluations
    from pathlib import Path

    report = run_static_resume_evaluations()
    nutrition_dataset = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "nutrition.jsonl"
    report["nutrition_grounding_dataset_cases"] = sum(
        1 for line in nutrition_dataset.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    return report


@router.post("/internal/evaluation/resume-metrics/nutrition/{case_index}", include_in_schema=False)
def evaluate_live_nutrition_case(
    case_index: int,
    _authorized: None = Depends(require_evaluation_key),
):
    """Run one synthetic nutrition query through the live NutritionAgent."""
    from evaluation.resume_metrics import evaluate_nutrition_case

    try:
        return evaluate_nutrition_case(case_index)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Do not return provider errors, prompt contents, or deployment details.
        raise HTTPException(
            status_code=502,
            detail=f"nutrition evaluation failed ({type(exc).__name__})",
        ) from exc
