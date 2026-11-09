"""Controlled-failure wrapper around optional cloud hybrid retrieval.

A retry without reranking is permitted only when the fitness cross-encoder itself
fails. Database/network failures are not mislabeled as reranker failures.
"""
import concurrent.futures as cf
import math
import os
from typing import Callable, Literal

from pydantic import BaseModel

Status = Literal["ok", "no_results", "unavailable", "invalid_query"]
DEFAULT_TIMEOUT_S = 8.0
MAX_TIMEOUT_S = 60.0
MAX_LIMIT = 10
MAX_CHARS = 600


class Evidence(BaseModel):
    ref: str
    id: int | str | None = None
    source: str | None = None
    page: int | None = None
    chunk_number: int | None = None
    text: str


class RetrievalOutcome(BaseModel):
    status: Status
    evidence: list[Evidence] = []
    reason: str | None = None
    degraded: list[str] = []


def _default_search() -> Callable | None:
    try:
        from rag.local_search import DB, local_search
        if DB.exists():
            return local_search
    except Exception:
        pass
    return _neon_search()


def _neon_search() -> Callable | None:
    try:
        from rag import neon
        if not os.getenv("NEON_DATABASE_URL"):
            return None
        from rag.hybrid import search_knowledge
        return search_knowledge
    except Exception:
        return None


def _call(search: Callable, query: str, limit: int, rerank: bool, timeout_s: float):
    executor = cf.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(search, query, limit=limit, rerank=rerank)
        return future.result(timeout=timeout_s)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def retrieve_evidence(
    query: str,
    limit: int = 3,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    search: Callable | None = None,
) -> RetrievalOutcome:
    if not isinstance(query, str) or not query.strip():
        return RetrievalOutcome(status="invalid_query", reason="empty query")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        return RetrievalOutcome(status="invalid_query", reason="invalid limit")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or not 0 < float(timeout_s) <= MAX_TIMEOUT_S
    ):
        return RetrievalOutcome(status="invalid_query", reason="invalid timeout")

    if search is None:
        search = _default_search()
        if search is None:
            return RetrievalOutcome(
                status="unavailable",
                reason="cloud knowledge store not available (offline or not configured)",
            )

    degraded: list[str] = []
    try:
        rows = _call(search, query.strip(), limit, True, float(timeout_s))
    except cf.TimeoutError:
        return RetrievalOutcome(status="unavailable", reason="timeout")
    except ValueError:
        return RetrievalOutcome(status="invalid_query", reason="ValueError")
    except Exception as first:
        try:
            from rag.hybrid import RerankerUnavailableError
        except Exception:
            RerankerUnavailableError = ()  # type: ignore[assignment,misc]
        if not isinstance(first, RerankerUnavailableError):
            return RetrievalOutcome(status="unavailable", reason=type(first).__name__)
        try:
            rows = _call(search, query.strip(), limit, False, float(timeout_s))
            degraded.append("reranker_unavailable")
        except cf.TimeoutError:
            return RetrievalOutcome(status="unavailable", reason="timeout")
        except Exception as second:
            return RetrievalOutcome(status="unavailable", reason=type(second).__name__)

    seen_ids, seen_text, evidence = set(), set(), []
    for row in rows or []:
        text = str(row.get("content", "")).strip()
        key = " ".join(text.lower().split())
        if not text or row.get("id") in seen_ids or key in seen_text:
            continue
        seen_ids.add(row.get("id"))
        seen_text.add(key)
        evidence.append(Evidence(
            ref=f"S{len(evidence) + 1}",
            id=row.get("id"),
            source=row.get("source"),
            page=row.get("page"),
            chunk_number=row.get("chunk_number"),
            text=text[:MAX_CHARS],
        ))
    if not evidence:
        return RetrievalOutcome(status="no_results", reason="nothing relevant found", degraded=degraded)
    return RetrievalOutcome(status="ok", evidence=evidence[:limit], degraded=degraded)


def grounded_context(outcome: RetrievalOutcome) -> str:
    if outcome.status == "ok":
        return "\n".join(
            f"[{item.ref}] {item.source or 'unknown source'}"
            + (f" p.{item.page}" if item.page else "")
            + f": {item.text}"
            for item in outcome.evidence
        )
    return (
        "No retrieved evidence is available ("
        + (outcome.reason or outcome.status)
        + "). Say that no source was found. Do not cite sources or invent evidence."
    )
