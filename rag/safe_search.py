"""Controlled-failure wrapper around the optional cloud hybrid retrieval (Step 7.5).
Never raises. Reports only exception TYPE names (messages can contain connection strings)."""
import concurrent.futures as cf
import os
from typing import Callable, Literal

from pydantic import BaseModel

Status = Literal["ok", "no_results", "unavailable", "invalid_query"]
DEFAULT_TIMEOUT_S = 8.0
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
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(search, query, limit=limit, rerank=rerank).result(timeout=timeout_s)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)  # a timed-out call may finish in the background


def retrieve_evidence(query: str, limit: int = 3, timeout_s: float = DEFAULT_TIMEOUT_S,
                      search: Callable | None = None) -> RetrievalOutcome:
    if not isinstance(query, str) or not query.strip():
        return RetrievalOutcome(status="invalid_query", reason="empty query")
    if search is None:
        search = _default_search()
        if search is None:
            return RetrievalOutcome(status="unavailable", reason="cloud knowledge store not available (offline or not configured)")
    degraded: list[str] = []
    try:
        rows = _call(search, query.strip(), limit, True, timeout_s)
    except cf.TimeoutError:
        return RetrievalOutcome(status="unavailable", reason="timeout")
    except ValueError:
        return RetrievalOutcome(status="invalid_query", reason="ValueError")
    except Exception as first:
        try:  # one bounded retry without the reranker
            rows = _call(search, query.strip(), limit, False, timeout_s)
            degraded.append("reranker_unavailable")
        except cf.TimeoutError:
            return RetrievalOutcome(status="unavailable", reason="timeout")
        except Exception:
            return RetrievalOutcome(status="unavailable", reason=type(first).__name__)
    seen_ids, seen_text, evidence = set(), set(), []
    for r in rows or []:
        text = str(r.get("content", "")).strip()
        key = " ".join(text.lower().split())
        if not text or r.get("id") in seen_ids or key in seen_text:
            continue
        seen_ids.add(r.get("id")); seen_text.add(key)
        evidence.append(Evidence(ref=f"S{len(evidence) + 1}", id=r.get("id"), source=r.get("source"),
                                 page=r.get("page"), chunk_number=r.get("chunk_number"), text=text[:MAX_CHARS]))
    if not evidence:
        return RetrievalOutcome(status="no_results", reason="nothing relevant found", degraded=degraded)
    return RetrievalOutcome(status="ok", evidence=evidence[:limit], degraded=degraded)


def grounded_context(outcome: RetrievalOutcome) -> str:
    if outcome.status == "ok":
        return "\n".join(f"[{e.ref}] {e.source or 'unknown source'}" + (f" p.{e.page}" if e.page else "") + f": {e.text}"
                         for e in outcome.evidence)
    return ("No retrieved evidence is available (" + (outcome.reason or outcome.status) + "). "
            "Say that no source was found. Do not cite sources or invent evidence.")
