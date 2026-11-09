"""Standalone nutrition cross-encoder reranking service."""
from __future__ import annotations

import hmac
import os
from typing import Any

# Keep ONNX Runtime CPU thread pools small on low-memory service instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ORT_INTRA_OP_THREADS", "1")
os.environ.setdefault("ORT_INTER_OP_THREADS", "1")

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from fastembed.rerank.cross_encoder import TextCrossEncoder

RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_model: TextCrossEncoder | None = None

app = FastAPI(title="Nutrition Reranker", version="1.0.0")


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    documents: list[str] = Field(min_length=1, max_length=100)


class RerankResponse(BaseModel):
    scores: list[float]


def _get_model() -> TextCrossEncoder:
    global _model
    if _model is None:
        _model = TextCrossEncoder(model_name=RERANKER_MODEL)
    return _model


def _authorize(api_key: str | None) -> None:
    expected = os.getenv("RERANKER_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="reranker authentication is not configured")
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(
    body: RerankRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _authorize(x_api_key)
    if any(not item.strip() or len(item) > 12000 for item in body.documents):
        raise HTTPException(status_code=422, detail="each document must contain 1–12000 characters")
    try:
        scores = [float(score) for score in _get_model().rerank(body.query, body.documents)]
        if len(scores) != len(body.documents):
            raise RuntimeError("cross-encoder returned an unexpected number of scores")
        return {"scores": scores}
    except Exception:
        # Do not leak model paths, input text, or runtime details to callers.
        raise HTTPException(status_code=503, detail="reranking is temporarily unavailable")
