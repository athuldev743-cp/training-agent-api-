# Nutrition Reranker Service

This service hosts the nutrition cross-encoder independently from the main Fitness API.
It does not connect to Neon or read/write either knowledge table.

## Deploy

Deploy this directory as a Docker service using the repository root as the build context
and `services/nutrition_reranker/Dockerfile` as the Dockerfile path.

Set one secret environment variable on the service:

- `RERANKER_API_KEY`: a long random secret.

The service exposes:

- `GET /health`: process health check; does not load the model.
- `POST /rerank`: authenticated cross-encoder scoring endpoint.

The model is loaded on the first authenticated rerank request. The service uses one
Uvicorn worker and limits ONNX CPU thread pools to reduce memory overhead.

## Connect the main API

Set these environment variables on the main Fitness API:

- `NUTRITION_RERANKER_URL`: the deployed service base URL, without a trailing slash.
- `NUTRITION_RERANKER_API_KEY`: the same secret as `RERANKER_API_KEY`.
- `NUTRITION_RERANKER_TIMEOUT`: optional HTTP timeout in seconds (default: 30).

If either URL or key is missing, or if the remote service fails, nutrition retrieval
continues with pgvector + PostgreSQL full-text search + Reciprocal Rank Fusion (RRF).
No local cross-encoder is loaded by the main API.

## Request contract

`POST /rerank` with header `X-API-Key` and JSON:

```json
{"query":"protein recovery","documents":["Document one","Document two"]}
```

Response scores preserve the input document order:

```json
{"scores":[0.42,0.81]}
```
