# Evaluation and release-readiness

This document records what can be measured in the repository and how to reproduce it. It deliberately separates implemented evaluation tooling from measured product performance.

## Current evaluation status

No reviewed routing dataset or reviewed retrieval relevance dataset is committed yet. Therefore, there are currently **no defensible accuracy, retrieval-quality, citation-validity, safety, or end-to-end latency scores to publish**. The benchmark commands below produce results only after reviewed datasets and a reachable test database are supplied.

| Area | Evaluation available | Current measured result |
|---|---|---|
| Agent routing | Labelled routing benchmark: accuracy, confusion matrix, per-route precision/recall/F1 | Not measured — labelled dataset required |
| Fitness retrieval | Vector-only vs hybrid RRF vs reranked retrieval; Precision@5, Recall@5, MRR@5, NDCG@5 | Not measured — reviewed chunk labels and Neon required |
| Nutrition retrieval | Same three-way comparison, with reranker fallback reporting | Not measured — reviewed chunk labels, Neon and reranker configuration required |
| Nutrition answer grounding | Existing citation validation and fail-closed agent behavior; manual audit protocol still required | Not measured as a dataset-level rate |
| Workout constraints | Existing planner applies equipment, experience and limitation filters | No aggregate constraint-satisfaction score |
| API performance | Retrieval p50/p95 only in the current retrieval benchmark | End-to-end API p50/p95 not measured |
| Cost | No request-level cost benchmark in this evaluation package | Not measured |

Do not convert test pass counts or individual successful examples into routing accuracy, retrieval quality, medical safety, or production reliability claims.

## 1. Routing benchmark

The orchestrator uses deterministic keyword-based classification to select the fitness agent, nutrition agent, both agents, or clarification. The benchmark evaluates that classifier against manually reviewed expected labels.

Prepare `evaluation/datasets/routing.jsonl` as described in [evaluation/datasets/README.md](evaluation/datasets/README.md), then run:

```powershell
python scripts/benchmark_routing.py --dataset evaluation/datasets/routing.jsonl --output artifacts/evaluation/routing.json
```

The output contains overall accuracy, a confusion matrix, per-route precision/recall/F1 and per-query predictions. Report the dataset size and class distribution with any score. Include edge cases and keep the labelled dataset under version control after review.

## 2. Retrieval benchmark

The benchmark uses the same embedding model and corpus for all three strategies:

1. `vector_only`: semantic pgvector nearest-neighbour search.
2. `hybrid_rrf`: semantic search + PostgreSQL full-text search, fused with Reciprocal Rank Fusion.
3. `hybrid_reranked`: hybrid RRF candidates reordered by the configured cross-encoder when available.

Prepare reviewed datasets with exact database chunk IDs, configure the usual Neon environment, and run:

```powershell
python scripts/benchmark_retrieval.py --domain fitness --dataset evaluation/datasets/fitness.jsonl --output artifacts/evaluation/fitness-retrieval.json --k 5
python scripts/benchmark_retrieval.py --domain nutrition --dataset evaluation/datasets/nutrition.jsonl --output artifacts/evaluation/nutrition-retrieval.json --k 5
```

The report contains macro-averaged Precision@5, Recall@5, MRR@5 and NDCG@5, p50/p95 retriever latency, reranker status counts and per-query results. Use the same reviewed queries, corpus snapshot and candidate limits across strategies. For a meaningful comparison, aim for at least 50 reviewed queries per domain and include a range of intents and difficulty levels. This is a recommended sample size, not a result already achieved.

Before timing each strategy, the benchmark runs one warm-up query to initialize model and database paths; that query is excluded from the reported quality metrics and latency percentiles. Reported p50/p95 values therefore represent warmed retriever-call latency in the benchmark process, not cold-start latency. They do not include authentication, API routing, LLM generation, network/client latency or the full user request.

## 3. Nutrition answer-grounding audit

The Nutrition Agent includes profile-aware retrieval, source attribution, citation validation and bounded retry/fail-closed behavior. A defensible citation-validity or unsupported-claim rate still requires a manually reviewed answer set.

For each reviewed case, record:
- query and non-sensitive profile conditions used;
- retrieved source IDs and the sources actually cited;
- whether every citation resolves to a retrieved source;
- whether each material factual claim is supported by the cited source;
- whether conflicting evidence or missing profile inputs were handled appropriately;
- whether the agent refused or asked for clarification when evidence was insufficient.

Calculate citation validity as valid citations divided by citations reviewed. Calculate unsupported-claim rate as unsupported material claims divided by material claims reviewed. Keep refusals/clarifications as a separate outcome category; do not count every refusal as a success. Do not publish these rates until the dataset and reviewer decisions are stored.

## 4. Workout constraint audit

The workout planner already filters exercise candidates using experience, available equipment and user limitations, and builds a four-week plan. An aggregate constraint-satisfaction rate is not currently produced by the evaluation package.

For a future reviewed test set, define expected constraints per profile before generating a plan. Check every prescribed exercise against equipment availability, difficulty/experience eligibility and limitation stress rules. Also verify that the generated plan has the requested training-day count where supported, valid exercise references, non-empty sessions and valid progression prescriptions. Report the number of profiles tested and each constraint separately; a single blended score can hide serious failures.

## 5. API performance and cost

Do not use retrieval p95 as an API p95 claim. End-to-end performance requires a separate load test against a controlled deployment and a defined workload. Record request mix, concurrency, warm-up, duration, sample count, error rate, p50/p95/p99 and deployment region. Exclude or separately report cold starts.

Cost per request requires provider/model usage data and pricing captured for the same evaluation window. The current benchmark does not calculate this.

## 6. Test suite

Run the repository test suite from the project root:

```powershell
pytest -q
```

Run the focused evaluation tests:

```powershell
pytest -q tests/test_ranking_metrics.py tests/test_retrieval_benchmark.py tests/test_routing_benchmark.py
```

A passing test suite validates tested code paths; it does not establish model quality or production readiness by itself.

## 7. Release-readiness checklist

Before describing the service as production-ready, verify these operational controls in the deployment environment:

- [ ] Secrets are stored in deployment secret management, not source control; rotate any exposed credentials.
- [ ] Authentication and authorization are tested for every user-data endpoint.
- [ ] Database migrations, backup/restore and retention procedures are documented and tested.
- [ ] Health/readiness checks distinguish application startup from database and dependency readiness.
- [ ] Structured logs and alerts cover API errors, database failures, LLM failures, retrieval failures and reranker fallback.
- [ ] Rate limits, request-size limits, timeouts and bounded retries are configured.
- [ ] User data is isolated by authenticated identity and is not included in logs or evaluation artifacts.
- [ ] Nutrition and exercise outputs include appropriate scope limits and escalation guidance; the system is not presented as a medical diagnostic service.
- [ ] A reviewed adversarial safety set and end-to-end load test have passed against the release candidate.

This checklist is a release gate, not a claim that every item has already been verified.

## 8. Secret-gated live retrieval evaluation

The API includes a temporary-use, read-only endpoint at `POST /internal/evaluation/retrieval`. It is deliberately excluded from OpenAPI. The endpoint returns 404 unless `EVALUATION_API_KEY` is configured in the deployment environment with a secret of at least 32 characters. Requests must provide the same value in the `X-Evaluation-Key` header. Never commit the secret or place it in a client-side application.

The endpoint accepts **one labelled query per request**, a domain (`fitness` or `nutrition`), and up to five candidate limits (5–100). It compares vector-only retrieval once, then RRF and cross-encoder retrieval for each candidate limit. Responses contain ranking metrics, server-side retrieval-call timings, reranker status and paired reranking overhead. They do not return chunk text.

The route is isolated from user profile, workout, history, program and account stores. It only performs SELECT-based retrieval against the fitness/nutrition knowledge tables and loads the existing embedding/reranking models. It does not write to application or user data. It is disabled by default when the evaluation secret is absent. Remove the deployment secret after the evaluation to disable access again.

Run the client from any working directory by supplying the dataset path. The dataset is read by the client and each labelled query is sent to the deployed API; the benchmark itself runs on the deployed service:

```powershell
$env:EVALUATION_API_KEY = "<the strong secret configured on the deployed API>"
python C:\path\to\training-agent-api\scripts\run_live_retrieval_sweep.py --base-url https://training-agent-api.fastapicloud.dev --domain fitness --dataset C:\path\to\fitness.jsonl --candidate-counts 10 20 30 50 75 --output C:\path\to\live-fitness-sweep.json
```

Repeat with `--domain nutrition` and the nutrition relevance dataset. The client performs a warm-up request first, then aggregates macro ranking metrics, p50/p95 retrieval latency and p50/p95 **paired** reranking overhead across the query set. A reranker fallback is never counted as a successful reranked result; incomplete reranker runs have null reranked quality/overhead summaries. The returned latency is warmed server-side retrieval-call latency, not end-to-end chat latency.

The endpoint processes one query per HTTP request to bound request duration. The existing API rate limit still applies. Start with a small `--max-cases 2` smoke test, then run the full reviewed dataset. Do not use a partial smoke-test report as a resume performance claim.

