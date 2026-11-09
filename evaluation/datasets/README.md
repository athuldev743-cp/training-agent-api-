# Reviewed retrieval datasets

Module 7 evaluates the live PostgreSQL retrievers against manually reviewed relevance labels.
Do not invent chunk IDs or treat synthetic labels as retrieval-quality evidence.

Create one JSONL file per domain. Each non-empty line must be a JSON object with:

- case_id: stable case identifier
- query: realistic user query
- relevant_chunk_ids: one or more exact IDs judged relevant by a human reviewer

Nutrition IDs must come from nutrition_knowledge.chunk_id.
Fitness IDs must come from fitness_knowledge.id.

Example structure only; replace the placeholder with real, reviewed IDs before running:

{"case_id":"nutrition-protein-01","query":"How much protein is useful for resistance training?","relevant_chunk_ids":["REPLACE_WITH_REVIEWED_CHUNK_ID"]}

Run from the repository root after setting the normal Neon connection environment:

- python scripts/benchmark_retrieval.py --domain nutrition --dataset evaluation/datasets/nutrition.jsonl --output artifacts/module-7/nutrition.json
- python scripts/benchmark_retrieval.py --domain fitness --dataset evaluation/datasets/fitness.jsonl --output artifacts/module-7/fitness.json

The report compares hybrid RRF with the reranked path and includes Precision@K,
Recall@K, MRR@K, NDCG@K, p50/p95 latency, and per-query reranker status.
Nutrition reranking may explicitly report fallback_rrf when its remote service is
not configured or unavailable. Fitness reranker failures are retried without
reranking only when the cross-encoder itself fails.

Precision@K uses K as the denominator, even if fewer than K documents are returned.
Metrics are macro-averaged across queries. Review labels and corpus versions should
be kept with each report so results remain reproducible.
