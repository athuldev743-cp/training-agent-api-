# Evaluation datasets

All benchmark labels must be reviewed by a person. Do not invent chunk IDs, use generated labels as relevance truth, or present a tiny smoke-test dataset as representative performance evidence.

## Retrieval relevance datasets

Create one JSONL file per domain. Each non-empty line:

- `case_id`: stable case identifier
- `query`: realistic user query
- `relevant_chunk_ids`: one or more exact IDs judged relevant by a human reviewer

Nutrition IDs come from `nutrition_knowledge.chunk_id`. Fitness IDs come from `fitness_knowledge.id`.

Example structure only; replace the placeholder with exact reviewed IDs:

```json
{"case_id":"nutrition-protein-01","query":"How much protein is useful for resistance training?","relevant_chunk_ids":["REPLACE_WITH_REVIEWED_CHUNK_ID"]}
```

Run from the repository root with the normal Neon connection environment configured:

```powershell
python scripts/benchmark_retrieval.py --domain nutrition --dataset evaluation/datasets/nutrition.jsonl --output artifacts/evaluation/nutrition-retrieval.json
python scripts/benchmark_retrieval.py --domain fitness --dataset evaluation/datasets/fitness.jsonl --output artifacts/evaluation/fitness-retrieval.json
```

Each report compares semantic-only vector retrieval, hybrid RRF, and hybrid RRF with the configured cross-encoder reranker. It records Precision@K, Recall@K, MRR@K, NDCG@K, p50/p95 retrieval latency, per-query results, and reranker fallback status.

## Routing classification dataset

Create `evaluation/datasets/routing.jsonl`. Each non-empty line contains a message and a manually reviewed expected route:

- `fitness`: fitness agent only
- `nutrition`: nutrition agent only
- `mixed`: both agents
- `clarify`: neither agent; ask the user to clarify

Example structure:

```json
{"case_id":"route-001","message":"Build me a four-day workout plan","expected_route":"fitness"}
```

Run:

```powershell
python scripts/benchmark_routing.py --dataset evaluation/datasets/routing.jsonl --output artifacts/evaluation/routing.json
```

The report includes accuracy, a confusion matrix, per-route precision/recall/F1, and per-query predictions. Include ambiguous, short, misspelled, and mixed-intent examples in a reviewed dataset. A balanced dataset is preferable to one dominated by a single route.

## Dataset and report hygiene

- Keep the dataset, corpus snapshot/version, model names, environment and generated JSON report together when publishing results.
- Do not compare reports generated from different corpora as if only the retrieval strategy changed.
- Retrieval latency is measured around the retriever call in the current process. It is not end-to-end API latency.
- Reranking fallback counts must be reported; a run where the remote nutrition reranker is unavailable is not evidence of reranker quality.
