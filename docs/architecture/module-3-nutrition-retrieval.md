# Module 3 — Nutrition hybrid retrieval and persistence

## Retrieval design

The nutrition retriever uses four stages:

1. **Dense retrieval:** `BAAI/bge-small-en-v1.5`, 384-dimensional vectors stored in pgvector.
2. **Lexical retrieval:** PostgreSQL full-text search over the generated `content_tsv` column.
3. **Fusion:** Reciprocal Rank Fusion (RRF), with `k=60`.
4. **Reranking:** optional FastEmbed cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`.

All SQL in this module targets only `nutrition_knowledge`; it does not read or write `fitness_knowledge`.

## Neon migrations

- Apply `sql/003_nutrition_knowledge.sql` once to create the isolated nutrition corpus and indexes.
- Apply `sql/004_nutrition_embedding_metadata.sql` once to add `embedding_model`, which records the model used for each stored vector.

Both migrations are additive and safe to re-run. Review them against the intended Neon project before execution.

## Build and persist nutrition chunks

First generate the Module 2 JSONL artifact:

```powershell
python -m nutrition_ingestion.cli --input tests/fixtures/nutrition --output build/nutrition_chunks.jsonl
```

Then install the retrieval dependencies and configure `NEON_DATABASE_URL` in the project environment:

```powershell
python -m pip install -e ".[ingestion,retrieval]"
python -m nutrition_ingestion.persist_cli --input build/nutrition_chunks.jsonl --batch-size 32
```

The persistence pipeline validates every JSONL row with the Module 2 Pydantic contract, generates 384-dimensional embeddings in batches, validates finite vector values, and writes through parameterized SQL. One database transaction covers the full JSONL file. Re-running the same file does not duplicate chunks; rows with missing vectors or a different recorded embedding model are re-embedded.

The command writes only to `nutrition_knowledge`. It does not modify the fitness corpus.

## Retrieval smoke test

After the table is populated:

```powershell
python -c "from rag.nutrition_retrieval import search_nutrition; print(search_nutrition('protein intake for resistance training', limit=5, rerank=False))"
```

Use `rerank=True` after confirming the base hybrid query works. First-time model use may download model weights.

## Evaluation boundaries

CI tests validate contracts, batching, vector dimensions, finite values, SQL isolation, upsert behavior, and rollback behavior using mocked model/database boundaries. CI does not connect to Neon and does not claim retrieval relevance.

A real benchmark must use a populated nutrition corpus and a reviewed set of nutrition questions with expected source/chunk IDs. Measure Recall@K, MRR@K, and nDCG@K for lexical-only, semantic-only, hybrid RRF, and hybrid RRF plus reranking. Also record latency and source attribution. Do not treat a small fixture corpus as evidence of production nutrition accuracy.

## Run a labeled retrieval benchmark

Prepare a UTF-8 JSONL file with one object per query. Each object must contain the query and at least one reviewed relevant chunk ID:

```json
{"query":"protein intake for resistance training","relevant_chunk_ids":["<64-character-chunk-id>"]}
```

Run both the hybrid and hybrid-plus-reranker evaluations:

```powershell
python scripts/benchmark_nutrition_retrieval.py --dataset evaluation/nutrition_retrieval.jsonl --output artifacts/module-3/retrieval_benchmark.json --k 5
```

The report includes Recall@K, MRR@K, nDCG@K, and p50/p95 query latency for each mode. Relevance labels must be reviewed by a subject-matter reviewer; the script measures ranking against those labels and does not establish medical correctness.
