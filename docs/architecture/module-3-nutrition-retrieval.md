# Module 3 — Nutrition hybrid retrieval

## Scope

This module adds a nutrition-specific retriever using the same retrieval strategy already used by the fitness knowledge base:

1. Dense semantic retrieval with `BAAI/bge-small-en-v1.5` (384 dimensions).
2. PostgreSQL full-text lexical retrieval.
3. Reciprocal Rank Fusion (RRF), with `k=60`.
4. Optional cross-encoder reranking with `Xenova/ms-marco-MiniLM-L-6-v2`.

The nutrition corpus is isolated in `nutrition_knowledge`; queries in this module never read `fitness_knowledge`.

## Database setup

Review and execute `sql/003_nutrition_knowledge.sql` against the intended Neon database. It creates the pgvector-backed table and HNSW, GIN, and source indexes. The migration is idempotent for the objects it creates. Confirm that the Neon project supports the `vector` extension before applying it.

This change does not automatically execute SQL against production, and it does not migrate or copy any fitness rows.

## Runtime dependencies

Install the optional retrieval extra:

```powershell
python -m pip install -e ".[ingestion,retrieval]"
```

The embedding and reranker models are downloaded on first use by FastEmbed. Plan for model download/cache and memory requirements in the deployment environment.

## API

```python
from rag.nutrition_retrieval import search_nutrition

hits = search_nutrition(
    "protein intake for resistance training",
    limit=5,
    candidate_limit=30,
    rerank=True,
)
```

Each hit includes chunk ID, content, source ID, source title, page, chunk number, and RRF score. Reranked results also include a reranker score.

## Current limitations

- This module provides the retrieval function and schema; it does not add an HTTP endpoint or connect it to the nutrition agent.
- It does not populate the table. Nutrition chunk persistence/embedding ingestion must be implemented and validated before meaningful live retrieval.
- CI uses mocked database/model boundaries. A real Neon integration test and retrieval-quality benchmark require a populated test corpus and configured database credentials.
- Scanned PDF OCR and scientific correctness remain outside retrieval mechanics.
