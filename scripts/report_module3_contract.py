"""Emit a machine-readable static contract report for Module 3 CI."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
retrieval = (root / "rag/nutrition_retrieval.py").read_text(encoding="utf-8")
store = (root / "nutrition_ingestion/store.py").read_text(encoding="utf-8")
migration = (root / "sql/003_nutrition_knowledge.sql").read_text(encoding="utf-8")
metadata_migration = (root / "sql/004_nutrition_embedding_metadata.sql").read_text(encoding="utf-8")

checks = {
    "semantic_retrieval_uses_pgvector": "embedding <=> %s::vector" in retrieval,
    "lexical_retrieval_uses_postgres_fts": "content_tsv @@ plainto_tsquery" in retrieval,
    "rrf_fusion_is_enabled": "SUM(score)" in retrieval and "RRF_K = 60" in retrieval,
    "reranking_is_optional": "if rerank and results" in retrieval,
    "retrieval_is_nutrition_scoped": "FROM nutrition_knowledge" in retrieval and re.search(r"\b(?:FROM|JOIN)\s+fitness_knowledge\b", retrieval, re.IGNORECASE) is None,
    "persistence_is_nutrition_scoped": "INSERT INTO nutrition_knowledge" in store and "fitness_knowledge" not in store,
    "embedding_dimension_is_384": "EMBEDDING_DIMENSIONS = 384" in store,
    "embedding_model_is_recorded": "embedding_model" in store and "ADD COLUMN IF NOT EXISTS embedding_model" in metadata_migration,
    "database_upsert_is_idempotent": "ON CONFLICT (chunk_id)" in store,
    "pgvector_schema_is_present": "VECTOR(384)" in migration,
}
report = {
    "module": "module-3-nutrition-retrieval-and-persistence",
    "evaluation_scope": "static contract checks only; no live Neon or relevance benchmark",
    "checks_passed": sum(checks.values()),
    "checks_total": len(checks),
    "checks": checks,
}
out = root / "artifacts/module-3/contract_report.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if not all(checks.values()):
    raise SystemExit("Module 3 contract report contains failed checks")
