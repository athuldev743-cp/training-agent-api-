from rag.hybrid import HYBRID_SQL
from rag.nutrition_retrieval import NUTRITION_HYBRID_SQL


def test_fitness_hybrid_retrieval_is_scoped_to_fitness_corpus():
    assert "FROM fitness_knowledge" in HYBRID_SQL
    assert "JOIN fitness_knowledge" in HYBRID_SQL
    assert "nutrition_knowledge" not in HYBRID_SQL
    assert "UNION ALL" in HYBRID_SQL
    assert "SUM(score)" in HYBRID_SQL


def test_nutrition_hybrid_retrieval_is_scoped_to_nutrition_corpus():
    assert "FROM nutrition_knowledge" in NUTRITION_HYBRID_SQL
    assert "JOIN nutrition_knowledge" in NUTRITION_HYBRID_SQL
    assert "fitness_knowledge" not in NUTRITION_HYBRID_SQL
    assert "UNION ALL" in NUTRITION_HYBRID_SQL
    assert "SUM(score)" in NUTRITION_HYBRID_SQL
