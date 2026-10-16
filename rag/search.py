"""Filtered search over the exercise knowledge base (no embeddings needed)."""
from engine.taxonomy import expand
from rag.store import Exercise

DIFFICULTY_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}


def search_exercises(
    exercises: list[Exercise],
    *,
    muscle: str | None = None,
    available_equipment: list[str] | None = None,
    max_difficulty: str | None = None,
    movement: str | None = None,
    query: str | None = None,
    primary_only: bool = False,
    limit: int = 10,
) -> list[Exercise]:
    """available_equipment=None means no equipment filter; [] means bodyweight only.
    muscle / max_difficulty of None or "any" mean no filter. muscle accepts muscles,
    groups (lower_body, push...) and aliases (legs, hams...). Unknown muscle -> no results.
    primary_only=True matches only exercises that are mainly FOR the muscle."""
    allowed = None
    requested = set()
    if available_equipment is not None:
        requested = {e.lower() for e in available_equipment} - {"bodyweight"}
        allowed = requested | {"bodyweight"}

    wanted_muscles = None
    if muscle and muscle.lower() != "any":
        wanted_muscles = expand(muscle)
        if not wanted_muscles:
            return []

    if max_difficulty and max_difficulty.lower() == "any":
        max_difficulty = None

    results = []
    for ex in exercises:
        if wanted_muscles is not None:
            matched = set(ex.primary if primary_only else ex.muscles)
            if not (wanted_muscles & matched):
                continue
        if movement and movement.lower() != ex.movement:
            continue
        if max_difficulty and DIFFICULTY_RANK[ex.difficulty] > DIFFICULTY_RANK[max_difficulty]:
            continue
        if allowed is not None and not set(ex.equipment) <= allowed:
            continue
        if query:
            text = f"{ex.name} {ex.id}".lower()
            if not all(token in text for token in query.lower().split()):
                continue
        results.append(ex)

    # Exercises that use the equipment the user asked for come first.
    results.sort(
        key=lambda e: (
            0 if requested & set(e.equipment) else 1,
            DIFFICULTY_RANK[e.difficulty],
            e.name,
        )
    )
    return results[:limit]
