"""Step 6.4: a substitution is checked against equipment, limitations, level and stimulus before it is applied."""
from rag.search import DIFFICULTY_RANK
from rag.store import Exercise


def check_swap(old_ids: set[str], new_id: str, exercises: list[Exercise], equipment: list[str],
               limitations: list[str], level: str) -> list[str]:
    by_id = {e.id: e for e in exercises}
    new = by_id.get(new_id)
    if new is None:
        return [f"unknown exercise {new_id}"]
    problems = []
    owned = set(equipment) | {"bodyweight"}
    if set(new.equipment) - owned:
        problems.append(f"needs equipment you lack: {sorted(set(new.equipment) - owned)}")
    if set(new.stresses) & set(limitations):
        problems.append(f"loads a limited area: {sorted(set(new.stresses) & set(limitations))}")
    if DIFFICULTY_RANK[new.difficulty] > DIFFICULTY_RANK[level]:
        problems.append(f"{new.difficulty} exercise is above your {level} level")
    olds = [by_id[i] for i in old_ids if i in by_id]
    if olds and not any(set(new.primary) & set(o.primary) for o in olds):
        problems.append("does not train the same main muscle as the exercise it replaces")
    return problems
