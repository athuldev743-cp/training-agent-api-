"""Explains which equipment would unlock a muscle the user cannot train yet."""
from engine.taxonomy import MUSCLES, expand
from rag.search import DIFFICULTY_RANK
from rag.store import Exercise


def unreachable_muscles(exercises: list[Exercise], muscles: list[str], equipment, difficulty: str) -> list[str]:
    """Requested muscles (groups expanded) that no allowed exercise mainly trains.
    equipment=None means no equipment filter, so everything is reachable."""
    allowed = None if equipment is None else set(equipment) | {"bodyweight"}
    targets = sorted(set().union(*(expand(m) for m in muscles)), key=MUSCLES.index)
    out = []
    for m in targets:
        reachable = any(
            m in e.primary
            and DIFFICULTY_RANK[e.difficulty] <= DIFFICULTY_RANK[difficulty]
            and (allowed is None or set(e.equipment) <= allowed)
            for e in exercises
        )
        if not reachable:
            out.append(m)
    return out


def unlock_options(exercises: list[Exercise], muscle: str, equipment, difficulty: str) -> list[list[str]]:
    """Extra equipment sets that would unlock this muscle, smallest first.
    Empty when the muscle is already reachable (or there is no equipment filter)."""
    if equipment is None:
        return []
    allowed = {"bodyweight"} | set(equipment)
    options = set()
    for e in exercises:
        if muscle not in e.primary or DIFFICULTY_RANK[e.difficulty] > DIFFICULTY_RANK[difficulty]:
            continue
        needed = tuple(sorted(set(e.equipment) - allowed))
        if not needed:
            return []
        options.add(needed)
    return [list(o) for o in sorted(options, key=lambda o: (len(o), o))]


def unlock_hint(exercises: list[Exercise], muscle: str, equipment, difficulty: str) -> str:
    name = muscle.replace("_", " ")
    options = unlock_options(exercises, muscle, equipment, difficulty)
    if not options:
        return f"No exercise in my library mainly trains {name} at your level yet."
    text = " or ".join(" + ".join(o).replace("_", " ") for o in options[:3])
    return f"With your equipment nothing mainly trains {name}. Adding {text} would unlock it."
