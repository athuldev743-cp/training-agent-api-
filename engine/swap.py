"""Programme swaps: replace an exercise (an easier or harder variation) in the remaining sessions."""
from engine.prescription import Prescription, load_templates, prescription_for
from engine.program import Program, ProgramExercise, _aim_range, _exercise_minutes, load_tiers
from rag.store import Exercise


def _minutes(items: list[ProgramExercise]) -> float:
    """Session time from each exercise's own stored range (not the level default)."""
    total = 0.0
    for e in items:
        rx = Prescription(template="programme", unit=e.unit, rep_min=e.rep_min, rep_max=e.rep_max,
                          sets_start=e.sets, sets_max=e.sets, rest_seconds=e.rest_seconds)
        total += _exercise_minutes(rx, e.sets)
    return round(total, 1)


def _redundant(a: Exercise, b: Exercise) -> bool:
    """The same exercise, or the same movement pattern for a shared main muscle."""
    return a.id == b.id or (a.movement == b.movement and bool(set(a.primary) & set(b.primary)))


def swap_exercise(program: Program, old_ids: set[str], new_id: str, level: str,
                  exercises: list[Exercise], after: tuple[int, int]) -> tuple[Program, int]:
    """Replaces every exercise in old_ids with new_id in all sessions after (week, day), keeping its position.
    If the new exercise would repeat a movement for a muscle already trained in that session, the old
    exercise is dropped without inserting a duplicate. The new exercise starts on `level`'s range.
    Returns (a new programme, number of changes). The original programme is not modified."""
    by_id = {e.id: e for e in exercises}
    tiers = load_tiers()
    templates = load_templates()
    rx = prescription_for(new_id, level, templates, program.goal)
    new_ex = by_id[new_id]
    result = program.model_copy(deep=True)
    changes = 0
    for wk in result.weeks:
        for session in wk.sessions:
            if (wk.number, session.day_number) <= after:
                continue
            olds = [i for i in session.exercises if i.exercise_id in old_ids]
            if not olds:
                continue
            changes += len(olds)
            rest = [i for i in session.exercises if i.exercise_id not in old_ids]
            inserted = any(_redundant(by_id[i.exercise_id], new_ex) for i in rest)  # redundant: just drop
            items = []
            for item in session.exercises:
                if item.exercise_id not in old_ids:
                    items.append(item)
                    continue
                if inserted:
                    continue
                inserted = True
                lo, hi = _aim_range(rx, wk.number)
                items.append(ProgramExercise(
                    exercise_id=new_id, name=new_ex.name, tier=tiers[new_id], unit=rx.unit,
                    sets=item.sets, rep_min=rx.rep_min, rep_max=rx.rep_max, aim_low=lo, aim_high=hi,
                    rest_seconds=rx.rest_seconds))
            session.exercises = items
            session.est_minutes = _minutes(items)
    return result, changes
