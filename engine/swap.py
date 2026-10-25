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
            old_minutes = session.est_minutes
            rx = prescription_for(new_id, level, templates, session.goal or program.goal)
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
            fresh = next((i for i in items if i.exercise_id == new_id), None)
            while fresh is not None and fresh.sets > 1 and session.est_minutes > old_minutes + 1e-9:
                fresh.sets -= 1  # a swap never makes a session longer than it was
                session.est_minutes = _minutes(items)
    by_day: dict[int, list] = {}
    for wk in result.weeks:
        for session in wk.sessions:
            if (wk.number, session.day_number) <= after:
                continue
            for i in session.exercises:
                if i.exercise_id == new_id:
                    by_day.setdefault(session.day_number, []).append((wk.number, session, i))
    for rows in by_day.values():  # sets never fall from one week to the next: cap earlier weeks at the later minimum
        cap = None
        for _, session, i in sorted(rows, key=lambda r: -r[0]):
            cap = i.sets if cap is None else min(cap, i.sets)
            i.sets = cap
            session.est_minutes = _minutes(session.exercises)
    return result, changes
