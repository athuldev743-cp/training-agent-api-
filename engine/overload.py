"""Progressive overload from logged sessions (double progression). Pure functions, no I/O.
Sessions are newest first, as HistoryStore.sessions_for returns them: {"performed_at", "sets": [set with
reps, weight_kg, rpe, completed]}. One noisy session never changes the plan; the model never does this math."""
from statistics import mean

DEFAULT_STEP_KG = 2.5
SMALL_STEP_KG = 1.0
EASY_RPE = 8.0  # every set at the top of the range at this effort or lower -> raise the load
HARD_RPE = 9.5  # average effort at or above this counts as too hard
FAILS_BEFORE_REGRESS = 2
DELOAD_SET_FRACTION = 0.6
VOLUME_DROP = 0.9


def step_kg(small_lift: bool = False) -> float:
    return SMALL_STEP_KG if small_lift else DEFAULT_STEP_KG


def session_stats(session: dict, planned_sets: int, rep_min: int, rep_max: int) -> dict:
    sets = session["sets"]
    avg = mean(s.rpe for s in sets)
    complete = all(s.completed for s in sets) and len(sets) >= planned_sets
    lowest = min(s.reps for s in sets)
    return {
        "load": max((s.weight_kg for s in sets), default=0.0),
        "lowest": lowest,
        "avg_rpe": avg,
        "failed": (not complete) or lowest < rep_min or avg >= HARD_RPE,
        "hit_top": complete and lowest >= rep_max and avg <= EASY_RPE,
    }


def next_target(planned_sets: int, rep_min: int, rep_max: int, sessions: list, loadable: bool = False,
                small_lift: bool = False) -> dict:
    """Target reps, sets and load for the next session of one exercise."""
    out = {"sets": planned_sets, "rep_min": rep_min, "rep_max": rep_max, "target_reps": rep_min,
           "target_load_kg": None}
    if not sessions:
        out["action"] = "start"
        out["reason"] = (f"No history yet: choose a load you can lift for {rep_min}-{rep_max} reps at RPE 6-7."
                         if loadable else f"No history yet: start at {rep_min} reps per set at RPE 6-7.")
        return out
    st = session_stats(sessions[0], planned_sets, rep_min, rep_max)
    load = st["load"] if loadable and st["load"] > 0 else None
    step = step_kg(small_lift)
    prev = session_stats(sessions[1], planned_sets, rep_min, rep_max) if len(sessions) > 1 else None
    if st["failed"]:
        if prev and prev["failed"]:
            out.update(action="regress", target_reps=rep_min,
                       target_load_kg=None if load is None else round(max(0.0, load - step), 1),
                       reason="Two hard or missed sessions in a row: go back one step and rebuild.")
        else:
            out.update(action="hold", target_reps=max(rep_min, st["lowest"]), target_load_kg=load,
                       reason="Missed reps or very hard effort once is normal: repeat the same target.")
    elif st["hit_top"]:
        if load is not None:
            out.update(action="add_load", target_reps=rep_min, target_load_kg=round(load + step, 1),
                       reason=f"Top of the range at RPE {EASY_RPE:g} or lower: add {step:g} kg and restart at {rep_min} reps.")
        else:
            out.update(action="top_of_range", target_reps=rep_max, target_load_kg=load,
                       reason="Top of the range: move to a harder variation or add a set.")
    else:
        out.update(action="add_rep", target_reps=max(rep_min, min(rep_max, st["lowest"] + 1)), target_load_kg=load,
                   reason="Good work: add one rep on every set.")
    return out


def deload_sets(sets: int) -> int:
    return max(1, round(sets * DELOAD_SET_FRACTION))


def deload_reason(weeks: list, failed_streak: int = 0) -> str | None:
    """weeks are newest first: {"avg_rpe", "volume"}. Returns why a deload week is needed, or None.
    Triggers: RPE at or above 9.5 two weeks running, two failed targets in a row on one lift, or volume
    falling while effort rises."""
    if failed_streak >= FAILS_BEFORE_REGRESS:
        return "two failed targets in a row on the same lift"
    if len(weeks) >= 2 and all(w["avg_rpe"] >= HARD_RPE for w in weeks[:2]):
        return "effort at RPE 9.5 or higher two weeks running"
    if len(weeks) >= 2 and weeks[0]["volume"] < VOLUME_DROP * weeks[1]["volume"] and weeks[0]["avg_rpe"] > weeks[1]["avg_rpe"]:
        return "volume fell while effort rose"
    return None
