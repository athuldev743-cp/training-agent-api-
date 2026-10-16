"""Smart, data-driven adaptation from logged sessions. No exercise is special-cased:
- easier / harder steps come from each exercise's regression / progression links,
- targets come from its rep range at the user's level,
- evidence comes from the sessions the user logged (newest first, as HistoryStore.sessions_for returns them).
Thresholds are this project's defaults, not clinical rules; they live here so they are easy to change."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from engine.prescription import Prescription
from rag.store import Exercise

PROMOTE_AFTER_SESSIONS = 2  # good sessions in a row, on different days, before moving to a harder variation
HIGH_EFFORT_RPE = 9.5  # average RPE at or above this counts as too hard
EASY_ENOUGH_RPE = 8.0  # average RPE at or below this counts as manageable
LOAD_STEP_PERCENT = 0.05  # roughly 5% heavier (the ACSM guidance is 2-10%)
MIN_LOAD_STEP_KG = 1.0
MAX_HOPS = 5


class Adaptation(BaseModel):
    action: Literal["start", "progress_reps", "hold", "regress", "promote", "add_load", "ladder_top"]
    exercise_id: str
    next_exercise_id: str | None = None
    aim_reps: int | None = None
    next_load_kg: float | None = None
    reason: str


def is_loadable(ex: Exercise) -> bool:
    """Only dumbbell exercises can take more load; bodyweight exercises progress through harder variations."""
    return "dumbbells" in ex.equipment


def next_load(current_kg: float) -> float:
    if current_kg <= 0:
        return 0.0
    step = max(MIN_LOAD_STEP_KG, round(current_kg * LOAD_STEP_PERCENT * 2) / 2)
    return round(current_kg + step, 1)


def ladder_down(exercise_id: str, by_id: dict, allowed_ids: set[str]) -> str | None:
    """First easier variation, walking down the regression chain, that this user is allowed to do and that trains the same main muscle."""
    current = by_id[exercise_id]
    for _ in range(MAX_HOPS):
        nxt = current.regression
        if nxt is None or nxt not in by_id:
            return None
        if nxt in allowed_ids and set(by_id[nxt].primary) & set(by_id[exercise_id].primary):
            return nxt
        current = by_id[nxt]
    return None


def ladder_up(exercise_id: str, by_id: dict, allowed_ids: set[str]) -> str | None:
    """The harder variation, only if this user is allowed to do it (level, phase, equipment, limits)."""
    nxt = by_id[exercise_id].progression
    return nxt if nxt in allowed_ids and set(by_id[nxt].primary) & set(by_id[exercise_id].primary) else None


def _stats(session: dict, rx: Prescription) -> dict:
    sets = session["sets"]
    done = [s for s in sets if s.completed]
    best = max((s.reps for s in done), default=0)
    avg_rpe = sum(s.rpe for s in sets) / len(sets)
    return {
        "date": datetime.fromisoformat(session["performed_at"]).date(),
        "best": best,
        "lowest": min(s.reps for s in sets),
        "avg_rpe": avg_rpe,
        "cannot_reach_min": best < rx.rep_min,
        "too_hard": avg_rpe >= HIGH_EFFORT_RPE,
        "hit_top": all(s.completed and s.reps >= rx.rep_max for s in sets) and avg_rpe <= EASY_ENOUGH_RPE,
    }


def decide(exercise_id: str, sessions: list[dict], rx: Prescription, by_id: dict,
           allowed_ids: set[str]) -> Adaptation:
    ex = by_id[exercise_id]
    unit = " s" if rx.unit == "seconds" else " reps"

    def result(action, reason, **kw):
        return Adaptation(action=action, exercise_id=exercise_id, reason=reason, **kw)

    if not sessions:
        return result("start", f"New exercise: start at {rx.rep_min}{unit} per set.", aim_reps=rx.rep_min)

    stats = [_stats(s, rx) for s in sessions]
    latest = stats[0]

    def step_back(why: str) -> Adaptation:
        easier = ladder_down(exercise_id, by_id, allowed_ids)
        if easier:
            return result(
                "regress",
                f"{why} Switch to {by_id[easier].name} and build up there; come back to {ex.name} once it "
                f"feels easy on {PROMOTE_AFTER_SESSIONS} different days.",
                next_exercise_id=easier)
        return result(
            "hold",
            f"{why} There is no easier option with your equipment and limits, so repeat it and aim for "
            f"{rx.rep_min}{unit}; fewer sets are fine.",
            aim_reps=rx.rep_min)

    if latest["cannot_reach_min"] and len(stats) > 1 and not stats[1]["cannot_reach_min"]:
        return result("hold", f"You managed {latest['best']}{unit} at best, below the {rx.rep_min} minimum, "
                              f"but the session before was fine. One off day is not a trend: repeat it and "
                              f"aim for {rx.rep_min}{unit}.", aim_reps=rx.rep_min)
    if latest["cannot_reach_min"]:
        return step_back(f"You managed {latest['best']}{unit} at best, below the {rx.rep_min} minimum.")

    if latest["too_hard"]:
        if len(stats) > 1 and stats[1]["too_hard"]:
            return step_back("The effort was too high two sessions in a row.")
        return result("hold", "That felt very hard. Repeat the same reps and stop 2-3 reps before failure.",
                      aim_reps=latest["lowest"])

    if latest["hit_top"]:
        recent = stats[:PROMOTE_AFTER_SESSIONS]
        enough = (len(recent) == PROMOTE_AFTER_SESSIONS
                  and all(s["hit_top"] for s in recent)
                  and len({s["date"] for s in recent}) == PROMOTE_AFTER_SESSIONS)
        if not enough:
            return result("hold", f"You reached the top of the range ({rx.rep_max}{unit}). Repeat it on "
                                  f"another day to confirm, then move up.", aim_reps=rx.rep_max)
        harder = ladder_up(exercise_id, by_id, allowed_ids)
        if harder:
            return result("promote", f"{PROMOTE_AFTER_SESSIONS} good sessions at the top of the range: "
                                     f"move to {by_id[harder].name}.", next_exercise_id=harder)
        if is_loadable(ex):
            load = max(s.weight_kg for s in sessions[0]["sets"])
            if load > 0:
                return result("add_load", f"Top of the range twice: add weight ({load:g} kg to "
                                          f"{next_load(load):g} kg) and start again at {rx.rep_min}{unit}.",
                              next_load_kg=next_load(load), aim_reps=rx.rep_min)
            return result("hold", "You reached the top of the range. Log the dumbbell weight you used so I "
                                  "can raise it.", aim_reps=rx.rep_max)
        return result("ladder_top", "You are at the top of the ladder for your level and equipment. Add a "
                                    "set, slow the tempo, or ask for a level check-in.")

    if latest["lowest"] >= rx.rep_max:
        return result("hold", "You are at the top of the range but it still felt hard, or a set was missed. "
                              "Repeat it until it feels easier (RPE 8 or lower).", aim_reps=rx.rep_max)

    aim = max(rx.rep_min, min(rx.rep_max, latest["lowest"] + 1))
    return result("progress_reps", f"Good work. Next time aim for {aim}{unit} on every set "
                                   f"(range {rx.rep_min}-{rx.rep_max}).", aim_reps=aim)
