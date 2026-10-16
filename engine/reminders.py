"""Deterministic reminder logic. Pure function: today's date, the programme and its session rows in,
messages out. The Android app will call this same logic from a scheduled background job."""
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel

from engine.program import Program

GAP_DAYS = 5  # days without a workout before a nudge
MAX_MISSED_SHOWN = 2


class Reminder(BaseModel):
    kind: Literal["no_program", "program_done", "missed", "workout_today", "workout_tomorrow",
                  "gap", "phase_change", "next_workout"]
    message: str
    session_id: int | None = None


def _when(d: date) -> str:
    return d.strftime("%A %d %b")


def compute_reminders(today: date, program: Program | None, sessions: list[dict]) -> list[Reminder]:
    if program is None:
        return [Reminder(kind="no_program", message=(
            "You don't have a routine yet. Want me to build a 4-week programme from your profile?"))]

    planned = sorted((r for r in sessions if r["status"] == "planned"), key=lambda r: r["planned_date"])
    if not planned:
        return [Reminder(kind="program_done", message=(
            "You finished your 4-week programme. Take an easier week (fewer sets, easy effort), then I "
            "can build the next one and check whether you are ready to move up a level."))]

    def info(r):
        return program.weeks[r["week"] - 1].sessions[r["day_number"] - 1]

    out: list[Reminder] = []

    missed = [r for r in planned if date.fromisoformat(r["planned_date"]) < today]
    today_rows = [r for r in planned if date.fromisoformat(r["planned_date"]) == today]
    for idx, r in enumerate(sorted(missed, key=lambda r: r["planned_date"], reverse=True)[:MAX_MISSED_SHOWN]):
        s = info(r)
        out.append(Reminder(kind="missed", session_id=r["id"], message=(
            f"You missed day {r['day_number']} of week {r['week']} ({s.label}), planned for "
            f"{r['planned_date']}. " + ("Do it today, or tell me to skip it." if idx == 0 and not today_rows else "Skip it, or tell me to move it. I won't stack it on top of another workout."))))

    has_today = False
    for r in planned:
        d = date.fromisoformat(r["planned_date"])
        s = info(r)
        if d == today:
            has_today = True
            out.append(Reminder(kind="workout_today", session_id=r["id"], message=(
                f"Workout today: day {r['day_number']} of week {r['week']} - {s.label}, "
                f"about {s.est_minutes:g} min.")))
        elif d == today + timedelta(days=1):
            out.append(Reminder(kind="workout_tomorrow", session_id=r["id"], message=(
                f"Tomorrow: day {r['day_number']} of week {r['week']} - {s.label}, "
                f"about {s.est_minutes:g} min.")))

    done = [r for r in sessions if r["status"] == "done" and r["completed_at"]]
    if done and not missed and not has_today:
        last = max(date.fromisoformat(r["completed_at"][:10]) for r in done)
        days = (today - last).days
        if days >= GAP_DAYS:
            out.append(Reminder(kind="gap", message=(
                f"It has been {days} days since your last workout. Even a short session keeps "
                f"the routine going.")))

    upcoming = [r for r in planned if date.fromisoformat(r["planned_date"]) >= today]
    if upcoming:
        nxt = upcoming[0]
        d = date.fromisoformat(nxt["planned_date"])
        if nxt["week"] == 3 and nxt["day_number"] == 1 and 0 <= (d - today).days <= 2:
            out.append(Reminder(kind="phase_change", session_id=nxt["id"], message=(
                f"Week 3 starts on {_when(d)}: harder variations and more sets, at a bit more effort.")))
        if not out:
            s = info(nxt)
            out.append(Reminder(kind="next_workout", session_id=nxt["id"], message=(
                f"Next workout: day {nxt['day_number']} of week {nxt['week']} ({s.label}) on {_when(d)}.")))
    return out
