"""Natural-language entry points to the coach flow. The model only picks the tool; these functions call the
same deterministic code as the slash commands. Read-only tools run directly. Tools that change stored data
(build_program over an active programme, finish_session, skip_session) ask for confirmation first, because a
small model can misroute and a message must never silently change a user's programme."""
from datetime import date

from agent.coach import (
    NO_ACTIVE,
    current_reminders,
    finish_session,
    skip_session,
    start_program,
    todays_workout,
    week_view,
)
from engine.explain import explain_programme
from engine.whatif import what_if
from rag.store import Exercise
from state.history import HistoryStore
from state.profile import UserProfile
from state.programs import ProgramStore

COACH_TOOLS = ("build_program", "todays_workout", "week_view", "finish_session", "skip_session", "show_reminders", "explain_programme")
NO_STORE = "I can't reach your saved programme right now."


def _next_description(programs: ProgramStore) -> str | None:
    active = programs.active_program()
    if active is None:
        return None
    row = programs.next_session(active["id"])
    session = active["program"].weeks[row["week"] - 1].sessions[row["day_number"] - 1]
    return f"day {row['day_number']} of week {row['week']} ({session.label})"


def run_coach_tool(decision: dict, exercises: list[Exercise], profile: UserProfile | None,
                   programs: ProgramStore | None, history: HistoryStore | None,
                   today: date | None = None, confirmed: bool = False) -> dict:
    tool = decision.get("tool")
    today = today or date.today()

    def out(status: str, reply: str, **data) -> dict:
        return {"tool": tool, "status": status, "data": data, "reply": reply}

    def ask(reply: str) -> dict:
        return out("confirm", reply + " Reply yes to confirm, or no to cancel.", pending=decision)

    if programs is None:
        return out("error", NO_STORE)

    if tool == "build_program":
        if profile is None:
            return out("needs_info", "I need your profile first. Run python setup_profile.py.")
        if not confirmed:
            if programs.active_program() is None:
                return ask("Build a 4-week programme from your profile, starting today?")
            return ask("You already have an active programme. Building a new one replaces it.")
        result = start_program(exercises, profile, programs, today, decision.get("days"), history=history)
        return out(result["status"], result["reply"], days=decision.get("days"))

    if tool == "todays_workout":
        return out("ok", todays_workout(programs, today))
    if tool == "week_view":
        return out("ok", week_view(programs))
    if tool == "show_reminders":
        items = current_reminders(programs, today)
        return out("ok", "\n".join(f"* {r.message}" for r in items) or "Nothing is due right now.")

    if tool == "explain_programme":
        latest = programs.latest_program()
        if latest is None:
            return out("ok", "You don't have a routine yet. Type /program and I'll build a 4-week programme from your profile.")
        return out("ok", explain_programme(latest["program"], exercises, profile))

    if tool == "what_if":
        if profile is None:
            return out("needs_info", "I need your profile first. Run python setup_profile.py.")
        latest = programs.latest_program()
        current = latest["program"].days_per_week if latest is not None else None
        result = what_if(exercises, profile, current, decision.get("add_equipment"), decision.get("days"), decision.get("minutes"))
        return out(result["status"], result["reply"])
    if tool in ("finish_session", "skip_session"):
        desc = _next_description(programs)
        if desc is None:
            return out("ok", NO_ACTIVE)
        if not confirmed:
            verb = "Mark" if tool == "finish_session" else "Skip"
            tail = " as done?" if tool == "finish_session" else "?"
            return ask(f"{verb} {desc}{tail}")
        if tool == "skip_session":
            return out("ok", skip_session(programs))
        if profile is None or history is None:
            return out("needs_info", "I need your profile and workout history to finish a session.")
        return out("ok", finish_session(exercises, profile, programs, history, today))

    return out("error", "Sorry, I don't know how to do that yet.")