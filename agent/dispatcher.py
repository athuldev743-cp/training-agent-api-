"""Executes the chosen tool with deterministic code and builds the reply text.
Only general_chat uses the model to write text."""
import re
import time
from statistics import mean

from pydantic import ValidationError

from agent.local_model import strip_think
from agent.coach_tools import COACH_TOOLS, run_coach_tool
from agent.grounded_chat import answer_chat
from agent.retrieved_chat import answer_chat_retrieved
from agent.router import available_tools, choose_tool
from agent.routing import route_message
from agent.whatif_route import regex_detect
from engine.models import SetResult, Target
from engine.progression import next_target
from engine.limits import extract_limitations
from engine.health import DISCLAIMER
from rag.resolve import resolve_exercise
from rag.search import search_exercises
from rag.store import Exercise, safe_exercises
from state.history import HistoryStore, SetLog

_BODYWEIGHT_ONLY = re.compile(
    r"\b(?:no equipment|without (?:any )?equipment|body-?weight only)\b", re.IGNORECASE
)
CHAT_MODE = "retrieved"  # "selected" = keyword topic rules (Step 23); "retrieved" = fact cards (Step 27)
_ASK_FOR = {
    "sets": "the number of sets",
    "reps": "the reps per set",
    "rpe": "how hard it felt (RPE 1-10)",
}


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _equipment_filter(text: str, extracted: list[str], profile_equipment):
    """Named equipment wins; 'no equipment' means bodyweight only; else the saved profile (None = no filter)."""
    if extracted:
        return extracted
    if _BODYWEIGHT_ONLY.search(text):
        return []
    return profile_equipment


def _fmt_target(t: Target) -> str:
    weight = f" @ {t.weight_kg:g} kg" if t.weight_kg else ""
    return f"{t.sets}x{t.reps}{weight}"


def _run_search(d: dict, text: str, exercises: list[Exercise], profile_equipment) -> dict:
    equipment = _equipment_filter(text, d.get("equipment") or [], profile_equipment)
    found = search_exercises(
        exercises,
        muscle=d.get("muscle"),
        available_equipment=equipment,
        max_difficulty=d.get("max_difficulty"),
        limit=5,
    )
    filters = {"muscle": d.get("muscle"), "equipment": equipment,
               "max_difficulty": d.get("max_difficulty")}
    if not found:
        reply = "I couldn't find matching exercises. Try a different muscle or equipment."
    else:
        muscle = d.get("muscle")
        label = "" if muscle in (None, "any") else muscle.replace("_", " ") + " "
        lines = [
            f"{i}. {e.name} ({e.difficulty}; needs {', '.join(e.equipment)})"
            for i, e in enumerate(found, 1)
        ]
        reply = f"Here are {len(found)} {label}exercises:\n" + "\n".join(lines)
    return {"tool": "search_exercises", "status": "ok",
            "data": {"filters": filters, "ids": [e.id for e in found]}, "reply": reply}


def _which_exercise(candidates: list[Exercise]) -> dict:
    names = ", ".join(e.name for e in candidates)
    if len(candidates) == 1:
        reply = f"Do you mean {names}? Please say the full exercise name."
    else:
        reply = f"Which exercise do you mean: {names}?"
    return {"tool": "calculate_progression", "status": "needs_info",
            "data": {"missing": ["exercise"], "candidates": [e.id for e in candidates]},
            "reply": reply}


def _decide_from_logs(exercise_id: str, sets: list[SetLog]):
    """Planned target = sets done, fewest reps in any set, heaviest weight; then apply the rules."""
    planned = Target(exercise=exercise_id, sets=min(len(sets), 10),
                     reps=max(1, min(s.reps for s in sets)),
                     weight_kg=max(s.weight_kg for s in sets))
    results = [SetResult(reps=s.reps, rpe=s.rpe, completed=s.completed) for s in sets]
    return planned, next_target(planned, results)


def _run_progression(d: dict, text: str, exercises: list[Exercise],
                     history: HistoryStore | None) -> dict:
    res = resolve_exercise(text, exercises)
    ex = res["exercise"] if res["status"] == "ok" else None
    keys = ("sets", "reps", "rpe")
    have_all = all(d.get(k) is not None for k in keys)
    have_none = all(d.get(k) is None for k in keys)

    # No numbers given: use the last logged session for that exercise.
    if have_none and history is not None:
        if res["status"] == "ambiguous":
            return _which_exercise(res["candidates"])
        if ex is not None:
            sessions = history.sessions_for(ex.id, limit=1)
            if not sessions:
                return {"tool": "calculate_progression", "status": "needs_info",
                        "data": {"missing": ["history"]},
                        "reply": (f"You haven't logged any {ex.name} sessions yet. Tell me the sets, "
                                  f"reps and RPE and I'll log it and work out your next target.")}
            logs = sessions[0]["sets"]
            planned, decision = _decide_from_logs(ex.id, logs)
            date = sessions[0]["performed_at"][:10]
            avg_rpe = mean(s.rpe for s in logs)
            reply = (f"Your last {ex.name} session ({date}): {_fmt_target(planned)} at RPE {avg_rpe:g}. "
                     f"Next target: {_fmt_target(decision.target)}. {decision.reason}")
            return {"tool": "calculate_progression", "status": "ok",
                    "data": {"source": "history", "exercise": ex.id,
                             "planned": planned.model_dump(), "next": decision.target.model_dump(),
                             "reason": decision.reason},
                    "reply": reply}

    if not have_all:
        missing = [k for k in keys if d.get(k) is None]
        ask = _join([_ASK_FOR[m] for m in missing])
        return {"tool": "calculate_progression", "status": "needs_info",
                "data": {"missing": missing},
                "reply": f"To work out your next target I need {ask}."}

    try:
        planned = Target(exercise="exercise", sets=d["sets"], reps=d["reps"],
                         weight_kg=d.get("weight_kg") or 0.0)
        results = [SetResult(reps=d["reps"], rpe=d["rpe"]) for _ in range(planned.sets)]
        decision = next_target(planned, results)
    except ValidationError:
        return {"tool": "calculate_progression", "status": "invalid_input", "data": {},
                "reply": "Those numbers look off (sets 1-10, reps 1-100, RPE 1-10). Can you check them?"}

    reply = (f"Last time: {_fmt_target(planned)} at RPE {d['rpe']:g}. "
             f"Next target: {_fmt_target(decision.target)}. {decision.reason}")
    data = {"source": "message", "planned": planned.model_dump(),
            "next": decision.target.model_dump(), "reason": decision.reason}

    if history is not None:
        if ex is not None:
            logs = [SetLog(exercise_id=ex.id, reps=planned.reps, weight_kg=planned.weight_kg,
                           rpe=d["rpe"]) for _ in range(planned.sets)]
            data["logged_workout_id"] = history.log_workout(logs)
            weight = f" @ {planned.weight_kg:g} kg" if planned.weight_kg else ""
            reply = (f"Logged {ex.name}: {planned.sets} sets of {planned.reps}{weight} at "
                     f"RPE {d['rpe']:g}. ") + reply
        elif res["status"] == "ambiguous":
            names = ", ".join(e.name for e in res["candidates"])
            reply += (f" I haven't logged it because I'm not sure which exercise you mean "
                      f"({names}). Say it again with the full exercise name to log it.")
    if history is not None and ex is None and res["status"] == "none":
        reply += " I couldn't tell which exercise that was, so I haven't logged it. Say the exercise name (for example: push-up) to log it."
    return {"tool": "calculate_progression", "status": "ok", "data": data, "reply": reply}


def run_tool(decision: dict, text: str, exercises: list[Exercise], profile_equipment=None,
             history: HistoryStore | None = None) -> dict:
    """Pure, model-free execution of a routing decision."""
    tool = decision.get("tool")
    if tool == "search_exercises":
        return _run_search(decision, text, exercises, profile_equipment)
    if tool == "calculate_progression":
        return _run_progression(decision, text, exercises, history)
    if tool == "general_chat":
        return {"tool": tool, "status": "needs_llm", "data": {}, "reply": ""}
    return {"tool": tool, "status": "error", "data": {}, "reply": "Sorry, I don't know how to do that yet."}


def chat_reply(llm, text: str) -> str:
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a friendly, concise fitness coach. Answer in at most 3 short sentences. Do not give medical advice; suggest seeing a professional for injuries or health conditions. If a question is not about fitness, training, exercise or healthy habits, say you can only help with fitness and do not answer it."},
            {"role": "user", "content": text + " /no_think"},
        ],
        max_tokens=160,
        temperature=0.4,
    )
    return strip_think(out["choices"][0]["message"]["content"])


def _handle_message_impl(llm, text: str, exercises: list[Exercise], profile=None, programs=None, today=None,
                   history: HistoryStore | None = None) -> dict:
    """profile (UserProfile) supplies default equipment; history (HistoryStore) enables logging."""
    profile_equipment = profile.equipment if profile else None
    t0 = time.perf_counter()
    from engine.safety import assess_symptoms
    safety = assess_symptoms(text)
    import re as _re
    from engine.limits import extract_limitations as _xl
    _alt = _re.search(r"\b(instead of|alternatives?|any \w+ exercises?|what exercises?|which exercises?|exercises? (for|that|to)|suggest|recommend)\b", text.lower())
    if safety.level == "urgent" or (safety.level == "caution" and not (_alt and _xl(text, exercises))):
        return {"tool": None, "status": "safety_" + safety.level, "data": {"matched": safety.matched},
                "reply": safety.reply, "route_s": 0.0, "prompt_tokens": 0,
                "total_s": time.perf_counter() - t0}
    from engine.safety import assess_policy
    _kind, _reply = assess_policy(text)
    if _kind:
        return {"tool": None, "status": "safety_" + _kind, "data": {}, "reply": _reply, "route_s": 0.0,
                "prompt_tokens": 0, "total_s": time.perf_counter() - t0}
    pre = regex_detect(text)
    if pre is not None:
        result = run_coach_tool({"tool": "what_if", **pre}, exercises, profile, programs, history, today)
        result["route_s"] = 0.0
        result["prompt_tokens"] = 0
        result["total_s"] = time.perf_counter() - t0
        return result
    has_programme = programs is None or programs.latest_program() is not None
    decision = route_message(llm, text, exercises, available_tools(has_programme))
    if not decision["data"]:
        result = {"tool": None, "status": "error", "data": {},
                  "reply": "Sorry, I couldn't understand that."}
    else:
        tool_exercises = exercises
        mentioned = []
        if decision["data"].get("tool") == "search_exercises":
            mentioned = extract_limitations(text, exercises)
            tool_exercises = safe_exercises(exercises, sorted(set(profile.limitations if profile else []) | set(mentioned)))
            named = resolve_exercise(text, exercises)
            if decision["data"].get("muscle") == "any" and named["status"] == "ok":
                decision["data"]["muscle"] = named["exercise"].primary[0]
                tool_exercises = [e for e in tool_exercises if e.id != named["exercise"].id]
        if decision["data"].get("tool") in COACH_TOOLS:
            result = run_coach_tool(decision["data"], exercises, profile, programs, history, today)
        else:
            result = run_tool(decision["data"], text, tool_exercises, profile_equipment, history)
        if decision["data"].get("tool") == "search_exercises" and mentioned:
            names = " and ".join(m.replace("_", " ") for m in mentioned)
            if result["data"].get("ids"):
                result["reply"] = f"You mentioned your {names}, so I left out exercises that load it.\n" + result["reply"] + "\n" + DISCLAIMER
            else:
                result["reply"] = ("I could not find an exercise for that muscle that avoids your " + names + " with your equipment. A physiotherapist or doctor can suggest alternatives.\n" + DISCLAIMER)
        from engine.scope import out_of_scope, SCOPE_REPLY
        if result["status"] == "needs_llm" and out_of_scope(text):
            result["reply"] = SCOPE_REPLY
            result["status"] = "out_of_scope"
        from engine.scope import out_of_scope, SCOPE_REPLY
        if result["status"] == "needs_llm" and out_of_scope(text):
            result["reply"] = SCOPE_REPLY
            result["status"] = "out_of_scope"
        if result["status"] == "needs_llm":
            chat = (answer_chat_retrieved(llm, text, exercises, profile, programs, history)
                    if CHAT_MODE == "retrieved" else answer_chat(llm, text, exercises, profile, programs, history))
            result["reply"] = chat["reply"]
            result["chat_prompt_tokens"] = chat["prompt_tokens"]
            result["status"] = "ok"
    result["route_s"] = decision["latency_s"]
    result["prompt_tokens"] = decision.get("prompt_tokens")
    result["total_s"] = time.perf_counter() - t0
    return result


def handle_message(llm, text, exercises, profile=None, programs=None, today=None, history=None) -> dict:
    """Contained entry point: an unexpected failure becomes a controlled error reply, never an exception."""
    t0 = time.perf_counter()
    if isinstance(text, str) and len(text) > MAX_MESSAGE_CHARS:
        return {"tool": None, "status": "too_long", "data": {}, "reply": f"That message is too long for me. Please keep it under {MAX_MESSAGE_CHARS} characters.",
                "route_s": 0.0, "prompt_tokens": 0, "total_s": 0.0}
    try:
        return _handle_message_impl(llm, text, exercises, profile, programs, today, history)
    except Exception as exc:
        from agent.cloud_llm import LLMUnavailable
        if isinstance(exc, LLMUnavailable):
            return {"tool": None, "status": "llm_unavailable", "data": {},
                    "reply": "The coach chat is unavailable right now because I cannot reach the language service. Your plan, workouts, logging and reminders still work.",
                    "route_s": 0.0, "prompt_tokens": 0, "total_s": time.perf_counter() - t0}
        return {"tool": None, "status": "error", "data": {"error": type(exc).__name__},
                "reply": "Sorry, something went wrong. Please try again.",
                "route_s": 0.0, "prompt_tokens": 0, "total_s": time.perf_counter() - t0}


MAX_MESSAGE_CHARS = 1000
