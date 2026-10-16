"""Turns a user message into a tool choice.
The model only picks the tool (tiny schema-constrained JSON, ~8 tokens).
Plain code then extracts every argument: search filters, progression numbers, training days."""
import json
import time

from engine.extract import (
    extract_difficulty,
    extract_equipment,
    extract_muscle,
    extract_progression,
)
from engine.extract_coach import extract_days

TOOLS = ["search_exercises", "calculate_progression", "build_program", "todays_workout", "week_view",
         "finish_session", "skip_session", "show_reminders", "explain_programme", "general_chat"]

SCHEMA = {
    "type": "object",
    "properties": {"tool": {"type": "string", "enum": TOOLS}},
    "required": ["tool"],
}

SYSTEM_PROMPT = """You are the router for a fitness coach app. Reply with ONE JSON object.

Tools:
- search_exercises: the user wants exercise ideas, or asks what to train or do for a muscle or body area (even if they do not say "exercises").
- calculate_progression: the user tells you what they did in training (sets, reps, effort), or asks what their next target for a named exercise should be.
- build_program: the user asks you to create a NEW routine, plan or programme (for a week or a month), possibly with a number of days.
- todays_workout: the user asks what their scheduled workout is today, or which session comes next in their existing plan.
- week_view: the user wants to see this week's schedule or plan.
- finish_session: the user says they finished or completed today's scheduled workout.
- skip_session: the user wants to skip a scheduled workout.
- show_reminders: the user asks about reminders, what is due, or whether they missed anything.
- explain_programme: the user asks about their own current plan: which split they follow, why it looks the way it does, why it has few exercises or short sessions, what each day trains, or whether core or a certain muscle is included.
- general_chat: greetings, thanks, nutrition, how to perform an exercise, which muscles an exercise works, questions about training terms, or anything else."""


PROGRAMME_TOOLS = ("todays_workout", "week_view", "finish_session", "skip_session", "show_reminders",
                   "explain_programme")
_PROMPT_LINES = SYSTEM_PROMPT.split("\n")


def available_tools(has_programme: bool) -> list[str]:
    """Tools that can apply in the current state. Without a saved programme the programme tools are hidden,
    so the model never sees them (shorter prompt, fewer ways to misroute). Order is kept."""
    return list(TOOLS) if has_programme else [t for t in TOOLS if t not in PROGRAMME_TOOLS]


def system_prompt_for(tools) -> str:
    """SYSTEM_PROMPT with only the tool lines in tools. With all tools it is identical to SYSTEM_PROMPT."""
    keep = set(tools)
    return "\n".join(line for line in _PROMPT_LINES
                     if not line.startswith("- ") or line[2:].split(":", 1)[0] in keep)


def schema_for(tools) -> dict:
    return {"type": "object", "properties": {"tool": {"type": "string", "enum": list(tools)}},
            "required": ["tool"]}


def choose_tool(llm, user_message: str, available=None) -> dict:
    tools = [t for t in TOOLS if t in (available or TOOLS)] or list(TOOLS)
    messages = [
        {"role": "system", "content": system_prompt_for(tools)},
        {"role": "user", "content": user_message + " /no_think"},
    ]
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=messages,
        response_format={"type": "json_object", "schema": schema_for(tools)},
        temperature=0.0,
        max_tokens=60,
    )
    latency = time.perf_counter() - t0
    model_raw = out["choices"][0]["message"]["content"]
    try:
        data = json.loads(model_raw)
        valid = True
    except json.JSONDecodeError:
        data, valid = None, False

    if data:
        tool = data.get("tool")
        if tool == "search_exercises":
            data = {
                "tool": tool,
                "muscle": extract_muscle(user_message),
                "equipment": extract_equipment(user_message),
                "max_difficulty": extract_difficulty(user_message),
            }
        elif tool == "calculate_progression":
            data = {"tool": tool, **extract_progression(user_message)}
        elif tool == "build_program":
            data = {"tool": tool, "days": extract_days(user_message)}

    return {
        "raw": json.dumps(data) if data else model_raw,
        "model_raw": model_raw,
        "data": data,
        "valid_json": valid,
        "n_tools": len(tools),
        "latency_s": latency,
        "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
    }