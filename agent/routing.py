"""Applicability guards around the model's tool choice (Step 29; weekday shortcut in Step 32; Step 32b keeps the model's
menu unchanged on a retry). The model proposes a tool; plain code checks that the message can actually feed it.
Data-changing tools need a completion or skip word (typos tolerated); progression needs numbers or a named exercise; the
today/week views are not for questions that name a weekday. A weekday question goes straight to general_chat (retrieval
answers it from the day card). Any other rejected tool is hidden and the model is asked again; on a retry a
data-changing tool is never accepted; after a few tries the answer is general_chat. A wrong re-route costs a slower chat
answer, never a data change. Hiding extra tools from the menu changes the model's picks for unrelated questions, so only
the rejected tool is hidden."""
import json
import re
from difflib import get_close_matches

from agent.router import TOOLS, choose_tool
from engine.extract import extract_progression
from rag.resolve import resolve_exercise

MAX_TRIES = 3
WRITE_TOOLS = ("build_program", "finish_session", "skip_session")  # tools that ask to change stored data
SCHEDULE_VIEWS = ("todays_workout", "week_view")
_WORD = re.compile(r"[a-z']+")
_FINISH = ("finish", "finished", "finishing", "complete", "completed", "completing", "done")
_SKIP = ("skip", "skipped", "skipping")
_WEEKDAY = re.compile(
    r"\b(?:(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?|mon|tues?|wed|thu|thur|thurs|fri)\b",
    re.IGNORECASE)


def _has_cue(text: str, cues: tuple) -> bool:
    for word in _WORD.findall(text.lower()):
        if word in cues:
            return True
        if len(word) >= 4 and get_close_matches(word, cues, n=1, cutoff=0.8):
            return True
    return False


def applicable(tool: str, text: str, exercises) -> bool:
    """Can this message feed the tool? Unknown tools are applicable (no guard)."""
    if tool == "finish_session":
        return _has_cue(text, _FINISH)
    if tool == "skip_session":
        return _has_cue(text, _SKIP)
    if tool in SCHEDULE_VIEWS:
        return not _WEEKDAY.search(text)
    if tool == "calculate_progression":
        p = extract_progression(text)
        if any(p[k] is not None for k in ("sets", "reps", "rpe")):
            return True
        return resolve_exercise(text, exercises)["status"] != "none"
    return True


def _accepted(tool, allowed, text, exercises, retry) -> bool:
    return tool in allowed and applicable(tool, text, exercises) and not (retry and tool in WRITE_TOOLS)


def route_message(llm, text: str, exercises, available=None, max_tries: int = MAX_TRIES) -> dict:
    """choose_tool plus the guards. The result has the same shape as choose_tool, with the summed routing time and
    prompt tokens of all tries and 'rerouted' = the tools that were proposed and rejected."""
    allowed = [t for t in TOOLS if t in (available or TOOLS)] or list(TOOLS)
    hidden, latency, tokens = [], 0.0, 0
    decision, forced_chat = None, False
    for _ in range(max_tries):
        decision = choose_tool(llm, text, available=allowed)
        latency += decision["latency_s"]
        tokens += decision.get("prompt_tokens") or 0
        data = decision["data"]
        tool = (data or {}).get("tool")
        if not data or _accepted(tool, allowed, text, exercises, bool(hidden)):
            break
        hidden.append(tool)
        if tool in SCHEDULE_VIEWS and _WEEKDAY.search(text):
            forced_chat = True  # a weekday question: the chat answers it from the matching day card
            break
        allowed = [t for t in allowed if t != tool] or ["general_chat"]
    data = decision["data"]
    tool = (data or {}).get("tool")
    if data and (forced_chat or not _accepted(tool, allowed, text, exercises, bool(hidden))):
        data = {"tool": "general_chat"}
    out = dict(decision)
    out.update(data=data, raw=json.dumps(data) if data else decision["raw"], latency_s=latency,
               prompt_tokens=tokens or decision.get("prompt_tokens"), rerouted=hidden)
    return out
