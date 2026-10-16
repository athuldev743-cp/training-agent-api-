"""Chat answers grounded by fact retrieval (Step 27, experimental). chat_cli and the dispatcher still use
agent/grounded_chat.py; this is a separate arm for evaluation. Same prompt and generation settings."""
import os

from agent.grounded_chat import GROUNDED_RULES, PLAIN
from agent.local_model import strip_think
from engine.context import exercise_facts
from engine.facts import build_facts, select_context
from rag.resolve import resolve_exercise


def retrieved_context(text: str, exercises, profile=None, programs=None, history=None) -> str:
    program, next_text = None, None
    if programs is not None:
        latest = programs.latest_program()
        if latest is not None:
            program = latest["program"]
            row = programs.next_session(latest["id"])
            if row is not None:
                label = program.weeks[row["week"] - 1].sessions[row["day_number"] - 1].label
                next_text = (f"Next session: day {row['day_number']} of week {row['week']} ({label}) "
                             f"on {row['planned_date']}.")
    recent = history.recent_workouts(3) if history is not None else []
    facts = build_facts(program, exercises, profile, next_text, recent)
    named = resolve_exercise(text, exercises)["status"] != "none"
    context = select_context(text, facts, exercise_named=named)
    if named:
        extra = exercise_facts(text, exercises)
        context = (context + "\n" + extra) if context else extra
    return context


def answer_chat_retrieved(llm, text: str, exercises, profile=None, programs=None, history=None) -> dict:
    context = retrieved_context(text, exercises, profile, programs, history)
    system = PLAIN + (f" {GROUNDED_RULES}\n\nFACTS:\n{context}" if context else "")
    refs, ev_status = [], "off"
    if not context and os.getenv("RAG_CHAT") == "1":
        from rag.safe_search import grounded_context
        outcome = EVIDENCE_SEARCH(text)
        ev_status, refs = outcome.status, [e.ref for e in outcome.evidence]
        rule = (" Answer only from the EVIDENCE below and cite sources like [S1]. If the evidence does not answer the question, say your sources do not cover it."
                if outcome.status == "ok" else " Say that you have no source for this question and answer only in general terms.")
        system += rule + "\n\nEVIDENCE:\n" + grounded_context(outcome)
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text + " /no_think"}],
        max_tokens=110,
        temperature=0.2,
    )
    return {"reply": strip_think(out["choices"][0]["message"]["content"]),
            "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
            "topics": ["retrieved"], "context_chars": len(context), "mode": "retrieved",
            "sources": refs, "evidence_status": ev_status}


def EVIDENCE_SEARCH(query):
    from rag.safe_search import retrieve_evidence
    return retrieve_evidence(query)
