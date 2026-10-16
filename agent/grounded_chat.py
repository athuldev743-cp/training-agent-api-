"""Chat answers grounded in the user's own state.
mode 'none' = plain chat (the old behaviour, used as the baseline), 'selected' = the context router,
'everything' = the naive approach of sending all state and the whole exercise library."""
from agent.local_model import strip_think
from engine.context import build_context, select_topics

PLAIN = ("You are a friendly, concise fitness coach. Answer in at most 3 short sentences. "
         "Do not give medical advice; suggest seeing a professional for injuries or health conditions. If a question is not about fitness, training, exercise or healthy habits, say you can only help with fitness and do not answer it.")
GROUNDED_RULES = ("For questions about the user's programme, profile, workout history or the app's exercises, use ONLY "
                  "the facts below and do not invent anything; if the facts do not contain the answer, say you don't "
                  "know and suggest /week, /today or /profile. For general fitness knowledge, answer briefly from "
                  "general knowledge.")


def answer_chat(llm, text: str, exercises, profile=None, programs=None, history=None, mode: str = "selected") -> dict:
    context, topics = "", []
    if mode != "none":
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
        topics = (select_topics(text, exercises) if mode == "selected"
                  else ["programme", "profile", "history", "exercise"])
        context = build_context(topics, text, exercises, profile, program, next_text, recent,
                                everything=(mode == "everything"))
    system = PLAIN + (f" {GROUNDED_RULES}\n\nFACTS:\n{context}" if context else "")
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text + " /no_think"}],
        max_tokens=110,
        temperature=0.2,
    )
    return {"reply": strip_think(out["choices"][0]["message"]["content"]),
            "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
            "topics": topics, "context_chars": len(context), "mode": mode}