"""Health limitations and pre-exercise screening. Deterministic; general guidance, not medical advice."""
import re

from state.profile import KNOWN_LIMITATIONS, UserProfile

DISCLAIMER = ("This app gives general fitness guidance, not medical advice. If something hurts or "
              "feels wrong, stop and talk to a doctor or physiotherapist.")

RED_FLAG_QUESTION = ("Do any of these apply to you? A heart condition or chest pain when exercising; "
                     "dizziness or fainting; high blood pressure; being pregnant; recovering from "
                     "surgery or a recent injury; or any other reason a doctor told you to limit "
                     "exercise (yes/no)")

_ALIASES = {
    "knees": "knee", "shoulders": "shoulder", "wrists": "wrist", "elbows": "elbow",
    "lower back": "lower_back", "low back": "lower_back", "back": "lower_back", "spine": "lower_back",
}


def normalize_limitations(text: str) -> list[str]:
    """'knees, lower back' -> ['knee', 'lower_back']; 'none' or blank -> []. Unknown -> ValueError."""
    parts = [p.strip().lower() for p in re.split(r"[,;/]|\band\b", text) if p.strip()]
    out: list[str] = []
    unknown: list[str] = []
    for part in parts:
        key = part.replace("-", " ").replace("_", " ")
        if key in ("none", "no", "nothing"):
            continue
        key = _ALIASES.get(key, key.replace(" ", "_"))
        if key in KNOWN_LIMITATIONS:
            if key not in out:
                out.append(key)
        else:
            unknown.append(part)
    if unknown:
        options = ", ".join(sorted(a.replace("_", " ") for a in KNOWN_LIMITATIONS))
        raise ValueError(f"I don't recognise: {', '.join(unknown)}. Choose from: {options}.")
    return out


def screening_from_answers(red_flag: bool, cleared_by_doctor: bool | None) -> str:
    if not red_flag:
        return "clear"
    return "cleared_by_doctor" if cleared_by_doctor else "needs_clearance"


def screening_gate(profile: UserProfile) -> str | None:
    """None if workout plans are allowed; otherwise the message to show instead of a plan."""
    if profile.screening == "not_asked":
        return ("Before I build a workout I need a quick health check. "
                "Please answer the health questions in your profile setup first.")
    if profile.screening == "needs_clearance":
        return ("Based on your answers, please get a doctor's go-ahead before starting a workout "
                "programme. Once you have it, update your health check in your profile.")
    return None
