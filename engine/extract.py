"""Deterministic entity extraction (no LLM needed): search filters and progression numbers."""
import re

from engine.taxonomy import ALIASES, GROUPS, MUSCLES


def _phrase_pattern(term: str) -> str:
    words = term.replace("_", " ").split()
    body = r"[\s-]".join(re.escape(w) for w in words)
    # The lookahead stops "push-up" / "pull-up" from matching the push / pull groups.
    return rf"\b{body}\b(?![-\s]?ups?\b)"


_TERMS = sorted({*MUSCLES, *GROUPS, *ALIASES}, key=len, reverse=True)
_MUSCLE_PATTERNS = [(t, re.compile(_phrase_pattern(t), re.IGNORECASE)) for t in _TERMS]

_EQUIPMENT = [
    ("dumbbells", r"\bdumb-?bells?\b"),
    ("bench", r"\bbench\b(?![-\s]?press)"),
    ("pull_up_bar", r"\b(?:pull|chin)[\s-]?up\s+bar\b"),
]

_LEVELS = [
    ("beginner", r"\b(?:beginners?|novice|new to (?:lifting|training|working out|the gym)"
                 r"|just (?:starting|started)|never (?:lifted|trained))\b"),
    ("advanced", r"\badvanced\b"),
    ("intermediate", r"\bintermediate\b"),
]


def extract_muscle(text: str) -> str:
    """Earliest muscle/group/alias mentioned, as a canonical name; 'any' if none."""
    best = None
    for term, pattern in _MUSCLE_PATTERNS:
        m = pattern.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), ALIASES.get(term, term))
    return best[1] if best else "any"


def extract_equipment(text: str) -> list[str]:
    return [name for name, pat in _EQUIPMENT if re.search(pat, text, re.IGNORECASE)]


def extract_difficulty(text: str) -> str:
    for level, pat in _LEVELS:
        if re.search(pat, text, re.IGNORECASE):
            return level
    return "any"


# ---- progression numbers -------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_SETS_REPS = [
    re.compile(r"\b(\d+)\s*[x\u00d7]\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*sets?\s*(?:of|x|\u00d7)\s*(\d+)\b", re.IGNORECASE),
]
_WEIGHT = re.compile(rf"\b{_NUM}\s*(?:kgs?|kilos?|kilograms?)\b", re.IGNORECASE)
_RPE = [
    re.compile(rf"\brpe\s*(?:of|was|is|:|=)?\s*{_NUM}\b", re.IGNORECASE),
    re.compile(rf"\b(?:felt|feels|feeling)\s+(?:like\s+)?(?:a|an)?\s*{_NUM}\b", re.IGNORECASE),
]


def extract_progression(text: str) -> dict:
    """sets, reps, weight_kg, rpe from a message. Missing sets/reps/rpe -> None.
    No weight mentioned -> 0.0 (bodyweight). Weight in pounds is not handled yet."""
    sets = reps = None
    for pattern in _SETS_REPS:
        m = pattern.search(text)
        if m:
            sets, reps = int(m.group(1)), int(m.group(2))
            break

    weight = 0.0
    w = _WEIGHT.search(text)
    if w:
        weight = float(w.group(1))

    rpe = None
    for pattern in _RPE:
        m = pattern.search(text)
        if m:
            rpe = float(m.group(1))
            break

    return {"exercise": None, "sets": sets, "reps": reps, "weight_kg": weight, "rpe": rpe}
