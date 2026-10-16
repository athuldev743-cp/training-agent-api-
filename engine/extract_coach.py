"""Extracts how many training days a week the user asked for (code, not the model)."""
import re

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_WORDS = "|".join(_NUMBER_WORDS)
_PATTERNS = [
    re.compile(r"\b(\d)\s*-?\s*days?\b", re.IGNORECASE),
    re.compile(r"\b(\d)\s*(?:x|times)\s*(?:a|per|/)?\s*(?:week|wk)\b", re.IGNORECASE),
    re.compile(rf"\b({_WORDS})\s*-?\s*days?\b", re.IGNORECASE),
    re.compile(rf"\b({_WORDS})\s*times\s*(?:a|per)\s*week\b", re.IGNORECASE),
]


def extract_days(text: str) -> int | None:
    for pattern in _PATTERNS:
        m = pattern.search(text)
        if m:
            raw = m.group(1).lower()
            value = _NUMBER_WORDS.get(raw) or int(raw)
            if 1 <= value <= 7:
                return value
    return None
