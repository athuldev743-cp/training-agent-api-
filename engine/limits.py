"""Body areas the user says hurt, found in a message (deterministic). A pain word plus wrist, knee, shoulder, elbow or lower
back. Exercise names that contain a body-part word (knee push-up, shoulder press) are removed first, derived from the
exercise library, so 'knee pushups hurt my wrists' gives only the wrist. Nothing is stored; 'sore' also fires on ordinary
muscle soreness, which is the conservative direction."""
import re

_PAIN = re.compile(r"\b(?:hurt|hurts|hurting|pain|painful|injur\w*|sore|ache|aches|aching|strain\w*|sprain\w*|tweak\w*"
                   r"|problem|problems|issue|issues|bad)\b", re.IGNORECASE)
_PARTS = {
    "wrist": r"\bwrists?\b",
    "knee": r"\bknees?\b",
    "shoulder": r"\bshoulders?\b",
    "elbow": r"\belbows?\b",
    "lower_back": r"\b(?:lower|low)\s+back\b|\bback\s+pain\b|\bspine\b|\b(?:my|the)\s+back\s+(?:hurts?|aches?|is\s+sore)\b",
}
_PART_WORDS = {"wrist", "knee", "shoulder", "elbow"}


def _strip_exercise_names(text: str, exercises) -> str:
    for e in exercises or []:
        tokens = re.findall(r"[a-z]+", e.name.lower())
        for i, t in enumerate(tokens):
            if t in _PART_WORDS:
                for window in (tokens[max(i - 1, 0):i + 1], tokens[i:i + 2]):
                    if len(window) == 2:
                        pattern = r"\b" + r"[\s-]*".join(re.escape(w) for w in window) + r"[a-z-]*"
                        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def extract_limitations(text: str, exercises=None) -> list[str]:
    cleaned = _strip_exercise_names(text, exercises)
    if not _PAIN.search(cleaned):
        return []
    return [name for name, pattern in _PARTS.items() if re.search(pattern, cleaned, re.IGNORECASE)]
