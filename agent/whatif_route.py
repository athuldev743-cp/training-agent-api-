"""Step 38c (experimental): decide, BEFORE the model router runs, whether a message is a what-if question. The router prompt is
not touched (adding a tool line to it regressed the held-out routing sets in Step 38b). Two detectors, both requiring that code
can extract a concrete change (equipment, days or minutes): regex (conditional or hypothetical framing words) and embed
(similarity to a one-sentence description with a small embedding model). Not wired into chat yet."""
import re

from agent.card_embed import cosine, embed_texts
from engine.whatif import extract_whatif

PREFIX = "Represent this sentence for searching relevant passages: "
DESCRIPTION = ("What would change in my workout plan if I had different equipment, trained a different number of days a "
               "week, or had less time per session")
_FRAMING = re.compile(
    r"\b(?:what if|what would (?:change|happen|be different)|if i|if we|suppose|supposing|imagine|say i|let'?s say"
    r"|how would (?:my|the|this) (?:plan|programme|program|routine|workouts?|sessions?) (?:look|change|be))\b",
    re.IGNORECASE)


def params_of(text: str) -> dict | None:
    p = extract_whatif(text)
    if p["add_equipment"] or p["days"] is not None or p["minutes"] is not None:
        return p
    return None


def regex_detect(text: str) -> dict | None:
    return params_of(text) if _FRAMING.search(text) else None


class EmbedDetector:
    def __init__(self, embedder, threshold: float):
        self.embedder, self.threshold = embedder, threshold
        self.vector = embed_texts(embedder, [DESCRIPTION])[0]

    def similarity(self, text: str) -> float:
        return cosine(embed_texts(self.embedder, [PREFIX + text])[0], self.vector)

    def detect(self, text: str) -> dict | None:
        p = params_of(text)
        if p is None:
            return None
        return p if self.similarity(text) >= self.threshold else None
