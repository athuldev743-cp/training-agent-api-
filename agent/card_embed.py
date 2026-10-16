"""Step 36 (experimental): choose the rationale card by semantic similarity with a small embedding model (bge-small-en-v1.5
as GGUF, run by the same llama-cpp-python as the chat model). No hand-written question-answer pairs: each card is
represented by its own title and one-line description (agent.card_router.GISTS) or by its own text. Not wired into chat."""
import math
from pathlib import Path

EMBED_PATH = Path(__file__).resolve().parent.parent / "models" / "bge-small-en-v1.5-q8_0.gguf"


def load_embedder(path: Path = EMBED_PATH):
    from llama_cpp import Llama
    return Llama(model_path=str(path), embedding=True, n_ctx=512, verbose=False)


def _normalise(v: list) -> list[float]:
    if v and isinstance(v[0], list):  # one vector per token: mean-pool
        n = len(v)
        v = [sum(col) / n for col in zip(*v)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    out = model.embed(list(texts))
    if out and isinstance(out[0], (int, float)):  # a single vector came back for a one-item list
        out = [out]
    return [_normalise(v) for v in out]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class CardIndex:
    def __init__(self, model, representations: dict[str, str]):
        self.model = model
        self.keys = list(representations)
        self.vectors = embed_texts(model, [representations[k] for k in self.keys])

    def scores(self, question: str, prefix: str = "") -> list[tuple[str, float]]:
        q = embed_texts(self.model, [prefix + question])[0]
        return sorted(((k, cosine(q, v)) for k, v in zip(self.keys, self.vectors)), key=lambda x: -x[1])

    def choose(self, question: str, sim_min: float, margin: float = 0.0, prefix: str = "") -> tuple[str, float, float]:
        """(card key or 'none', best similarity, gap to the second best)."""
        ranked = self.scores(question, prefix)
        best_key, best = ranked[0]
        gap = best - ranked[1][1] if len(ranked) > 1 else best
        if best < sim_min or gap < margin:
            return "none", best, gap
        return best_key, best, gap
