"""Finds which knowledge-base exercise a message is talking about (deterministic).
Exact names resolve, also glued or spaced spellings (pushups, push ups, pullup, calfraises); vague mentions return
candidates to ask about instead of guessing."""
import re

from rag.store import Exercise

_GENERIC_HEADS = {"up"}  # 'push-up', 'pull-up' end in 'up'; do not match on that alone
_EXTRA_TOKENS = 2        # how many more message tokens than name tokens may be glued into one word


def _stem(token: str) -> str:
    if len(token) >= 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]


def _contains(seq: list[str], sub: list[str]) -> bool:
    n = len(sub)
    return n > 0 and any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


def _glued(seq: list[str], name: list[str]) -> bool:
    """True when a run of consecutive message tokens, written together as one word, equals the exercise name written
    as one word (pushups, push ups -> pushup; calfraises -> calfraise). Only whole tokens are joined, never substrings."""
    if len(name) < 2:
        return False
    target = "".join(name)
    for i in range(len(seq)):
        joined = ""
        for tok in seq[i:i + len(name) + _EXTRA_TOKENS]:
            joined += tok
            if joined == target:
                return True
            if len(joined) >= len(target):
                break
    return False


def resolve_exercise(text: str, exercises: list[Exercise]) -> dict:
    """status: 'ok' (one exercise named), 'ambiguous' (ask which), or 'none'."""
    toks = _tokens(text)

    exact = []
    for ex in exercises:
        name = _tokens(ex.name)
        if _contains(toks, name) or _glued(toks, name):
            exact.append((len(name), ex))
    if exact:
        longest = max(n for n, _ in exact)
        best = [ex for n, ex in exact if n == longest]
        if len(best) == 1:
            return {"status": "ok", "exercise": best[0], "candidates": best}
        return {"status": "ambiguous", "exercise": None, "candidates": best}

    candidates = []
    for ex in exercises:
        head = _tokens(ex.name)[-1]
        if head not in _GENERIC_HEADS and head in toks:
            candidates.append(ex)
    if candidates:
        return {"status": "ambiguous", "exercise": None, "candidates": candidates[:5]}
    return {"status": "none", "exercise": None, "candidates": []}
