"""Fact retrieval for chat (Step 27, fixes in Step 28). The user's state is written as small one-sentence facts (index
cards). For a question, the best-matching cards are pulled by word overlap (rare words count more, typos and a few
synonyms are forgiven). A short cheat sheet (core facts: level, goal, equipment, split, days) is added for any question
that is clearly about the user. No topic labels per question, no model. Experimental: chat still uses engine/context.py.
Step 28: synonyms apply to the question only (not the card text); bare digits never score; week references such as
"weeks 3 and 4" become week:3 / week:4 tabs on both sides; the cheat sheet says "sessions a week"."""
import math
import re
from dataclasses import dataclass
from difflib import get_close_matches
from statistics import mean

from engine.context import locked_pulling, profile_facts
from engine.program import SPLIT_NAMES, Program
from rag.store import Exercise
from state.profile import UserProfile

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MIN_SCORE = 1.5   # a card must match with at least this much rarity-weighted overlap
TOP_K = 4         # at most this many non-core cards
MAX_CHARS = 1500

_WORD = re.compile(r"[a-z0-9]+")
_WEEK_REF = re.compile(r"\bweeks?\s+(\d+)(?:\s*(?:-|to|and|&)\s*(\d+))?")
_STOP = frozenset(
    "a an the is are am was were be do does did i me my mine we us our you your it its of on in at to for and or so "
    "that this these those what which how when where why who whom can could should would will have has had about "
    "again there than then too very just also please tell show give any some if with as by from into out up not no".split())
_PERSONAL = frozenset({"i", "me", "my", "mine", "we", "our", "us", "am"})
_ABBREV = {"mon": "monday", "tue": "tuesday", "tues": "tuesday", "wed": "wednesday", "thu": "thursday",
           "thur": "thursday", "thurs": "thursday", "fri": "friday"}  # no sat/sun: ordinary words


def stem(word: str) -> str:
    """Crude suffix stripper so exercise/exercises, log/logged, programme/program compare equal."""
    w = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2]:
        w = w[:-1]
    return w


# vocabulary normalisation for QUESTIONS only (generic words, not per-question labels)
_SYNONYMS = {stem(k): stem(v) for k, v in {
    "workout": "session", "routine": "program", "plan": "program", "schedule": "program",
    "abs": "core", "ab": "core", "leg": "lower", "last": "recent", "latest": "recent", "previous": "recent",
    "free": "rest", "off": "rest"}.items()}
# words every programme fact contains: they never retrieve a card, but they do bring the cheat sheet
_DOMAIN = frozenset(stem(w) for w in ("session", "program", "week", "day", "train", "exercise"))


@dataclass(frozen=True)
class Fact:
    text: str
    core: bool = False


def _week_tokens(text: str) -> list[str]:
    """'weeks 3 and 4' or 'Weeks 3-4' -> week:3, week:4 (like a page-number tab on the card)."""
    out = []
    for m in _WEEK_REF.finditer(text.lower()):
        lo = int(m.group(1))
        hi = int(m.group(2) or lo)
        if 1 <= lo <= hi <= 12:
            out += [f"week:{n}" for n in range(lo, hi + 1)]
    return out


def _tokens(text: str, synonyms: bool = False) -> list[str]:
    out = _week_tokens(text)
    for w in _WORD.findall(text.lower()):
        w = _ABBREV.get(w, w)
        if w in _STOP:
            continue
        t = stem(w)
        if synonyms:
            t = _SYNONYMS.get(t, t)
        if len(t) >= 2 or t.isdigit():
            out.append(t)
    return out


def build_facts(program: Program | None, exercises: list[Exercise], profile: UserProfile | None,
                next_text: str | None, recent: list[dict]) -> list[Fact]:
    facts = [Fact(profile_facts(profile), core=True)]
    if program is None:
        facts.append(Fact("PROGRAMME: none yet (the user can type /program to build one).", core=True))
    else:
        sessions = program.weeks[0].sessions
        days = [s.weekday for s in sessions]
        avg = mean(s.est_minutes for s in sessions)
        facts.append(Fact(
            f"PROGRAMME: {len(program.weeks)}-week {SPLIT_NAMES[program.split]} split, {program.days_per_week} days a "
            f"week ({', '.join(days)}), so {program.days_per_week} sessions a week; sessions are about {avg:.0f} "
            f"minutes on average.", core=True))
        for s in sessions:
            names = "; ".join(e.name for e in s.exercises)
            facts.append(Fact(f"{s.weekday} session (day {s.day_number}, {s.label}, about {s.est_minutes:g} minutes, "
                              f"{len(s.exercises)} exercises, trains {', '.join(s.muscles)}): {names}."))
        rest = [d for d in WEEKDAYS if d not in days]
        if rest:
            facts.append(Fact("Rest days (no session): " + ", ".join(rest) + "."))
        facts.append(Fact(f"The programme lasts {len(program.weeks)} weeks. Weeks 3-4 add harder variations and more sets."))
        facts += [Fact(f"Note: {n}") for n in program.notes]
        locked = locked_pulling(exercises, program.equipment)
        if locked:
            facts.append(Fact("Back work (rows, pull-ups, chin-ups) is limited because it needs equipment you do not "
                              "have yet: " + ", ".join(i.replace("_", " ") for i in locked) + "."))
    if next_text:
        facts.append(Fact(next_text))
    if recent:
        names = {e.id: e.name for e in exercises}
        for i, w in enumerate(recent[:3]):
            done = ", ".join(names.get(x, x) for x in w["exercises"])
            facts.append(Fact(f"{'Most recent' if i == 0 else 'Earlier'} logged workout, on "
                              f"{w['performed_at'][:10]}: {done} ({w['n_sets']} sets)."))
    else:
        facts.append(Fact("No workouts logged yet."))
    return facts


def _index(facts: list[Fact]):
    tokens = [set(_tokens(f.text)) for f in facts]
    n = max(len(facts), 1)
    df: dict[str, int] = {}
    for s in tokens:
        for t in s:
            df[t] = df.get(t, 0) + 1
    return tokens, {t: math.log(1 + n / c) for t, c in df.items()}


def retrieve(question: str, facts: list[Fact], k: int = TOP_K, min_score: float = MIN_SCORE) -> list[Fact]:
    """The best-matching non-core facts. Question words not seen in any fact are corrected to the closest word that is
    (typo tolerance, words of 5+ letters only). Domain words and bare digits never score."""
    tokens, idf = _index(facts)
    vocab = list(idf)
    q = set()
    for t in _tokens(question, synonyms=True):
        if t not in idf and len(t) >= 5 and ":" not in t:
            close = get_close_matches(t, vocab, n=1, cutoff=0.8)
            t = close[0] if close else t
        q.add(t)
    scored = []
    for i, f in enumerate(facts):
        if f.core:
            continue
        score = sum(idf[t] for t in q & tokens[i] if t not in _DOMAIN and not t.isdigit())
        if score >= min_score:
            scored.append((-score, i))
    scored.sort()
    return [facts[i] for _, i in scored[:k]]


def select_context(question: str, facts: list[Fact], exercise_named: bool = False) -> str:
    """Facts text for the model, or '' when the question needs none."""
    personal = bool(set(_WORD.findall(question.lower())) & _PERSONAL)
    if exercise_named and not personal:
        return ""  # exercise facts only (the caller adds them)
    hits = retrieve(question, facts)
    domain = bool(set(_tokens(question, synonyms=True)) & _DOMAIN)
    if not (hits or personal or domain):
        return ""
    context = "\n".join([f.text for f in facts if f.core] + [f.text for f in hits])
    if len(context) > MAX_CHARS:
        context = context[:MAX_CHARS].rsplit("\n", 1)[0]
    return context
