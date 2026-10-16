"""Deterministic symptom triage (Step 6.5). It never diagnoses: it only decides when to stop and refer."""
import re
from typing import Literal

from pydantic import BaseModel

Level = Literal["none", "caution", "urgent"]

_URGENT = [
    r"\bchest (pain|pressure|tightness)\b", r"\bpain in (my |the )?chest\b",
    r"\b(short(ness)? of breath|can'?t breathe|cannot breathe|trouble breathing|struggling to breathe)\b",
    r"\b(faint(ed|ing)?|passed out|blacked out|black(ing)? out|lost consciousness)\b",
    r"\b(dizz(y|iness)|light[- ]?headed)\b",
    r"\b(numb(ness)?|tingling|pins and needles)\b",
    r"\b(palpitations?|heart (racing|pounding|fluttering)|irregular heartbeat)\b",
    r"\b(heard|felt) (a )?(pop|snap|crack|tear)\b",
    r"\b(sudden|severe|sharp|stabbing|shooting|excruciating|unbearable) (\w+ )?pain\b",
    r"\b(hit my head|head injury|concussion)\b",
]
_CAUTION = [
    r"\bpain(?!-free)(ful)?\b", r"\bhurts?\b", r"\bhurting\b", r"\binjur(y|ed|ies)\b",
    r"\bswollen\b", r"\bswelling\b", r"\b(sprain|sprained|strain|strained)\b", r"\bpulled (a |my )?muscle\b",
]
_NEGATION = re.compile(r"\b(no|not|never|without|nothing|none|neither|nor|free of|zero|don'?t|didn'?t|doesn'?t|isn'?t|aren'?t|haven'?t|hasn'?t)\b")
_CLAUSE_BREAK = re.compile(r"[.,;:!?]|\bbut\b|\bthough\b|\bhowever\b")

URGENT_REPLY = ("Please stop exercising now. What you describe can be a sign of something that needs to be "
                "checked by a medical professional, and I can't tell what is causing it. If it is severe, "
                "getting worse, or involves chest pain, trouble breathing, fainting or numbness, call your "
                "local emergency number or get urgent care. I won't suggest any exercise until you have "
                "been checked.")
CAUTION_REPLY = ("I'm sorry that hurts. I can't tell what is causing it, so please stop any exercise that "
                 "brings it on. If it continues, gets worse, or comes with swelling, numbness or weakness, "
                 "see a doctor or physiotherapist before training that area again.")


class SafetyAssessment(BaseModel):
    level: Level
    reply: str | None = None
    matched: list[str] = []


def _hits(patterns: list[str], text: str) -> list[str]:
    out = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            clause = _CLAUSE_BREAK.split(text[max(0, m.start() - 30):m.start()])[-1]
            if not _NEGATION.search(clause):
                out.append(m.group(0))
    return out


def assess_symptoms(text: str) -> SafetyAssessment:
    t = text.lower().replace("\u2019", "'")
    urgent = _hits(_URGENT, t)
    if urgent:
        return SafetyAssessment(level="urgent", reply=URGENT_REPLY, matched=urgent)
    caution = _hits(_CAUTION, t)
    if caution:
        return SafetyAssessment(level="caution", reply=CAUTION_REPLY, matched=caution)
    return SafetyAssessment(level="none")


_CONDITION = [
    r"\bheart (condition|disease|problem|problems|failure|attack|surgery)\b",
    r"\b(high blood pressure|hypertension|hypertensive)\b",
    r"\b(pregnant|pregnancy|postpartum)\b",
    r"\b(recovering from|just had|post[- ]?op\w*)\b.{0,25}\b(surgery|operation|injury|fracture|break)\b",
    r"\bdoctor (told|said|advised)\b.{0,40}\b(limit|avoid|not|stop|rest)\b",
]
_BYPASS = [
    r"\b(ignore|bypass|disable|override|skip|forget)\b.{0,25}\b(safety|rules|health check|screening|restrictions|limitations|guidelines)\b",
    r"\bpretend\b.{0,30}\b(health check|screening|cleared|safe)\b",
]
_MAX_EFFORT = [r"\bmax(imal|imum)?[- ]effort\b", r"\b1 ?rm test\b", r"\bone[- ]rep max test\b", r"\btest my (1 ?rm|max)\b"]

CONDITION_REPLY = ("Thanks for telling me. I can't judge whether training is safe with that, so please check with a "
                   "doctor before starting or changing a programme. I won't build or adjust a plan, or say an "
                   "exercise is safe for you, until you have their go-ahead.")
BYPASS_REPLY = ("I can't skip or switch off my safety checks. They apply to every plan, whatever the request. If you "
                "want a plan, answer the health check in your profile and I'll build it within those limits.")
MAX_EFFORT_REPLY = ("I don't prescribe max-effort tests or all-out sessions, and I can't check your readiness from "
                    "here. I can build a progressive programme that works up gradually instead.")


def assess_policy(text: str):
    """(kind, reply) for stated medical conditions, attempts to bypass safety rules, or max-effort requests; else (None, None)."""
    t = text.lower().replace("\u2019", "'")
    for kind, pats, reply in (("bypass", _BYPASS, BYPASS_REPLY), ("condition", _CONDITION, CONDITION_REPLY),
                              ("max_effort", _MAX_EFFORT, MAX_EFFORT_REPLY)):
        if _hits(pats, t):
            return kind, reply
    return None, None

_MAX_EFFORT += [
    r"\b(test|find|check|determine|establish|measure)\b.{0,20}\b(one|1)[- ]?rep[- ]?max\b",
    r"\b(max|maximal|maximum) (lift|attempt|test)\b",
    r"\ball[- ]out (test|session|workout|effort)\b",
]
