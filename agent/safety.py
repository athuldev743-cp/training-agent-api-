"""Deterministic safety gates shared by domain agents.

These checks do not diagnose. They stop requests that need a clinician or urgent
assessment before retrieval or model generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from engine.safety import assess_symptoms


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    status: str = "ok"
    message: str | None = None
    reason: str | None = None


_MEDICAL_NUTRITION = (
    (r"\b(diabetes|diabetic|blood sugar|glycemic)\b", "diabetes"),
    (r"\b(kidney disease|renal disease|kidney failure|dialysis)\b", "kidney disease"),
    (r"\b(liver disease|cirrhosis|fatty liver|hepatitis)\b", "liver disease"),
    (r"\b(cancer|chemotherapy|radiotherapy)\b", "cancer treatment"),
    (r"\b(pregnan\w*|breastfeeding|lactating|postpartum)\b", "pregnancy or breastfeeding"),
    (r"\b(eating disorder|anorexia|bulimia|binge eating|purging)\b", "eating disorder"),
    (r"\b(food allergy|allergic to|anaphylaxis|celiac disease|coeliac disease)\b", "food allergy or coeliac disease"),
)
_MEDICAL_REQUEST = re.compile(
    r"\b(treat|cure|reverse|manage|control|prevent|heal|medication|therapeutic|"
    r"diet for|meal plan for|what should i eat for|supplement dose)\b",
    re.IGNORECASE,
)
_BYPASS = re.compile(
    r"\b(ignore|bypass|disable|override|skip)\b.{0,30}\b(safety|medical|health|"
    r"screening|restriction|warning)\b",
    re.IGNORECASE,
)

URGENT_NUTRITION_REPLY = (
    "I can't assess urgent symptoms. Please seek immediate medical care or contact "
    "your local emergency number if symptoms are severe or rapidly worsening."
)
MEDICAL_NUTRITION_REPLY = (
    "I can provide general nutrition education, but I can't create a treatment diet "
    "or manage that condition. Please ask your treating clinician or registered "
    "dietitian for advice based on your medical history."
)
BYPASS_REPLY = "I can't bypass safety checks. I can still help with general, non-medical information."


def assess_nutrition_request(message: str) -> SafetyDecision:
    text = message.strip()
    if _BYPASS.search(text):
        return SafetyDecision(False, "refused", BYPASS_REPLY, "safety_bypass")

    symptoms = assess_symptoms(text)
    if symptoms.level == "urgent":
        return SafetyDecision(False, "refused", symptoms.reply or URGENT_NUTRITION_REPLY, "urgent_symptoms")

    lowered = text.lower()
    for pattern, condition in _MEDICAL_NUTRITION:
        if re.search(pattern, lowered) and _MEDICAL_REQUEST.search(lowered):
            return SafetyDecision(False, "refused", MEDICAL_NUTRITION_REPLY, condition)

    return SafetyDecision(True)
