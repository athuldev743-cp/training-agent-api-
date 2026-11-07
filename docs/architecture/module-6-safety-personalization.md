# Module 6 — Safety and Personalization

## Safety boundary

- Fitness continues to use the existing deterministic symptom triage, medical-policy checks, profile screening gate, and limitation-aware exercise filtering.
- Nutrition now has a deterministic pre-retrieval safety gate for urgent symptoms, attempts to bypass safeguards, and requests for treatment diets or condition management.
- High-risk medical nutrition requests are refused with a clinician / registered-dietitian referral before retrieval or LLM generation.
- General education about a condition remains allowed when the user is not asking for diagnosis, treatment, or a therapeutic plan.
- The gate does not diagnose, infer a condition, or create medical advice.

## Personalization boundary

The current `UserProfile` contains age, height, weight, goal, experience, equipment, limitations, and exercise screening. It does not contain allergies, dietary preferences, medications, diagnoses, or clinician instructions. Agents must not invent these fields or infer them from a message. Fitness receives the validated profile and existing limitation/screening logic; nutrition may only use explicitly supplied profile fields when relevant and must ask for missing information rather than guessing.

## Validation

`tests/test_agent_safety.py` validates medical-condition triage, urgent symptom handling, safety-bypass refusal, and allowed general education. `tests/test_nutrition_agent.py` validates that blocked requests do not call retrieval or the LLM.
