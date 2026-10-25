"""Per-exercise sets and rep ranges by level and goal (deterministic starting defaults, not medical advice).
The base table is the general and muscle goals: loaded reps follow the ACSM novice (8-12) and intermediate (6-12)
guidance, bodyweight ranges centre on about 8 / 12 / 15 reps, and rest follows the ACSM summaries of 2-3 minutes for
multi-joint and 1-2 minutes for single-joint lifts (summaries of the 2009 position stand differ slightly).
GOAL_OVERRIDES change reps and rest for strength, fat_loss and endurance. Those numbers are this project's own
drafts (only the direction for endurance, more than 15 reps with under 90 s rest, follows the ACSM summary)
and need review by a qualified trainer."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "data" / "exercise_prescription.json"
LEVELS = ("beginner", "intermediate", "advanced")
GOALS = ("general", "muscle", "strength", "fat_loss", "endurance")

# template -> (unit, {level: (rep_min, rep_max, sets_start, sets_max, rest_seconds)})
TEMPLATES = {
    "loaded_compound": ("reps", {
        "beginner": (8, 12, 2, 3, 90),
        "intermediate": (6, 12, 3, 4, 120),
        "advanced": (5, 10, 4, 5, 150),
    }),
    "bodyweight_compound": ("reps", {
        "beginner": (6, 10, 2, 3, 60),
        "intermediate": (10, 14, 3, 4, 60),
        "advanced": (12, 18, 3, 5, 60),
    }),
    "isolation": ("reps", {
        "beginner": (10, 15, 2, 3, 60),
        "intermediate": (10, 15, 3, 4, 60),
        "advanced": (8, 15, 3, 4, 60),
    }),
    "core_reps": ("reps", {
        "beginner": (10, 15, 2, 3, 45),
        "intermediate": (12, 20, 3, 4, 45),
        "advanced": (15, 25, 3, 4, 45),
    }),
    "hold": ("seconds", {
        "beginner": (20, 30, 2, 3, 45),
        "intermediate": (30, 45, 3, 4, 45),
        "advanced": (45, 60, 3, 4, 45),
    }),
    "hard_bodyweight": ("reps", {
        "beginner": (3, 5, 2, 3, 120),
        "intermediate": (4, 8, 3, 4, 120),
        "advanced": (6, 12, 4, 5, 120),
    }),
}

# goal -> template -> level -> (rep_min, rep_max, sets_start, sets_max, rest_seconds). Missing entries use the base table.
GOAL_OVERRIDES = {
    "strength": {
        "loaded_compound": {
            "beginner": (6, 10, 2, 3, 120),
            "intermediate": (4, 8, 3, 4, 150),
            "advanced": (3, 6, 4, 5, 180),
        },
        "bodyweight_compound": {
            "beginner": (5, 8, 2, 3, 90),
            "intermediate": (6, 10, 3, 4, 90),
            "advanced": (8, 12, 3, 5, 90),
        },
    },
    "fat_loss": {
        "loaded_compound": {
            "beginner": (10, 15, 2, 3, 60),
            "intermediate": (10, 15, 3, 4, 60),
            "advanced": (10, 15, 3, 5, 60),
        },
        "bodyweight_compound": {
            "beginner": (8, 12, 2, 3, 45),
            "intermediate": (12, 16, 3, 4, 45),
            "advanced": (15, 20, 3, 5, 45),
        },
        "isolation": {
            "beginner": (12, 18, 2, 3, 45),
            "intermediate": (12, 18, 3, 4, 45),
            "advanced": (12, 18, 3, 4, 45),
        },
    },
    "endurance": {
        "loaded_compound": {
            "beginner": (12, 18, 2, 3, 45),
            "intermediate": (15, 20, 3, 4, 45),
            "advanced": (15, 25, 3, 4, 45),
        },
        "bodyweight_compound": {
            "beginner": (10, 15, 2, 3, 45),
            "intermediate": (15, 20, 3, 4, 45),
            "advanced": (20, 30, 3, 5, 30),
        },
        "isolation": {
            "beginner": (15, 20, 2, 3, 30),
            "intermediate": (15, 25, 3, 4, 30),
            "advanced": (20, 30, 3, 4, 30),
        },
        "core_reps": {
            "beginner": (15, 20, 2, 3, 30),
            "intermediate": (20, 30, 3, 4, 30),
            "advanced": (25, 40, 3, 4, 30),
        },
    },
}


TEMPLATES["conditioning"] = ("seconds", {"beginner": (20, 30, 2, 3, 40), "intermediate": (30, 40, 3, 4, 30), "advanced": (30, 45, 3, 5, 30)})
TEMPLATES["plyometric"] = ("reps", {"beginner": (5, 8, 2, 3, 60), "intermediate": (6, 10, 3, 4, 60), "advanced": (6, 12, 3, 5, 75)})
TEMPLATES["skill_bodyweight"] = ("reps", {"beginner": (2, 4, 2, 3, 120), "intermediate": (3, 6, 3, 4, 120), "advanced": (4, 8, 3, 5, 120)})
GOAL_OVERRIDES["endurance"]["conditioning"] = {"beginner": (30, 40, 2, 3, 30), "intermediate": (40, 50, 3, 4, 20), "advanced": (45, 60, 3, 5, 20)}
GOAL_OVERRIDES["fat_loss"]["conditioning"] = {"beginner": (30, 40, 2, 3, 30), "intermediate": (40, 45, 3, 4, 25), "advanced": (40, 50, 3, 5, 20)}
GOAL_OVERRIDES["strength"]["conditioning"] = {"beginner": (15, 20, 2, 3, 60), "intermediate": (15, 20, 3, 4, 60), "advanced": (15, 20, 3, 4, 60)}


TEMPLATES["cardio"] = ("seconds", {"beginner": (90, 150, 2, 3, 60), "intermediate": (120, 180, 2, 3, 60), "advanced": (150, 210, 2, 3, 60)})
GOAL_OVERRIDES["endurance"]["cardio"] = {"beginner": (150, 210, 2, 3, 45), "intermediate": (210, 270, 2, 3, 45), "advanced": (270, 360, 2, 3, 45)}
GOAL_OVERRIDES["fat_loss"]["cardio"] = {"beginner": (60, 90, 2, 3, 60), "intermediate": (90, 120, 2, 3, 60), "advanced": (90, 150, 2, 3, 60)}


TEMPLATES["loaded_unilateral"] = ("reps", {"beginner": (8, 12, 2, 3, 90), "intermediate": (8, 12, 3, 4, 90), "advanced": (6, 10, 3, 5, 120)})
GOAL_OVERRIDES["strength"]["loaded_unilateral"] = {"beginner": (6, 10, 2, 3, 90), "intermediate": (6, 10, 3, 4, 120), "advanced": (5, 8, 3, 5, 150)}
GOAL_OVERRIDES["fat_loss"]["loaded_unilateral"] = {"beginner": (10, 15, 2, 3, 60), "intermediate": (10, 15, 3, 4, 60), "advanced": (10, 15, 3, 5, 60)}
GOAL_OVERRIDES["endurance"]["loaded_unilateral"] = {"beginner": (12, 18, 2, 3, 45), "intermediate": (15, 20, 3, 4, 45), "advanced": (15, 20, 3, 4, 45)}


class Prescription(BaseModel):
    template: str
    unit: Literal["reps", "seconds"]
    rep_min: int
    rep_max: int
    sets_start: int  # where a new program begins
    sets_max: int  # where later weeks can progress to
    rest_seconds: int


def load_templates(path: Path = TEMPLATE_PATH) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prescription_for(exercise_id: str, level: str, templates: dict[str, str] | None = None,
                     goal: str = "general") -> Prescription:
    """Sets, rep range and rest for an exercise at a level and goal. An unknown goal behaves like general."""
    if templates is None:
        templates = load_templates()
    name = templates[exercise_id]  # KeyError if an exercise has no template (a test guards this)
    unit, levels = TEMPLATES[name]
    override = GOAL_OVERRIDES.get(goal, {}).get(name, {})
    rep_min, rep_max, sets_start, sets_max, rest = override.get(level, levels[level])
    return Prescription(template=name, unit=unit, rep_min=rep_min, rep_max=rep_max,
                        sets_start=sets_start, sets_max=sets_max, rest_seconds=rest)


def format_prescription(p: Prescription, sets: int | None = None) -> str:
    sets = sets or p.sets_start
    suffix = " s" if p.unit == "seconds" else ""
    return f"{sets} sets of {p.rep_min}-{p.rep_max}{suffix}, rest {p.rest_seconds} s"
