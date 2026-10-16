"""Exercise knowledge base: typed records loaded from data/exercises.json,
plus body-area stress labels from data/exercise_stress.json."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from state.profile import KNOWN_LIMITATIONS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "exercises.json"
STRESS_PATH = Path(__file__).resolve().parent.parent / "data" / "exercise_stress.json"


class Exercise(BaseModel):
    id: str
    name: str
    muscles: list[str]  # every muscle the exercise works
    primary: list[str]  # the muscles it is mainly for (a subset of muscles)
    equipment: list[str]
    difficulty: Literal["beginner", "intermediate", "advanced"]
    movement: str
    stresses: list[str] = Field(default_factory=list)  # body areas it loads heavily
    alternatives: list[str] = Field(default_factory=list)
    progression: str | None = None
    regression: str | None = None


def load_exercises(path: Path = DATA_PATH, stress_path: Path = STRESS_PATH) -> list[Exercise]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tags = {}
    if Path(stress_path).exists():
        tags = json.loads(Path(stress_path).read_text(encoding="utf-8"))
    return [Exercise(**{**item, "stresses": tags.get(item["id"], [])}) for item in raw]


def safe_exercises(exercises: list[Exercise], limitations: list[str] | None) -> list[Exercise]:
    """Drops every exercise that loads a body area the user has a limitation on."""
    if not limitations:
        return list(exercises)
    avoid = set(limitations)
    return [e for e in exercises if not (set(e.stresses) & avoid)]


def check_references(exercises: list[Exercise]) -> list[str]:
    """Return a list of problems; an empty list means the data is consistent."""
    known = {e.id for e in exercises}
    problems = []
    for e in exercises:
        links = list(e.alternatives) + [e.progression, e.regression]
        for ref in links:
            if ref is not None and ref not in known:
                problems.append(f"{e.id} points to unknown exercise '{ref}'")
        if not e.primary:
            problems.append(f"{e.id} has no primary muscle")
        for m in e.primary:
            if m not in e.muscles:
                problems.append(f"{e.id} primary '{m}' is not in its muscles")
        for s in e.stresses:
            if s not in KNOWN_LIMITATIONS:
                problems.append(f"{e.id} has unknown stress area '{s}'")
    return problems
