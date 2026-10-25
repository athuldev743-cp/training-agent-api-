"""User profile model: who the user is, what they can train with, and health limits."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

GYM_EQUIPMENT = {"barbell", "squat_rack", "cable", "machine"}
KNOWN_EQUIPMENT = {"bodyweight", "dumbbells", "bench", "pull_up_bar", "row_bar", "cardio_machine"} | GYM_EQUIPMENT | {"full_gym"}  # full_gym = everything
KNOWN_LIMITATIONS = {"knee", "lower_back", "shoulder", "wrist", "elbow"}

# not_asked / needs_clearance block workout plans; clear / cleared_by_doctor allow them.
Screening = Literal["not_asked", "clear", "cleared_by_doctor", "needs_clearance"]


class UserProfile(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    age: int | None = Field(default=None, ge=10, le=100)
    height_cm: float | None = Field(default=None, ge=100, le=250)
    weight_kg: float | None = Field(default=None, ge=30, le=300)
    goal: Literal["strength", "muscle", "fat_loss", "endurance", "general"] = "general"
    experience: Literal["beginner", "intermediate", "advanced"] = "beginner"
    # None = not asked yet (no filter). [] = owns nothing (bodyweight only).
    equipment: list[str] | None = None
    session_minutes: int = Field(default=45, ge=5, le=180)
    limitations: list[str] = Field(default_factory=list)  # body areas to avoid loading
    screening: Screening = "not_asked"

    @field_validator("equipment")
    @classmethod
    def _check_equipment(cls, value):
        if value is None:
            return None
        cleaned = list(dict.fromkeys(item.strip().lower() for item in value))
        unknown = set(cleaned) - KNOWN_EQUIPMENT
        if unknown:
            raise ValueError(f"unknown equipment: {sorted(unknown)}")
        # Bodyweight is always available, so it is not stored as owned equipment.
        if "full_gym" in cleaned:  # a gym has everything: expand, but keep the marker so the app can show one tick
            cleaned = list(dict.fromkeys(cleaned + sorted(KNOWN_EQUIPMENT - {"bodyweight", "full_gym"})))
        return [item for item in cleaned if item != "bodyweight"]

    @field_validator("limitations")
    @classmethod
    def _check_limitations(cls, value):
        cleaned = list(dict.fromkeys(item.strip().lower() for item in value))
        unknown = set(cleaned) - KNOWN_LIMITATIONS
        if unknown:
            raise ValueError(f"unknown limitation: {sorted(unknown)}")
        return cleaned
