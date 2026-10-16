"""Shapes for a proposed workout plan and the limits it must respect."""
from pydantic import BaseModel, Field


class PlannedExercise(BaseModel):
    exercise_id: str = Field(min_length=1)
    sets: int = Field(ge=1, le=10)
    reps: int = Field(ge=1, le=100)
    weight_kg: float = Field(default=0.0, ge=0)
    rest_seconds: int = Field(default=60, ge=0, le=600)


class Constraints(BaseModel):
    available_equipment: list[str] = Field(default_factory=list)
    max_minutes: int = Field(ge=5, le=180)
    max_difficulty: str | None = None
    max_exercises: int = Field(default=8, ge=1, le=20)
    max_total_sets: int = Field(default=24, ge=1, le=60)
    limitations: list[str] = Field(default_factory=list)
