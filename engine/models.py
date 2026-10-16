"""Core data shapes for the fitness engine. Pydantic rejects invalid values."""
from pydantic import BaseModel, Field


class SetResult(BaseModel):
    """What actually happened in one set."""
    reps: int = Field(ge=0, le=100)
    rpe: float = Field(ge=1, le=10)
    completed: bool = True


class Target(BaseModel):
    """What was planned (or what is planned next) for one exercise."""
    exercise: str = Field(min_length=1)
    sets: int = Field(ge=1, le=10)
    reps: int = Field(ge=1, le=100)
    weight_kg: float = Field(default=0.0, ge=0)


class Decision(BaseModel):
    """The engine's answer: the next target plus a human-readable reason."""
    target: Target
    reason: str
