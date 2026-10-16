"""Deterministic plan validator: the LLM proposes, this decides PASS or REVISE."""
from pydantic import BaseModel

from engine.plan import Constraints, PlannedExercise
from rag.search import DIFFICULTY_RANK
from rag.store import Exercise

SECONDS_PER_REP = 4


class Violation(BaseModel):
    code: str
    message: str
    exercise_id: str | None = None


class ValidationResult(BaseModel):
    passed: bool
    violations: list[Violation]
    estimated_minutes: float


def estimate_minutes(plan: list[PlannedExercise]) -> float:
    seconds = sum(p.sets * (p.reps * SECONDS_PER_REP + p.rest_seconds) for p in plan)
    return round(seconds / 60, 1)


def validate_plan(
    plan: list[PlannedExercise],
    constraints: Constraints,
    exercises: list[Exercise],
) -> ValidationResult:
    if not plan:
        return ValidationResult(
            passed=False,
            violations=[Violation(code="EMPTY_PLAN", message="The plan has no exercises.")],
            estimated_minutes=0.0,
        )

    by_id = {e.id: e for e in exercises}
    allowed = {e.lower() for e in constraints.available_equipment} | {"bodyweight"}
    violations: list[Violation] = []
    seen: set[str] = set()

    for item in plan:
        if item.exercise_id in seen:
            violations.append(Violation(
                code="DUPLICATE_EXERCISE",
                message=f"'{item.exercise_id}' appears more than once.",
                exercise_id=item.exercise_id,
            ))
        seen.add(item.exercise_id)

        ex = by_id.get(item.exercise_id)
        if ex is None:
            violations.append(Violation(
                code="UNKNOWN_EXERCISE",
                message=f"'{item.exercise_id}' is not in the knowledge base.",
                exercise_id=item.exercise_id,
            ))
            continue

        missing = sorted(set(ex.equipment) - allowed)
        if missing:
            violations.append(Violation(
                code="EQUIPMENT_MISSING",
                message=f"{ex.name} needs equipment the user lacks: {', '.join(missing)}.",
                exercise_id=ex.id,
            ))

        if constraints.limitations and set(ex.stresses or []) & set(constraints.limitations):
            violations.append(Violation(
                code="LIMITATION_VIOLATED",
                message=f"{ex.name} loads an area the user limits: {', '.join(constraints.limitations)}.",
                exercise_id=ex.id,
            ))

        if (
            constraints.max_difficulty
            and DIFFICULTY_RANK[ex.difficulty] > DIFFICULTY_RANK[constraints.max_difficulty]
        ):
            violations.append(Violation(
                code="DIFFICULTY_TOO_HIGH",
                message=f"{ex.name} is {ex.difficulty}; the limit is {constraints.max_difficulty}.",
                exercise_id=ex.id,
            ))

    if len(plan) > constraints.max_exercises:
        violations.append(Violation(
            code="TOO_MANY_EXERCISES",
            message=f"{len(plan)} exercises; the limit is {constraints.max_exercises}.",
        ))

    total_sets = sum(p.sets for p in plan)
    if total_sets > constraints.max_total_sets:
        violations.append(Violation(
            code="TOO_MANY_SETS",
            message=f"{total_sets} total sets; the limit is {constraints.max_total_sets}.",
        ))

    minutes = estimate_minutes(plan)
    if minutes > constraints.max_minutes:
        violations.append(Violation(
            code="TOO_LONG",
            message=f"Estimated {minutes} min; the budget is {constraints.max_minutes} min.",
        ))

    return ValidationResult(
        passed=not violations,
        violations=violations,
        estimated_minutes=minutes,
    )
