"""Deterministic progression rules. The LLM never does this math."""
from statistics import mean

from engine.models import Decision, SetResult, Target

REP_FLOOR = 8
REP_CEILING = 15
WEIGHT_STEP_KG = 2.5


def next_target(planned: Target, results: list[SetResult]) -> Decision:
    if not results:
        raise ValueError("results must contain at least one set")

    all_done = len(results) >= planned.sets and all(r.completed for r in results)
    avg_rpe = mean(r.rpe for r in results)

    # Too hard, or sets were missed: step back.
    if not all_done or avg_rpe > 9:
        if planned.reps <= REP_FLOOR and planned.weight_kg > 0:
            new_weight = max(0.0, planned.weight_kg - WEIGHT_STEP_KG)
            return Decision(
                target=planned.model_copy(update={"weight_kg": new_weight}),
                reason=f"Too hard at the rep floor: weight down {WEIGHT_STEP_KG} kg.",
            )
        new_reps = max(REP_FLOOR, planned.reps - 2)
        return Decision(
            target=planned.model_copy(update={"reps": new_reps}),
            reason=("Too hard or sets missed: reps down 2." if new_reps != planned.reps else "Already at the lowest rep target with no weight to cut: hold here and consider an easier variation."),
        )

    # Hard but doable: repeat.
    if avg_rpe > 8:
        return Decision(target=planned, reason="Hard but doable: hold the same target.")

    # Room to grow: add reps.
    add = 2 if avg_rpe <= 6 else 1
    new_reps = planned.reps + add

    if new_reps <= REP_CEILING:
        return Decision(
            target=planned.model_copy(update={"reps": new_reps}),
            reason=f"Felt easy: reps up {add}.",
        )

    # Passed the rep ceiling.
    if planned.weight_kg > 0:
        return Decision(
            target=planned.model_copy(
                update={"reps": REP_FLOOR, "weight_kg": planned.weight_kg + WEIGHT_STEP_KG}
            ),
            reason=f"Hit the rep ceiling: weight up {WEIGHT_STEP_KG} kg, reps reset.",
        )
    return Decision(
        target=planned.model_copy(update={"reps": REP_CEILING}),
        reason="Hit the rep ceiling with bodyweight: move to a harder variation.",
    )
