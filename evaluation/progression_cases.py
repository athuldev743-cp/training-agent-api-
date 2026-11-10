"""Check engine.progression.next_target against hand-written expected cases."""
import json
from pathlib import Path

from engine.models import SetResult, Target
from engine.progression import next_target


def run_progression_cases(path: Path) -> dict:
    cases = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    out, passed = [], 0
    for c in cases:
        d = next_target(Target(exercise="x", sets=c["planned_sets"], reps=c["planned_reps"], weight_kg=c["planned_weight_kg"]),
                        [SetResult(**r) for r in c["results"]])
        ok = d.target.reps == c["expected_reps"] and abs(d.target.weight_kg - c["expected_weight_kg"]) < 1e-9
        passed += ok
        out.append({"case_id": c["case_id"], "note": c["note"], "correct": ok,
                    "got": [d.target.reps, d.target.weight_kg], "expected": [c["expected_reps"], c["expected_weight_kg"]]})
    return {"evaluation": "progression-decisions", "total": len(cases), "passed": passed, "cases": out}
