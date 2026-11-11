"""Check generated 4-week programmes against hard constraints. All rules are derived from the exercise library and profile, not from model labels."""
from __future__ import annotations

import datetime
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from engine.prescription import GOALS
from engine.program import DEFAULT_DAYS, build_program
from rag.search import DIFFICULTY_RANK
from rag.store import load_exercises
from state.profile import UserProfile

CATEGORIES = ("equipment", "difficulty", "limitations", "exercise_reference",
              "session_count", "four_week_structure", "prescription", "set_progression")
EQUIPMENT_SETS = [None, [], ["dumbbells"], ["dumbbells", "bench"],
                  ["dumbbells", "bench", "pull_up_bar", "row_bar"], ["full_gym"]]
LIMITATION_SETS = [[], ["knee"], ["shoulder"], ["lower_back"], ["wrist", "elbow"], ["knee", "shoulder"]]
LEVELS = ("beginner", "intermediate", "advanced")
GUARDRAIL_CASES = [
    {"case_id": "guard-001", "profile": {"name": "g", "screening": "not_asked"}, "expect_status": "blocked_health"},
    {"case_id": "guard-002", "profile": {"name": "g", "screening": "needs_clearance"}, "expect_status": "blocked_health"},
    {"case_id": "guard-003", "profile": {"name": "g", "screening": "clear", "height_cm": 150, "weight_kg": 300}, "expect_status": "needs_info"},
    {"case_id": "guard-004", "profile": {"name": "g", "screening": "clear", "age": 14, "experience": "advanced"}, "expect_status": "needs_info"},
]


def generate_profiles(n: int = 100, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        goals = rng.sample(list(GOALS), rng.choice([1, 1, 2]))
        rows.append({
            "case_id": f"wk-{i:03d}",
            "profile": {
                "name": "eval", "screening": "clear", "experience": rng.choice(LEVELS),
                "equipment": rng.choice(EQUIPMENT_SETS), "limitations": rng.choice(LIMITATION_SETS),
                "goal": goals[0], "session_minutes": rng.choice([30, 45, 60, 90]),
            },
            "days": rng.choice([None, 2, 3, 4, 5, 6]),
            "goals": goals,
            "expect_status": "ok",
        })
    return rows + GUARDRAIL_CASES


def check_program(program, profile: UserProfile, requested_days, by_id) -> dict[str, list[str]]:
    v: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    have = set(profile.equipment or []) | {"bodyweight"}
    rank = DIFFICULTY_RANK[profile.experience]
    limits = set(profile.limitations)
    last_sets: dict[str, int] = {}
    for wk in program.weeks:
        for s in wk.sessions:
            ids = [e.exercise_id for e in s.exercises]
            if not ids or len(ids) != len(set(ids)):
                v["four_week_structure"].append(f"w{wk.number} d{s.day_number}: empty or duplicate exercises")
            for e in s.exercises:
                ex = by_id.get(e.exercise_id)
                if ex is None:
                    v["exercise_reference"].append(f"unknown id {e.exercise_id}")
                    continue
                if not set(ex.equipment) <= have:
                    v["equipment"].append(f"{ex.id} needs {sorted(set(ex.equipment) - have)}")
                if DIFFICULTY_RANK[ex.difficulty] > rank:
                    v["difficulty"].append(f"{ex.id} is {ex.difficulty} for {profile.experience}")
                if limits & set(ex.stresses or []):
                    v["limitations"].append(f"{ex.id} stresses {sorted(limits & set(ex.stresses))}")
                if not (e.sets >= 1 and e.rep_min >= 1 and e.rep_min <= e.aim_low <= e.aim_high <= e.rep_max and e.rest_seconds >= 0):
                    v["prescription"].append(f"{e.exercise_id} w{wk.number}: sets={e.sets} rep {e.rep_min}-{e.rep_max} aim {e.aim_low}-{e.aim_high} rest {e.rest_seconds}")
                if e.exercise_id in last_sets and e.sets < last_sets[e.exercise_id]:
                    v["set_progression"].append(f"{e.exercise_id}: sets fell to {e.sets} in week {wk.number}")
                last_sets[e.exercise_id] = e.sets
    expected_days = max(2, min(6, requested_days if requested_days else DEFAULT_DAYS[profile.experience]))
    if program.days_per_week != expected_days:
        v["session_count"].append(f"days_per_week {program.days_per_week} != expected {expected_days}")
    for wk in program.weeks:
        if len(wk.sessions) != expected_days:
            v["session_count"].append(f"week {wk.number} has {len(wk.sessions)} sessions")
        if len({s.weekday for s in wk.sessions}) != len(wk.sessions):
            v["session_count"].append(f"week {wk.number} repeats a weekday")
    if [w.number for w in program.weeks] != [1, 2, 3, 4]:
        v["four_week_structure"].append("weeks are not numbered 1-4")
    if [w.phase for w in program.weeks] != ["foundation", "foundation", "build", "build"]:
        v["four_week_structure"].append("phases are not foundation, foundation, build, build")
    return v


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_workout_benchmark(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    exercises = load_exercises()
    by_id = {e.id: e for e in exercises}
    cases = [json.loads(x) for x in Path(dataset_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    cat_fail = {c: 0 for c in CATEGORIES}
    status_counts: dict[str, int] = {}
    clean = generated = guard_total = guard_pass = 0
    no_plan, failures = [], []
    for c in cases:
        profile = UserProfile(**c["profile"])
        result = build_program(exercises, profile, c.get("days"), c.get("goals"))
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if c["expect_status"] != "ok":
            guard_total += 1
            guard_pass += result["status"] == c["expect_status"]
            continue
        if result["status"] != "ok":
            no_plan.append({"case_id": c["case_id"], "status": result["status"]})
            continue
        generated += 1
        viol = check_program(result["program"], profile, c.get("days"), by_id)
        bad = {k: x for k, x in viol.items() if x}
        for k in bad:
            cat_fail[k] += 1
        if bad:
            failures.append({"case_id": c["case_id"], "profile": c["profile"], "days": c.get("days"), "violations": {k: x[:3] for k, x in bad.items()}})
        else:
            clean += 1
    report = {
        "evaluation": "workout-constraints",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset": str(dataset_path),
        "library_exercise_count": len(exercises),
        "profiles_total": len(cases) - guard_total,
        "plans_generated": generated,
        "no_plan": no_plan,
        "profiles_with_zero_hard_violations": clean,
        "zero_violation_rate_of_generated": clean / generated if generated else None,
        "per_category_pass": {c: {"passed": generated - cat_fail[c], "total": generated} for c in CATEGORIES},
        "guardrail_cases": {"correct_status": guard_pass, "total": guard_total},
        "status_counts": status_counts,
        "failures": failures,
        "note": "Constraints are rule-derived from the exercise library and profile. Progression decisions (engine/progression.py) are evaluated separately.",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
