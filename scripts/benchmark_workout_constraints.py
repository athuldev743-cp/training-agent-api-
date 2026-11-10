"""CLI: evaluate generated programmes against hard constraints."""
import argparse
import json
from pathlib import Path

from evaluation.workout_constraints import generate_profiles, run_workout_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--generate", action="store_true", help="(re)write the seeded synthetic dataset first")
    a = p.parse_args()
    if a.generate:
        a.dataset.parent.mkdir(parents=True, exist_ok=True)
        a.dataset.write_text("\n".join(json.dumps(r) for r in generate_profiles()) + "\n", encoding="utf-8")
    r = run_workout_benchmark(a.dataset, a.output)
    print(f"profiles={r['profiles_total']} generated={r['plans_generated']} no_plan={len(r['no_plan'])} "
          f"zero_violation={r['profiles_with_zero_hard_violations']} rate={r['zero_violation_rate_of_generated']}")
    for k, v in r["per_category_pass"].items():
        print(f"  {k}: {v['passed']}/{v['total']}")
    print("guardrails:", r["guardrail_cases"], "statuses:", r["status_counts"])
    for f in r["failures"][:8]:
        print("FAIL", f["case_id"], f["profile"], f["days"], f["violations"])
    for n in r["no_plan"][:8]:
        print("NO_PLAN", n)


if __name__ == "__main__":
    main()
