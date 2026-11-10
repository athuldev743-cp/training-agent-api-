"""CLI: evaluate the deterministic safety layer."""
import argparse
from pathlib import Path

from evaluation.safety import run_safety_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run_safety_benchmark(a.dataset, a.output)
    print(f"passed {r['passed']}/{r['total']} ({r['pass_rate']:.1%}); critical failures: {len(r['critical_failures'])}")
    for k, v in r["per_category"].items():
        print(f"  {k}: {v['passed']}/{v['total']}")
    for c in r["per_case"]:
        if not c["passed"]:
            print("  FAIL", c["case_id"], c["category"], "|", c["message"], "|", c["problems"])


if __name__ == "__main__":
    main()
