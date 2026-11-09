"""CLI for Module 7 fitness and nutrition retrieval evaluation."""
import argparse
import json
from pathlib import Path

from evaluation.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fitness or nutrition retrieval against reviewed chunk IDs."
    )
    parser.add_argument("--domain", choices=("fitness", "nutrition"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = run_benchmark(
        args.domain, args.dataset, args.output, k=args.k
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
