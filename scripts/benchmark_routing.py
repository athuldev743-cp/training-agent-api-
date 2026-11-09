"""CLI for evaluating the deterministic multi-agent request classifier."""
import argparse
import json
from pathlib import Path

from evaluation.routing import run_routing_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate request routing against a manually reviewed JSONL dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_routing_benchmark(args.dataset, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
