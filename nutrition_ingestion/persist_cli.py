"""CLI for embedding and persisting a Module 2 nutrition JSONL artifact."""
import argparse
import json
from pathlib import Path

from nutrition_ingestion.store import persist_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings and persist nutrition chunks to Neon."
    )
    parser.add_argument("--input", type=Path, required=True, help="Module 2 nutrition JSONL file")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = persist_jsonl(args.input, batch_size=args.batch_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
