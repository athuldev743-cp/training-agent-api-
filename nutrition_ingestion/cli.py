"""CLI entry point for nutrition document ingestion."""
import argparse
import json
from pathlib import Path
from nutrition_ingestion.pipeline import ingest_directory

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and chunk nutrition knowledge documents.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()
    report = ingest_directory(args.input, args.output, max_chars=args.max_chars, overlap=args.overlap)
    print(json.dumps(report.model_dump(mode="json"), indent=2))

if __name__ == "__main__":
    main()
