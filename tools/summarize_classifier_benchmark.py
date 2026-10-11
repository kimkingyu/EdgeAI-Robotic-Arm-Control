#!/usr/bin/env python3
"""Generate offline Stage-A JSON/CSV/HTML reports without running models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Existing directory containing run.json and results/*.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace the three existing report files")
    args = parser.parse_args(argv)
    # Keep --help usable even when optional vision/framework packages are absent.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.vision.classification_benchmark import BenchmarkError, aggregate_results, write_report

    try:
        summary = aggregate_results(args.run_dir)
        paths = write_report(args.run_dir, summary, overwrite=args.overwrite)
    except (BenchmarkError, OSError) as error:
        print(f"summarize_classifier_benchmark: {error}", file=sys.stderr)
        return 2
    print(f"{summary['run_id']}: {summary['status']}; {summary['configuration_count']} configurations")
    for key, value in paths.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
