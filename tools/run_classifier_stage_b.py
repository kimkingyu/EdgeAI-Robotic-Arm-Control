#!/usr/bin/env python3
"""Stage B: audited M4 FP32 input, serial CPU quantization/evaluation and static reports.

Use a fresh directory. --resume validates the same run rather than retrying
failed jobs or mixing measurement rounds. QB drift groups alone permit one
predeclared whole-group retry. Frozen pipeline/RSS observations use separate
workers and are not inference microbenchmark samples. No data are downloaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _positive(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("positive finite integer required") from exc
    if not 1 <= value <= 86400:
        raise argparse.ArgumentTypeError("timeout must be between 1 and 86400 seconds")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Prepared schema1 manifest; calibration256/dev128/holdout1000")
    parser.add_argument("--fp32-run", type=Path, help="Complete M4 --mode formal output directory or run.json")
    parser.add_argument("--output-dir", type=Path, help="New Stage B output directory; parent must exist")
    parser.add_argument("--resume", action="store_true", help="Strict SHA/config/code/environment matched continuation; no hidden retries")
    parser.add_argument("--allow-fp32-failures", action="store_true", help="Expose genuine failed M4 configurations; never bypass corruption or missing results")
    parser.add_argument("--calibration-timeout-s", type=_positive, default=900)
    parser.add_argument("--worker-timeout-s", type=_positive, default=900)
    parser.add_argument("--holdout-timeout-s", type=_positive, default=900,
                        help="Explicit finite holdout budget (seconds); same value required on resume")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        from src.vision.classification_stage_b import run_stage_b, worker
        if args.worker:
            if args.job is None:
                parser.error("--worker requires --job")
            return worker(args.job)
        if args.job is not None:
            parser.error("--job is reserved for --worker")
        if args.manifest is None or args.fp32_run is None or args.output_dir is None:
            parser.error("--manifest, --fp32-run and --output-dir are required")
        summary = run_stage_b(args.manifest, args.fp32_run, args.output_dir, resume=args.resume,
                              allow_fp32_failures=args.allow_fp32_failures, seed=args.seed,
                              calibration_timeout_s=args.calibration_timeout_s,
                              worker_timeout_s=args.worker_timeout_s, holdout_timeout_s=args.holdout_timeout_s)
        print(f"Stage B: {summary['status']}")
        print(f"Report: {args.output_dir.resolve() / 'report.html'}")
        for model, result in summary.get("final", {}).items():
            print(f"{model}: {result['status']} (no holdout reselection)")
        return 0 if summary["status"] == "completed" else 2
    except KeyboardInterrupt:
        print("Stage B interrupted; immutable job evidence retained", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
