#!/usr/bin/env python3
"""Reduce M0 raw reports to auditable tracked summaries, without discarding batches."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from benchmark_preprocess import STAGES, describe_ns


def summarize(paths):
    result = {"schema_version": 1, "scope": "M0 OpenCV baseline only; no MLIR speedup measured",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "measured_sample_count": 0, "reports": []}
    for path in paths:
        encoded = path.read_bytes()
        report = json.loads(encoded)
        if report["backend"] != "opencv" or report["schema_version"] != 1:
            raise ValueError("unsupported raw report: " + str(path))
        entry = {key: report[key] for key in ("config", "environment", "inputs", "methodology",
                                              "started_utc", "finished_utc")}
        entry.update({"raw_file": path.name, "raw_bytes": len(encoded),
                      "raw_sha256": hashlib.sha256(encoded).hexdigest(), "cases": []})
        for case in report["cases"]:
            raw = []
            batches = []
            for batch in case["batches"]:
                samples = batch["raw_samples"]
                if len(samples) != batch["sample_count"] or len(samples) != report["config"]["iterations"]:
                    raise ValueError("sample count mismatch")
                if sum(row["total_ns"] for row in samples) != batch["sum_total_ns"]:
                    raise ValueError("batch duration mismatch")
                if any(row["total_ns"] != sum(row[key] for key in STAGES[:-1]) for row in samples):
                    raise ValueError("stage timings do not sum to total")
                raw.extend(samples)
                batches.append({key: value for key, value in batch.items() if key != "raw_samples"})
            if len(batches) != report["config"]["batches"] or not raw:
                raise ValueError("missing measured batches")
            aggregate = {"sample_count": len(raw),
                         "stages": {key: describe_ns([row[key] for row in raw]) for key in STAGES},
                         "serial_pipeline_fps": len(raw) * 1e9 / sum(row["total_ns"] for row in raw)}
            entry["cases"].append({"mode": case["mode"], "output_shape": case["output_shape"],
                                   "reference_checksums": case["reference_checksums"],
                                   "batches": batches, "batch_summary": case["summary"], "aggregate": aggregate})
            result["measured_sample_count"] += len(raw)
        result["reports"].append(entry)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", nargs="+", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.report.exists() or not args.report.parent.is_dir():
            raise ValueError("report must be new and its parent must already exist")
        result = summarize(args.raw)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print("input -> output | mode | mean / P50 / P95 / P99 ms")
        for entry in result["reports"]:
            size = "x".join(map(str, entry["config"]["input_size"])) if entry["config"]["input_size"] else "native"
            for case in entry["cases"]:
                times = case["aggregate"]["stages"]["total_ns"]
                print("{} -> {} | {} | {}".format(size, case["output_shape"][1], case["mode"],
                      " / ".join("{:.4f}".format(times[key] / 1e6) for key in ("mean_ns", "p50_ns", "p95_ns", "p99_ns"))))
        print("Measured samples: " + str(result["measured_sample_count"]))
        return 0
    except Exception as error:
        print("summarize_preprocess_benchmark: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
