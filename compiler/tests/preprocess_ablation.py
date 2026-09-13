#!/usr/bin/env python3
"""M4 ablation: time real MLIR kernels against the real OpenCV baseline.

Every timed call validates its own output, so an empty or wrong kernel cannot
look fast. Variants are interleaved per repetition to spread thermal drift.
"""
import argparse
from datetime import datetime, timezone
import ctypes as C
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import cv2
from preprocess_reference import reference_resize_rgb
from preprocess_aot_checks import Kernel, require, sha


def describe(samples):
    ordered = sorted(samples)
    def percentile(q):
        rank = (len(ordered) - 1) * q
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return {"count": len(ordered), "mean_ns": statistics.mean(ordered),
            "min_ns": ordered[0], "p50_ns": percentile(.5), "p95_ns": percentile(.95),
            "p99_ns": percentile(.99), "max_ns": ordered[-1]}


def sysfs():
    snapshot = {}
    for name, pattern in (("thermal_millidegrees", "/sys/class/thermal/thermal_zone*/temp"),
                          ("frequency_khz", "/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq")):
        values = {}
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            try:
                values[str(path)] = path.read_text().strip()
            except OSError as error:
                values[str(path)] = "unavailable: " + str(error)
        snapshot[name] = values
    return snapshot


class OpenCVBaseline:
    """Same two steps as the existing pipeline: resize then BGR->RGB."""
    variant = "opencv_baseline"

    def __init__(self, height, width):
        self.height, self.width = height, width

    def run_once(self, image):
        start = time.perf_counter_ns()
        resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        elapsed = time.perf_counter_ns() - start
        return elapsed, rgb


class MlirKernel:
    def __init__(self, library, height, width, variant):
        self.kernel = Kernel(library, height, width, variant)
        self.variant = variant
        self.height, self.width = height, width
        self.contexts = {}
        self.output = np.empty((height, width, 3), dtype=np.uint8)

    def context(self, shape):
        if shape not in self.contexts:
            self.contexts[shape] = self.kernel.context(*shape)
        return self.contexts[shape]

    def run_once(self, image):
        h, w = image.shape[:2]
        source = np.ascontiguousarray(image)
        ctx = self.context((h, w))
        start = time.perf_counter_ns()
        status = self.kernel.run(ctx, source.ctypes.data, source.nbytes, w * 3,
                                 self.output.ctypes.data, self.output.nbytes, self.width * 3)
        elapsed = time.perf_counter_ns() - start
        require(status == 0, "kernel returned " + str(status))
        return elapsed, self.output

    def close(self):
        for ctx in self.contexts.values():
            self.kernel.destroy(ctx)
        self.contexts.clear()


def measure(args):
    require(platform.machine() == "aarch64", "run on the board")
    require(not args.report.exists() and args.report.parent.is_dir(), "report must be new")
    manifest = json.loads(args.manifest.read_text())
    require(manifest["status"] == "built", "incomplete build manifest")
    base = args.manifest.parent
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    files = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:8]
    require(len(files) == 8, "8 real images required")
    images, provenance = [], []
    for path in files:
        encoded = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(decoded is not None, "cannot decode " + str(path))
        image = np.ascontiguousarray(cv2.resize(decoded, (args.input[1], args.input[0]),
                                               interpolation=cv2.INTER_LINEAR))
        images.append(image)
        provenance.append({"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(),
                           "decoded_shape": list(decoded.shape), "benchmark_shape": list(image.shape)})
    result = {"schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "single-threaded CPU preprocessing latency; no model accuracy or end-to-end claim",
              "input_shape": list(args.input), "profile": list(args.profile),
              "warmup": args.warmup, "iterations": args.iterations, "images": provenance,
              "manifest_sha256": sha(args.manifest), "script_sha256": sha(Path(__file__)),
              "opencv": cv2.__version__, "numpy": np.__version__, "python": sys.version,
              "methodology": ["Each timed call validates its full output against the Q11 reference.",
                              "Variants are interleaved within every repetition, not run in blocks.",
                              "Timing covers the preprocessing call only: no decode, no allocation of the output.",
                              "OpenCV baseline is the existing resize + cvtColor path, single-threaded.",
                              "Process affinity is set by the caller (taskset); no governor or sysfs writes."],
              "sysfs_before": sysfs(), "variants": []}
    height, width = args.profile
    expected = [reference_resize_rgb(image, height, width) for image in images]
    runners = [OpenCVBaseline(height, width)]
    try:
        for item in manifest["variants"]:
            if item.get("status") != "built" or tuple(item["profile"]) != tuple(args.profile):
                continue
            runners.append(MlirKernel(base / item["library"], height, width, item["variant"]))
        require(len(runners) > 1, "no kernels for this profile")
        samples = {runner.variant: [] for runner in runners}
        mismatch = {runner.variant: 0 for runner in runners}
        for index in range(args.warmup + args.iterations):
            image = images[index % len(images)]
            reference = expected[index % len(images)]
            for runner in runners:
                elapsed, output = runner.run_once(image)
                if not np.array_equal(output, reference):
                    # OpenCV may differ by one level; the MLIR kernels must not.
                    require(runner.variant == "opencv_baseline",
                            "MLIR kernel output differs from the reference: " + runner.variant)
                    require(int(np.abs(output.astype(np.int16) - reference.astype(np.int16)).max()) <= 1,
                            "OpenCV exceeded the frozen 1-level tolerance")
                    mismatch[runner.variant] += 1
                if index >= args.warmup:
                    samples[runner.variant].append(elapsed)
        baseline = describe(samples["opencv_baseline"])
        for runner in runners:
            stats = describe(samples[runner.variant])
            record = {"variant": runner.variant, "timing_ns": stats,
                      "outputs_differing_from_reference": mismatch[runner.variant]}
            if runner.variant != "opencv_baseline":
                record["p50_ratio_vs_opencv"] = stats["p50_ns"] / baseline["p50_ns"]
                record["mean_ratio_vs_opencv"] = stats["mean_ns"] / baseline["mean_ns"]
            result["variants"].append(record)
        result["status"] = "measured"
    except Exception as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        for runner in runners:
            if isinstance(runner, MlirKernel):
                runner.close()
        result["sysfs_after"] = sysfs()
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    for record in result["variants"]:
        ratio = record.get("p50_ratio_vs_opencv")
        print("{:<24} P50 {:9.1f} us{}".format(record["variant"], record["timing_ns"]["p50_ns"] / 1000,
              "" if ratio is None else "  x{:.2f} vs OpenCV".format(ratio)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--profile", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--input", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        measure(args)
    except Exception as error:
        print("preprocess_ablation: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
