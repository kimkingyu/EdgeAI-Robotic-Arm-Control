#!/usr/bin/env python3
"""M0 OpenCV CPU baseline only: resize -> BGR2RGB -> uint8 NHWC view.

Real JPG/PNG files are mandatory. Decode, optional input adaptation, reference
construction, validation and SHA-256 are outside operation timing. No MLIR/RKNN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import cv2
except (ImportError, OSError) as exc:
    np = cv2 = None
    DEPENDENCY_ERROR = str(exc)
else:
    DEPENDENCY_ERROR = None

STAGES = ("resize_ns", "cvtColor_ns", "batch_view_ns", "total_ns")


def nonnegative_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def positive_int(value):
    number = nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def input_dimensions(value):
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("input size must be WxH")
    return tuple(positive_int(part) for part in parts)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--image-count", type=positive_int, default=16)
    parser.add_argument("--sizes", nargs="+", type=positive_int, default=[640, 448],
                        help="square output sizes, in pixels")
    parser.add_argument("--input-size", type=input_dimensions,
                        help="WxH derived input; adaptation is outside timing")
    parser.add_argument("--mode", choices=("hot", "stream", "both"), default="both")
    parser.add_argument("--warmup", type=nonnegative_int, default=30,
                        help="discarded, validated iterations before EACH batch")
    parser.add_argument("--iterations", type=positive_int, default=1000,
                        help="measured samples per batch")
    parser.add_argument("--batches", type=positive_int, default=5)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--report", required=True, type=Path,
                        help="new JSON file; parent directory must already exist")
    return parser.parse_args(argv)


def require_dependencies():
    if cv2 is None or np is None:
        raise RuntimeError(f"NumPy and OpenCV are required: {DEPENDENCY_ERROR}")


def select_images(directory, count):
    directory = Path(directory)
    if count <= 0 or not directory.is_dir():
        raise ValueError(f"invalid image count or missing images directory: {directory}")
    paths = sorted((p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                   key=lambda p: p.name)
    if len(paths) < count:
        raise ValueError(f"need {count} JPG/PNG files, found {len(paths)} in {directory}")
    return paths[:count]


def array_sha256(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def load_inputs(paths, input_size=None):
    require_dependencies()
    images, records = [], []
    for path in paths:
        path = Path(path)
        try:
            encoded = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error) as exc:
            raise ValueError(f"cannot read/decode image {path}: {exc}") from exc
        if image is None or image.size == 0:
            raise ValueError(f"corrupt or undecodable image: {path}")
        original_shape = list(image.shape)
        if input_size is not None:
            image = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
        image.setflags(write=False)
        records.append({"path": str(path.resolve()),
                        "file_sha256": hashlib.sha256(encoded).hexdigest(),
                        "original_shape": original_shape, "input_shape": list(image.shape),
                        "input_sha256": array_sha256(image), "derived_input": input_size is not None,
                        "input_transform": "derived input: untimed INTER_LINEAR resize"
                        if input_size is not None else "none; decoded BGR input"})
        images.append(image)
    return images, records


def reference_checksums(image, size):
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    # Independent channel reversal checks BGR2RGB, without calling cvtColor.
    rgb = np.ascontiguousarray(resized[:, :, ::-1])
    return {"resize_sha256": array_sha256(resized), "output_sha256": array_sha256(rgb)}


def timed_preprocess(image, size):
    start = time.perf_counter_ns()
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    after_resize = time.perf_counter_ns()
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    after_color = time.perf_counter_ns()
    output = np.expand_dims(rgb, axis=0)
    end = time.perf_counter_ns()
    return (resized, rgb, output), {
        "resize_ns": after_resize - start, "cvtColor_ns": after_color - after_resize,
        "batch_view_ns": end - after_color, "total_ns": end - start}


def validate_output(image, products, size, expected):
    resized, rgb, output = products
    for array, shape in zip(products, ((size, size, 3), (size, size, 3), (1, size, size, 3))):
        if array.shape != shape or array.dtype != np.uint8 or not array.flags.c_contiguous:
            raise ValueError("output shape/dtype/C-contiguity validation failed")
    if not np.shares_memory(rgb, output) or rgb.ctypes.data != output.ctypes.data:
        raise ValueError("batch dimension must be an exact RGB view, not a copy")
    if (np.shares_memory(image, resized) or np.shares_memory(resized, rgb)
            or np.shares_memory(image, rgb)):
        raise ValueError("resize and color stages must allocate independent outputs")
    # The batch view covers all RGB bytes; hashing it also validates RGB pixels.
    actual = {"resize_sha256": array_sha256(resized), "output_sha256": array_sha256(output)}
    if actual != expected:
        raise ValueError("output checksum validation failed")


def describe_ns(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty samples")

    def percentile(q):
        rank = (len(ordered) - 1) * q
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)

    return {"sample_count": len(ordered), "mean_ns": statistics.mean(ordered),
            "min_ns": ordered[0], "max_ns": ordered[-1], "p50_ns": percentile(.50),
            "p95_ns": percentile(.95), "p99_ns": percentile(.99)}


def summarize_batch(raw):
    stages = {name: describe_ns([row[name] for row in raw]) for name in STAGES}
    total = sum(row["total_ns"] for row in raw)
    if total <= 0:
        raise ValueError("sum of pipeline timings must be positive")
    return {"sample_count": len(raw), "stages": stages, "sum_total_ns": total,
            "serial_pipeline_fps": len(raw) * 1e9 / total,
            "validation_overhead": describe_ns([row["validation_ns"] for row in raw])}


def run_batch(images, size, mode, warmup, iterations, expected):
    raw = []
    for count, measured in ((warmup, False), (iterations, True)):
        for step in range(count):
            index = 0 if mode == "hot" else step % len(images)
            products, timing = timed_preprocess(images[index], size)
            validation_start = time.perf_counter_ns()
            validate_output(images[index], products, size, expected[index])
            validation_ns = time.perf_counter_ns() - validation_start
            if measured:
                raw.append({"input_index": index, **timing, "validation_ns": validation_ns})
            del products  # Destruction and sample bookkeeping are outside operation timing.
    return {"warmup_samples_excluded": warmup, **summarize_batch(raw), "raw_samples": raw}


def summarize_batches(batches):
    return {"batch_count": len(batches),
            "batch_means": {name: describe_ns([b["stages"][name]["mean_ns"] for b in batches])
                            for name in STAGES},
            "serial_pipeline_fps": sum(b["sample_count"] for b in batches) * 1e9
            / sum(b["sum_total_ns"] for b in batches)}


def sysfs_snapshot(root=Path("/sys")):
    root = Path(root)
    patterns = {"thermal_temp": ("class/thermal/thermal_zone*/temp", "millidegrees_C"),
                "thermal_type": ("class/thermal/thermal_zone*/type", "text"),
                "current_frequency": ("devices/system/cpu/cpufreq/policy*/scaling_cur_freq", "kHz"),
                "hardware_frequency": ("devices/system/cpu/cpufreq/policy*/cpuinfo_cur_freq", "kHz"),
                "minimum_frequency": ("devices/system/cpu/cpufreq/policy*/scaling_min_freq", "kHz"),
                "maximum_frequency": ("devices/system/cpu/cpufreq/policy*/scaling_max_freq", "kHz"),
                "governor": ("devices/system/cpu/cpufreq/policy*/scaling_governor", "text")}
    groups = {}
    for name, (pattern, unit) in patterns.items():
        readings = []
        try:
            for path in sorted(root.glob(pattern)):
                reading = {"path": str(path), "unit": unit}
                try:
                    reading["value"] = path.read_text(encoding="utf-8").strip()
                except OSError as exc:
                    reading["error"] = str(exc)
                readings.append(reading)
            successes = sum("value" in reading for reading in readings)
            status = "unavailable" if not successes else "read" if successes == len(readings) else "partial"
            groups[name] = {"status": status, "readings": readings}
        except OSError as exc:
            groups[name] = {"status": "unavailable", "error": str(exc), "readings": readings}
    return {"utc": datetime.now(timezone.utc).isoformat(), "groups": groups}


def environment():
    try:
        affinity = {"cpus": sorted(os.sched_getaffinity(0)), "status": "read"}
    except (AttributeError, OSError) as exc:
        affinity = {"cpus": None, "status": "unavailable", "reason": str(exc)}
    return {"python": sys.version, "python_executable": sys.executable,
            "numpy": np.__version__, "opencv": cv2.__version__,
            "opencv_build_information": cv2.getBuildInformation(),
            "opencv_threads": cv2.getNumThreads(), "opencv_opencl": cv2.ocl.useOpenCL(),
            "opencv_optimized": cv2.useOptimized(), "platform": platform.platform(),
            "uname": platform.uname()._asdict(), "cpu_count": os.cpu_count(),
            "cpu_affinity": affinity, "clock": vars(time.get_clock_info("perf_counter")),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def run_benchmark(args):
    require_dependencies()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("report must be a new file in an existing directory")
    paths = select_images(args.images_dir, args.image_count)
    cv2.setNumThreads(args.threads)
    cv2.ocl.setUseOpenCL(False)
    images, inputs = load_inputs(paths, args.input_size)
    report = {"schema_version": 1, "backend": "opencv", "scope": "M0 CPU baseline; not MLIR",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": environment(), "inputs": inputs, "cases": [],
              "methodology": {
                  "pipeline": "INTER_LINEAR resize -> BGR2RGB -> expand_dims(axis=0); uint8 NHWC",
                  "decode": "imdecode(IMREAD_COLOR), untimed; original_shape is decoded BGR before adaptation",
                  "allocation": "fresh resize/RGB arrays on every call; batch is a shared-memory view",
                  "total_ns": "per-call wall interval across all three stages, including timer overhead",
                  "excluded": "decode, derived input resize, references, validation/SHA-256, destruction, bookkeeping",
                  "validation": "every output including warmup; full resize and output SHA-256 outside timing",
                  "validation_effect": "validation cost is reported separately; it can still alter caches/thermals",
                  "fps": "sample_count * 1e9 / sum(total_ns); NOT sustained wall-clock throughput",
                  "percentile": "linear interpolation at (N-1)*q; batch_means summarizes per-batch means",
                  "hot": "repeat the first lexicographically sorted selected image",
                  "stream": "round-robin decoded RAM inputs; no disk timing or cache-flush/cold-cache claim",
                  "warmup": "before each batch; stream restarts at input 0 for measured samples",
                  "sysfs": "read-only boundary snapshots, not continuous thermal/frequency monitoring"}}
    modes = ("hot", "stream") if args.mode == "both" else (args.mode,)
    for size in args.sizes:
        for mode in modes:
            active = images[:1] if mode == "hot" else images
            expected = [reference_checksums(image, size) for image in active]
            batches = []
            for index in range(args.batches):
                before = sysfs_snapshot()
                batch = run_batch(active, size, mode, args.warmup, args.iterations, expected)
                batch.update({"batch_index": index, "sysfs_before": before, "sysfs_after": sysfs_snapshot()})
                batches.append(batch)
            report["cases"].append({"output_shape": [1, size, size, 3], "mode": mode,
                                    "reference_checksums": expected, "batches": batches,
                                    "summary": summarize_batches(batches)})
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10 or newer is required")
        report = run_benchmark(args)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    except Exception as exc:
        print(f"benchmark_preprocess: error: {exc}", file=sys.stderr)
        return 1
    print(f"OpenCV M0 report: {args.report}; FPS is timing-derived, not sustained throughput")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
