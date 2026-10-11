"""Stage-A benchmark data contracts and offline reporting; standard library only.

No model is loaded by this module. Missing, failed and untrusted measurements stay
visible; they never acquire zero latency or synthetic accuracy values.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import re
import statistics
import tempfile
from typing import Any

SCHEMA_VERSION = 1
MODEL_IDS = ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0")
VARIANT_IDS = ("B0", "F1", "L1", "FL1", "O0", "O1")
THREAD_SWEEP_VARIANTS = ("B0", "O1")
SPLITS = ("smoke", "calibration", "development", "holdout")
DATASET_REQUIREMENTS = {"development": 128, "performance": 64,
                        "calibration": 256, "holdout": 1000}
RESULT_STATUSES = frozenset({"passed", "failed_correctness", "failed_runtime",
                             "failed_timeout", "unsupported", "blocked_data",
                             "awaiting_dataset", "interrupted"})
RUN_STATUSES = RESULT_STATUSES | {"running", "completed", "completed_with_failures",
                                  "complete", "failed", "prepared"}
TIMING_FIELDS = ("preprocess_ms", "input_adapt_ms", "inference_ms",
                 "postprocess_ms", "pipeline_wall_ms")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "classifier_benchmark.json"


class BenchmarkError(ValueError):
    """An invalid protocol, dataset or report operation."""


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"{name} must be an integer, not bool")
    if value < minimum or (maximum is not None and value > maximum):
        raise BenchmarkError(f"{name} is outside the permitted budget")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{name} must be a finite number, not bool")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise BenchmarkError(f"{name} must be finite") from error
    if not math.isfinite(number) or number < minimum or (positive and number <= 0):
        raise BenchmarkError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _finite_tree(value: Any, name: str = "JSON") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BenchmarkError(f"{name} contains NaN or infinity")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{name} must be a nonempty string")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise BenchmarkError(f"{name} must be a SHA256 hex digest")
    return value.lower()


def _version(value: dict, name: str) -> None:
    if _integer(value.get("schema_version"), f"{name}.schema_version", minimum=1) != SCHEMA_VERSION:
        raise BenchmarkError(f"{name}: unsupported schema_version")


def _json_bytes(value: Any) -> bytes:
    _finite_tree(value)
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise BenchmarkError(f"Cannot serialize finite JSON: {error}") from error


def _read_json(path: Path) -> tuple[dict, bytes]:
    def reject_constant(value: str) -> None:
        raise BenchmarkError(f"JSON contains invalid constant {value}")

    def unique_keys(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise BenchmarkError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant,
                           object_pairs_hook=unique_keys)
        _object(value, str(path))
        _finite_tree(value)
        return value, raw
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise BenchmarkError(f"Cannot read {path}: {error}") from error


def _fixed_list(value: Any, allowed: tuple, name: str) -> None:
    if (not isinstance(value, list) or any(type(item) is not type(allowed[0]) for item in value)
            or len(value) != len(allowed) or set(value) != set(allowed)):
        raise BenchmarkError(f"{name} must contain each supported ID exactly once: {list(allowed)}")


def _validate_config(value: dict) -> dict:
    _object(value, "config")
    _finite_tree(value, "config")
    _version(value, "config")
    _fixed_list(value.get("models"), MODEL_IDS, "models")
    _fixed_list(value.get("variants"), VARIANT_IDS, "variants")
    if value.get("thread_sweep") != [1, 2, 4] or any(type(v) is not int for v in value.get("thread_sweep", [])):
        raise BenchmarkError("thread_sweep must be [1, 2, 4]; only B0/O1 are swept")
    _fixed_list(value.get("thread_sweep_variants", list(THREAD_SWEEP_VARIANTS)),
                THREAD_SWEEP_VARIANTS, "thread_sweep_variants")
    for key in ("baseline_threads", "inter_op_threads", "batch_size"):
        if _integer(value.get(key, 1), key, minimum=1) != 1:
            raise BenchmarkError(f"Stage A fixes {key}=1")
    for key, expected in (("weights_version", "IMAGENET1K_V1"), ("dtype", "float32"), ("device", "cpu")):
        if value.get(key, expected) != expected:
            raise BenchmarkError(f"Stage A fixes {key}={expected}")
    _integer(value.get("seed"), "seed", minimum=0, maximum=2**32 - 1)
    for key in ("rtol", "atol"):
        if _number(value.get(key), key) != 1e-4:
            raise BenchmarkError(f"Stage A fixes {key}=1e-4; no silent tolerance relaxation")
    _number(value.get("worker_timeout_s"), "worker_timeout_s", positive=True)
    for mode in ("smoke", "formal"):
        budget = _object(value.get(mode), mode)
        if set(budget) != {"warmup", "iterations", "repeats"}:
            raise BenchmarkError(f"{mode} must specify warmup, iterations and repeats")
        _integer(budget["warmup"], f"{mode}.warmup", maximum=100000)
        _integer(budget["iterations"], f"{mode}.iterations", minimum=1, maximum=1000000)
        _integer(budget["repeats"], f"{mode}.repeats", minimum=1, maximum=1000)
        if (budget["warmup"] + budget["iterations"]) * budget["repeats"] > 10000000:
            raise BenchmarkError(f"{mode} total budget exceeds 10000000 calls per configuration")
    gates = _object(value.get("dataset_requirements"), "dataset_requirements")
    if set(gates) != set(DATASET_REQUIREMENTS):
        raise BenchmarkError("dataset_requirements must specify all four fixed data gates")
    for key, expected in DATASET_REQUIREMENTS.items():
        if _integer(gates[key], f"dataset_requirements.{key}", minimum=1) != expected:
            raise BenchmarkError(f"Stage A fixes dataset_requirements.{key}={expected}")
    ort = _object(value.get("ort", {}), "ort")
    if ort.get("providers", ["CPUExecutionProvider"]) != ["CPUExecutionProvider"]:
        raise BenchmarkError("Only CPUExecutionProvider is supported")
    if ort.get("execution_mode", "sequential") != "sequential":
        raise BenchmarkError("ORT execution_mode must be sequential")
    for key in ("intra_op_spinning", "inter_op_spinning"):
        if not isinstance(ort.get(key, False), bool):
            raise BenchmarkError(f"ort.{key} must be bool")
    adoption = _object(value.get("adoption", {}), "adoption")
    for key, default in (("min_speedup", 1.05), ("max_p95_ratio", 1.05), ("min_ci_lower", 1.0)):
        _number(adoption.get(key, default), f"adoption.{key}", minimum=1.0)
    accuracy = _object(value.get("quantization_accuracy_budget_pp", {}), "quantization_accuracy_budget_pp")
    for key in ("top1", "top5"):
        if _number(accuracy.get(key, 1.0), f"quantization_accuracy_budget_pp.{key}") > 100:
            raise BenchmarkError("Accuracy budget must be between 0 and 100 percentage points")
    return copy.deepcopy(value)


def load_config(path: str | Path | None = None) -> dict:
    """Load and validate the frozen, CPU-only three-model Stage-A protocol."""
    value, _ = _read_json(Path(path) if path is not None else _DEFAULT_CONFIG)
    return _validate_config(value)


def expand_experiments(config: dict, *, include_thread_sweep: bool = True) -> list[dict]:
    """18 main configurations plus 12 B0/O1 thread configurations, without repeats."""
    config = _validate_config(config)
    if not isinstance(include_thread_sweep, bool):
        raise BenchmarkError("include_thread_sweep must be bool")
    result = []
    for model in config["models"]:
        for variant in config["variants"]:
            result.append({"id": f"{model}.{variant}.t1", "model_id": model,
                           "variant_id": variant, "threads": 1})
        if include_thread_sweep:
            for variant in THREAD_SWEEP_VARIANTS:
                for threads in config["thread_sweep"]:
                    if threads != 1:
                        result.append({"id": f"{model}.{variant}.t{threads}", "model_id": model,
                                       "variant_id": variant, "threads": threads})
    return result


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path, *, check_files: bool = True) -> dict:
    """Validate local bytes and split leakage, leaving image decoding to the worker."""
    if not isinstance(check_files, bool):
        raise BenchmarkError("check_files must be bool")
    manifest_path = Path(path).resolve()
    value, raw = _read_json(manifest_path)
    _version(value, "manifest")
    root_text = _text(value.get("root"), "manifest.root")
    root = Path(root_text)
    if not root.is_absolute():
        root = manifest_path.parent / root
    root = root.resolve()
    if check_files and not root.is_dir():
        raise BenchmarkError(f"Manifest root is not a directory: {root}")
    images = value.get("images")
    if not isinstance(images, list):
        raise BenchmarkError("manifest.images must be a list")
    seen_ids, seen_hashes, sources = set(), set(), {}
    resolved_images = []
    for index, original in enumerate(images):
        entry = copy.deepcopy(_object(original, f"images[{index}]"))
        image_id = _text(entry.get("id"), f"images[{index}].id")
        relative = _text(entry.get("path"), f"images[{index}].path")
        posix = PurePosixPath(relative.replace("\\", "/"))
        windows = PureWindowsPath(relative)
        if (posix.is_absolute() or windows.is_absolute() or windows.drive
                or any(":" in part for part in posix.parts)):
            raise BenchmarkError(f"Image path must be relative to root: {relative}")
        resolved = (root / Path(*posix.parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise BenchmarkError(f"Image path escapes root: {relative}") from error
        if resolved == root:
            raise BenchmarkError(f"Image path is not a file: {relative}")
        digest = _hash(entry.get("sha256"), f"images[{index}].sha256")
        split = entry.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise BenchmarkError(f"Unsupported split for {image_id}: {split}")
        label = entry.get("label")
        if label is not None:
            _integer(label, f"images[{index}].label", maximum=999)
        if image_id in seen_ids:
            raise BenchmarkError(f"Duplicate image id: {image_id}")
        if digest in seen_hashes:
            raise BenchmarkError(f"Duplicate image SHA256 (split leakage): {image_id}")
        source_id = entry.get("source_id")
        if source_id is not None:
            _text(source_id, f"images[{index}].source_id")
            if source_id in sources and sources[source_id] != split:
                raise BenchmarkError(f"source_id crosses splits: {source_id}")
            sources[source_id] = split
        for key in ("source", "source_url", "license"):
            if entry.get(key) is not None and not isinstance(entry[key], str):
                raise BenchmarkError(f"images[{index}].{key} must be a string or null")
        if check_files:
            try:
                if not resolved.is_file() or resolved.stat().st_size <= 0:
                    raise BenchmarkError(f"Missing or empty image file: {relative}")
                if _file_hash(resolved) != digest:
                    raise BenchmarkError(f"Image byte SHA256 mismatch: {relative}")
            except OSError as error:
                raise BenchmarkError(f"Cannot validate image {relative}: {error}") from error
        seen_ids.add(image_id)
        seen_hashes.add(digest)
        entry["label"] = label
        entry["resolved_path"] = str(resolved)
        resolved_images.append(entry)
    result = copy.deepcopy(value)
    result["images"] = resolved_images
    result["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    result["resolved_root"] = str(root)
    return result


def dataset_readiness(manifest: dict) -> dict:
    """Data gates are counts, not a claim of representativeness or label quality."""
    images = _object(manifest, "manifest").get("images")
    if not isinstance(images, list):
        raise BenchmarkError("manifest.images must be a list")
    counts, labeled = Counter(), Counter()
    for entry in images:
        entry = _object(entry, "image")
        split = entry.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise BenchmarkError("Invalid image split")
        counts[split] += 1
        if entry.get("label") is not None:
            _integer(entry["label"], "image.label", maximum=999)
            labeled[split] += 1
    smoke_count = counts["smoke"] + counts["development"]
    performance_count = min(counts["development"], DATASET_REQUIREMENTS["performance"])
    formal_missing = []
    if counts["development"] < DATASET_REQUIREMENTS["development"]:
        formal_missing.append(f"development {counts['development']}/128")
    if performance_count < DATASET_REQUIREMENTS["performance"]:
        formal_missing.append(f"performance {performance_count}/64")
    quant_missing = []
    for key, actual, minimum in (("calibration", counts["calibration"], 256),
                                 ("development labeled", labeled["development"], 128),
                                 ("holdout labeled", labeled["holdout"], 1000)):
        if actual < minimum:
            quant_missing.append(f"{key} {actual}/{minimum}")
    return {
        "smoke": {"ready": smoke_count >= 1, "reason": "Exploratory only; not dataset accuracy" if smoke_count else "No smoke/development images",
                  "count": smoke_count},
        "formal": {"ready": not formal_missing, "reason": "Data count gates met; fixed development subset only" if not formal_missing else "awaiting_dataset: " + "; ".join(formal_missing),
                   "development_count": counts["development"], "performance_count": performance_count},
        "quantization": {"ready": not quant_missing, "reason": "Count/label gates met; provenance and label mapping still require confirmation" if not quant_missing else "blocked_data: " + "; ".join(quant_missing),
                         "calibration_count": counts["calibration"],
                         "development_labeled_count": labeled["development"],
                         "holdout_labeled_count": labeled["holdout"]},
    }


def select_images(manifest: dict, *, mode: str = "smoke", seed: int = 20261001) -> list[dict]:
    """Stable ordering plus seeded sampling; formal always uses exactly 64 dev images."""
    _integer(seed, "seed", maximum=2**32 - 1)
    if mode not in ("smoke", "formal"):
        raise BenchmarkError("mode must be smoke or formal")
    readiness = dataset_readiness(manifest)
    if not readiness[mode]["ready"]:
        raise BenchmarkError(readiness[mode]["reason"])
    allowed = {"development"} if mode == "formal" else {"smoke", "development"}
    entries = [entry for entry in manifest["images"] if entry["split"] in allowed]
    for entry in entries:
        _text(entry.get("id"), "image.id")
    entries = sorted(entries, key=lambda entry: entry["id"])
    if len({entry["id"] for entry in entries}) != len(entries):
        raise BenchmarkError("Duplicate image id in selection")
    random.Random(seed).shuffle(entries)
    return copy.deepcopy(entries[:64] if mode == "formal" else entries[:min(8, len(entries))])


def _describe(values: list, *, allow_zero: bool = False) -> dict:
    if not values:
        raise BenchmarkError("Cannot summarize empty measurements")
    numbers = [_number(value, "latency_ms", positive=not allow_zero) for value in values]
    ordered = sorted(numbers)
    scale = max(ordered)
    scaled = [value / scale for value in numbers] if scale else numbers

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    result = {"count": len(numbers), "p50_ms": percentile(0.50), "p95_ms": percentile(0.95),
              "mean_ms": statistics.fmean(scaled) * scale,
              "std_ms": statistics.pstdev(scaled) * scale,
              "min_ms": ordered[0], "max_ms": ordered[-1]}
    _finite_tree(result, "statistics")
    return result


def summarize(values: Any) -> dict:
    """Linear quantiles and population standard deviation; no numpy dependency."""
    try:
        return _describe(list(values))
    except TypeError as error:
        raise BenchmarkError("Measurements must be an iterable of positive numbers") from error


def stage_stats(samples: list[dict]) -> dict:
    """Actual wall-clock distributions, never the sum of stage medians."""
    if not isinstance(samples, list) or not samples:
        raise BenchmarkError("Stage statistics require samples")
    result = {}
    for field in TIMING_FIELDS:
        if all(field in sample for sample in samples):
            result[field] = _describe([sample[field] for sample in samples],
                                      allow_zero=field not in ("inference_ms", "pipeline_wall_ms"))
    if all(all(field in sample for field in TIMING_FIELDS) for sample in samples):
        sums = [sum(sample[field] for field in TIMING_FIELDS[:-1]) for sample in samples]
        overhead = [sample["pipeline_wall_ms"] - total for sample, total in zip(samples, sums)]
        if any(value < -1e-7 * max(1.0, sample["pipeline_wall_ms"]) for value, sample in zip(overhead, samples)):
            raise BenchmarkError("Pipeline stages exceed measured wall time")
        result["stage_sum_ms"] = _describe(sums)
        result["wrapper_overhead_ms"] = _describe([max(0.0, value) for value in overhead], allow_zero=True)
    return result


def _experiment(value: Any, known: dict[str, dict]) -> dict:
    value = _object(value, "experiment")
    name = _text(value.get("id"), "experiment.id")
    if name not in known:
        raise BenchmarkError(f"Unsupported experiment: {name}")
    expected = known[name]
    for key in ("model_id", "variant_id"):
        if value.get(key) != expected[key]:
            raise BenchmarkError(f"experiment.{key} disagrees with id")
    if _integer(value.get("threads"), "experiment.threads", minimum=1) != expected["threads"]:
        raise BenchmarkError("experiment.threads disagrees with id")
    return expected


def _correctness(validation: Any) -> bool | None:
    if not isinstance(validation, dict):
        return None
    if "passed" in validation:
        if not isinstance(validation["passed"], bool):
            raise BenchmarkError("validation.passed must be bool")
        return validation["passed"]
    if "allclose" in validation:
        if not isinstance(validation["allclose"], bool):
            raise BenchmarkError("validation.allclose must be bool")
        return validation["allclose"]
    checks = validation.get("checks")
    if isinstance(checks, list) and checks:
        states = [_correctness(check) for check in checks]
        return all(states) if all(state is not None for state in states) else None
    return None


def _validate_readiness(value: Any) -> dict:
    value = _object(value, "dataset_readiness")
    fields = {"smoke": ("count",), "formal": ("development_count", "performance_count"),
              "quantization": ("calibration_count", "development_labeled_count", "holdout_labeled_count")}
    minimums = {"smoke": (1,), "formal": (128, 64), "quantization": (256, 128, 1000)}
    for stage, keys in fields.items():
        gate = _object(value.get(stage), f"dataset_readiness.{stage}")
        if not isinstance(gate.get("ready"), bool):
            raise BenchmarkError(f"dataset_readiness.{stage}.ready must be bool")
        _text(gate.get("reason"), f"dataset_readiness.{stage}.reason")
        numbers = [_integer(gate.get(key), f"dataset_readiness.{stage}.{key}") for key in keys]
        if gate["ready"] != all(number >= minimum for number, minimum in zip(numbers, minimums[stage])):
            raise BenchmarkError(f"dataset_readiness.{stage} contradicts fixed data gates")
    return copy.deepcopy(value)


def _validate_run(value: dict) -> tuple[dict, list[dict], set[str]]:
    _version(value, "run")
    _text(value.get("run_id"), "run_id")
    if value.get("mode") not in ("smoke", "formal"):
        raise BenchmarkError("run.mode must be smoke or formal")
    config = _validate_config(value.get("config"))
    _hash(value.get("config_sha256"), "run.config_sha256")
    if value.get("manifest_sha256") is not None:
        _hash(value["manifest_sha256"], "run.manifest_sha256")
    fingerprint = value.get("code_fingerprint")
    if isinstance(fingerprint, str):
        _hash(fingerprint, "run.code_fingerprint")
    elif not isinstance(fingerprint, dict) or not fingerprint:
        raise BenchmarkError("run.code_fingerprint must be a SHA256 or a nonempty JSON object")
    _object(value.get("environment"), "run.environment")
    _validate_readiness(value.get("dataset_readiness"))
    if not isinstance(value.get("status"), str) or value["status"] not in RUN_STATUSES:
        raise BenchmarkError(f"Invalid run.status: {value.get('status')}")
    if not isinstance(value.get("limitations"), list) or any(not isinstance(item, str) for item in value["limitations"]):
        raise BenchmarkError("run.limitations must be a list of strings")
    if not isinstance(value.get("prepared_models"), (dict, list)):
        raise BenchmarkError("run.prepared_models must be an object or list")
    all_experiments = expand_experiments(config)
    known = {experiment["id"]: experiment for experiment in all_experiments}
    scheduled = value.get("experiments")
    if not isinstance(scheduled, list):
        raise BenchmarkError("run.experiments must be a list of experiment objects")
    ids = [_experiment(experiment, known)["id"] for experiment in scheduled]
    if len(ids) != len(set(ids)):
        raise BenchmarkError("run.experiments contains duplicate configurations")
    return config, all_experiments, set(ids)


def _validate_result(value: dict, run: dict, known: dict, scheduled: set[str]) -> None:
    _version(value, "result")
    for key in ("config_sha256", "manifest_sha256", "code_fingerprint"):
        if key not in value or value[key] != run.get(key):
            raise BenchmarkError(f"result {key} does not match this run")
    experiment = _experiment(value.get("experiment"), known)
    if experiment["id"] not in scheduled:
        raise BenchmarkError("Result was not scheduled in this run")
    repeats = run["config"][run["mode"]]["repeats"]
    _integer(value.get("block"), "result.block", maximum=repeats - 1)
    status = value.get("status")
    if not isinstance(status, str) or status not in RESULT_STATUSES:
        raise BenchmarkError(f"Invalid result.status: {status}")
    kind = value.get("kind")
    if kind not in ("inference", "pipeline", "memory"):
        raise BenchmarkError(f"Invalid measurement kind: {kind}")
    pid = value.get("worker_pid")
    if status == "passed" or pid is not None:
        _integer(pid, "worker_pid", minimum=1)
    if value.get("worker_identity") is not None:
        identity = _text(value["worker_identity"], "worker_identity")
        ticks = _integer(value.get("worker_creation_ticks"), "worker_creation_ticks", minimum=1)
        source = _text(value.get("worker_identity_source"), "worker_identity_source")
        prefix = f"{pid}:{ticks}"
        if not (identity == prefix or identity.startswith(prefix + ":")) or not source:
            raise BenchmarkError("Process identity does not bind PID and creation marker")
    samples = value.get("samples", [])
    if not isinstance(samples, list):
        raise BenchmarkError("result.samples must be a list")
    for sample in samples:
        _object(sample, "sample")
        _text(sample.get("image_id"), "sample.image_id")
        for field in TIMING_FIELDS:
            if field in sample:
                _number(sample[field], field, positive=field in ("inference_ms", "pipeline_wall_ms"))
    for field in ("load_ms", "first_inference_ms"):
        if value.get(field) is not None:
            _number(value[field], field, positive=True)
    if status != "passed":
        return  # Partial observations stay visible, but never count as successful samples.
    if run.get("manifest_sha256") is None:
        raise BenchmarkError("Passed measurements require a real manifest SHA256")
    for field in ("model_metadata", "variant_metadata", "validation"):
        _object(value.get(field), field)
    if kind in ("inference", "pipeline"):
        if len(samples) != run["config"][run["mode"]]["iterations"]:
            raise BenchmarkError("Measured sample count differs from the frozen iterations budget")
        if any("inference_ms" not in sample for sample in samples):
            raise BenchmarkError("Missing inference_ms in successful samples")
        if _correctness(value["validation"]) is not True:
            raise BenchmarkError("Passed result lacks passing numerical correctness validation")
        if kind == "pipeline":
            if any(any(field not in sample for field in TIMING_FIELDS) for sample in samples):
                raise BenchmarkError("Pipeline result requires all four stages and actual wall time")
            stage_stats(samples)
    else:
        memory = _object(value.get("memory"), "memory")
        for field in ("rss_baseline_bytes", "rss_loaded_bytes", "rss_peak_bytes"):
            _integer(memory.get(field), f"memory.{field}", minimum=1)
        if memory["rss_peak_bytes"] < memory["rss_loaded_bytes"]:
            raise BenchmarkError("RSS peak cannot be smaller than loaded RSS")
        _number(memory.get("sampling_interval_ms"), "memory.sampling_interval_ms", positive=True)
        if not isinstance(memory.get("limitations"), list) or any(not isinstance(v, str) for v in memory["limitations"]):
            raise BenchmarkError("memory.limitations must be a list of strings")
        if samples and len(samples) != run["config"][run["mode"]]["iterations"]:
            raise BenchmarkError("Memory observation sample count differs from the frozen budget")


def _invalidate(record: dict, reason: str) -> None:
    record["trusted"] = False
    record["status"] = "untrusted"
    if reason not in record["issues"]:
        record["issues"].append(reason)


def _read_results(directory: Path, run: dict, known: dict, scheduled: set[str]) -> list[dict]:
    records = []
    keys, pids = defaultdict(list), defaultdict(list)
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*.json")):
        record = {"source_file": f"results/{path.name}", "trusted": False,
                  "status": "untrusted", "original_status": None, "issues": [], "data": None}
        try:
            value, raw = _read_json(path)
            record["data"] = value
            record["original_status"] = value.get("status")
            record["record_sha256"] = hashlib.sha256(raw).hexdigest()
            _validate_result(value, run, known, scheduled)
            record.update({"trusted": True, "status": value["status"]})
        except (BenchmarkError, OSError, TypeError) as error:
            _invalidate(record, str(error))
        records.append(record)
        value = record["data"]
        if isinstance(value, dict):
            experiment = value.get("experiment")
            if (isinstance(experiment, dict) and isinstance(experiment.get("id"), str)
                    and type(value.get("block")) is int and isinstance(value.get("kind"), str)):
                keys[(experiment["id"], value["block"], value["kind"])].append(record)
            if type(value.get("worker_pid")) is int and value["worker_pid"] > 0:
                # An integer PID can be recycled after process exit. Native
                # creation markers disambiguate new collectors; legacy rows
                # without a marker keep the conservative PID-only rule.
                marker = value.get("worker_identity")
                pids[(value["worker_pid"], marker if isinstance(marker, str) else None)].append(record)
    for entries in keys.values():
        if len(entries) > 1:
            for record in entries:
                _invalidate(record, "Duplicate experiment/block/kind; neither duplicate is selected")
    for (pid, marker), entries in pids.items():
        if len(entries) > 1:
            for record in entries:
                _invalidate(record, f"worker_pid {pid} / process instance {marker or 'legacy identity missing'} reused across independent jobs")
    return records


def _kind_summary(records: list[dict], kind: str) -> dict:
    relevant = [record for record in records if record["data"].get("kind") == kind]
    passed = [record for record in relevant if record["trusted"] and record["status"] == "passed"]
    passed.sort(key=lambda record: record["data"]["block"])
    blocks = []
    samples = []
    for record in passed:
        value = record["data"]
        block = {"block": value["block"], "worker_pid": value["worker_pid"],
                 "source_file": record["source_file"], "sample_count": len(value.get("samples", []))}
        if kind == "memory":
            block["memory"] = copy.deepcopy(value["memory"])
        else:
            block["stats"] = stage_stats(value["samples"])
            block["image_ids"] = [sample["image_id"] for sample in value["samples"]]
            samples.extend(value["samples"])
        blocks.append(block)
    result = {"block_count": len(blocks), "sample_count": len(samples), "blocks": blocks,
              "status_counts": dict(Counter(record["status"] for record in relevant)),
              "stats": stage_stats(samples) if samples else {}}
    if kind == "memory":
        result["independent_process"] = True
        for field in ("rss_baseline_bytes", "rss_loaded_bytes", "rss_peak_bytes"):
            values = [block["memory"][field] for block in blocks]
            result[field] = {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else None
    else:
        for field in ("load_ms", "first_inference_ms"):
            values = [record["data"][field] for record in passed if record["data"].get(field) is not None]
            result[field] = summarize(values) if values else None
    return result


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def _comparison(candidate: dict, baseline: dict | None, *, baseline_id: str,
                kind: str, seed: int, expected_blocks: int, attribution: str) -> dict:
    result = {"baseline_id": baseline_id, "kind": kind, "attribution": attribution,
              "status": "unavailable", "speedup": None, "pooled_p50_ratio": None,
              "p95_ratio": None, "paired_block_count": 0, "block_ratios": [],
              "ci95": None, "reason": "No trusted passing paired measurements"}
    if baseline is None:
        result["reason"] = "Matching-factor baseline not measured; thread/backend factors cannot be mixed"
        return result
    field = "pipeline_wall_ms" if kind == "pipeline" else "inference_ms"
    candidate_kind, baseline_kind = candidate[kind], baseline[kind]
    candidate_blocks = {block["block"]: block for block in candidate_kind["blocks"]}
    baseline_blocks = {block["block"]: block for block in baseline_kind["blocks"]}
    mismatch = []
    ratios = []
    for index in sorted(candidate_blocks.keys() & baseline_blocks.keys()):
        cand, base = candidate_blocks[index], baseline_blocks[index]
        if cand["image_ids"] != base["image_ids"]:
            mismatch.append(index)
            continue
        ratio = base["stats"][field]["p50_ms"] / cand["stats"][field]["p50_ms"]
        if not math.isfinite(ratio) or ratio <= 0:
            result["reason"] = "Non-finite paired speed ratio"
            return result
        ratios.append(ratio)
        result["block_ratios"].append({"block": index, "speedup": ratio})
    result["paired_block_count"] = len(ratios)
    if not ratios:
        if mismatch:
            result["reason"] = f"Paired image sequence mismatch in blocks {mismatch}"
        return result
    result["speedup"] = math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
    result["estimator"] = "geometric mean of paired block P50 ratios (baseline/candidate)"
    result["pooled_p50_ratio"] = baseline_kind["stats"][field]["p50_ms"] / candidate_kind["stats"][field]["p50_ms"]
    result["p95_ratio"] = candidate_kind["stats"][field]["p95_ms"] / baseline_kind["stats"][field]["p95_ms"]
    result["status"] = "available" if len(ratios) == expected_blocks and not mismatch else "partial"
    result["reason"] = ("All planned repeat blocks paired" if result["status"] == "available"
                        else "Missing/failed blocks or incompatible image sequences; not an adoption comparison")
    if len(ratios) >= 5:
        material = f"{seed}:{baseline_id}:{candidate['id']}:{kind}".encode("utf-8")
        bootstrap_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        rng = random.Random(bootstrap_seed)
        logs = [math.log(ratio) for ratio in ratios]
        draws = [math.exp(statistics.fmean(rng.choices(logs, k=len(logs)))) for _ in range(2000)]
        result["ci95"] = {"low": _percentile(draws, 0.025), "high": _percentile(draws, 0.975),
                          "confidence": 0.95, "method": "paired_block_bootstrap",
                          "resampling_unit": "repeat_block", "block_count": len(ratios),
                          "resamples": 2000, "seed": bootstrap_seed,
                          "limitation": "Exploratory interval; five blocks do not guarantee generalization"}
    else:
        result["ci_reason"] = "At least 5 independent paired repeat blocks required; iterations are not independent blocks"
    _finite_tree(result, "comparison")
    return result


def _row_status(row: dict, run: dict, repeats: int) -> str:
    states = row["status_counts"]
    for status in ("untrusted", "interrupted", "failed_correctness", "failed_runtime", "failed_timeout", "unsupported", "blocked_data", "awaiting_dataset"):
        if states.get(status):
            return status
    if row["scheduled"] and row["inference"]["block_count"] < repeats:
        if run["status"] == "interrupted":
            return "interrupted"
        if run["status"] in ("blocked_data", "awaiting_dataset", "unsupported"):
            return run["status"]
        return "incomplete" if states else "not_run"
    return "passed" if states.get("passed") else "not_scheduled"


def aggregate_results(run_dir: str | Path) -> dict:
    """Read existing artifacts only. Corrupt result records are visible and excluded."""
    directory = Path(run_dir)
    run, raw = _read_json(directory / "run.json")
    config, experiments, scheduled = _validate_run(run)
    known = {experiment["id"]: experiment for experiment in experiments}
    records = _read_results(directory / "results", run, known, scheduled)
    grouped = defaultdict(list)
    for record in records:
        value = record["data"]
        if isinstance(value, dict) and isinstance(value.get("experiment"), dict):
            name = value["experiment"].get("id")
            if isinstance(name, str) and name in known:
                grouped[name].append(record)
    rows = []
    repeats = config[run["mode"]]["repeats"]
    for experiment in experiments:
        relevant = grouped[experiment["id"]]
        row = dict(experiment)
        row.update({"scheduled": row["id"] in scheduled,
                    "status_counts": dict(Counter(record["status"] for record in relevant)),
                    "record_files": [record["source_file"] for record in relevant],
                    "inference": _kind_summary(relevant, "inference"),
                    "pipeline": _kind_summary(relevant, "pipeline"),
                    "memory": _kind_summary(relevant, "memory"),
                    "validation": [], "predictions": [], "model_metadata": [],
                    "variant_metadata": [], "errors": [], "comparisons": {}})
        for record in relevant:
            value = record["data"]
            context = {"block": value.get("block"), "kind": value.get("kind"),
                       "trusted": record["trusted"], "source_file": record["source_file"]}
            for key in ("validation", "predictions", "model_metadata", "variant_metadata"):
                if key in value:
                    row[key].append({**context, "value": copy.deepcopy(value[key])})
            if record["issues"] or value.get("error") or value.get("traceback"):
                row["errors"].append({**context, "status": record["status"], "issues": list(record["issues"]),
                                      "error": copy.deepcopy(value.get("error")), "traceback": value.get("traceback")})
        row["status"] = _row_status(row, run, repeats)
        rows.append(row)
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        model, variant, threads = row["model_id"], row["variant_id"], row["threads"]
        definitions = {"overall": (f"{model}.B0.t1", "Whole execution-stack comparison to same-model B0.t1")}
        if threads > 1:
            definitions["factor"] = (f"{model}.{variant}.t1", "Thread scaling only; inter-op stays 1")
            definitions["threads"] = definitions["factor"]
        else:
            definitions["factor"] = (f"{model}.{'O0' if variant == 'O1' else 'B0'}.t1", "ORT runtime optimization only" if variant == "O1" else "Variant versus same-model eager baseline")
        if variant == "FL1":
            definitions["fusion_given_layout"] = (f"{model}.L1.t1", "Fusion increment with channels_last fixed")
            definitions["layout_given_fusion"] = (f"{model}.F1.t1", "Layout increment with fusion fixed")
        if variant == "O1":
            definitions["ort_incremental"] = (f"{model}.O0.t{threads}", "ORT optimization increment at matching thread count")
        for name, (baseline_id, attribution) in definitions.items():
            row["comparisons"][name] = _comparison(row, by_id.get(baseline_id), baseline_id=baseline_id,
                                                     kind="inference", seed=config["seed"],
                                                     expected_blocks=repeats, attribution=attribution)
        overall = row["comparisons"]["overall"]
        ci = overall["ci95"]
        gate = (run["mode"] == "formal" and run["dataset_readiness"]["formal"]["ready"]
                and row["status"] == "passed" and overall["status"] == "available" and ci is not None
                and overall["speedup"] >= config.get("adoption", {}).get("min_speedup", 1.05)
                and ci["low"] > config.get("adoption", {}).get("min_ci_lower", 1.0)
                and overall["p95_ratio"] <= config.get("adoption", {}).get("max_p95_ratio", 1.05))
        row["recommendation"] = {"eligible": False, "screening_gate_passed": bool(gate),
                                 "status": "exploratory_only" if run["mode"] == "smoke" else "confirmation_required",
                                 "reason": "Smoke is never a formal adoption recommendation" if run["mode"] == "smoke" else "Independent frozen-candidate confirmation is outside Stage A"}
    integrity_errors = [{"source_file": record["source_file"], "issues": list(record["issues"])}
                        for record in records if not record["trusted"]]
    image_ids = set()
    for record in records:
        if record["trusted"] and record["status"] == "passed":
            image_ids.update(sample["image_id"] for sample in record["data"].get("samples", []))
    limitations = list(run["limitations"])
    additions = ["CPU batch=1 only; not NPU/robot/camera performance",
                 "O0 versus B0 combines export and execution backend; exporter may already fold nodes",
                 "RSS comes from independent sampled processes; sampling may miss peaks and includes dependencies",
                 "SHA/source_id checks cannot detect every unmarked near-duplicate",
                 "Top-5 prediction agreement is not dataset accuracy; missing labels never pass accuracy gates",
                 "Stage B requires dev128/performance64, calibration256, dev128 labeled and holdout1000 labeled",
                 "No frozen-candidate independent confirmation; no formal adoption recommendation"]
    if run["mode"] == "smoke":
        additions.append("smoke/exploratory_only: latency results do not justify formal adoption or dataset accuracy")
    if len(image_ids) == 1 or run["dataset_readiness"]["smoke"]["count"] == 1:
        additions.append("single_image: same-image/cache exploration only; not representative multi-image performance")
    for text in additions:
        if text not in limitations:
            limitations.append(text)
    summary_status = "untrusted" if integrity_errors else ("interrupted" if run["status"] == "interrupted" else run["status"])
    if summary_status in ("completed", "complete", "passed") and any(row["scheduled"] and row["status"] in ("not_run", "incomplete") for row in rows):
        summary_status = "incomplete"
    summary = {"schema_version": SCHEMA_VERSION, "run_id": run["run_id"], "mode": run["mode"],
               "status": summary_status, "run_status": run["status"], "run_sha256": hashlib.sha256(raw).hexdigest(),
               "config": config, "config_sha256": run["config_sha256"], "manifest_sha256": run["manifest_sha256"],
               "code_fingerprint": copy.deepcopy(run["code_fingerprint"]), "environment": copy.deepcopy(run["environment"]),
               "dataset_readiness": copy.deepcopy(run["dataset_readiness"]), "stage_b": {"status": "not_executed",
                   "formal": copy.deepcopy(run["dataset_readiness"]["formal"]),
                   "quantization": copy.deepcopy(run["dataset_readiness"]["quantization"])},
               "prepared_models": copy.deepcopy(run["prepared_models"]),
               "configuration_count": len(rows), "scheduled_configuration_count": len(scheduled),
               "measured_unique_image_count": len(image_ids), "experiments": rows,
               "records": records, "integrity_errors": integrity_errors,
               "status_counts": dict(Counter(row["status"] for row in rows)), "limitations": limitations,
               "methodology": {"ratio": "baseline/candidate; geometric mean of paired block P50 ratios",
                   "ci": "paired block bootstrap, 2000 resamples; >=5 independent blocks only",
                   "quantiles": "linear interpolation", "std": "population",
                   "iteration_independence": False, "formal_adoption_recommended": False,
                   "throughput": "Not inferred from 1000/P50; no whole-system throughput claim"}}
    _finite_tree(summary, "summary")
    return summary


def _escape(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return html.escape(value, quote=True)


def _display(value: Any, digits: int = 3) -> str:
    return "—" if value is None else _escape(f"{value:.{digits}f}")


def _metric(row: dict, key: str, field: str = "inference_ms", kind: str = "inference") -> Any:
    return row[kind]["stats"].get(field, {}).get(key)


def _ratio(row: dict, name: str, key: str = "speedup") -> Any:
    return row["comparisons"].get(name, {}).get(key)


def _csv_report(summary: dict) -> str:
    fields = ["id", "model_id", "variant_id", "threads", "status", "scheduled", "blocks", "samples",
              "p50_ms", "p95_ms", "mean_ms", "std_ms", "overall_baseline", "overall_speedup",
              "overall_ci95_low", "overall_ci95_high", "factor_baseline", "factor_speedup",
              "o1_vs_o0_speedup", "thread_speedup", "pipeline_wall_p50_ms",
              "rss_baseline_median_bytes", "rss_loaded_median_bytes", "rss_peak_median_bytes",
              "load_p50_ms", "first_inference_p50_ms", "validation", "predictions", "errors", "recommendation"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in summary["experiments"]:
        overall, factor = row["comparisons"]["overall"], row["comparisons"]["factor"]
        ci = overall["ci95"] or {}
        item = {key: row[key] for key in ("id", "model_id", "variant_id", "threads", "status", "scheduled")}
        item.update({"blocks": row["inference"]["block_count"], "samples": row["inference"]["sample_count"],
                     **{key: _metric(row, key) for key in ("p50_ms", "p95_ms", "mean_ms", "std_ms")},
                     "overall_baseline": overall["baseline_id"], "overall_speedup": overall["speedup"],
                     "overall_ci95_low": ci.get("low"), "overall_ci95_high": ci.get("high"),
                     "factor_baseline": factor["baseline_id"], "factor_speedup": factor["speedup"],
                     "o1_vs_o0_speedup": _ratio(row, "ort_incremental"), "thread_speedup": _ratio(row, "threads"),
                     "pipeline_wall_p50_ms": _metric(row, "p50_ms", "pipeline_wall_ms", "pipeline"),
                     "load_p50_ms": (row["inference"]["load_ms"] or {}).get("p50_ms"),
                     "first_inference_p50_ms": (row["inference"]["first_inference_ms"] or {}).get("p50_ms")})
        for field in ("rss_baseline_bytes", "rss_loaded_bytes", "rss_peak_bytes"):
            item[field.replace("_bytes", "_median_bytes")] = (row["memory"][field] or {}).get("median")
        for field in ("validation", "predictions", "errors", "recommendation"):
            item[field] = json.dumps(row[field], ensure_ascii=False, sort_keys=True, allow_nan=False)
        for key, value in item.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                item[key] = "'" + value  # Prevent formula interpretation in spreadsheet viewers.
        writer.writerow(item)
    return output.getvalue()


def _render_html(summary: dict) -> str:
    parts = ["<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
             "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
             "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; script-src 'none'; base-uri 'none'; form-action 'none'\">",
             "<title>多模型阶段 A 对照报告</title><style>body{font:14px system-ui,sans-serif;margin:24px;color:#182233;background:#f5f7fa}h1,h2,h3{color:#20384d}table{border-collapse:collapse;width:100%;background:white;margin:12px 0}th,td{border:1px solid #cbd3dd;padding:7px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:1000px}svg{max-width:100%;height:auto;background:white}.notice{padding:12px;border-left:5px solid #c66;background:#fff}details{margin:8px 0}.muted{color:#536374}</style></head><body>",
             "<h1>多模型阶段 A 对照报告</h1>",
             f"<p>运行：{_escape(summary['run_id'])}；模式：{_escape(summary['mode'])}；状态：{_escape(summary['status'])}；配置：{_escape(summary['configuration_count'])}（计划执行 {_escape(summary['scheduled_configuration_count'])}）</p>",
             "<p class=\"notice\">阶段 A / CPU / batch=1。smoke 与 single_image 只支持探索；没有正式采用推荐。失败、缺数据与不支持项不填 0。Top-5 一致性不是数据集准确率。</p>",
             "<h2>阶段 B 数据门槛</h2><p>Development ≥128；固定性能子集64；Calibration ≥256；带标签Development ≥128；带标签Holdout ≥1000。阶段 B 尚未执行。</p>",
             f"<pre>{_escape(summary['dataset_readiness'])}</pre>",
             "<h2>范围与限制</h2><ul>"]
    parts.extend(f"<li>{_escape(value)}</li>" for value in summary["limitations"])
    parts.append("</ul><h2>三模型 × 30 配置</h2><p>速度比=对照/候选；整体对照为同模型 B0.t1。O1/O0 只在同线程数对照；线程变化另对同方案t1。CI按独立重复块重采样，至少5块，不拿迭代当独立样本。</p>")
    for model in summary["config"]["models"]:
        rows = [row for row in summary["experiments"] if row["model_id"] == model]
        parts.append(f"<h3>{_escape(model)}</h3>")
        valid = [row for row in rows if _metric(row, "p50_ms") is not None]
        if valid:
            maximum = max(_metric(row, "p50_ms") for row in valid)
            height = 30 + 26 * len(valid)
            parts.append(f'<svg viewBox="0 0 740 {height}" role="img" aria-label="推理P50绝对时延毫秒"><text x="10" y="20" fill="#20384d">推理 P50（ms）：数值越低越快</text>')
            for index, row in enumerate(valid):
                y = 40 + 26 * index
                value = _metric(row, "p50_ms")
                width = 470 * value / maximum
                parts.append(f'<text x="10" y="{y}" fill="#20384d">{_escape(row["variant_id"])} t{row["threads"]}</text><rect x="105" y="{y-14}" width="{width:.2f}" height="18" fill="#3576a8"></rect><text x="{115+width:.2f}" y="{y}" fill="#20384d">{_display(value)} ms</text>')
            parts.append("</svg>")
        parts.append("<table><thead><tr><th>方案/线程</th><th>状态</th><th>块/样本</th><th>推理P50/P95 ms</th><th>对B0.t1 / 95%块CI</th><th>因素自身对照</th><th>O1/O0</th><th>线程/t1</th><th>流水线wall P50 ms</th><th>独立RSS baseline/loaded/peak MiB</th><th>数值误差 / Top-5 / 采用</th></tr></thead><tbody>")
        for row in rows:
            overall, factor = row["comparisons"]["overall"], row["comparisons"]["factor"]
            ci = overall["ci95"]
            interval = "无CI（不足5独立块或无有效对照）" if ci is None else f"[{_display(ci['low'])}, {_display(ci['high'])}]"
            rss = [((row["memory"][field] or {}).get("median")) for field in ("rss_baseline_bytes", "rss_loaded_bytes", "rss_peak_bytes")]
            rss_text = " / ".join(_display(value / (1024**2)) if value is not None else "—" for value in rss)
            diagnostic = {"validation": row["validation"], "predictions": row["predictions"], "recommendation": row["recommendation"]}
            cells = [f"{_escape(row['variant_id'])} / {_escape(row['threads'])}", _escape(row["status"]),
                     f"{row['inference']['block_count']} / {row['inference']['sample_count']}",
                     f"{_display(_metric(row,'p50_ms'))} / {_display(_metric(row,'p95_ms'))}",
                     f"{_display(overall['speedup'])} / {interval}<br>{_escape(overall['status'])}",
                     f"{_escape(factor['baseline_id'])}<br>{_display(factor['speedup'])}",
                     _display(_ratio(row, "ort_incremental")), _display(_ratio(row, "threads")),
                     _display(_metric(row, "p50_ms", "pipeline_wall_ms", "pipeline")), rss_text,
                     f"<details><summary>数值/Top-5/限制</summary><pre>{_escape(diagnostic)}</pre></details>"]
            parts.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        parts.append("</tbody></table>")
        for row in rows:
            details = {"id": row["id"], "status_counts": row["status_counts"], "inference": row["inference"],
                       "pipeline": row["pipeline"], "memory": row["memory"], "comparisons": row["comparisons"],
                       "model_metadata": row["model_metadata"], "variant_metadata": row["variant_metadata"], "errors": row["errors"]}
            parts.append(f"<details><summary>{_escape(row['id'])}：块/PID、加载/首轮、阶段、独立内存与原始错误</summary><pre>{_escape(details)}</pre></details>")
    parts.append("<h2>完整性错误（不静默丢弃）</h2>")
    parts.append(f"<pre>{_escape(summary['integrity_errors'])}</pre>")
    failed = [{"file": record["source_file"], "status": record["status"], "original_status": record["original_status"],
               "issues": record["issues"], "data": record["data"]} for record in summary["records"] if record["status"] != "passed"]
    parts.append(f"<details><summary>全部失败、blocked与interrupted记录</summary><pre>{_escape(failed)}</pre></details>")
    audit = {key: summary[key] for key in ("config_sha256", "manifest_sha256", "code_fingerprint", "environment", "prepared_models", "methodology")}
    parts.append(f"<details><summary>冻结协议、环境与溯源</summary><pre>{_escape(audit)}</pre></details></body></html>")
    return "".join(parts)


def write_report(run_dir: str | Path, summary: dict, *, overwrite: bool = False) -> dict:
    """Stage all reports, then atomically publish; summary.json is the last marker."""
    if not isinstance(overwrite, bool):
        raise BenchmarkError("overwrite must be bool")
    directory = Path(run_dir)
    if not directory.is_dir():
        raise BenchmarkError("Report directory must already exist")
    _object(summary, "summary")
    _version(summary, "summary")
    json_content = _json_bytes(summary)
    try:
        csv_content = _csv_report(summary).encode("utf-8")
        html_content = _render_html(summary).encode("utf-8")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise BenchmarkError(f"Invalid report summary: {error}") from error
    outputs = {"summary_json": directory / "summary.json", "summary_csv": directory / "summary.csv",
               "report_html": directory / "report.html"}
    for path in outputs.values():
        if path.is_symlink() or (path.exists() and (not overwrite or not path.is_file())):
            raise BenchmarkError(f"Refusing to overwrite report: {path}")
    contents = {"summary_json": json_content, "summary_csv": csv_content, "report_html": html_content}
    staged, backups, published = {}, {}, []
    try:
        for key, path in outputs.items():
            if path.exists():
                backups[key] = path.read_bytes()
            with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                             dir=directory, delete=False) as handle:
                staged[key] = Path(handle.name)
                handle.write(contents[key])
                handle.flush()
                os.fsync(handle.fileno())
        _read_json(staged["summary_json"])
        if len(list(csv.DictReader(io.StringIO(csv_content.decode("utf-8"))))) != len(summary["experiments"]):
            raise BenchmarkError("Temporary CSV verification failed")
        for key in ("summary_csv", "report_html", "summary_json"):
            path = outputs[key]
            if overwrite:
                os.replace(staged[key], path)
            else:
                os.link(staged[key], path)  # Atomic no-clobber; a concurrent writer cannot be overwritten.
            published.append(key)
        return {key: str(path) for key, path in outputs.items()}
    except (OSError, BenchmarkError) as error:
        rollback_errors = []
        for key in reversed(published):
            try:
                if key in backups:
                    with tempfile.NamedTemporaryFile(mode="wb", prefix=".report-rollback.", suffix=".tmp",
                                                     dir=directory, delete=False) as handle:
                        backup_path = Path(handle.name)
                        handle.write(backups[key])
                    try:
                        os.replace(backup_path, outputs[key])
                    finally:
                        backup_path.unlink(missing_ok=True)
                else:
                    outputs[key].unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        note = f"; rollback errors: {rollback_errors}" if rollback_errors else ""
        raise BenchmarkError(f"Report publication failed: {error}{note}") from error
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
