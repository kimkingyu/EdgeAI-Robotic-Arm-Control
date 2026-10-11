#!/usr/bin/env python3
"""Isolated, CPU-only classifier experiments; no hardware or implicit downloads."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_FILES = (
    "src/vision/__init__.py", "src/vision/classification.py",
    "src/vision/classification_variants.py", "src/vision/classification_benchmark.py",
    "src/vision/classification_controls.py", "tools/run_classifier_benchmark.py", "tools/export_classifier_onnx.py",
    "tools/summarize_classifier_benchmark.py", "requirements-resnet.txt",
)
LIMITATIONS = [
    "CPU-only, batch=1. No camera, robot, training, RKNN or NPU benchmark.",
    "Smoke is exploratory, not dataset accuracy or a formal adoption recommendation.",
    "Timing starts from prepared tensors or decoded RGB, not file I/O.",
    "Memory uses a separate process and sampled RSS, not model-exclusive memory.",
    "Configuration selection and final holdout evaluation require disjoint datasets.",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value, *, overwrite=False):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and not overwrite:
        raise ValueError(f"Output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        if path.exists() and not overwrite:
            raise ValueError(f"Output appeared while writing: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_sample_log(output, result):
    """Publish the real raw samples separately, with an integrity hash."""
    if not result["samples"]:
        return
    target = Path(output).with_suffix(".samples.jsonl")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for sample in result["samples"]:
                row = {"experiment_id": result["experiment"]["id"], "kind": result["kind"],
                       "block": result["block"], "worker_pid": result["worker_pid"],
                       "worker_identity": result.get("worker_identity"), **sample}
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if target.exists():
            raise ValueError("Raw sample output already exists")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    result["samples_path"] = str(target)
    result["samples_sha256"] = sha256(target)


def source_fingerprint():
    files = {name: sha256(ROOT / name) if (ROOT / name).is_file() else None
             for name in SOURCE_FILES}
    return canonical_hash(files), files


def environment_info():
    versions = {}
    for distribution in ("torch", "torchvision", "numpy", "Pillow", "onnx", "onnxruntime"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    git = {}
    for key, command in (("head", ["git", "rev-parse", "HEAD"]),
                         ("status", ["git", "status", "--short"])):
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", timeout=10)
            git[key] = completed.stdout.strip() if completed.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            git[key] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(), "physical_cpu_count": None,
            "cpu_topology": "unverified", "power_policy": "not changed or verified",
            "versions": versions, "git": git}


class RssReader:
    """Native working-set reader; no extra package or system configuration."""
    def __init__(self):
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
            kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            function = psapi.GetProcessMemoryInfo
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            function.restype = wintypes.BOOL
            self._native = (ctypes, Counters, function, kernel.GetCurrentProcess())
            self.source = "Windows GetProcessMemoryInfo.WorkingSetSize"
        elif sys.platform.startswith("linux") and Path("/proc/self/statm").is_file():
            self._page_size = os.sysconf("SC_PAGE_SIZE")
            self.source = "Linux /proc/self/statm resident pages"
        else:
            raise RuntimeError("Sampled current RSS is unsupported on this platform")

    def read(self):
        if sys.platform == "win32":
            ctypes, counter_type, function, process = self._native
            counters = counter_type()
            counters.cb = ctypes.sizeof(counters)
            if not function(process, ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
            return int(counters.WorkingSetSize)
        return int(Path("/proc/self/statm").read_text().split()[1]) * self._page_size


class RssSampler:
    def __init__(self, reader, interval=0.02):
        self.reader, self.interval = reader, interval
        self.values = []
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        try:
            while not self._stop.is_set():
                self.values.append(self.reader.read())
                self._stop.wait(self.interval)
        except Exception as exc:
            self.error = str(exc)

    def __enter__(self):
        self.values.append(self.reader.read())
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)
        self.values.append(self.reader.read())


def timed_pipeline(image, prepare, adapt, infer, post, *, clock=time.perf_counter_ns):
    """Input adaptation belongs to its own stage; total wall time is measured."""
    wall_start = clock()
    started = clock()
    tensor = prepare(image)
    prepared = clock()
    value = adapt(tensor)
    adapted = clock()
    logits = infer(value)
    inferred = clock()
    prediction = post(logits)
    finished = clock()
    timing = {"preprocess_ms": (prepared - started) / 1e6,
              "input_adapt_ms": (adapted - prepared) / 1e6,
              "inference_ms": (inferred - adapted) / 1e6,
              "postprocess_ms": (finished - inferred) / 1e6}
    timing["stages_total_ms"] = sum(timing.values())
    timing["pipeline_wall_ms"] = (clock() - wall_start) / 1e6
    timing["wrapper_overhead_ms"] = timing["pipeline_wall_ms"] - timing["stages_total_ms"]
    if any(not math.isfinite(value) or value < 0 for value in timing.values()):
        raise ValueError("Invalid stage accounting")
    return prediction, timing


def _numpy_logits(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def worker_process_identity(pid=None):
    """Identify an OS process instance, not a reusable integer PID."""
    actual_pid = os.getpid() if pid is None else pid
    if type(actual_pid) is not int or actual_pid < 1:
        raise ValueError("Invalid process ID")
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, actual_pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), "OpenProcess for identity failed")
        try:
            creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
            ticks = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
            source = "Windows GetProcessTimes creation FILETIME (100ns since 1601)"
            identity = f"{actual_pid}:{ticks}"
        finally:
            kernel.CloseHandle(handle)
    elif sys.platform.startswith("linux"):
        raw = Path(f"/proc/{actual_pid}/stat").read_text()
        ticks = int(raw[raw.rfind(")") + 2:].split()[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        source = "Linux /proc PID stat start ticks plus boot_id"
        identity = f"{actual_pid}:{ticks}:{boot_id}"
    else:
        raise RuntimeError("Native process-instance identity is unsupported on this platform")
    if ticks <= 0:
        raise ValueError("OS process creation marker must be positive")
    return {"worker_pid": actual_pid, "worker_identity": identity,
            "worker_creation_ticks": ticks, "worker_identity_source": source}


def _job_base(job):
    return {"schema_version": 1, "run_id": job["run_id"], "status": "running",
            "experiment": job["experiment"], "block": job["block"],
            **worker_process_identity(), "kind": job["kind"],
            "config_sha256": job["config_sha256"],
            "manifest_sha256": job["manifest_sha256"],
            "code_fingerprint": job["code_fingerprint"],
            "job_sha256": canonical_hash(job), "warmup": job["warmup"],
            "iterations": job["iterations"],
            "expected_sample_count": job["iterations"] if job["kind"] in ("inference", "pipeline", "control") else 0,
            "samples": [], "model_metadata": job["model_metadata"],
            "limitations": list(LIMITATIONS)}


def _validate_job(job_path, job):
    run_dir = Path(job["run_dir"]).resolve()
    if job_path.resolve().parent != run_dir / "jobs":
        raise ValueError("Worker job must be inside this run's jobs directory")
    output = Path(job["output_path"]).resolve()
    result_folder = {"profile": "profiles", "control": "controls"}.get(job["kind"], "results")
    if output.parent != run_dir / result_folder or output.suffix != ".json":
        raise ValueError("Worker result must be inside this run's assigned result directory")
    if job.get("schema_version") != 1 or job["kind"] not in ("inference", "pipeline", "memory", "profile", "control"):
        raise ValueError("Invalid worker schema or workload")
    experiment = job["experiment"]
    if experiment["model_id"] not in ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"):
        raise ValueError("Unregistered model")
    if experiment["variant_id"] not in ("B0", "F1", "L1", "FL1", "O0", "O1"):
        raise ValueError("Unregistered variant")
    for field in ("threads",):
        if type(experiment[field]) is not int or experiment[field] < 1:
            raise ValueError(f"Invalid {field}")
    for field, minimum in (("iterations", 1), ("warmup", 0), ("block", 0)):
        if type(job[field]) is not int or job[field] < minimum:
            raise ValueError(f"Invalid {field}")
    for field in ("rtol", "atol"):
        if isinstance(job[field], bool) or not math.isfinite(job[field]) or job[field] < 0:
            raise ValueError("Invalid numerical tolerance")
    fingerprint, _ = source_fingerprint()
    if fingerprint != job["code_fingerprint"]:
        raise ValueError("Experiment source changed after freezing the run")
    for path_key, hash_key in (("weights_path", "weights_sha256"),
                               ("reference_path", "reference_sha256")):
        if sha256(job[path_key]) != job[hash_key]:
            raise ValueError(f"Artifact hash mismatch: {path_key}")
    for entry in job["images"]:
        if sha256(entry["resolved_path"]) != entry["sha256"]:
            raise ValueError("Input image changed after freezing the run")
    reference = Path(job["reference_path"]).resolve()
    if run_dir / "prepared" not in reference.parents:
        raise ValueError("Reference must belong to this run's prepared data")
    if experiment["variant_id"].startswith("O"):
        if sha256(job["onnx_path"]) != job["onnx_sha256"]:
            raise ValueError("ONNX artifact hash mismatch")
    return output


def worker(job_path):
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = _validate_job(job_path, job)
    result = _job_base(job)
    rss_reader, rss_baseline = None, None
    if job["kind"] == "memory":
        try:
            rss_reader = RssReader()
            rss_baseline = rss_reader.read()
        except Exception as exc:
            result.update(status="unsupported", error=str(exc))
            atomic_json(output, result)
            return 2
    try:
        import numpy as np
        from src.vision.classification import (
            configure_cpu_threads, load_classifier, open_rgb_image,
            prepare_image, compare_logits, topk_predictions,
        )
        from src.vision.classification_variants import build_torch_variant, create_ort_session

        experiment = job["experiment"]
        model_id, variant_id, threads = (experiment[name]
                                        for name in ("model_id", "variant_id", "threads"))
        with np.load(job["reference_path"], allow_pickle=False) as reference:
            inputs = reference["inputs"].copy()
            expected = reference["logits"].copy()
        if (inputs.dtype != np.float32 or expected.dtype != np.float32
                or inputs.shape != (len(job["images"]), 3, 224, 224)
                or expected.shape != (len(job["images"]), 1000)
                or not np.isfinite(inputs).all() or not np.isfinite(expected).all()):
            raise ValueError("Prepared input/reference contract mismatch")
        categories = job["model_metadata"]["categories"]
        if len(categories) != 1000:
            raise ValueError("Class mapping must contain 1000 categories")
        started = time.perf_counter_ns()
        profile_enabled = job["kind"] == "profile"
        if variant_id.startswith("O"):
            session = create_ort_session(
                job["onnx_path"], threads=threads, variant_id=variant_id,
                profiling=profile_enabled,
                profile_file_prefix=str(Path(job["run_dir"]) / "profiles" / job["job_name"])
                if profile_enabled else None,
            )
            engine_input = [np.ascontiguousarray(inputs[index:index + 1]) for index in range(len(inputs))]
            infer = lambda value: session.run(["logits"], {"images": value})[0]
            adapt = lambda tensor: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
            context = nullcontext()
            if job["kind"] == "pipeline":
                # torchvision preprocessing must use the same CPU thread budget.
                configure_cpu_threads(threads, inter_op_threads=1)
            result["variant_metadata"] = {
                "backend": "onnxruntime", "memory_format": "contiguous",
                "providers": session.get_providers(), "fused_pair_count": None,
                "graph_optimization": "ORT_DISABLE_ALL" if variant_id == "O0" else "ORT_ENABLE_ALL",
                "intra_op_threads": threads, "inter_op_threads": 1,
                "execution_mode": "ORT_SEQUENTIAL", "allow_spinning": 0,
                "onnx_sha256": job["onnx_sha256"],
            }
        else:
            import torch
            configure_cpu_threads(threads, inter_op_threads=1)
            original, metadata = load_classifier(model_id, weights_path=job["weights_path"],
                                                  download_weights=False, threads=threads)
            model, variant_metadata = build_torch_variant(original, model_id, variant_id)
            del original
            memory_format = (torch.channels_last if variant_metadata["memory_format"] == "channels_last"
                             else torch.contiguous_format)
            engine_input = [torch.from_numpy(inputs[index:index + 1]).contiguous(memory_format=memory_format)
                            for index in range(len(inputs))]
            infer = model
            adapt = lambda tensor: tensor.contiguous(memory_format=memory_format)
            context = torch.inference_mode()
            result["model_metadata"] = metadata
            result["variant_metadata"] = {**variant_metadata, "backend": "pytorch",
                                           "intra_op_threads": threads, "inter_op_threads": 1}
        result["load_ms"] = (time.perf_counter_ns() - started) / 1e6
        prepare = lambda image: prepare_image(image, model_id=model_id)
        post = lambda logits: topk_predictions(logits, categories, k=5)
        sequence = job["image_sequence"]
        if (len(sequence) != job["iterations"] or any(type(index) is not int or index < 0
                                                     or index >= len(inputs) for index in sequence)):
            raise ValueError("Invalid frozen image sequence")
        comparisons = []
        with context:
            for index, tensor in enumerate(engine_input):
                began = time.perf_counter_ns()
                actual = infer(tensor)
                call_ms = (time.perf_counter_ns() - began) / 1e6
                if index == 0:
                    result["first_inference_ms"] = call_ms
                    result["predictions"] = post(actual)
                comparisons.append(compare_logits(expected[index:index + 1], _numpy_logits(actual),
                                                  rtol=job["rtol"], atol=job["atol"]))
            result["validation"] = {
                "allclose": all(entry["allclose"] for entry in comparisons),
                "passed": all(entry["passed"] for entry in comparisons),
                "image_count": len(comparisons),
                "max_abs_error": max(entry["max_abs_error"] for entry in comparisons),
                "mean_abs_error": sum(entry["mean_abs_error"] for entry in comparisons) / len(comparisons),
                "top1_equal": all(entry["top1_equal"] for entry in comparisons),
                "top5_equal": all(entry["top5_equal"] for entry in comparisons),
                "rtol": job["rtol"], "atol": job["atol"], "per_image": comparisons,
            }
            if not result["validation"]["passed"]:
                result.update(status="failed_correctness", error="FP32 output comparison failed")
                atomic_json(output, result)
                return 2
            decoded = ([open_rgb_image(entry["resolved_path"]) for entry in job["images"]]
                       if job["kind"] == "pipeline" else None)
            for index in range(job["warmup"]):
                image_index = index % len(inputs)
                if decoded is None:
                    infer(engine_input[image_index])
                else:
                    timed_pipeline(decoded[image_index], prepare, adapt, infer, post)
            measured_wall = time.perf_counter_ns()
            if job["kind"] == "memory":
                loaded = rss_reader.read()
                with RssSampler(rss_reader) as sampler:
                    for index in sequence:
                        infer(engine_input[index])
                if sampler.error:
                    raise RuntimeError(f"RSS sampler failed: {sampler.error}")
                result["memory"] = {
                    "rss_baseline_bytes": rss_baseline, "rss_loaded_bytes": loaded,
                    "rss_peak_bytes": max(sampler.values), "rss_source": rss_reader.source,
                    "sampling_interval_ms": sampler.interval * 1000,
                    "sample_count": len(sampler.values),
                    "limitations": ["Sampled RSS can miss brief peaks.",
                                    "Includes runtime, prepared inputs, dependencies and allocator caches; not model-exclusive memory.",
                                    "Memory observation is separate from timed performance workers."],
                }
            elif job["kind"] == "profile":
                if variant_id.startswith("O"):
                    for index in sequence[:5]:
                        infer(engine_input[index])
                    profile_path = session.end_profiling()
                    result["profile"] = {"backend": "onnxruntime", "trace_path": profile_path,
                                           "included_in_timing": False}
                else:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                record_shapes=True) as profiler:
                        for index in sequence[:5]:
                            infer(engine_input[index])
                    trace_path = Path(job["run_dir"]) / "profiles" / f"{job['job_name']}.trace.json"
                    profiler.export_chrome_trace(str(trace_path))
                    hottest = sorted(profiler.key_averages(), key=lambda event: event.self_cpu_time_total,
                                     reverse=True)[:20]
                    result["profile"] = {
                        "backend": "pytorch", "trace_path": str(trace_path), "included_in_timing": False,
                        "operators": [{"name": entry.key, "calls": entry.count,
                                       "self_cpu_time_us": entry.self_cpu_time_total,
                                       "cpu_time_us": entry.cpu_time_total} for entry in hottest],
                    }
            else:
                for index in sequence:
                    if decoded is not None:
                        _, timing = timed_pipeline(decoded[index], prepare, adapt, infer, post)
                    else:
                        began = time.perf_counter_ns()
                        infer(engine_input[index])
                        timing = {"inference_ms": (time.perf_counter_ns() - began) / 1e6}
                    result["samples"].append({"image_id": job["images"][index]["id"], **timing})
            result["measured_wall_ms"] = (time.perf_counter_ns() - measured_wall) / 1e6
        result["sample_count"] = len(result["samples"])
        result["runtime_modules"] = {name: name in sys.modules for name in ("torch", "torchvision", "onnxruntime")}
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed_runtime", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    write_sample_log(output, result)
    atomic_json(output, result)
    return 0 if result["status"] == "passed" else 2


def inline_manifest(image_path):
    path = Path(image_path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Image does not exist or is empty: {path}")
    source_path = path.with_name(f"{path.stem}.source.json")
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else {"kind": "user_provided"}
    entry = {"id": "image_0000", "path": path.name, "resolved_path": str(path),
             "sha256": sha256(path), "split": "smoke", "label": None,
             "source_id": sha256(path), "source": source, "license": "not independently verified"}
    payload = {"schema_version": 1, "root": str(path.parent), "images": [entry]}
    return {**payload, "manifest_sha256": canonical_hash(payload), "inline": True}


def _worker_environment(threads):
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
    return environment


def child_python(threads):
    """Bypass the Windows venv redirector without losing its site-packages.

    Running the base executable with the launcher's venv environment keeps the
    spawned PID equal to the real worker PID, including timeout termination.
    """
    environment = _worker_environment(threads)
    executable = sys.executable
    base = getattr(sys, "_base_executable", None)
    if os.name == "nt" and sys.prefix != sys.base_prefix and base and Path(base).is_file():
        environment["__PYVENV_LAUNCHER__"] = sys.executable
        executable = base
    return executable, environment


def _prepare_models(run_dir, config, images, *, download_weights=False, models=None, variants=None):
    import numpy as np
    import torch
    from src.vision.classification import load_classifier, open_rgb_image, prepare_image

    prepared = {}
    for model_id in models or config["models"]:
        print(f"Preparing {model_id}: fixed official weights and reference outputs", flush=True)
        model, metadata = load_classifier(model_id, download_weights=download_weights, threads=1)
        if metadata.get("category_mapping_assumed") or metadata.get("weights_source") == "local_custom":
            raise ValueError("Formal registry experiments require fixed official V1 weights")
        model_dir = run_dir / "prepared" / model_id
        model_dir.mkdir(parents=True, exist_ok=False)
        inputs, logits = [], []
        with torch.inference_mode():
            for entry in images:
                if sha256(entry["resolved_path"]) != entry["sha256"]:
                    raise ValueError("Input image changed after manifest validation")
                tensor = prepare_image(open_rgb_image(entry["resolved_path"]), model_id=model_id)
                inputs.append(tensor.numpy()[0])
                logits.append(model(tensor).detach().cpu().numpy()[0])
        reference_path = model_dir / "reference.npz"
        np.savez(reference_path, inputs=np.stack(inputs).astype(np.float32),
                 logits=np.stack(logits).astype(np.float32))
        item = {"model_metadata": metadata, "weights_path": metadata["weights_path"],
                "weights_sha256": metadata["weights_sha256"],
                "reference_path": str(reference_path), "reference_sha256": sha256(reference_path)}
        if any(variant.startswith("O") for variant in (variants or config["variants"])):
            exporter_fingerprint = canonical_hash({
                "sources": {name: sha256(ROOT / name) for name in (
                    "src/vision/classification.py", "src/vision/classification_variants.py",
                    "tools/export_classifier_onnx.py")},
                "versions": {name: importlib.metadata.version(name) for name in (
                    "torch", "torchvision", "onnx", "onnxruntime")},
            })
            cache_tag = f"{metadata['weights_sha256'][:12]}-{exporter_fingerprint[:12]}"
            cache_dir = ROOT / "models" / "weights" / "classifier-benchmark" / model_id / cache_tag
            cache_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = cache_dir / "model-fp32.onnx"
            metadata_path = onnx_path.with_suffix(".json")
            if not onnx_path.exists() and not metadata_path.exists():
                parameters = {"model_id": model_id, "image_paths": [entry["resolved_path"] for entry in images],
                              "output_path": str(onnx_path), "weights_path": metadata["weights_path"],
                              "download_weights": False, "threads": 1, "overwrite": False}
                code = ("import json,sys; from tools.export_classifier_onnx import export_and_verify; "
                        "export_and_verify(**json.loads(sys.argv[1]))")
                executable, environment = child_python(1)
                with (run_dir / "logs" / f"export-{model_id}.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run([executable, "-c", code, json.dumps(parameters)],
                                               cwd=ROOT, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=900)
                if completed.returncode != 0:
                    raise RuntimeError(f"ONNX export failed; see logs/export-{model_id}.log")
            if not onnx_path.is_file() or not metadata_path.is_file():
                raise ValueError(f"Incomplete cached ONNX pair: {onnx_path}")
            exported = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (exported["model"]["weights_sha256"] != metadata["weights_sha256"]
                    or exported["onnx"]["sha256"] != sha256(onnx_path)):
                raise ValueError(f"Cached ONNX provenance mismatch: {onnx_path}")
            item.update(onnx_path=str(onnx_path), onnx_sha256=sha256(onnx_path),
                        export_metadata=exported)
        prepared[model_id] = item
        del model
    return prepared


def make_job(run, run_dir, experiment, block, kind, images, budget):
    prepared = run["prepared_models"][experiment["model_id"]]
    sequence = []
    generator = random.Random(run["config"]["seed"] + block)
    while len(sequence) < budget["iterations"]:
        cycle = list(range(len(images)))
        generator.shuffle(cycle)
        sequence.extend(cycle)
    name = f"{experiment['id']}.{kind}.b{block}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Unsafe experiment job ID")
    return {"schema_version": 1, "run_id": run["run_id"], "run_dir": str(run_dir),
            "job_name": name, "experiment": experiment, "block": block, "kind": kind,
            "config_sha256": run["config_sha256"], "manifest_sha256": run["manifest_sha256"],
            "code_fingerprint": run["code_fingerprint"],
            "warmup": budget["warmup"], "iterations": budget["iterations"],
            "image_sequence": sequence[:budget["iterations"]], "images": images,
            "rtol": run["config"]["rtol"], "atol": run["config"]["atol"],
            "output_path": str(run_dir / {"profile": "profiles", "control": "controls"}.get(kind, "results") / f"{name}.json"), **prepared}


def run_job(job, run_dir, timeout):
    path = run_dir / "jobs" / f"{job['job_name']}.json"
    output = Path(job["output_path"])
    atomic_json(path, job)
    process = None
    try:
        executable, environment = child_python(job["experiment"]["threads"])
        with (run_dir / "logs" / f"{job['job_name']}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [executable, str(Path(__file__).resolve()), "--worker", str(path)],
                cwd=ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                launch_identity = worker_process_identity(process.pid)
                returncode = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                process.kill()
                process.wait(timeout=10)
                raise
        if output.is_file():
            result = json.loads(output.read_text(encoding="utf-8"))
            if (result.get("job_sha256") != canonical_hash(job)
                    or any(result.get(key) != value for key, value in launch_identity.items())):
                raise RuntimeError("Worker result provenance mismatch")
            if returncode != 0 and result.get("status") == "passed":
                raise RuntimeError("Worker returned failure after a passed result")
            return result
        error, status = f"Worker produced no result (exit {returncode})", "failed_runtime"
    except subprocess.TimeoutExpired:
        error, status = f"Worker exceeded timeout {timeout}s", "failed_runtime"
    except KeyboardInterrupt:
        result = _job_base(job)
        result.update(status="interrupted", worker_pid=process.pid if process else None,
                      error="Experiment interrupted by user", samples=[],
                      worker_identity=None, worker_creation_ticks=None, worker_identity_source=None)
        atomic_json(output, result, overwrite=True)
        raise
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        error, status = f"{type(exc).__name__}: {exc}", "failed_runtime"
    result = _job_base(job)
    result.update(status=status, worker_pid=process.pid if process else None, error=error, samples=[],
                  worker_identity=None, worker_creation_ticks=None, worker_identity_source=None)
    atomic_json(output, result, overwrite=True)
    return result


def resume_result(job, run_dir):
    """Validate each resumed job; archive failures outside the aggregation input."""
    output = Path(job["output_path"])
    if not output.is_file():
        return None
    previous = json.loads(output.read_text(encoding="utf-8"))
    if (previous.get("job_sha256") != canonical_hash(job)
            or previous.get("config_sha256") != job["config_sha256"]
            or previous.get("manifest_sha256") != job["manifest_sha256"]
            or previous.get("code_fingerprint") != job["code_fingerprint"]):
        raise ValueError("Resume result fingerprint mismatch")
    if previous.get("status") == "passed":
        if previous.get("sample_count") != len(previous.get("samples", [])):
            raise ValueError("Resume sample accounting mismatch")
        return previous
    archive = run_dir / "archived-results"
    archive.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    output.replace(archive / f"{output.stem}.{stamp}.json")
    old_job = run_dir / "jobs" / f"{job['job_name']}.json"
    if old_job.exists():
        old_job.replace(archive / f"{old_job.stem}.{stamp}.job.json")
    sample_file = output.with_suffix(".samples.jsonl")
    if sample_file.exists():
        sample_file.replace(archive / f"{output.stem}.{stamp}.samples.jsonl")
    return None


def _publish(run_dir, *, overwrite=True):
    from src.vision.classification_benchmark import aggregate_results, write_report
    summary = aggregate_results(run_dir)
    paths = write_report(run_dir, summary, overwrite=overwrite)
    return summary, paths


def _resume(run_dir, args, config, manifest, fingerprint):
    path = run_dir / "run.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    if (previous["config_sha256"] != canonical_hash(config)
            or previous["manifest_sha256"] != manifest["manifest_sha256"]
            or previous["code_fingerprint"] != fingerprint or previous["mode"] != args.mode):
        raise ValueError("Resume requires matching source, config, mode and dataset hashes")
    for item in previous["prepared_models"].values():
        for file_key, hash_key in (("weights_path", "weights_sha256"),
                                   ("reference_path", "reference_sha256"),
                                   ("onnx_path", "onnx_sha256")):
            if file_key in item and sha256(item[file_key]) != item[hash_key]:
                raise ValueError(f"Resume artifact mismatch: {file_key}")
    return previous


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return value


def nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal"), default="preflight")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image", type=Path, help="One unlabelled image; exploratory smoke only")
    parser.add_argument("--models", nargs="+", choices=("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"))
    parser.add_argument("--variants", nargs="+", choices=("B0", "F1", "L1", "FL1", "O0", "O1"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--no-thread-sweep", action="store_true")
    parser.add_argument("--no-observations", action="store_true", help="Skip separate pipeline/RSS observations")
    parser.add_argument("--profile", action="store_true", help="Separate diagnostic traces; never timed")
    parser.add_argument("--warmup", type=nonnegative)
    parser.add_argument("--iterations", type=positive)
    parser.add_argument("--repeats", type=positive)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    try:
        from src.vision.classification_benchmark import (
            load_config, load_manifest, dataset_readiness, select_images, expand_experiments,
        )
        if args.image and args.manifest:
            raise ValueError("Choose either --image or --manifest")
        config = load_config(args.config)
        mode = "formal" if args.mode == "formal" else "smoke"
        for field in ("warmup", "iterations", "repeats"):
            if getattr(args, field) is not None:
                config[mode][field] = getattr(args, field)
        manifest = (load_manifest(args.manifest) if args.manifest
                    else inline_manifest(args.image or ROOT / "data" / "resnet-demo" / "dog.jpg"))
        readiness = dataset_readiness(manifest)
        experiments = expand_experiments(config, include_thread_sweep=not args.no_thread_sweep)
        if args.models:
            experiments = [entry for entry in experiments if entry["model_id"] in args.models]
        if args.variants:
            experiments = [entry for entry in experiments if entry["variant_id"] in args.variants]
        if not experiments:
            raise ValueError("Empty experiment selection")
        active_models = [name for name in config["models"] if any(e["model_id"] == name for e in experiments)]
        active_variants = [name for name in config["variants"] if any(e["variant_id"] == name for e in experiments)]
        fingerprint, sources = source_fingerprint()
        dependencies = {name: importlib.util.find_spec(name) is not None
                        for name in ("torch", "torchvision", "numpy", "PIL", "onnx", "onnxruntime")}
        if args.mode == "preflight":
            print(json.dumps({"schema_version": 1, "stage": "A", "dataset_readiness": readiness,
                              "experiment_count": len(experiments), "dependencies": dependencies,
                              "source_complete": all(value is not None for value in sources.values()),
                              "no_download_or_model_execution": True, "environment": environment_info(),
                              "limitations": LIMITATIONS}, ensure_ascii=False, indent=2))
            return 0 if all(dependencies.values()) and readiness["smoke"]["ready"] else 2
        if not all(dependencies.values()):
            raise RuntimeError("Required CPU demo dependencies are missing; no Mock fallback")
        images = select_images(manifest, mode=mode, seed=config["seed"])
        run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
        run_dir = (args.output_dir or ROOT / "build" / "classifier-benchmark" / run_id).resolve()
        if args.resume:
            run = _resume(run_dir, args, config, manifest, fingerprint)
            if run["experiments"] != experiments:
                raise ValueError("Resume requires the same model/variant/thread experiment selection")
            current = environment_info()
            for key in ("versions", "processor", "machine", "logical_cpu_count"):
                if run["environment"].get(key) != current.get(key):
                    raise ValueError(f"Resume environment changed: {key}")
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError("Run directory is not empty; choose another directory or --resume")
            for folder in ("jobs", "results", "logs", "prepared", "profiles", "controls"):
                (run_dir / folder).mkdir(parents=True, exist_ok=True)
            run = {"schema_version": 1, "run_id": run_id, "created_at": datetime.now().astimezone().isoformat(),
                   "mode": mode, "status": "running", "phase": "preparing", "config": config,
                   "config_sha256": canonical_hash(config), "manifest_sha256": manifest["manifest_sha256"],
                   "code_fingerprint": fingerprint, "source_hashes": sources,
                   "environment": environment_info(), "dataset_readiness": readiness,
                   "experiments": experiments, "images": images, "prepared_models": {},
                   "selection": {"models": active_models, "variants": active_variants,
                                 "include_thread_sweep": not args.no_thread_sweep},
                   "observations_enabled": not args.no_observations, "profile_enabled": args.profile,
                   "limitations": LIMITATIONS + (["Only one distinct image; not formal performance or accuracy."]
                                                  if len(images) == 1 else [])}
            atomic_json(run_dir / "run.json", run)
            atomic_json(run_dir / "dataset.json", manifest)
            try:
                run["prepared_models"] = _prepare_models(
                    run_dir, config, images, download_weights=args.download_weights,
                    models=active_models, variants=active_variants,
                )
            except Exception as exc:
                run.update(status="failed_runtime", phase="preparation_failed",
                           error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                atomic_json(run_dir / "run.json", run, overwrite=True)
                _publish(run_dir)
                raise
        if run["observations_enabled"] != (not args.no_observations) or run["profile_enabled"] != args.profile:
            raise ValueError("Resume must retain the observation/profile settings")
        run.update(status="running", phase="measuring")
        atomic_json(run_dir / "run.json", run, overwrite=True)
        budget = config[mode]
        completed_results = []
        diagnostics = []
        try:
            if mode == "formal":
                from src.vision.classification_controls import run_formal_groups
                run["drift_protocol"] = {"threshold": 1.10, "max_attempts": 2,
                                         "scope": "model/block", "control": "B0.t1"}
                atomic_json(run_dir / "run.json", run, overwrite=True)
                completed_results.extend(run_formal_groups(
                    run, run_dir, experiments, images, budget, make_job=make_job,
                    run_job=run_job, resume_result=resume_result, write_json=atomic_json,
                    resuming=args.resume,
                ))
            else:
                for block in range(budget["repeats"]):
                    order = list(experiments)
                    random.Random(config["seed"] + block).shuffle(order)
                    for experiment in order:
                        job = make_job(run, run_dir, experiment, block, "inference", images, budget)
                        previous = resume_result(job, run_dir) if args.resume else None
                        result = previous or run_job(job, run_dir, config["worker_timeout_s"])
                        completed_results.append(result)
                        print(f"block {block + 1}/{budget['repeats']} {experiment['id']}: {result['status']}", flush=True)
            kinds = ([] if args.no_observations else ["pipeline", "memory"]) + (["profile"] if args.profile else [])
            for kind in kinds:
                for experiment in experiments:
                    if experiment["threads"] != 1:
                        continue
                    job = make_job(run, run_dir, experiment, 0, kind, images, budget)
                    previous = resume_result(job, run_dir) if args.resume else None
                    result = previous or run_job(job, run_dir, config["worker_timeout_s"])
                    (diagnostics if kind == "profile" else completed_results).append(result)
                    print(f"{kind} {experiment['id']}: {result['status']}", flush=True)
        except KeyboardInterrupt:
            run["status"] = "interrupted"
            atomic_json(run_dir / "run.json", run, overwrite=True)
            _publish(run_dir)
            return 130
        except Exception as exc:
            run.update(status="failed_runtime", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
            atomic_json(run_dir / "run.json", run, overwrite=True)
            _publish(run_dir)
            raise
        statuses = Counter(entry["status"] for entry in completed_results)
        all_results = completed_results + diagnostics
        run.update(status="completed" if all(entry["status"] == "passed" for entry in all_results)
                   else "completed_with_failures", phase="finished",
                   completed_at=datetime.now().astimezone().isoformat(), result_status_counts=dict(statuses),
                   diagnostic_profile_count=len(diagnostics),
                   diagnostic_status_counts=dict(Counter(entry["status"] for entry in diagnostics)))
        atomic_json(run_dir / "run.json", run, overwrite=True)
        summary, _ = _publish(run_dir)
        atomic_json(run_dir / "complete.json", {"run_id": run["run_id"], "status": run["status"],
                                                "code_fingerprint": fingerprint,
                                                "summary_sha256": sha256(run_dir / "summary.json")}, overwrite=args.resume)
        print(f"Report: {run_dir / 'report.html'}", flush=True)
        print(f"Dataset readiness: {json.dumps(readiness, ensure_ascii=False)}", flush=True)
        return 0 if run["status"] == "completed" else 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
