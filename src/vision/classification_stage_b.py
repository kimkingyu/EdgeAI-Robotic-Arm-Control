"""Stage B CPU orchestration. The parent imports only the standard library.

This schema is independent of Stage A. No downloads, implicit mock mode, dataset
acquisition, hardware access or persistent model/session belongs in this module.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0")
FAMILY_VARIANTS = ("QB", "Q1", "Q2", "Q3")
EXECUTABLE_VARIANTS = ("O1",) + FAMILY_VARIANTS
THREADS = (1, 2, 4)
SCHEMA_VERSION = 1
SCREENING = {"repeats": 5, "iterations": 200, "warmup": 20}
CONFIRMATION = {"repeats": 5, "iterations": 500, "warmup": 20}
OBSERVATIONS = {"iterations": 200, "warmup": 20}
PREPROCESS = {
    "layout": "NCHW", "color_space": "RGB", "resize": 256,
    "center_crop": 224, "interpolation": "bilinear", "antialias": True,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "input_shape": [1, 3, 224, 224],
}
SOURCE_FILES = (
    "src/vision/classification_stage_b.py", "tools/run_classifier_stage_b.py",
    "src/vision/classification_quantization.py", "tools/quantize_classifier_onnx.py",
    "src/vision/classification.py", "src/vision/classification_variants.py",
    "src/vision/classification_benchmark.py", "tools/run_classifier_benchmark.py",
    "tools/export_classifier_onnx.py", "configs/classifier_dataset.json",
    "configs/classifier_benchmark.json", "src/vision/__init__.py", "requirements-resnet.txt",
)
OPTIONAL_SOURCE_FILES = ("src/vision/classification_controls.py",)
LIMITATIONS = [
    "仅本机 CPU、batch=1；不代表 RK3588/NPU、摄像头或机械臂性能。",
    "数据仅覆盖 Oxford-IIIT Pet 的 20 个映射犬种；所有输出/排名仍覆盖完整 1000 类。",
    "不是完整 ImageNet 验证；官方 split 及 SHA/source_id 不证明预训练未见或不存在近重复。",
    "最终候选仅限同模型 ORT O1.t1/2/4 与 QB/Q 家族；PyTorch 仅保留独立比较，不偷换后端。",
    "INT8 增量只相对同期 QB.t1；QB 预处理及 raw 导出变化不归因于 INT8。",
    "每模型至多新增一个量化+已筛选线程组合；不做参数笛卡尔积或 holdout 调参。",
    "5 个配对进程块的 bootstrap 区间是探索性不确定性；迭代不是独立实验单位。",
    "静态 QDQ 不等于全图整数执行；实际 profile 仅作核覆盖诊断，不参与计时。",
    "流水线从已解码 performance RGB 开始；磁盘解码不计时，流水线总墙钟直接测量，不能相加阶段 P50。",
    "独立 RSS 进程观测 prepared-input ORT 推理，含依赖、输入缓存与分配器；20ms 采样可能漏峰，不是模型独占内存。",
    "QB 漂移哨兵与筛选/确认样本分离，最多整组复测一次；没有合法 QB 时明确缺控制且不宣称采用收益。",
    "所有作业串行、有限超时；一般失败不自动重试，只有预声明漂移组允许一次复测；失败不填零。",
]


class StageBError(ValueError):
    """A failed data/provenance/selection guard, never a synthetic pass."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _int(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise StageBError(f"{name}: invalid integer")
    return value


def _number(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise StageBError(f"{name}: invalid finite number")
    return float(value)


def _process_lifecycle(value, pid=None):
    """Validate the stored OS process instance; never synthesize historical markers."""
    actual_pid = _int(value.get("worker_pid"), "worker PID", 1)
    if pid is not None and actual_pid != pid:
        raise StageBError("Process lifecycle PID mismatch")
    ticks = _int(value.get("worker_creation_ticks"), "OS process creation ticks", 1)
    identity = value.get("worker_identity")
    source = value.get("worker_identity_source")
    if (not isinstance(identity, str) or not (identity == f"{actual_pid}:{ticks}" or identity.startswith(f"{actual_pid}:{ticks}:"))
            or not isinstance(source, str) or not source.strip()):
        raise StageBError("Missing/corrupt OS process-instance identity; PID alone is insufficient")
    return {"worker_pid": actual_pid, "worker_identity": identity,
            "worker_creation_ticks": ticks, "worker_identity_source": source}


def _digest(value, name="sha256"):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise StageBError(f"{name}: invalid SHA256")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StageBError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda item: (_ for _ in ()).throw(StageBError(f"nonfinite: {item}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageBError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageBError("JSON root must be an object")
    # Reject overflow (e.g. 1e999), not just NaN/Infinity tokens.
    _json_bytes(value)
    return value


def _json_bytes(value):
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                           allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StageBError(f"JSON must be finite and serializable: {exc}") from exc


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path, value, *, overwrite=False):
    """Publish one file without deleting the old one first."""
    path = Path(path)
    data = _json_bytes(value)
    if not path.parent.is_dir():
        raise StageBError(f"Parent directory does not exist: {path.parent}")
    if path.is_symlink() or (path.exists() and not overwrite):
        raise StageBError(f"Refusing overwrite: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".stage-b-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink() or (path.exists() and not overwrite):
            raise StageBError(f"Refusing overwrite at publish: {path}")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def code_fingerprint():
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    sources.update({name: sha256(ROOT / name) if (ROOT / name).is_file() else None for name in OPTIONAL_SOURCE_FILES})
    return canonical_hash(sources), sources


def environment_info():
    versions = {}
    for name in ("numpy", "Pillow", "torch", "torchvision", "onnx", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
            "machine": platform.machine(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "versions": versions,
            "power_policy": "unknown", "affinity_modified": False}


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBError(f"Path escapes approved directory: {path}") from exc
    return path


def _check_artifact(path, digest, name="artifact", *, allow_empty=False):
    digest = _digest(digest, f"{name}.sha256")
    path = Path(path)
    if not path.is_file() or (not allow_empty and path.stat().st_size <= 0) or sha256(path) != digest:
        raise StageBError(f"{name} SHA256 mismatch: {path}")
    return path.resolve()


def load_stage_b_manifest(path, *, check_files=True):
    """Read split metadata, but never open holdout image bytes before freezing."""
    if type(check_files) is not bool:
        raise StageBError("check_files must be bool")
    from src.vision.classification_benchmark import load_manifest
    manifest = load_manifest(path, check_files=False)
    root = Path(manifest["resolved_root"])
    if not root.is_dir():
        raise StageBError("Dataset root does not exist")
    config = read_json(ROOT / "configs/classifier_dataset.json")
    classes = config.get("classes")
    if not isinstance(classes, list) or len(classes) != 20:
        raise StageBError("Expected the frozen 20-breed mapping")
    mapping = {row["prefix"]: row for row in classes}
    indices = {_int(row["imagenet_index"], "imagenet_index", maximum=999) for row in classes}
    if len(indices) != 20 or len(mapping) != 20:
        raise StageBError("20 distinct full1000-way indices required")
    source_name = manifest.get("source_metadata")
    if not isinstance(source_name, str) or Path(source_name).is_absolute():
        raise StageBError("source_metadata must reference a local relative file")
    source_path = _inside(root / source_name, root)
    source = read_json(source_path)
    subset = source.get("subset", {})
    if (type(source.get("schema_version")) is not int or source.get("schema_version") != 1 or source.get("weights_version") != "IMAGENET1K_V1"
            or subset.get("class_count") != 20 or subset.get("species") != "dog"
            or subset.get("mapping") != classes or source.get("license") != config.get("license")):
        raise StageBError("Dataset source/mapping/weights contract mismatch")
    mirror = source.get("mirror")
    if mirror is not None:
        if (not isinstance(mirror, dict) or not isinstance(mirror.get("dataset_id"), str)
                or not mirror["dataset_id"].strip() or not isinstance(mirror.get("revision"), str)
                or re.fullmatch(r"[0-9a-f]{40}", mirror["revision"]) is None
                or mirror.get("card_license") != source["license"]
                or subset.get("original_split_roles") != {"train": ["calibration", "development"], "test": ["holdout"]}):
            raise StageBError("Pinned mirror revision/license/split policy mismatch")
        names, files = mirror.get("label_names"), mirror.get("files")
        if not isinstance(names, dict) or not isinstance(files, dict) or set(files) != {"train", "test"}:
            raise StageBError("Mirror requires independent label names and complete train/test LFS records")
        for split, file in files.items():
            digest = _digest(file.get("sha256_expected"), "mirror LFS reference")
            if (file.get("sha256_observed") != digest or file.get("sha256_matches_lfs_reference") is not True
                    or file.get("unchanged_during_preparation") is not True or file.get("split") != split
                    or file.get("mirror_relative_path") != f"{split}.parquet"):
                raise StageBError("Mirror LFS hash/provenance declarations are incomplete")
            _int(file.get("row_count"), "mirror row count", 1)
    counts = {split: 0 for split in ("calibration", "development", "holdout")}
    for entry in manifest["images"]:
        split = entry["split"]
        if split not in counts:
            raise StageBError("Stage B accepts only calibration/development/holdout splits")
        counts[split] += 1
        row = mapping.get(entry.get("breed"))
        if row is None or entry.get("canonical_name") != row["canonical_name"]:
            raise StageBError(f"Unmapped breed: {entry['id']}")
        expected = None if split == "calibration" else row["imagenet_index"]
        if entry.get("label") != expected or (expected is not None and type(entry["label"]) is not int):
            raise StageBError("Calibration labels must be null; labeled splits use genuine full1000 indices")
        official = "test" if split == "holdout" else "train" if mirror is not None else "trainval"
        if mirror is not None:
            annotation = entry.get("original_annotation", {})
            row_index = _int(annotation.get("mirror_row_index"), "mirror row index", maximum=mirror["files"][official]["row_count"] - 1)
            label_index = _int(annotation.get("label"), "independent mirror breed label", maximum=36)
            truth = f"mirror:{mirror['dataset_id']}@{mirror['revision']}:{official}.parquet#row={row_index}"
            if (entry.get("ground_truth_source") != truth or annotation.get("ground_truth_source") != truth
                    or annotation.get("source_original_split") != official or annotation.get("image_id") != entry["id"]
                    or annotation.get("label_cat_dog") != 1 or annotation.get("label_name") != entry["breed"]
                    or mirror["label_names"].get(str(label_index)) != entry["breed"]):
                raise StageBError("Mirror independent truth/revision/row/breed mapping mismatch")
        elif entry.get("ground_truth_source") != f"annotations/{official}.txt":
            raise StageBError("Official annotation truth provenance is missing")
        if entry.get("source_original_split") != official or not entry.get("source_id"):
            raise StageBError("Source split/identity provenance is missing")
        if not entry.get("source") or not entry.get("license"):
            raise StageBError("Image source/license is missing")
        for key in ("width", "height"):
            _int(entry.get(key), key, minimum=1)
        if check_files and split != "holdout":
            _check_artifact(entry["resolved_path"], entry["sha256"], "image")
    if counts != {"calibration": 256, "development": 128, "holdout": 1000}:
        raise StageBError(f"blocked_data: exact approved split counts required: {counts}")
    if subset.get("counts") != counts:
        raise StageBError("Source split counts mismatch")
    completion_path = root / "prepared.complete.json"
    completion = read_json(completion_path)
    source_sha = sha256(source_path)
    if (completion.get("status") != "complete"
            or completion.get("manifest_sha256") != manifest["manifest_sha256"]
            or completion.get("source_sha256") != source_sha or completion.get("counts") != counts):
        raise StageBError("Dataset completion SHA/counts mismatch")
    manifest.update(manifest_path=str(Path(path).resolve()), source_path=str(source_path),
                    source_sha256=source_sha, source=source,
                    dataset_completion_sha256=sha256(completion_path),
                    dataset_scope={"dataset": manifest.get("dataset"), "class_count": 20,
                                   "species": "dog", "output_class_count": 1000,
                                   "ranking": "full1000-way, no subset masking", "mapping": classes,
                                   "counts": counts, "is_full_imagenet_validation": False,
                                   "pretraining_unseen_proven": False})
    return manifest


def audit_fp32_controls(directory, run, records):
    """M4 nominal inference must belong to a complete accepted sentinel ledger."""
    directory = Path(directory).resolve()
    path = directory / "controls" / "ledger.json"
    ledger = read_json(path)
    if (type(ledger.get("schema_version")) is not int or ledger.get("schema_version") != 1
            or any(ledger.get(key) != run.get(key) for key in ("config_sha256", "manifest_sha256", "code_fingerprint"))
            or ledger.get("threshold") != 1.10 or ledger.get("max_attempts") != 2):
        raise StageBError("M4 ledger identity/protocol mismatch")
    groups = ledger.get("groups")
    expected_groups = {f"{model}.b{block}" for model in MODELS for block in range(5)}
    if not isinstance(groups, dict) or set(groups) != expected_groups:
        raise StageBError("M4 ledger is missing complete model/block groups")
    dependencies = {str(path): sha256(path)}
    for group_key, group in groups.items():
        if group.get("status") != "accepted":
            raise StageBError("M4 unstable/unaccepted groups cannot enter Stage B")
        attempt = _int(group.get("attempt"), "M4 control attempt", maximum=1)
        factor = _number(group.get("drift_factor"), "M4 drift factor", True)
        if not 1 <= factor <= 1.10 or group.get("threshold") != 1.10:
            raise StageBError("M4 accepted group violates the drift budget")
        model, block = group_key.rsplit(".b", 1)
        before_after, instances = [], []
        for role in ("before", "after"):
            control_path = _inside(group.get(f"{role}_path", ""), directory / "controls")
            result = read_json(control_path)
            job_path = directory / "jobs" / control_path.name
            job = read_json(job_path)
            if (result.get("status") != "passed" or result.get("worker_pid") != group.get(f"{role}_pid")
                    or result.get("job_sha256") != canonical_hash(job)
                    or result.get("kind") != "control" or result.get("block") != int(block)
                    or result.get("validation", {}).get("passed") is not True
                    or job.get("control_role") != role or job.get("attempt") != attempt
                    or job.get("experiment", {}).get("model_id") != model
                    or job.get("experiment", {}).get("variant_id") != "B0"
                    or job.get("experiment", {}).get("threads") != 1
                    or any(result.get(key) != run.get(key) or job.get(key) != run.get(key)
                           for key in ("run_id", "config_sha256", "manifest_sha256", "code_fingerprint"))):
                raise StageBError("M4 sentinel job/PID/identity/correctness binding mismatch")
            _int(result.get("worker_pid"), "M4 sentinel PID", 1)
            instance = _process_lifecycle(result)["worker_identity"]
            if group.get(f"{role}_identity") != instance:
                raise StageBError("M4 ledger sentinel lifecycle identity mismatch")
            instances.append(instance)
            samples = result.get("samples", [])
            if len(samples) != 200 or job.get("warmup") != 20 or job.get("iterations") != 200:
                raise StageBError("M4 sentinel formal budget mismatch")
            values = [_number(row.get("inference_ms"), "M4 control timing", True) for row in samples]
            sequence = job.get("image_sequence", [])
            if (len(sequence) != 200 or any(type(index) is not int or not 0 <= index < 64 for index in sequence)
                    or [row.get("image_id") for row in samples] != [run["images"][index]["id"] for index in sequence]):
                raise StageBError("M4 sentinel input sequence mismatch")
            before_after.append(statistics.median(values))
            dependencies[str(control_path)] = sha256(control_path)
            dependencies[str(job_path)] = sha256(job_path)
            if result.get("samples_path") is not None:
                _check_artifact(result["samples_path"], result.get("samples_sha256"), "M4 sentinel JSONL")
                dependencies[str(Path(result["samples_path"]).resolve())] = result["samples_sha256"]
        if instances[0] == instances[1] or abs(max(before_after) / min(before_after) - factor) > 1e-12:
            raise StageBError("M4 ledger drift/PID does not match its raw sentinels")
    expected = {(row["id"], block) for row in run["experiments"] for block in range(5)}
    seen = set()
    for wrapped in records:
        record = wrapped.get("data", wrapped)
        if record.get("kind") != "inference":
            continue  # Legacy pipeline/RSS observations are independent, not bracketed groups.
        key = (record.get("experiment", {}).get("id"), record.get("block"))
        if key not in expected or key in seen:
            raise StageBError("M4 nominal ledger measurement inventory is invalid")
        seen.add(key)
        group_key = f"{record['experiment']['model_id']}.b{record['block']}"
        group = groups[group_key]
        if (record.get("environment_accepted") is not True or record.get("drift_control_group") != group_key
                or record.get("attempt") != group["attempt"] or record.get("drift_factor") != group["drift_factor"]):
            raise StageBError("M4 inference measurement is not environment_accepted by its ledger")
    if seen != expected:
        raise StageBError("M4 ledger lacks complete accepted nominal measurements")
    return {"path": str(path), "sha256": sha256(path), "ledger": ledger, "dependencies": dependencies}


def audit_fp32_run(path, manifest, *, allow_failures=False):
    """Audit original Stage A bytes. Never rerun M4 or rewrite its reports."""
    if type(allow_failures) is not bool:
        raise StageBError("allow_failures must be bool")
    from src.vision.classification_benchmark import aggregate_results, expand_experiments
    path = Path(path).resolve()
    directory = path.parent if path.name == "run.json" else path
    run_path = directory / "run.json"
    run = read_json(run_path)
    if (run.get("schema_version") != 1 or run.get("mode") != "formal"
            or run.get("status") not in ("completed", "completed_with_failures")):
        raise StageBError("Stage A smoke/incomplete runs are not formal FP32 evidence")
    if run.get("config", {}).get("formal") != SCREENING:
        raise StageBError("M4 must use the unchanged formal20/200/5 budget")
    if run.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise StageBError("M4 and Stage B must use the same manifest")
    expected = expand_experiments(run["config"])
    if run.get("experiments") != expected or len(expected) != 30:
        raise StageBError("Complete original 30-configuration FP32 matrix is required")
    images = run.get("images")
    if not isinstance(images, list) or len(images) != 64 or len({row.get("id") for row in images}) != 64:
        raise StageBError("M4 reference workload must contain 64 distinct development images")
    dev = {entry["id"]: entry for entry in manifest["images"] if entry["split"] == "development"}
    for image in images:
        original = dev.get(image.get("id"))
        if original is None or any(image.get(key) != original.get(key) for key in (
                "id", "sha256", "label", "source_id", "resolved_path", "split")):
            raise StageBError("M4 workload/source/labels differ from the development manifest")
    summary = aggregate_results(directory)
    if summary.get("integrity_errors") or summary.get("status") in ("incomplete", "untrusted", "interrupted"):
        raise StageBError("FP32 artifacts/results are corrupt or incomplete; failures flag cannot bypass integrity")
    rows = summary["experiments"]
    failures = [row for row in rows if not row.get("scheduled") or row.get("status") != "passed"]
    if (failures or run["status"] != "completed") and not allow_failures:
        raise StageBError("All scheduled FP32 configurations must pass; --allow-fp32-failures is explicit")
    if any(row.get("status") in ("incomplete", "not_run", "untrusted") for row in rows):
        raise StageBError("Missing FP32 results are not genuine visible failures")
    dependencies = {str(run_path): sha256(run_path)}
    completion_path, summary_path = directory / "complete.json", directory / "summary.json"
    completion = read_json(completion_path)
    if (completion.get("run_id") != run.get("run_id") or completion.get("status") != run.get("status")
            or completion.get("code_fingerprint") != run.get("code_fingerprint")
            or completion.get("summary_sha256") != sha256(summary_path)):
        raise StageBError("M4 completion/summary SHA binding mismatch")
    dependencies[str(completion_path)] = sha256(completion_path)
    dependencies[str(summary_path)] = completion["summary_sha256"]
    control_audit = audit_fp32_controls(directory, run, summary["records"])
    dependencies.update(control_audit["dependencies"])
    prepared = run.get("prepared_models", {})
    if set(prepared) != set(MODELS):
        raise StageBError("All three fixed model references are required")
    category_table = None
    for model in MODELS:
        item = prepared[model]
        metadata = item.get("model_metadata", {})
        if (metadata.get("model_id") != model or metadata.get("weights_source") != "official_imagenet1k_v1"
                or metadata.get("category_mapping_assumed") is not False
                or metadata.get("matches_official_hash") is not True
                or metadata.get("input_shape") != [1, 3, 224, 224]
                or metadata.get("dtype") != "float32" or metadata.get("device") != "cpu"
                or metadata.get("preprocess") != PREPROCESS
                or metadata.get("weights_sha256") != item.get("weights_sha256")):
            raise StageBError(f"FP32 official weights/normalization contract mismatch: {model}")
        categories = metadata.get("categories")
        if not isinstance(categories, list) or len(categories) != 1000 or len(set(categories)) != 1000:
            raise StageBError("All models must retain the full1000 category mapping")
        if category_table is None:
            category_table = categories
        if categories != category_table:
            raise StageBError("Cross-model category tables differ")
        for mapping in manifest["dataset_scope"]["mapping"]:
            if categories[mapping["imagenet_index"]] != mapping["canonical_name"]:
                raise StageBError("Dataset labels disagree with the full1000 category table")
        for name in ("weights", "reference", "onnx"):
            artifact = _check_artifact(item.get(f"{name}_path", ""), item.get(f"{name}_sha256"), f"M4 {name}")
            dependencies[str(artifact)] = item[f"{name}_sha256"]
        exported = item.get("export_metadata", {})
        sidecar_path = Path(item["onnx_path"]).with_suffix(".json")
        if read_json(sidecar_path) != exported:
            raise StageBError("M4 export metadata differs from the original ONNX sidecar")
        dependencies[str(sidecar_path.resolve())] = sha256(sidecar_path)
        contract = exported.get("onnx", {})
        checks = exported.get("validation", {})
        if (exported.get("status") != "verified" or exported.get("model_id") != model
                or exported.get("model", {}).get("weights_sha256") != item["weights_sha256"]
                or contract.get("sha256") != item["onnx_sha256"]
                or contract.get("input_shape") != [1, 3, 224, 224]
                or contract.get("output_shape") != [1, 1000]
                or contract.get("input_name") != "images" or contract.get("output_name") != "logits"
                or contract.get("input_dtype") != "float32" or contract.get("output_dtype") != "float32"
                or contract.get("opset") != 17 or contract.get("normalization_in_graph") is not False
                or contract.get("external_data") is not False
                or checks.get("provider") != "CPUExecutionProvider"):
            raise StageBError("Raw ONNX/reference export validation contract failed")
        exported_images = checks.get("results")
        cached_count = _int(checks.get("verified_images_count"), "cached export image count", 1)
        if (not isinstance(exported_images, list) or len(exported_images) != cached_count
                or any(row.get("comparison", {}).get("passed") is not True for row in exported_images)
                or exported.get("model", {}).get("preprocess") != metadata["preprocess"]):
            raise StageBError("Cached FP32 export lacks independent passing per-image validation")
        # The cache may have been verified on a prior smoke image. Current-run proof
        # comes from O0/O1 jobs below, not from rewriting or relabeling this sidecar.
    workload_evidence = {(model, variant): set() for model in MODELS for variant in ("O0", "O1")}
    for record in summary["records"]:
        result = record["data"]
        _process_lifecycle(result)
        result_path = directory / record["source_file"]
        job_path = directory / "jobs" / result_path.name
        job = read_json(job_path)
        if result.get("job_sha256") != canonical_hash(job):
            raise StageBError("M4 job SHA mismatch")
        if any(job.get(key) != run.get(key) for key in ("run_id", "config_sha256", "manifest_sha256", "code_fingerprint")):
            raise StageBError("M4 job belongs to another run")
        model_item = prepared[result["experiment"]["model_id"]]
        if any(job.get(key) != model_item.get(key) for key in (
                "weights_sha256", "reference_sha256", "onnx_sha256")):
            raise StageBError("M4 job artifact hashes differ from the prepared references")
        if job.get("images") != images:
            raise StageBError("M4 job input workload differs")
        if result["kind"] == "inference" and result["experiment"]["variant_id"] in ("O0", "O1") and result["experiment"]["threads"] == 1:
            validation = result.get("validation", {})
            comparisons = validation.get("per_image")
            if (result["status"] != "passed" or validation.get("passed") is not True
                    or validation.get("allclose") is not True or type(validation.get("image_count")) is not int
                    or validation["image_count"] != 64 or not isinstance(comparisons, list) or len(comparisons) != 64
                    or validation.get("rtol") != 1e-4 or validation.get("atol") != 1e-4
                    or job.get("reference_path") != model_item["reference_path"]
                    or job.get("reference_sha256") != model_item["reference_sha256"]
                    or job.get("onnx_path") != model_item["onnx_path"]):
                raise StageBError("Current M4 O0/O1.t1 lacks 64-image Torch-reference numerical proof")
            for index, comparison in enumerate(comparisons):
                if not isinstance(comparison, dict) or comparison.get("allclose") is not True or comparison.get("passed") is not True:
                    raise StageBError("Current M4 64-image per-image numerical comparison failed")
                for field, expected_key in (("id", "id"), ("image_id", "id"), ("sha256", "sha256"), ("image_sha256", "sha256")):
                    if field in comparison and comparison[field] != images[index][expected_key]:
                        raise StageBError("Current M4 per-image proof order/SHA differs from job.images")
            _check_artifact(job["reference_path"], model_item["reference_sha256"], "current M4 Torch reference")
            workload_evidence[(result["experiment"]["model_id"], result["experiment"]["variant_id"])].add(result["block"])
        if result["status"] == "passed" and result["kind"] in ("inference", "pipeline"):
            sequence = job.get("image_sequence", [])
            if (len(sequence) != 200 or any(type(index) is not int or not 0 <= index < 64 for index in sequence)
                    or [row["image_id"] for row in result["samples"]] != [images[index]["id"] for index in sequence]):
                raise StageBError("M4 input sequence/result mismatch")
        dependencies[str(job_path.resolve())] = sha256(job_path)
        dependencies[str(result_path.resolve())] = record["record_sha256"]
        if result.get("samples_path") is not None:
            log_path = Path(result["samples_path"])
            if not log_path.is_absolute():
                log_path = directory / log_path
            _check_artifact(log_path, result.get("samples_sha256"), "M4 timing JSONL")
            raw_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            if len(raw_rows) != len(result.get("samples", [])) or any(
                    row != {"experiment_id": result["experiment"]["id"], "kind": result["kind"],
                            "block": result["block"], "worker_pid": result["worker_pid"],
                            "worker_identity": result["worker_identity"], **sample}
                    for row, sample in zip(raw_rows, result["samples"])):
                raise StageBError("M4 raw timing JSONL differs from its result")
            dependencies[str(log_path.resolve())] = result["samples_sha256"]
    if any(blocks != set(range(5)) for blocks in workload_evidence.values()):
        raise StageBError("Current M4 must contain all five complete 64-image O0/O1.t1 validation blocks per model")
    return {"run_dir": str(directory), "run_path": str(run_path), "run_sha256": sha256(run_path),
            "snapshot_sha256": canonical_hash(dependencies), "dependencies": dependencies,
            "run": run, "summary": summary, "visible_failures": [row["id"] for row in failures],
            "controls": control_audit, "allow_failures": bool(allow_failures)}


def fixed_sequence(seed, block, iterations, count=64):
    _int(seed, "seed")
    _int(block, "block")
    _int(iterations, "iterations", 1)
    _int(count, "count", 1)
    generator = random.Random(seed + block)
    sequence = []
    while len(sequence) < iterations:
        cycle = list(range(count))
        generator.shuffle(cycle)
        sequence.extend(cycle)
    return sequence[:iterations]


def _percentile(values, fraction):
    values = sorted(values)
    location = (len(values) - 1) * fraction
    lower = int(location)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (location - lower)


def describe(values):
    values = [_number(item, "timing", positive=True) for item in values]
    if not values:
        raise StageBError("No timing samples; failure is not a zero")
    return {"count": len(values), "p50": _percentile(values, .5), "p95": _percentile(values, .95),
            "mean": statistics.mean(values), "std": statistics.pstdev(values),
            "min": min(values), "max": max(values)}


def paired_performance(baseline, candidate, *, seed=20261001, bootstrap_samples=2000):
    """Resample paired process blocks, never individual timing iterations."""
    empty = {"status": "incomplete", "speedup": None, "ci95": None, "p95_ratio": None,
             "gate_passed": False, "resampling_unit": "paired process block", "blocks": [],
             "baseline_stats": None, "candidate_stats": None, "failure_reasons": []}
    if len(baseline) != 5 or len(candidate) != 5:
        empty["failure_reasons"].append("Five complete paired blocks are required")
        return empty
    left = sorted(baseline, key=lambda row: row["block"])
    right = sorted(candidate, key=lambda row: row["block"])
    if [row["block"] for row in left] != list(range(5)) or [row["block"] for row in right] != list(range(5)):
        empty["failure_reasons"].append("Missing or duplicate process blocks")
        return empty
    ratios, blocks, left_samples, right_samples = [], [], [], []
    pids = set()
    for ref, cand in zip(left, right):
        if (ref.get("status") != "passed" or cand.get("status") != "passed"
                or ref.get("environment_accepted") is not True or cand.get("environment_accepted") is not True
                or ref.get("model_id") != cand.get("model_id")
                or ref.get("round_id") != cand.get("round_id")
                or ref.get("image_sequence") != cand.get("image_sequence")
                or ref.get("iterations") != cand.get("iterations")
                or ref.get("warmup") != cand.get("warmup")):
            empty["failure_reasons"].append("Failed, cross-model, cross-round or unpaired workload")
            return empty
        for row in (ref, cand):
            pid = _int(row.get("worker_pid"), "worker_pid", 1)
            try:
                identity = _process_lifecycle(row, pid)["worker_identity"]
            except StageBError:
                empty["failure_reasons"].append("Missing OS process-instance identity; PID recycling is not proof of independence")
                return empty
            if identity in pids:
                empty["failure_reasons"].append("Process instance reused across independent blocks")
                return empty
            pids.add(identity)
        samples_ref, samples_cand = ref.get("samples", []), cand.get("samples", [])
        if (len(samples_ref) != ref["iterations"] or len(samples_cand) != cand["iterations"]
                or [(row.get("image_id"), row.get("image_sha256")) for row in samples_ref]
                != [(row.get("image_id"), row.get("image_sha256")) for row in samples_cand]):
            empty["failure_reasons"].append("Sample sequence/hash mismatch")
            return empty
        stats_ref = describe([row["inference_ms"] for row in samples_ref])
        stats_cand = describe([row["inference_ms"] for row in samples_cand])
        ratio = stats_ref["p50"] / stats_cand["p50"]
        ratios.append(math.log(ratio))
        blocks.append({"block": ref["block"], "baseline_pid": ref["worker_pid"],
                       "candidate_pid": cand["worker_pid"], "baseline_identity": ref["worker_identity"],
                       "candidate_identity": cand["worker_identity"], "speedup": ratio,
                       "baseline_stats": stats_ref, "candidate_stats": stats_cand})
        left_samples.extend(row["inference_ms"] for row in samples_ref)
        right_samples.extend(row["inference_ms"] for row in samples_cand)
    generator = random.Random(seed)
    _int(bootstrap_samples, "bootstrap_samples", 1)
    samples = [math.exp(statistics.mean(generator.choices(ratios, k=5))) for _ in range(bootstrap_samples)]
    low, high = _percentile(samples, .025), _percentile(samples, .975)
    base_stats, cand_stats = describe(left_samples), describe(right_samples)
    ratio = math.exp(statistics.mean(ratios))
    p95_ratio = cand_stats["p95"] / base_stats["p95"]
    reasons = []
    if ratio < 1.05:
        reasons.append("Speedup < 1.05")
    if low <= 1.0:
        reasons.append("Paired block CI lower bound <= 1")
    if p95_ratio > 1.05:
        reasons.append("P95 worsens by more than 5%")
    return {"status": "available", "speedup": ratio, "ci95": {"low": low, "high": high},
            "p95_ratio": p95_ratio, "gate_passed": not reasons, "blocks": blocks,
            "baseline_stats": base_stats, "candidate_stats": cand_stats,
            "resampling_unit": "paired process block", "bootstrap_samples": bootstrap_samples,
            "seed": seed, "failure_reasons": reasons,
            "note": "Five blocks; exploratory interval, not 1000 independent inference experiments"}


def accuracy_gate(comparison, *, split="development"):
    """Recheck numeric budgets; a forged 'passed' string is insufficient."""
    if not isinstance(comparison, dict) or comparison.get("split") != split or comparison.get("paired") is not True:
        return {"passed": False, "status": "missing_or_unpaired"}
    statuses = []
    for metric in ("top1", "top5"):
        data = comparison.get(metric, {})
        try:
            drop = _number(data.get("drop_pp"), "drop_pp")
            ci = data.get("ci95_pp")
            if not isinstance(ci, list) or len(ci) != 2:
                raise StageBError("Missing accuracy CI")
            lower, upper = [_number(value, "accuracy CI") for value in ci]
            if lower > upper or not lower <= drop <= upper:
                raise StageBError("Invalid accuracy CI")
            status = "failed" if drop > 1.0 or lower > 1.0 else "inconclusive" if upper > 1.0 else "passed"
        except StageBError:
            status = "invalid"
        statuses.append(status)
    status = "passed" if all(item == "passed" for item in statuses) else (
        "failed" if "failed" in statuses else "inconclusive" if "inconclusive" in statuses else "invalid")
    return {"passed": status == "passed", "status": status}


def determine_int8_evidence_status(rows, qualified_int8=None):
    """Determine INT8 evidence status strictly from screening/development rows and failures.

    States:
      - 'qualified_available': at least one INT8 candidate passed all screening gates
      - 'evaluated_no_benefit': all INT8 candidates completed valid evaluation but failed gates
      - 'failed_runtime': all INT8 candidates failed due to runtime/execution errors
      - 'unsupported': all INT8 candidates are unsupported
      - 'partial_failure': mixed outcomes (some evaluated, some failed runtime/unsupported)
      - 'not_evaluated': all INT8 candidates failed without valid evaluation (mixed failure kinds)
      - 'none': no INT8 candidate rows
    """
    int8_rows = [row for row in rows if row.get("variant_id") in ("Q1", "Q2", "Q3")]
    if not int8_rows:
        return "none"
    if qualified_int8:
        return "qualified_available"

    def _is_evaluated(row):
        dev = row.get("development")
        return (row.get("phase") in ("screening", "combination")
                and row.get("status") == "passed"
                and row.get("validation_passed") is True
                and isinstance(dev, dict)
                and dev.get("split") == "development")

    def _failure_kind(row):
        status = row.get("status")
        if status == "unsupported":
            return "unsupported"
        if status == "failed_runtime":
            return "failed_runtime"
        for err in row.get("errors", []):
            if err.get("status") == "unsupported":
                return "unsupported"
            if err.get("status") == "failed_runtime":
                return "failed_runtime"
        if isinstance(status, str) and "fail" in status:
            return "failed_runtime"
        return "not_evaluated"

    evaluated = [r for r in int8_rows if _is_evaluated(r)]
    if len(evaluated) == len(int8_rows):
        return "evaluated_no_benefit"

    non_evaluated = [r for r in int8_rows if not _is_evaluated(r)]
    kinds = set(_failure_kind(r) for r in non_evaluated)

    if evaluated:
        return "partial_failure"
    if kinds == {"failed_runtime"}:
        return "failed_runtime"
    if kinds == {"unsupported"}:
        return "unsupported"
    return "not_evaluated"


def select_candidate(model_id, rows):
    """Pure selection: only validated development and screening records are read."""
    if model_id not in MODELS:
        raise StageBError("Unknown model")
    eligible, legal_fp32, reasons = [], [], []
    for original in rows:
        # Do not even copy a holdout/final field into a selection record.
        allowed = ("id", "model_id", "variant_id", "threads", "status", "validation_passed",
                   "artifact_path", "artifact_sha256", "baseline_artifact_path", "baseline_artifact_sha256",
                   "baseline_threads", "baseline_variant_id", "phase", "development", "development_comparison",
                   "performance", "runtime_profile", "size_bytes", "graph_statistics", "errors", "measurement_jobs")
        row = {key: copy.deepcopy(original[key]) for key in allowed if key in original}
        if row.get("phase") not in ("screening", "combination"):
            reasons.append({"id": row.get("id"), "reason": "Only development screening rounds can select candidates"})
            continue
        if row.get("model_id") != model_id or row.get("variant_id") not in EXECUTABLE_VARIANTS:
            raise StageBError("Candidate model/variant mismatch")
        if (type(row.get("threads")) is not int or row.get("threads") not in THREADS
                or type(row.get("baseline_threads")) is not int or row.get("baseline_threads") != 1):
            raise StageBError("Candidate thread contract mismatch")
        dev = row.get("development")
        if not isinstance(dev, dict) or dev.get("split") != "development":
            reasons.append({"id": row.get("id"), "reason": "No development-only evaluation"})
            continue
        if dev.get("model_id") != model_id or dev.get("artifact_sha256") != row.get("artifact_sha256"):
            raise StageBError("Development artifact/model binding mismatch")
        if row.get("status") != "passed" or row.get("validation_passed") is not True:
            reasons.append({"id": row.get("id"), "reason": row.get("status", "invalid")})
            continue
        quantized = row["variant_id"] in ("Q1", "Q2", "Q3")
        comparison = row.get("development_comparison")
        acc = accuracy_gate(comparison)
        performance = row.get("performance", {})
        try:
            perf_passed = (performance.get("status") == "available"
                           and _number(performance.get("speedup"), "speedup", True) >= 1.05
                           and _number(performance.get("ci95", {}).get("low"), "CI lower", True) > 1
                           and _number(performance.get("p95_ratio"), "p95_ratio", True) <= 1.05
                           and performance.get("gate_passed") is True)
        except (StageBError, AttributeError):
            perf_passed = False
        if quantized:
            profile = row.get("runtime_profile", {})
            events = profile.get("int8_operator_events")
            count = len(events) if isinstance(events, list) else events if type(events) is int else 0
            has_int = (profile.get("providers") == ["CPUExecutionProvider"]
                       and count > 0 and bool(profile.get("int8_operator_counts")))
            if row.get("baseline_variant_id") != "QB":
                raise StageBError("INT8 attribution must use same-model contemporaneous QB.t1")
            if not acc["passed"] or not has_int or not perf_passed:
                reasons.append({"id": row["id"], "reason": f"accuracy={acc['status']}; integer_runtime={has_int}; performance={perf_passed}"})
                continue
        else:
            if row.get("baseline_variant_id") != "O1":
                raise StageBError("FP32 attribution must use same-model contemporaneous raw O1.t1")
            if not acc["passed"]:
                reasons.append({"id": row["id"], "reason": f"FP32 development accuracy={acc['status']}"})
                continue
            legal_fp32.append(row)
            if not perf_passed:
                continue
        eligible.append(row)
    quantized = [row for row in eligible if row["variant_id"] in ("Q1", "Q2", "Q3")]
    baselines = [row for row in legal_fp32 if row["variant_id"] == "O1" and row["threads"] == 1]
    int8_evidence_status = determine_int8_evidence_status(rows, quantized)

    def _selection_reason(is_quantized, int8_status):
        if is_quantized:
            return "INT8 gates passed and lowest same-workload nominal P50"
        if int8_status == "evaluated_no_benefit":
            return "No INT8 adoption benefit over selected FP32; conservative raw-baseline fallback or FP32 candidate"
        if int8_status == "failed_runtime":
            return "All INT8 candidates failed runtime preparation/execution without evaluation; falling back to legal FP32 candidate"
        if int8_status == "unsupported":
            return "All INT8 candidates unsupported; falling back to legal FP32 candidate"
        if int8_status == "partial_failure":
            return "INT8 evaluation incomplete due to partial runtime/preparation failures; falling back to legal FP32 candidate"
        if int8_status == "qualified_available":
            return "Selected FP32 candidate has lower nominal P50 than qualified INT8 candidates"
        return "INT8 candidates not evaluated; falling back to legal FP32 candidate"

    # Ratios to QB and raw O1 are different denominators: never rank them as if equal.
    # Choose the lowest nominal same-workload P50 among gated candidates and the legal raw baseline.
    # This is one disclosed latency objective, not an accuracy/size/performance composite score.
    pool = eligible + [row for row in baselines if row not in eligible]
    if pool:
        def latency_key(row):
            stats = row.get("performance", {}).get("candidate_stats") or {}
            value = stats.get("p50")
            latency = _number(value, "candidate P50", True) if value is not None else float("inf")
            return latency, row["variant_id"] in ("Q1", "Q2", "Q3"), row["id"]
        chosen = sorted(pool, key=latency_key)[0]
        is_quantized = chosen["variant_id"] in ("Q1", "Q2", "Q3")
        retained = chosen["variant_id"] == "O1" and chosen["threads"] == 1
        return {"model_id": model_id, "status": "retained_fp32" if retained else "confirmation_required", "candidate": chosen,
                "selection_basis": "development_only", "int8_screening_passed": is_quantized,
                "int8_evidence_status": int8_evidence_status,
                "reason": _selection_reason(is_quantized, int8_evidence_status),
                "latency_selection_policy": "lowest same-model screening P50; different QB/raw ratios are never ranked together",
                "qualified_int8_count": len(quantized), "rejected": reasons}
    if legal_fp32:
        baselines = [row for row in legal_fp32 if row["variant_id"] == "O1" and row["threads"] == 1]
        chosen = sorted(baselines or legal_fp32, key=lambda row: row["id"])[0]
        return {"model_id": model_id, "status": "retained_fp32", "candidate": chosen,
                "selection_basis": "development_only", "int8_screening_passed": False,
                "int8_evidence_status": int8_evidence_status,
                "reason": _selection_reason(False, int8_evidence_status), "rejected": reasons}
    return {"model_id": model_id, "status": "blocked_correctness", "candidate": None,
            "selection_basis": "development_only", "int8_screening_passed": False,
            "int8_evidence_status": int8_evidence_status,
            "reason": "No legal full1000-way FP32 candidate/baseline; holdout remains blocked", "rejected": reasons}


def validate_frozen(value, *, manifest_sha256, code=None, family_files=None, check_files=True):
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("status") != "frozen" or value.get("manifest_sha256") != manifest_sha256):
        raise StageBError("holdout guard: frozen schema/status/manifest mismatch")
    _digest(value.get("manifest_sha256"))
    _digest(value.get("code_fingerprint"))
    if code is not None and value["code_fingerprint"] != code:
        raise StageBError("holdout guard: code fingerprint mismatch")
    if not isinstance(value.get("created_at"), str) or not value["created_at"]:
        raise StageBError("holdout guard: missing creation timestamp")
    candidates = value.get("candidates")
    if (not isinstance(candidates, list) or len(candidates) != 3
            or {row.get("model_id") for row in candidates} != set(MODELS)):
        raise StageBError("holdout guard: all three models must be frozen together")
    for row in candidates:
        if (row.get("variant_id") not in EXECUTABLE_VARIANTS or row.get("threads") not in THREADS
                or type(row["threads"]) is not int or row.get("baseline_threads") != 1
                or type(row["baseline_threads"]) is not int
                or row.get("selection_basis") != "development_only"):
            raise StageBError("holdout guard: candidate contract mismatch")
        for key in ("artifact_sha256", "baseline_artifact_sha256", "family_sha256"):
            _digest(row.get(key), key)
        if check_files:
            _check_artifact(row.get("artifact_path", ""), row["artifact_sha256"], "frozen candidate")
            _check_artifact(row.get("baseline_artifact_path", ""), row["baseline_artifact_sha256"], "frozen baseline")
        if family_files is not None:
            path = family_files.get(row["model_id"])
            if path is None or sha256(path) != row["family_sha256"]:
                raise StageBError("holdout guard: family SHA mismatch")
            family = read_json(path)
            if family.get("manifest_sha256") != manifest_sha256 or family.get("model_id") != row["model_id"]:
                raise StageBError("holdout guard: family model/manifest mismatch")
            if row["variant_id"] in FAMILY_VARIANTS:
                artifact = family.get("variants", {}).get(row["variant_id"], {})
                if artifact.get("status") != "prepared" or artifact.get("onnx_sha256") != row["artifact_sha256"]:
                    raise StageBError("holdout guard: candidate is not the selected family artifact")
            elif family.get("input_model", {}).get("sha256") != row["artifact_sha256"]:
                raise StageBError("holdout guard: O1 must execute the original same-weight raw ONNX")
            baseline = (family.get("variants", {}).get("QB", {}) if row["variant_id"] in ("Q1", "Q2", "Q3")
                        else family.get("input_model", {}))
            expected = baseline.get("onnx_sha256", baseline.get("sha256"))
            if expected != row["baseline_artifact_sha256"]:
                raise StageBError("holdout guard: baseline family binding mismatch")
    return copy.deepcopy(value)


def freeze_candidates(selections, manifest_sha256, fingerprint, family_files, output_path):
    if len(selections) != 3 or any(row.get("candidate") is None for row in selections):
        raise StageBError("Cannot freeze until every model has a legal candidate")
    candidates = []
    for selection in selections:
        row = selection["candidate"]
        candidates.append({"model_id": row["model_id"], "variant_id": row["variant_id"],
                           "artifact_sha256": row["artifact_sha256"],
                           "baseline_artifact_sha256": row["baseline_artifact_sha256"],
                           "threads": row["threads"], "baseline_threads": 1,
                           "selection_basis": "development_only",
                           "family_sha256": sha256(family_files[row["model_id"]]),
                           "artifact_path": row["artifact_path"],
                           "baseline_artifact_path": row["baseline_artifact_path"]})
    value = {"schema_version": 1, "status": "frozen", "manifest_sha256": manifest_sha256,
             "candidates": candidates, "created_at": _now(), "code_fingerprint": fingerprint,
             "limitations": LIMITATIONS + ["Holdout failure does not permit changing this frozen selection."]}
    validate_frozen(value, manifest_sha256=manifest_sha256, code=fingerprint, family_files=family_files)
    atomic_json(output_path, value)
    return value, sha256(output_path)


def _reject_mock(value):
    if isinstance(value, dict):
        if value.get("synthetic") is True or value.get("evidence_kind") in ("mock", "test_mock", "synthetic"):
            raise StageBError("Test/mock values cannot be published as a real report")
        for item in value.values():
            _reject_mock(item)
    elif isinstance(value, list):
        for item in value:
            _reject_mock(item)


def _escape(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return html.escape(str(value), quote=True)


def _csv_cell(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else ("" if value is None else str(value))
    if text.lstrip(" \t\r\n\ufeff").startswith(("=", "+", "-", "@")):
        text = "'" + text
    return text


def _report_csv(summary):
    fields = ("phase", "model_id", "variant_id", "threads", "status", "baseline_variant_id",
              "size_bytes", "p50_ms", "p95_ms", "speedup", "ci95", "p95_ratio",
              "int8_operator_events", "development_top1", "development_top5",
              "development_drop_ci", "final_holdout", "final_adoption", "errors",
              "observation_role", "pipeline_stages_and_wall", "independent_rss", "frozen_sha256")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(fields)
    for row in summary.get("fp32_stage_a", {}).get("experiments", []):
        stats = row.get("inference", {}).get("stats", {}).get("inference_ms", {})
        comparison = row.get("comparisons", {}).get("overall", {})
        record = {"phase": "M4_original_FP32", **{key: row.get(key) for key in ("model_id", "variant_id", "threads", "status")},
                  "p50_ms": stats.get("p50"), "p95_ms": stats.get("p95"),
                  "speedup": comparison.get("speedup"), "ci95": comparison.get("ci95"),
                  "p95_ratio": comparison.get("p95_ratio"), "errors": row.get("errors")}
        writer.writerow([_csv_cell(record.get(field)) for field in fields])
    for row in summary.get("rows", []):
        performance = row.get("performance", {})
        stats = performance.get("candidate_stats") or row.get("timing_stats") or {}
        dev = row.get("development") or {}
        final = summary.get("final", {}).get(row["model_id"], {})
        record = {**row, "p50_ms": stats.get("p50"), "p95_ms": stats.get("p95"),
                  "speedup": performance.get("speedup"), "ci95": performance.get("ci95"),
                  "p95_ratio": performance.get("p95_ratio"),
                  "int8_operator_events": row.get("runtime_profile", {}).get("int8_operator_events"),
                  "development_top1": dev.get("top1_accuracy"), "development_top5": dev.get("top5_accuracy"),
                  "development_drop_ci": row.get("development_comparison"),
                  "final_holdout": final.get("holdout_comparison") if final.get("candidate_id") == row.get("id") else None,
                  "final_adoption": final.get("status") if final.get("candidate_id") == row.get("id") else None}
        writer.writerow([_csv_cell(record.get(field)) for field in fields])
    frozen = {row["model_id"]: row for row in summary.get("frozen", {}).get("value", {}).get("candidates", [])} if summary.get("frozen") else {}
    for model, roles in summary.get("observations", {}).items():
        for role, kinds in roles.items():
            for kind, observation in kinds.items():
                payload = observation.get("payload") or {}
                chosen = frozen.get(model, {})
                record = {"phase": "observations", "model_id": model, "variant_id": kind,
                          "threads": chosen.get("baseline_threads") if role == "baseline" else chosen.get("threads"),
                          "status": observation.get("status"), "errors": observation.get("error"),
                          "observation_role": role, "pipeline_stages_and_wall": payload.get("pipeline_stats"),
                          "independent_rss": payload.get("memory"), "frozen_sha256": observation.get("frozen_sha256")}
                writer.writerow([_csv_cell(record.get(field)) for field in fields])
    return stream.getvalue()


def _render_html(summary):
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src \'none\'; base-uri \'none\'">',
             '<title>分类阶段 B · CPU 离线报告</title><style>body{font:15px sans-serif;margin:2em}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.4em;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:1em 0}</style></head><body>',
             '<h1>阶段 B：FP32 / QB / Q1 / Q2 / Q3</h1>',
             '<p>无综合分；完整 1000-way 排名；失败/退化/不确定均保留。筛选、冻结确认与最终 holdout 分开。</p>']
    parts.append(f'<p>状态：{_escape(summary.get("status"))} · Run：{_escape(summary.get("run_id"))}</p>')
    for title, value in (("数据 scope、来源、许可与 SHA", summary.get("dataset")),
                         ("测量协议与隔离", summary.get("methodology")),
                         ("限制与归因", summary.get("limitations"))):
        parts.append(f'<details open><summary>{_escape(title)}</summary><pre>{_escape(value)}</pre></details>')
    parts.append('<h2>M4 原始多图 FP32（不与同期 QB 混用作 INT8 基线）</h2><table><tr><th>模型/配置</th><th>状态</th><th>推理/流水线/内存/启动</th><th>正确性/原始独立比较</th></tr>')
    for row in summary.get("fp32_stage_a", {}).get("experiments", []):
        parts.append('<tr><td>' + _escape(row.get("id")) + '</td><td>' + _escape(row.get("status")) +
                     '</td><td><pre>' + _escape({key: row.get(key) for key in ("inference", "pipeline", "memory")}) +
                     '</pre></td><td><pre>' + _escape({key: row.get(key) for key in ("validation", "comparisons", "errors")}) + '</pre></td></tr>')
    parts.append('</table><h2>同期筛选 / 有限组合（INT8 只对 QB.t1）</h2><table><tr><th>模型/配置/阶段</th><th>状态/体积</th><th>P50/P95、配对块 CI</th><th>开发 Top1/Top5 与下降 pp CI</th><th>实际整数 / 浮点覆盖</th></tr>')
    for row in summary.get("rows", []):
        values = (f'{row.get("id")} / {row.get("phase")}',
                  {key: row.get(key) for key in ("status", "size_bytes", "validation_passed", "errors")},
                  {key: row.get(key) for key in ("baseline_variant_id", "timing_stats", "performance")},
                  {key: row.get(key) for key in ("development", "development_comparison")},
                  {key: row.get(key) for key in ("graph_statistics", "runtime_profile")})
        parts.append('<tr>' + ''.join('<td><pre>' + _escape(value) + '</pre></td>' for value in values) + '</tr>')
    parts.append('</table>')
    for title, value in (("仅开发集选择依据", summary.get("selections")),
                         ("统一冻结配置及 SHA", summary.get("frozen")),
                         ("冻结基线/候选的独立分类流水线和 RSS（不同于微基准）", summary.get("observations")),
                         ("QB 前后哨兵、至多一次组复测及原始拒绝记录", summary.get("drift_controls")),
                         ("独立 5×500 确认与最终 holdout（失败后不改选）", summary.get("final")),
                         ("family / 原始作业 / PID / 样本与产物哈希索引", summary.get("evidence")),
                         ("失败与未支持记录", summary.get("failures"))):
        parts.append(f'<details><summary>{_escape(title)}</summary><pre>{_escape(value)}</pre></details>')
    parts.append('</body></html>')
    return ''.join(parts)


def write_report(output_dir, summary, *, overwrite=False):
    """Stage all report bytes, publish the completion marker last, rollback errors."""
    _reject_mock(summary)
    directory = Path(output_dir)
    if not directory.is_dir():
        raise StageBError("Report directory must already exist")
    files = {"summary.json": _json_bytes(summary), "report.html": _render_html(summary).encode("utf-8"),
             "results.csv": _report_csv(summary).encode("utf-8-sig")}
    marker = {"schema_version": 1, "stage": "B", "run_id": summary.get("run_id"),
              "status": summary.get("status"),
              "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["report.complete.json"] = _json_bytes(marker)
    for name in files:
        target = directory / name
        if target.is_symlink() or (target.exists() and not overwrite):
            raise StageBError(f"Refusing report overwrite: {target}")
    published = []
    with tempfile.TemporaryDirectory(dir=directory, prefix=".report-stage-b-") as temporary:
        staging = Path(temporary)
        old = staging / "previous"
        old.mkdir()
        for name, data in files.items():
            (staging / name).write_bytes(data)
            if (directory / name).exists():
                shutil.copy2(directory / name, old / name)
        try:
            for name in files:
                if (directory / name).is_symlink() or ((directory / name).exists() and not overwrite):
                    raise StageBError("Report publication collision")
                os.replace(staging / name, directory / name)
                published.append(name)
        except BaseException:
            for name in reversed(published):
                if (old / name).exists():
                    os.replace(old / name, directory / name)
                elif (directory / name).exists():
                    (directory / name).unlink()
            raise
    return {name: str(directory / name) for name in files}


def _evaluation_check(value, job, manifest):
    split = job["split"]
    images = sorted((entry for entry in manifest["images"] if entry["split"] == split), key=lambda row: row["id"])
    if (value.get("manifest_sha256") != job["manifest_sha256"] or value.get("split") != split
            or value.get("model_id") != job["model_id"] or value.get("artifact_sha256") != job["artifact_sha256"]
            or value.get("image_count") != len(images)):
        raise StageBError("Evaluator manifest/model/artifact/split binding mismatch")
    rows = value.get("per_image")
    if not isinstance(rows, list) or len(rows) != len(images):
        raise StageBError("Evaluator is missing genuine labeled image results")
    for original, row in zip(images, rows):
        if any(row.get(key) != original.get(key) for key in ("id", "label", "sha256")):
            raise StageBError("Evaluator image id/label/SHA mismatch")
        _int(row.get("label"), "full1000 label", maximum=999)
        _int(row.get("top1"), "full1000 top1", maximum=999)
        top5 = row.get("top5")
        if not isinstance(top5, list) or len(top5) != 5 or len(set(top5)) != 5:
            raise StageBError("Full1000-way top5 contract mismatch")
        for index in top5:
            _int(index, "full1000 top5", maximum=999)
        if (row["top1"] != top5[0] or type(row.get("top1_correct")) is not bool
                or type(row.get("top5_correct")) is not bool
                or row["top1_correct"] != (row["top1"] == row["label"])
                or row["top5_correct"] != (row["label"] in top5)):
            raise StageBError("Accuracy fields cannot substitute model predictions for ground truth")
    for metric in ("top1", "top5"):
        actual = sum(row[f"{metric}_correct"] for row in rows) / len(rows)
        if abs(_number(value.get(f"{metric}_accuracy"), metric) - actual) > 1e-12:
            raise StageBError("Evaluator summary accuracy disagrees with the genuine per-image labels")
    session = value.get("session", {})
    if (session.get("providers") != ["CPUExecutionProvider"]
            or session.get("threads", session.get("intra_op_num_threads", session.get("intra_op_threads"))) != job["threads"]):
        raise StageBError("Evaluator CPU/thread contract mismatch")
    if split == "holdout":
        frozen = value.get("frozen", {})
        if frozen.get("sha256") != job.get("frozen_sha256"):
            raise StageBError("Evaluator did not execute with the frozen holdout guard")


def _profile_check(profile, job):
    if profile.get("artifact_sha256") != job["artifact_sha256"]:
        raise StageBError("Profile artifact SHA mismatch")
    if profile.get("providers") != ["CPUExecutionProvider"]:
        raise StageBError("Profile must identify actual CPU kernels")
    events = profile.get("int8_operator_events")
    if not isinstance(events, list) and type(events) is not int:
        raise StageBError("Profile integer events must be a list or a count")
    counts = profile.get("int8_operator_counts")
    if not isinstance(counts, dict):
        raise StageBError("Profile integer operator counts are missing")
    total = sum(_int(value, "integer events", 1) for value in counts.values())
    if (len(events) if isinstance(events, list) else _int(events, "integer event count")) != total:
        raise StageBError("Profile integer event/count mismatch")
    _check_artifact(profile.get("trace_path", ""), profile.get("trace_sha256"), "runtime trace")


def _family_check(family, job, manifest):
    if (family.get("schema_version") != 1 or family.get("model_id") != job["model_id"]
            or family.get("manifest_sha256") != job["manifest_sha256"]
            or family.get("input_model", {}).get("sha256") != job["raw_onnx_sha256"]):
        raise StageBError("Family input graph/model/manifest binding mismatch")
    calibration = family.get("calibration", {})
    images = sorted((entry for entry in manifest["images"] if entry["split"] == "calibration"), key=lambda row: row["id"])
    random.Random(job["seed"]).shuffle(images)
    if calibration.get("count") != 256 or calibration.get("image_ids") != [row["id"] for row in images]:
        raise StageBError("Family calibration image count/order mismatch")
    recorded = calibration.get("images", [])
    if len(recorded) != 256 or any(any(row.get(key) != original.get(key) for key in ("id", "sha256", "source_id"))
                                   for row, original in zip(recorded, images)):
        raise StageBError("Family calibration sample SHA/source mismatch")
    if set(family.get("variants", {})) != set(FAMILY_VARIANTS):
        raise StageBError("Family must report QB and all Q1/Q2/Q3, including unsupported")
    for variant, row in family["variants"].items():
        if row.get("status") not in ("prepared", "unsupported", "failed_runtime", "failed_correctness"):
            raise StageBError("Unknown family artifact status")
        if row["status"] == "prepared":
            path = _inside(row.get("path", ""), job["family_output_dir"])
            _check_artifact(path, row.get("onnx_sha256"), variant)
            if row.get("size_bytes") != path.stat().st_size or row.get("validation", {}).get("passed") is not True:
                raise StageBError("Prepared family artifact lacks size/validation evidence")
            config = row.get("config", {})
            if variant != "QB" and config.get("parent_onnx_sha256", config.get("parent_qb_sha256")) != family["variants"]["QB"]["onnx_sha256"]:
                raise StageBError("Q variant is not derived from the same QB")
        elif not (row.get("reason") or row.get("error")):
            raise StageBError("Family failures must contain genuine reasons")
    path = Path(job["family_output_dir"]) / "family.json"
    if read_json(path) != family:
        raise StageBError("Prepared family does not match its on-disk record")


def make_job(run, run_dir, model_id, variant_id, threads, *, kind, phase,
             block=0, role="candidate", artifact=None, budget=None, **extra):
    """Create a closed, hashed job protocol; no arbitrary callable/plugin jobs."""
    if model_id not in MODELS or variant_id not in EXECUTABLE_VARIANTS or threads not in THREADS:
        raise StageBError("Job model/variant/thread not in Stage B protocol")
    if kind not in ("reference", "prepare", "measure", "pipeline", "memory", "control", "evaluate", "profile", "compare"):
        raise StageBError("Unknown Stage B worker job")
    if re.fullmatch(r"[A-Za-z0-9_-]+", phase) is None or role not in ("baseline", "candidate"):
        raise StageBError("Unsafe phase/role")
    _int(block, "block", maximum=4)
    run_dir = Path(run_dir).resolve()
    item = run["fp32_prepared"][model_id]
    name = f"{phase}.{model_id}.{variant_id}.t{threads}.{kind}.{role}.b{block}"
    if "attempt" in extra:
        _int(extra["attempt"], "drift attempt", maximum=1)
        name += f".a{extra['attempt']}"
    if "control_role" in extra:
        if extra["control_role"] not in ("before", "after"):
            raise StageBError("Unknown sentinel role")
        name += f".{extra['control_role']}"
    phase_seed = run["config"]["seed"] + {"combination": 10000, "confirmation": 20000}.get(phase, 0)
    images = run["images"]
    job = {"schema_version": 1, "stage": "B", "run_id": run["run_id"], "run_dir": str(run_dir),
           "round_id": f"{run['run_id']}:{phase}", "job_name": name, "model_id": model_id,
           "variant_id": variant_id, "threads": threads, "kind": kind, "phase": phase,
           "role": role, "block": block, "config_sha256": run["config_sha256"],
           "manifest_sha256": run["manifest_sha256"], "manifest_path": run["manifest_path"],
           "source_path": run["source_path"], "source_sha256": run["source_sha256"],
           "code_fingerprint": run["code_fingerprint"], "seed": phase_seed,
           "fp32_snapshot_sha256": run["fp32_snapshot_sha256"], "images": copy.deepcopy(images),
           "weights_path": item["weights_path"], "weights_sha256": item["weights_sha256"],
           "reference_path": item["reference_path"], "reference_sha256": item["reference_sha256"],
           "raw_onnx_path": item["onnx_path"], "raw_onnx_sha256": item["onnx_sha256"],
           "model_metadata": item["model_metadata"],
           "output_path": str(run_dir / "results" / f"{name}.json")}
    if artifact is not None:
        job.update(artifact_path=artifact["path"], artifact_sha256=artifact["onnx_sha256"])
    if budget is not None:
        job.update(iterations=budget["iterations"], warmup=budget["warmup"],
                   image_sequence=fixed_sequence(phase_seed, block, budget["iterations"]))
        job["input_sequence_sha256"] = canonical_hash({"reference_sha256": job["reference_sha256"],
            "image_sequence": job["image_sequence"], "images": [{"id": row["id"], "sha256": row["sha256"]} for row in images]})
    job.update(extra)
    if phase == "confirmation" or kind in ("pipeline", "memory"):
        frozen = run.get("frozen")
        if not isinstance(frozen, dict):
            raise StageBError("confirmation guard: all models must be frozen first")
        job.update(frozen_path=frozen["path"], frozen_sha256=frozen["sha256"])
    return job


def _guard_frozen_job(job, run, fingerprint):
    _check_artifact(job.get("frozen_path", ""), job.get("frozen_sha256"), "frozen")
    frozen = validate_frozen(read_json(job["frozen_path"]), manifest_sha256=job["manifest_sha256"],
                             code=fingerprint, family_files=run.get("family_files"))
    chosen = next(row for row in frozen["candidates"] if row["model_id"] == job["model_id"])
    if job["kind"] == "control":
        qb = run.get("families", {}).get(job["model_id"], {}).get("variants", {}).get("QB", {})
        if qb.get("status") != "prepared" or job.get("artifact_sha256") != qb.get("onnx_sha256") or job.get("threads") != 1:
            raise StageBError("frozen guard: sentinel must be the contemporaneous family QB.t1")
    elif job["kind"] != "compare":
        baseline = job.get("role") == "baseline"
        artifact = chosen["baseline_artifact_sha256"] if baseline else chosen["artifact_sha256"]
        threads = chosen["baseline_threads"] if baseline else chosen["threads"]
        variant = ("QB" if chosen["variant_id"] in ("Q1", "Q2", "Q3") else "O1") if baseline else chosen["variant_id"]
        if (job.get("artifact_sha256"), job.get("threads"), job.get("variant_id")) != (artifact, threads, variant):
            raise StageBError("frozen guard: candidate/baseline role, artifact, variant or threads mismatch")
    return frozen


def validate_job(job_path, job):
    """Worker guard runs before importing any heavy dependency or reading pixels."""
    run_dir = Path(job.get("run_dir", "")).resolve()
    name = job.get("job_name", "")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
        raise StageBError("Unsafe job name")
    if Path(job_path).resolve() != run_dir / "jobs" / f"{name}.json":
        raise StageBError("Job must be the dispatched local file")
    if Path(job.get("output_path", "")).resolve() != run_dir / "results" / f"{name}.json":
        raise StageBError("Job output must be its own result file")
    if type(job.get("schema_version")) is not int or job.get("schema_version") != 1 or job.get("stage") != "B":
        raise StageBError("Not a Stage B job")
    if (job.get("model_id") not in MODELS or job.get("variant_id") not in EXECUTABLE_VARIANTS
            or type(job.get("threads")) is not int or job["threads"] not in THREADS
            or job.get("kind") not in ("reference", "prepare", "measure", "pipeline", "memory", "control", "evaluate", "profile", "compare")):
        raise StageBError("Job whitelist mismatch")
    fingerprint, _ = code_fingerprint()
    if job.get("code_fingerprint") != fingerprint:
        raise StageBError("Worker code fingerprint mismatch")
    run = read_json(run_dir / "run.json")
    for key in ("run_id", "manifest_sha256", "config_sha256", "code_fingerprint", "fp32_snapshot_sha256"):
        if job.get(key) != run.get(key):
            raise StageBError(f"Job/run mismatch: {key}")
    for name, path_key, hash_key in (("manifest", "manifest_path", "manifest_sha256"),
                                    ("source", "source_path", "source_sha256"),
                                    ("weights", "weights_path", "weights_sha256"),
                                    ("reference", "reference_path", "reference_sha256"),
                                    ("raw ONNX", "raw_onnx_path", "raw_onnx_sha256")):
        _check_artifact(job.get(path_key, ""), job.get(hash_key), name)
    if job["model_metadata"].get("preprocess") != PREPROCESS:
        raise StageBError("Worker normalization contract mismatch")
    if job["kind"] in ("measure", "pipeline", "memory", "control", "evaluate", "profile"):
        _check_artifact(job.get("artifact_path", ""), job.get("artifact_sha256"))
    if job["kind"] in ("measure", "pipeline", "memory", "control"):
        if job.get("iterations") not in (200, 500) or job.get("warmup") != 20:
            raise StageBError("Worker formal budget mismatch")
        expected = fixed_sequence(job["seed"], job["block"], job["iterations"])
        if job.get("image_sequence") != expected or len(job.get("images", [])) != 64:
            raise StageBError("Worker input sequence mismatch")
        expected_sha = canonical_hash({"reference_sha256": job["reference_sha256"], "image_sequence": expected,
                                      "images": [{"id": row["id"], "sha256": row["sha256"]} for row in job["images"]]})
        if job.get("input_sequence_sha256") != expected_sha:
            raise StageBError("Worker input sequence SHA mismatch")
    if job["kind"] in ("evaluate", "compare"):
        if job.get("split") not in ("development", "holdout"):
            raise StageBError("Unknown evaluation/comparison split")
        if job["split"] == "holdout":
            _guard_frozen_job(job, run, fingerprint)
    if job["kind"] in ("pipeline", "memory") or (job["kind"] in ("measure", "control") and job.get("phase") == "confirmation"):
        _guard_frozen_job(job, run, fingerprint)
    if job["kind"] in ("pipeline", "memory") and (job.get("phase") != "observations" or job.get("iterations") != 200):
        raise StageBError("Observation protocol requires frozen observations phase, warm20/200 samples")
    if job["kind"] == "prepare":
        if Path(job.get("family_output_dir", "")).resolve() != run_dir / "families" / job["model_id"]:
            raise StageBError("Family output escapes its model directory")
    if job["kind"] == "profile":
        _inside(job.get("profile_output_dir", ""), run_dir / "profiles")
    if job["kind"] == "compare":
        for prefix in ("reference_eval", "candidate_eval"):
            _check_artifact(job.get(f"{prefix}_path", ""), job.get(f"{prefix}_sha256"), prefix)
    return Path(job["output_path"])


def _result_base(job, *, pid=None, identity=None):
    if identity is None:
        if pid is not None:
            raise StageBError("Parent failures need captured process-instance identity, not an invented PID marker")
        from tools.run_classifier_benchmark import worker_process_identity
        identity = worker_process_identity()
    lifecycle = _process_lifecycle(identity, pid)
    keys = ("run_id", "round_id", "config_sha256", "manifest_sha256", "code_fingerprint",
            "fp32_snapshot_sha256", "job_name", "model_id", "variant_id", "threads", "kind", "phase", "role", "block")
    result = {key: job[key] for key in keys}
    result.update(schema_version=1, stage="B", status="failed_runtime", job_sha256=canonical_hash(job),
                  **lifecycle, synthetic=False, evidence_kind="cpu_worker",
                  started_at=_now(), samples=[], payload=None, artifact_refs=[])
    for key in ("iterations", "warmup", "image_sequence", "input_sequence_sha256", "frozen_sha256", "attempt", "control_role"):
        if key in job:
            result[key] = copy.deepcopy(job[key])
    return result


def _load_reference(job):
    import numpy as np
    with np.load(job["reference_path"], allow_pickle=False) as archive:
        if set(archive.files) != {"inputs", "logits"}:
            raise StageBError("Reference NPZ has unexpected arrays")
        inputs, logits = archive["inputs"].copy(), archive["logits"].copy()
    if (inputs.shape != (64, 3, 224, 224) or logits.shape != (64, 1000)
            or inputs.dtype != np.float32 or logits.dtype != np.float32
            or not np.isfinite(inputs).all() or not np.isfinite(logits).all()):
        raise StageBError("M4 full1000 reference/normalization arrays are invalid")
    return inputs, logits


def _audit_reference_worker(job):
    import numpy as np
    from src.vision.classification import get_model_spec, open_rgb_image, prepare_image, configure_cpu_threads
    configure_cpu_threads(1)
    spec = get_model_spec(job["model_id"])
    metadata = job["model_metadata"]
    if spec["preprocess"] != metadata["preprocess"] or spec["categories"] != metadata["categories"]:
        raise StageBError("Current V1 category/normalization contract differs from M4")
    if spec["weights_id"] != metadata.get("weights_id") or not job["weights_sha256"].startswith(spec["hash_prefix"]):
        raise StageBError("M4 weights are not the pinned official V1 weights")
    inputs, _ = _load_reference(job)
    records = []
    for index, image in enumerate(job["images"]):
        _check_artifact(image["resolved_path"], image["sha256"], "performance image")
        actual = prepare_image(open_rgb_image(image["resolved_path"]), model_id=job["model_id"]).detach().cpu().numpy()[0]
        if not np.array_equal(actual, inputs[index]):
            raise StageBError("M4 prepared input differs from the current canonical source normalization")
        records.append({"id": image["id"], "sha256": image["sha256"],
                        "input_sha256": hashlib.sha256(np.ascontiguousarray(inputs[index]).tobytes()).hexdigest()})
    return {"passed": True, "image_count": 64, "input_shape": [64, 3, 224, 224],
            "output_shape": [64, 1000], "reference_sha256": job["reference_sha256"],
            "weights_sha256": job["weights_sha256"], "manifest_sha256": job["manifest_sha256"],
            "preprocess": spec["preprocess"], "per_image": records}


def _checked_runtime(job, result):
    import numpy as np
    from src.vision.classification_variants import create_ort_session
    inputs, expected = _load_reference(job)
    began = time.perf_counter_ns()
    # Reuse the ORT adapter without adding Stage B identifiers to Stage A's whitelist.
    session = create_ort_session(job["artifact_path"], threads=job["threads"], variant_id="O1")
    input_info, output_info = session.get_inputs(), session.get_outputs()
    if (len(input_info) != 1 or len(output_info) != 1 or input_info[0].name != "images"
            or output_info[0].name != "logits" or input_info[0].shape != [1, 3, 224, 224]
            or output_info[0].shape != [1, 1000] or input_info[0].type != "tensor(float)"
            or output_info[0].type != "tensor(float)" or session.get_providers() != ["CPUExecutionProvider"]):
        raise StageBError("Runtime must retain FP32 I/O and full1000 CPU output contract")
    result["load_ms"] = (time.perf_counter_ns() - began) / 1e6
    arrays = [np.ascontiguousarray(inputs[index:index + 1]) for index in range(64)]
    input_hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in arrays]
    validations = []
    require_equivalence = job["variant_id"] in ("O1", "QB")
    for index, value in enumerate(arrays):
        began = time.perf_counter_ns()
        actual = session.run(["logits"], {"images": value})[0]
        if index == 0:
            result["first_inference_ms"] = (time.perf_counter_ns() - began) / 1e6
        if actual.shape != (1, 1000) or actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise StageBError("Nonfinite, wrong dtype/shape or subset-masked logits")
        difference = np.abs(actual[0] - expected[index])
        allclose = bool(np.allclose(actual[0], expected[index], rtol=1e-4, atol=1e-4))
        validations.append({"id": job["images"][index]["id"], "sha256": job["images"][index]["sha256"],
                            "allclose": allclose, "max_abs_error": float(difference.max()),
                            "mean_abs_error": float(difference.mean()),
                            "top1_equal": int(np.argmax(actual[0])) == int(np.argmax(expected[index]))})
    numerical_passed = all(row["allclose"] for row in validations)
    result["validation"] = {"passed": numerical_passed if require_equivalence else True,
                            "image_count": 64, "finite_full1000": True, "rtol": 1e-4, "atol": 1e-4,
                            "numerical_equivalence_required": require_equivalence,
                            "numerical_equivalence_passed": numerical_passed, "per_image": validations,
                            "note": "INT8 adoption requires genuine development labels, not FP32 allclose"}
    if require_equivalence and not numerical_passed:
        result.update(status="failed_correctness", error="FP32 allclose hard gate failed")
        return None, None, None
    return session, arrays, input_hashes


def _measure_worker(job, result):
    import numpy as np
    session, arrays, input_hashes = _checked_runtime(job, result)
    if session is None:
        return
    for index in range(job["warmup"]):
        session.run(["logits"], {"images": arrays[index % 64]})
    began = time.perf_counter_ns()
    for sequence_index, image_index in enumerate(job["image_sequence"]):
        tick = time.perf_counter_ns()
        actual = session.run(["logits"], {"images": arrays[image_index]})[0]
        elapsed = (time.perf_counter_ns() - tick) / 1e6
        if actual.shape != (1, 1000) or actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise StageBError("Invalid runtime output during measured sequence")
        image = job["images"][image_index]
        result["samples"].append({"sequence_index": sequence_index, "input_index": image_index,
                                  "image_id": image["id"], "image_sha256": image["sha256"],
                                  "input_sha256": input_hashes[image_index], "inference_ms": elapsed})
    result["measured_wall_ms"] = (time.perf_counter_ns() - began) / 1e6
    result["payload"] = {"timing_stats": describe([row["inference_ms"] for row in result["samples"]]),
                         "throughput_per_s": job["iterations"] * 1000 / result["measured_wall_ms"],
                         "scope": "serial batch=1 measured loop wall, not system throughput",
                         "providers": session.get_providers(), "intra_op_threads": job["threads"],
                         "inter_op_threads": 1, "allow_spinning": 0, "graph_optimization": "ORT_ENABLE_ALL",
                         "execution_mode": "ORT_SEQUENTIAL", "artifact_sha256": job["artifact_sha256"]}
    result["status"] = "passed"
    del session


def _runtime_payload(job, session):
    return {"artifact_sha256": job["artifact_sha256"], "providers": session.get_providers(),
            "intra_op_threads": job["threads"], "inter_op_threads": 1, "allow_spinning": 0,
            "graph_optimization": "ORT_ENABLE_ALL", "execution_mode": "ORT_SEQUENTIAL"}


def _workload_sample(job, index, input_hashes):
    image_index = job["image_sequence"][index]
    image = job["images"][image_index]
    return {"sequence_index": index, "input_index": image_index, "image_id": image["id"],
            "image_sha256": image["sha256"], "input_sha256": input_hashes[image_index]}


def collect_pipeline_samples(job, decoded, prepare, adapt, infer, post, input_hashes, *, clock=time.perf_counter_ns):
    """An injectable synchronous loop; wall time comes from timed_pipeline, never P50 sums."""
    from tools.run_classifier_benchmark import timed_pipeline
    for index in range(job["warmup"]):
        timed_pipeline(decoded[index % len(decoded)], prepare, adapt, infer, post, clock=clock)
    samples = []
    for index, image_index in enumerate(job["image_sequence"]):
        _, timing = timed_pipeline(decoded[image_index], prepare, adapt, infer, post, clock=clock)
        samples.append({**_workload_sample(job, index, input_hashes), **timing})
    return samples


def collect_memory_samples(job, arrays, infer, input_hashes, reader, sampler):
    """Observe RSS in a separate process. No inference latency sample is manufactured."""
    for index in range(job["warmup"]):
        infer(arrays[index % len(arrays)])
    loaded = reader.read()
    workload = []
    with sampler:
        for index, image_index in enumerate(job["image_sequence"]):
            infer(arrays[image_index])
            workload.append(_workload_sample(job, index, input_hashes))
    if sampler.error:
        raise StageBError(f"RSS sampler failed: {sampler.error}")
    values = [_int(value, "sampled RSS", 1) for value in sampler.values]
    if not values:
        raise StageBError("Independent memory worker produced no RSS samples")
    return workload, loaded, values


def _write_rss_log(output, values):
    path = Path(output).with_suffix(".rss.jsonl")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for index, value in enumerate(values):
            handle.write(json.dumps({"sample_index": index, "rss_bytes": value}, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": sha256(path)}


def _observation_worker(job, result):
    reader, baseline = None, None
    if job["kind"] == "memory":
        from tools.run_classifier_benchmark import RssReader
        reader = RssReader()
        baseline = reader.read()  # Before NumPy/ORT imports and session/reference loading.
    session, arrays, input_hashes = _checked_runtime(job, result)
    if session is None:
        return
    infer = lambda value: session.run(["logits"], {"images": value})[0]
    payload = _runtime_payload(job, session)
    if job["kind"] == "pipeline":
        import numpy as np
        from src.vision.classification import configure_cpu_threads, open_rgb_image, prepare_image, topk_predictions
        from src.vision.classification_benchmark import stage_stats
        configure_cpu_threads(job["threads"], inter_op_threads=1)
        categories = job["model_metadata"]["categories"]
        if len(categories) != 1000:
            raise StageBError("Pipeline postprocess must rank all 1000 classes")
        decoded = []
        for image in job["images"]:
            _check_artifact(image["resolved_path"], image["sha256"], "performance RGB source")
            decoded.append(open_rgb_image(image["resolved_path"]))
        prepare = lambda image: prepare_image(image, model_id=job["model_id"])
        adapt = lambda tensor: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        post = lambda logits: topk_predictions(logits, categories, k=5)
        for index, image in enumerate(decoded):
            if not np.array_equal(adapt(prepare(image)), arrays[index]):
                raise StageBError("Pipeline source/normalization differs from frozen performance reference")
        result["samples"] = collect_pipeline_samples(job, decoded, prepare, adapt, infer, post, input_hashes)
        payload.update(pipeline_stats=stage_stats(result["samples"]),
                       scope="decoded performance RGB -> canonical preprocessing -> ORT input adaptation -> inference -> full1000 softmax/top5",
                       excludes=["disk decode", "imports", "session loading", "reference validation", "warmup"],
                       wall_note="pipeline_wall_ms measured directly; never sum phase P50 values",
                       prediction_class_count=1000, independent_process=True)
    else:
        from tools.run_classifier_benchmark import RssSampler
        sampler = RssSampler(reader, interval=.02)
        workload, loaded, values = collect_memory_samples(job, arrays, infer, input_hashes, reader, sampler)
        result["samples"] = workload
        trace = _write_rss_log(job["output_path"], values)
        result["artifact_refs"].append(trace)
        payload.update(memory={"rss_baseline_bytes": baseline, "rss_loaded_bytes": loaded,
                       "rss_peak_bytes": max([loaded] + values), "rss_source": reader.source,
                       "sampling_interval_ms": sampler.interval * 1000, "sample_count": len(values),
                       "rss_samples": values, "raw_rss": trace, "workload_iterations": job["iterations"],
                       "timing_included": False, "independent_process": True,
                       "dependency_versions": environment_info()["versions"],
                       "runtime_modules": {name: name in sys.modules for name in ("numpy", "torch", "torchvision", "onnxruntime", "PIL")},
                       "scope": "prepared-input ORT inference, runtime + all 64 prepared inputs + allocator caches; not model-exclusive RSS",
                       "limitations": ["20ms sampling can miss brief peaks; before/after samples are also included",
                                       "Different backend dependency loads are included, not subtracted",
                                       "This process does not supply latency or pipeline timing samples"]})
    result["payload"], result["status"] = payload, "passed"
    del session


def _write_samples(output, result):
    if not result["samples"]:
        return
    path = Path(output).with_suffix(".samples.jsonl")
    if path.exists():
        raise StageBError("Raw timing JSONL already exists")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in result["samples"]:
            value = {"job_sha256": result["job_sha256"], **_process_lifecycle(result), **row}
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    result["samples_path"], result["samples_sha256"] = str(path), sha256(path)
    result["artifact_refs"].append({"path": str(path), "sha256": result["samples_sha256"]})


def worker(job_path):
    job = read_json(job_path)
    output = validate_job(job_path, job)
    if output.exists():
        raise StageBError("Worker may not overwrite previous evidence")
    result = _result_base(job)
    try:
        kind = job["kind"]
        if kind == "reference":
            result["payload"] = _audit_reference_worker(job)
        elif kind in ("measure", "control"):
            _measure_worker(job, result)
        elif kind in ("pipeline", "memory"):
            _observation_worker(job, result)
        else:
            from src.vision import classification_quantization as quant
            if kind == "prepare":
                result["payload"] = quant.prepare_quantized_family(
                    job["model_id"], job["raw_onnx_path"], job["manifest_path"], job["family_output_dir"],
                    overwrite=False, seed=job["seed"])
                for path in Path(job["family_output_dir"]).iterdir():
                    if path.is_file():
                        result["artifact_refs"].append({"path": str(path.resolve()), "sha256": sha256(path)})
            elif kind == "evaluate":
                result["payload"] = quant.evaluate_classifier(
                    job["model_id"], job["artifact_path"], job["manifest_path"], split=job["split"], threads=job["threads"],
                    frozen_path=job.get("frozen_path"), expected_frozen_sha256=job.get("frozen_sha256"))
            elif kind == "profile":
                inputs, _ = _load_reference(job)
                result["payload"] = quant.profile_quantized_execution(
                    job["artifact_path"], inputs[:1], job["profile_output_dir"], threads=job["threads"])
                result["artifact_refs"].append({"path": result["payload"]["trace_path"],
                                                "sha256": result["payload"]["trace_sha256"]})
            else:
                reference = read_json(job["reference_eval_path"])["payload"]
                candidate = read_json(job["candidate_eval_path"])["payload"]
                if reference.get("split") != job["split"] or candidate.get("split") != job["split"]:
                    raise StageBError("Accuracy comparison split mismatch")
                result["payload"] = quant.compare_accuracies(reference, candidate, seed=job["seed"], bootstrap_samples=2000)
        if kind not in ("measure", "pipeline", "memory", "control"):
            result["status"] = "passed"
        # Rehash trusted source files after each operation, without reading holdout pixels in the parent.
        for path, digest in ((job["manifest_path"], job["manifest_sha256"]),
                             (job["source_path"], job["source_sha256"]),
                             (job["raw_onnx_path"], job["raw_onnx_sha256"]),
                             (job["reference_path"], job["reference_sha256"]),
                             (job["weights_path"], job["weights_sha256"])):
            _check_artifact(path, digest)
        if "artifact_path" in job:
            _check_artifact(job["artifact_path"], job["artifact_sha256"])
        if "frozen_path" in job:
            _check_artifact(job["frozen_path"], job["frozen_sha256"])
    except Exception as exc:
        result.update(status="failed_runtime", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    result["completed_at"] = _now()
    result["runtime_modules"] = {name: name in sys.modules for name in ("torch", "torchvision", "onnxruntime")}
    _write_samples(output, result)
    atomic_json(output, result)
    return 0 if result["status"] == "passed" else 2


def validate_result(job, result, *, expected_pid, expected_identity=None, manifest=None, input_hashes=None):
    """Bind child SHA/PID/sequence/bytes before using any reported metric."""
    _reject_mock(result)
    if (type(result.get("schema_version")) is not int or result.get("schema_version") != 1 or result.get("stage") != "B"
            or result.get("job_sha256") != canonical_hash(job)
            or type(result.get("worker_pid")) is not int or result["worker_pid"] != expected_pid):
        raise StageBError("Worker job SHA/PID binding mismatch")
    _int(expected_pid, "successful child PID", 1)
    lifecycle = _process_lifecycle(result, expected_pid)
    if expected_identity is not None and lifecycle != _process_lifecycle(expected_identity, expected_pid):
        raise StageBError("Worker OS process-instance identity differs from the launch receipt")
    for key in ("run_id", "round_id", "config_sha256", "manifest_sha256", "code_fingerprint", "fp32_snapshot_sha256",
                "job_name", "model_id", "variant_id", "threads", "kind", "phase", "role", "block"):
        if result.get(key) != job.get(key):
            raise StageBError(f"Worker result job binding mismatch: {key}")
    if result.get("status") not in ("passed", "unsupported", "failed_runtime", "failed_correctness", "interrupted"):
        raise StageBError("Unknown child status")
    if any(key in job and result.get(key) != job[key] for key in ("frozen_sha256", "attempt", "control_role")):
        raise StageBError("Worker frozen SHA/drift-attempt/sentinel-role binding mismatch")
    for reference in result.get("artifact_refs", []):
        _inside(reference.get("path", ""), job["run_dir"])
        _check_artifact(reference["path"], reference.get("sha256"), "worker output")
    if result["status"] != "passed":
        if not result.get("error"):
            raise StageBError("A failed result must expose its real reason")
        return result
    payload = result.get("payload")
    if not isinstance(payload, dict):
        raise StageBError("Successful worker is missing actual payload")
    if job["kind"] in ("measure", "pipeline", "memory", "control"):
        if any(result.get(key) != job.get(key) for key in ("iterations", "warmup", "image_sequence", "input_sequence_sha256")):
            raise StageBError("Worker sequence/budget SHA mismatch")
        samples = result.get("samples", [])
        if len(samples) != job["iterations"] or result.get("validation", {}).get("passed") is not True:
            raise StageBError("Successful measurement is incomplete or failed correctness")
        for index, sample in enumerate(samples):
            input_index = job["image_sequence"][index]
            image = job["images"][input_index]
            if (sample.get("sequence_index") != index or sample.get("input_index") != input_index
                    or sample.get("image_id") != image["id"] or sample.get("image_sha256") != image["sha256"]):
                raise StageBError("Worker sample input sequence/image SHA mismatch")
            _digest(sample.get("input_sha256"), "normalized input SHA")
            if input_hashes is not None and sample["input_sha256"] != input_hashes[image["id"]]:
                raise StageBError("Worker sample normalization SHA mismatch")
            if job["kind"] != "memory":
                _number(sample.get("inference_ms"), "inference_ms", True)
            elif any(key.endswith("_ms") for key in sample):
                raise StageBError("Memory samples must not masquerade as latency samples")
        _check_artifact(result.get("samples_path", ""), result.get("samples_sha256"), "raw timing JSONL")
        raw = Path(result["samples_path"]).read_text(encoding="utf-8").splitlines()
        if len(raw) != len(samples) or any(json.loads(line) != {"job_sha256": result["job_sha256"],
                **_process_lifecycle(result), **sample} for line, sample in zip(raw, samples)):
            raise StageBError("Raw timing JSONL differs from the successful child samples")
        if (payload.get("artifact_sha256") != job["artifact_sha256"] or payload.get("providers") != ["CPUExecutionProvider"]
                or payload.get("intra_op_threads") != job["threads"] or payload.get("inter_op_threads") != 1
                or payload.get("execution_mode") != "ORT_SEQUENTIAL" or payload.get("graph_optimization") != "ORT_ENABLE_ALL"
                or payload.get("allow_spinning") != 0):
            raise StageBError("Measurement artifact/provider/thread/runtime configuration mismatch")
        if job["variant_id"] in ("O1", "QB") and result["validation"].get("numerical_equivalence_passed") is not True:
            raise StageBError("FP32 hard numerical gate cannot be bypassed")
        if job["kind"] == "pipeline":
            from src.vision.classification_benchmark import stage_stats, TIMING_FIELDS
            if any(any(key not in row for key in TIMING_FIELDS) for row in samples):
                raise StageBError("Pipeline requires every measured stage and direct wall clock")
            if payload.get("pipeline_stats") != stage_stats(samples) or payload.get("prediction_class_count") != 1000:
                raise StageBError("Pipeline wall/stage statistics or full1000 postprocess contract mismatch")
        if job["kind"] == "memory":
            memory = payload.get("memory", {})
            for field in ("rss_baseline_bytes", "rss_loaded_bytes", "rss_peak_bytes", "sample_count"):
                _int(memory.get(field), field, 1)
            if (memory.get("timing_included") is not False or memory.get("independent_process") is not True
                    or memory.get("sampling_interval_ms") != 20.0 or memory.get("workload_iterations") != 200):
                raise StageBError("Memory observation scope/interval/budget contract mismatch")
            values = memory.get("rss_samples", [])
            if len(values) != memory["sample_count"] or any(type(value) is not int or value < 1 for value in values):
                raise StageBError("RSS raw sample inventory mismatch")
            if memory["rss_peak_bytes"] != max([memory["rss_loaded_bytes"]] + values):
                raise StageBError("RSS peak is not derived from actual independent samples")
            trace = memory.get("raw_rss", {})
            _check_artifact(trace.get("path", ""), trace.get("sha256"), "raw RSS JSONL")
            raw_rss = [json.loads(line) for line in Path(trace["path"]).read_text(encoding="utf-8").splitlines()]
            if raw_rss != [{"sample_index": index, "rss_bytes": value} for index, value in enumerate(values)]:
                raise StageBError("Raw RSS JSONL does not match the sampled values")
    elif job["kind"] == "reference":
        if (payload.get("passed") is not True or payload.get("reference_sha256") != job["reference_sha256"]
                or payload.get("weights_sha256") != job["weights_sha256"]
                or payload.get("image_count") != 64 or payload.get("preprocess") != PREPROCESS):
            raise StageBError("Reference audit was not bound to M4/source normalization")
        per_image = payload.get("per_image", [])
        if len(per_image) != 64 or any(any(row.get(key) != original[key] for key in ("id", "sha256"))
                                      for row, original in zip(per_image, job["images"])):
            raise StageBError("Reference audit sample identities mismatch")
        for row in per_image:
            _digest(row.get("input_sha256"))
    elif job["kind"] == "evaluate" and manifest is not None:
        _evaluation_check(payload, job, manifest)
    elif job["kind"] == "profile":
        _profile_check(payload, job)
    elif job["kind"] == "prepare" and manifest is not None:
        _family_check(payload, job, manifest)
    elif job["kind"] == "compare":
        if (payload.get("split") != job["split"] or payload.get("manifest_sha256") != job["manifest_sha256"]
                or payload.get("model_id") != job["model_id"] or payload.get("paired") is not True):
            raise StageBError("Accuracy comparison model/manifest/split mismatch")
        ref = read_json(job["reference_eval_path"])["payload"]
        cand = read_json(job["candidate_eval_path"])["payload"]
        if (payload.get("reference_artifact_sha256") != ref.get("artifact_sha256")
                or payload.get("candidate_artifact_sha256") != cand.get("artifact_sha256")
                or payload.get("image_count") != ref.get("image_count")):
            raise StageBError("Accuracy comparison is not tied to its eval artifacts")
    return result


class Dispatcher:
    """Sequential finite-timeout child dispatcher with immutable jobs/results."""

    def __init__(self, directory, run, manifest):
        self.directory = Path(directory).resolve()
        self.run = run
        self.manifest = manifest
        self.results = {}
        self.input_hashes = {}

    def _remember(self, result, job, receipt_path):
        self.results[job["job_name"]] = result
        for path in (Path(job["output_path"]), self.directory / "jobs" / f"{job['job_name']}.json", receipt_path):
            self.run["artifacts"][str(path)] = sha256(path)
        for reference in result.get("artifact_refs", []):
            self.run["artifacts"][reference["path"]] = reference["sha256"]
        if job["kind"] == "reference" and result["status"] == "passed":
            self.input_hashes[job["model_id"]] = {row["id"]: row["input_sha256"] for row in result["payload"]["per_image"]}
        if result["status"] != "passed" and not any(row["job_name"] == job["job_name"] for row in self.run["failures"]):
            self.run["failures"].append({"job_name": job["job_name"], "status": result["status"],
                                         "error": result.get("error"), "worker_pid": result["worker_pid"],
                                         "result_sha256": sha256(job["output_path"]), "no_retry": True})
        self.save()
        return result

    def save(self):
        atomic_json(self.directory / "run.json", self.run, overwrite=True)

    def execute(self, job, timeout):
        _int(timeout, "finite worker timeout", 1, 86400)
        job_path = self.directory / "jobs" / f"{job['job_name']}.json"
        output = Path(job["output_path"])
        receipt_path = self.directory / "receipts" / f"{job['job_name']}.json"
        if output.exists():
            if read_json(job_path) != job:
                raise StageBError("Resume job fingerprint changed")
            receipt = read_json(receipt_path)
            if (receipt.get("job_sha256") != canonical_hash(job)
                    or receipt.get("result_sha256") != sha256(output) or receipt.get("status") != "finished"):
                raise StageBError("Resume child receipt/result SHA binding mismatch")
            result = read_json(output)
            validate_result(job, result, expected_pid=receipt["worker_pid"], expected_identity=receipt, manifest=self.manifest,
                            input_hashes=self.input_hashes.get(job["model_id"]))
            if receipt.get("returncode") != 0 and result["status"] == "passed":
                raise StageBError("Resume cannot accept a failed process's passed result")
            return self._remember(result, job, receipt_path)
        if job_path.exists() or receipt_path.exists():
            raise StageBError("Incomplete dispatched job requires manual diagnosis; no hidden rerun or mixed round")
        atomic_json(job_path, job)
        # Holdout guard is checked in both the parent and child before execution.
        if ((job["kind"] in ("evaluate", "compare") and job.get("split") == "holdout")
                or job["kind"] in ("pipeline", "memory")
                or (job["kind"] in ("measure", "control") and job.get("phase") == "confirmation")):
            validate_job(job_path, job)
        from tools.run_classifier_benchmark import child_python
        executable, environment = child_python(job["threads"])
        log_path = self.directory / "logs" / f"{job['job_name']}.log"
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen([executable, str(ROOT / "tools/run_classifier_stage_b.py"),
                                        "--worker", "--job", str(job_path)], cwd=ROOT,
                                       env=environment, stdout=log, stderr=subprocess.STDOUT)
            from tools.run_classifier_benchmark import worker_process_identity
            try:
                native_identity = _process_lifecycle(worker_process_identity(process.pid), process.pid)
            except BaseException:
                process.kill()
                process.wait(timeout=10)
                raise
            receipt = {"schema_version": 1, "run_id": job["run_id"], "job_sha256": canonical_hash(job),
                       **native_identity, "launched_at": _now(), "timeout_s": timeout,
                       "status": "launched", "returncode": None, "result_sha256": None}
            atomic_json(receipt_path, receipt)
            failure = None
            try:
                returncode = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                process.kill()
                process.wait(timeout=10)
                returncode = process.returncode
                failure = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed_runtime"
                if not output.exists():
                    result = _result_base(job, pid=process.pid, identity=native_identity)
                    result.update(status=failure, evidence_kind="dispatcher_failure", completed_at=_now(),
                                  error="User interrupted the job" if failure == "interrupted" else f"Worker exceeded finite timeout {timeout}s")
                    atomic_json(output, result)
            if not output.exists():
                result = _result_base(job, pid=process.pid, identity=native_identity)
                result.update(status="failed_runtime", evidence_kind="dispatcher_failure", completed_at=_now(),
                              error=f"Worker exited {returncode} without a result")
                atomic_json(output, result)
            result = read_json(output)
            validate_result(job, result, expected_pid=process.pid, expected_identity=native_identity, manifest=self.manifest,
                            input_hashes=self.input_hashes.get(job["model_id"]))
            if returncode != 0 and result["status"] == "passed":
                raise StageBError("Child exited with failure after declaring success")
            receipt.update(status="finished", completed_at=_now(), returncode=returncode,
                           result_sha256=sha256(output), log_sha256=sha256(log_path))
            atomic_json(receipt_path, receipt, overwrite=True)
        self.run["artifacts"][str(log_path)] = sha256(log_path)
        value = self._remember(result, job, receipt_path)
        if failure == "interrupted":
            raise KeyboardInterrupt
        return value


def _config(seed, calibration_timeout_s, worker_timeout_s, holdout_timeout_s, allow_failures):
    _int(seed, "seed", 0, 2**32 - 1)
    for timeout in (calibration_timeout_s, worker_timeout_s, holdout_timeout_s):
        _int(timeout, "timeout", 1, 86400)
    return {"seed": seed, "screening": SCREENING, "confirmation": CONFIRMATION, "observations": OBSERVATIONS,
            "drift_control": {"variant": "QB.t1", "threshold": 1.10, "max_attempts": 2, "sentinel_iterations": 200},
            "threads": [1, 2, 4], "baseline_threads": 1, "inter_op_threads": 1,
            "providers": ["CPUExecutionProvider"], "batch_size": 1, "rtol": 1e-4, "atol": 1e-4,
            "calibration_timeout_s": calibration_timeout_s, "worker_timeout_s": worker_timeout_s,
            "holdout_timeout_s": holdout_timeout_s, "allow_fp32_failures": bool(allow_failures),
            "bootstrap_samples": 2000, "accuracy_budget_pp": 1.0, "min_speedup": 1.05,
            "min_speedup_ci_lower": 1.0, "max_p95_ratio": 1.05, "max_combinations_per_model": 1,
            "automatic_retries": 0, "selection_policy": "development_only; same-model ORT, INT8 versus QB.t1"}


def validate_resume(run, *, config, manifest, audit, fingerprint, environment):
    if (type(run.get("schema_version")) is not int or run.get("schema_version") != 1 or run.get("stage") != "B" or run.get("mode") != "formal"
            or run.get("config") != config or run.get("config_sha256") != canonical_hash(config)
            or run.get("manifest_sha256") != manifest["manifest_sha256"]
            or run.get("source_sha256") != manifest["source_sha256"]
            or run.get("dataset_completion_sha256") != manifest["dataset_completion_sha256"]
            or run.get("fp32_snapshot_sha256") != audit["snapshot_sha256"]
            or run.get("code_fingerprint") != fingerprint or run.get("environment") != environment):
        raise StageBError("Resume config/manifest/source/FP32/code/environment fingerprint mismatch")
    for path, digest in run.get("artifacts", {}).items():
        _inside(path, run["run_dir"])
        _check_artifact(path, digest, "resume artifact", allow_empty=True)
    for path, digest in audit["dependencies"].items():
        _check_artifact(path, digest, "resume FP32 dependency")
    return run


def _raw_artifact(run, model):
    item = run["fp32_prepared"][model]
    return {"path": item["onnx_path"], "onnx_sha256": item["onnx_sha256"],
            "size_bytes": Path(item["onnx_path"]).stat().st_size, "status": "prepared",
            "validation": {"passed": True}, "graph_statistics": item["export_metadata"]["onnx"].get("graph_statistics")}


def _source_fp32_passed(audit, model, threads):
    return any(row["model_id"] == model and row["variant_id"] == "O1" and row["threads"] == threads
               and row["status"] == "passed" for row in audit["summary"]["experiments"])


def _eval(dispatcher, model, variant, threads, artifact, phase, *, split="development", role="candidate", frozen=None):
    run = dispatcher.run
    extra = {"split": split}
    if frozen is not None:
        extra.update(frozen_path=frozen["path"], frozen_sha256=frozen["sha256"])
    job = make_job(run, dispatcher.directory, model, variant, threads, kind="evaluate", phase=phase,
                   role=role, artifact=artifact, **extra)
    timeout = run["config"]["holdout_timeout_s"] if split == "holdout" else run["config"]["worker_timeout_s"]
    result = dispatcher.execute(job, timeout)
    return {"result": result, "path": job["output_path"], "sha256": sha256(job["output_path"]),
            "payload": result["payload"] if result["status"] == "passed" else None}


def _compare(dispatcher, model, variant, threads, reference, candidate, phase, split):
    if reference.get("payload") is None or candidate.get("payload") is None:
        return {"split": split, "model_id": model, "status": "missing_evaluation", "paired": False,
                "budget_passed": False, "error": "Genuine labeled baseline/candidate evaluation failed"}
    guard = {}
    if split == "holdout":
        frozen = dispatcher.run.get("frozen")
        if not isinstance(frozen, dict):
            raise StageBError("holdout guard: comparison requires the unified freeze")
        guard = {"frozen_path": frozen["path"], "frozen_sha256": frozen["sha256"]}
    job = make_job(dispatcher.run, dispatcher.directory, model, variant, threads, kind="compare", phase=phase,
                   split=split, reference_eval_path=reference["path"], reference_eval_sha256=reference["sha256"],
                   candidate_eval_path=candidate["path"], candidate_eval_sha256=candidate["sha256"], **guard)
    result = dispatcher.execute(job, dispatcher.run["config"]["worker_timeout_s"])
    return result["payload"] if result["status"] == "passed" else {
        "split": split, "model_id": model, "status": result["status"], "paired": False,
        "budget_passed": False, "error": result.get("error")}


def _profile(dispatcher, model, variant, threads, artifact, phase="diagnostic"):
    job = make_job(dispatcher.run, dispatcher.directory, model, variant, threads, kind="profile", phase=phase,
                   artifact=artifact, profile_output_dir=str(dispatcher.directory / "profiles" / f"{model}.{variant}.t{threads}"))
    result = dispatcher.execute(job, dispatcher.run["config"]["worker_timeout_s"])
    return result["payload"] if result["status"] == "passed" else {
        "status": result["status"], "int8_operator_events": [], "int8_operator_counts": {},
        "providers": [], "error": result.get("error"), "precision_note": "No verified integer kernel evidence"}


def _run_observations(dispatcher, selections):
    observations = {}
    for selection in selections:
        chosen = selection["candidate"]
        model = chosen["model_id"]
        observations[model] = {}
        for role in ("baseline", "candidate"):
            artifact = {"path": chosen["baseline_artifact_path"] if role == "baseline" else chosen["artifact_path"],
                        "onnx_sha256": chosen["baseline_artifact_sha256"] if role == "baseline" else chosen["artifact_sha256"]}
            variant = chosen["baseline_variant_id"] if role == "baseline" else chosen["variant_id"]
            threads = 1 if role == "baseline" else chosen["threads"]
            observations[model][role] = {}
            for kind in ("pipeline", "memory"):
                job = make_job(dispatcher.run, dispatcher.directory, model, variant, threads, kind=kind,
                               phase="observations", role=role, artifact=artifact, budget=OBSERVATIONS)
                result = dispatcher.execute(job, dispatcher.run["config"]["worker_timeout_s"])
                observations[model][role][kind] = {"status": result["status"], "payload": result.get("payload"),
                    "error": result.get("error"), **_process_lifecycle(result), "job_name": job["job_name"],
                    "frozen_sha256": job["frozen_sha256"], "samples_sha256": result.get("samples_sha256"),
                    "samples_path": result.get("samples_path"), "iterations": 200, "warmup": 20,
                    "separate_from_microbenchmark": True}
    dispatcher.run["observations"] = observations
    dispatcher.save()
    return observations


def _rejected_group(dispatcher, jobs, phase, model, block, attempt):
    directory = dispatcher.directory / "rejected-results" / f"{phase}.{model}.b{block}.a{attempt}"
    directory.mkdir(parents=True, exist_ok=True)
    references = []
    for job in jobs:
        for folder, suffix in (("jobs", ".json"), ("results", ".json"), ("results", ".samples.jsonl"),
                               ("receipts", ".json"), ("logs", ".log")):
            source = dispatcher.directory / folder / f"{job['job_name']}{suffix}"
            if not source.exists():
                continue
            target = directory / f"{folder}-{source.name}"
            if target.exists() and sha256(target) != sha256(source):
                raise StageBError("Rejected group history changed on resume")
            if not target.exists():
                shutil.copy2(source, target)
            digest = sha256(target)
            dispatcher.run["artifacts"][str(target)] = digest
            references.append({"path": str(target), "sha256": digest})
    return references


def _measure_round(dispatcher, configurations, phase, budget):
    """QB sentinels are untallied independent jobs; at most one whole-group drift retry."""
    from src.vision.classification_controls import drift_factor
    measurements = {row["id"]: [] for row in configurations}
    controls_dir = dispatcher.directory / "controls"
    controls_dir.mkdir(exist_ok=True)
    for block in range(5):
        models = sorted({row["model_id"] for row in configurations})
        random.Random(dispatcher.run["config"]["seed"] + block).shuffle(models)
        for model in models:
            rows = [row for row in configurations if row["model_id"] == model]
            random.Random(dispatcher.run["config"]["seed"] + block).shuffle(rows)
            group_key = f"{phase}.{model}.b{block}"
            ledger_path = controls_dir / f"{group_key}.json"
            identity = {key: dispatcher.run[key] for key in ("run_id", "config_sha256", "manifest_sha256", "code_fingerprint")}
            ledger = read_json(ledger_path) if ledger_path.exists() else {
                "schema_version": 1, **identity, "phase": phase, "group": group_key,
                "threshold": 1.10, "max_attempts": 2, "attempts": [], "status": "pending"}
            if any(ledger.get(key) != value for key, value in identity.items()):
                raise StageBError("Stage B drift ledger identity changed")
            qb = dispatcher.run.get("families", {}).get(model, {}).get("variants", {}).get("QB", {})
            controlled = qb.get("status") == "prepared"
            for attempt in range(2 if controlled else 1):
                jobs, results = [], []
                def sentinel(role):
                    job = make_job(dispatcher.run, dispatcher.directory, model, "QB", 1, kind="control", phase=phase,
                                   block=block, role="baseline", artifact=qb, budget=OBSERVATIONS,
                                   attempt=attempt, control_role=role)
                    jobs.append(job)
                    return dispatcher.execute(job, dispatcher.run["config"]["worker_timeout_s"])
                before = sentinel("before") if controlled else None
                for row in rows:
                    job = make_job(dispatcher.run, dispatcher.directory, model, row["variant_id"], row["threads"],
                                   kind="measure", phase=phase, block=block, role=row.get("role", "candidate"),
                                   artifact=row["artifact"], budget=budget, attempt=attempt)
                    jobs.append(job)
                    results.append((row["id"], dispatcher.execute(job, dispatcher.run["config"]["worker_timeout_s"])))
                after = sentinel("after") if controlled else None
                factor = drift_factor(before, after) if controlled else None
                accepted = controlled and factor is not None and factor <= 1.10
                status = "accepted" if accepted else "unstable" if controlled else "uncontrolled"
                decision = {"attempt": attempt, "status": status, "drift_factor": factor,
                            "before_job": before.get("job_name") if before else None,
                            "after_job": after.get("job_name") if after else None,
                            "before_pid": before.get("worker_pid") if before else None,
                            "after_pid": after.get("worker_pid") if after else None,
                            "nominal_jobs": [result["job_name"] for _, result in results],
                            "source_result_sha256": {job["job_name"]: sha256(job["output_path"]) for job in jobs},
                            "missing_control_reason": None if controlled else "No legal prepared family QB; no Stage B drift adoption claim"}
                if not accepted and controlled:
                    decision["rejected_artifacts"] = _rejected_group(dispatcher, jobs, phase, model, block, attempt)
                if len(ledger["attempts"]) > attempt:
                    if ledger["attempts"][attempt] != decision:
                        raise StageBError("Stage B drift decision changed on replay")
                else:
                    ledger["attempts"].append(decision)
                    ledger["status"] = status
                    atomic_json(ledger_path, ledger, overwrite=ledger_path.exists())
                dispatcher.run["artifacts"][str(ledger_path)] = sha256(ledger_path)
                dispatcher.run.setdefault("drift_controls", {})[group_key] = ledger
                dispatcher.save()
                if accepted or not controlled or attempt == 1:
                    for key, original in results:
                        record = copy.deepcopy(original)
                        record.update(environment_accepted=True if accepted else False if controlled else None,
                                      drift_factor=factor, drift_control_group=group_key, attempt=attempt,
                                      control_acceptance_path=str(ledger_path), control_acceptance_sha256=sha256(ledger_path))
                        if controlled and not accepted:
                            record.update(status="failed_runtime", samples=[], payload=None,
                                          error="QB drift gate failed after at most one whole-group retry; rejected raw records preserved")
                        measurements[key].append(record)
                    break
    return measurements


def _row(model, variant, threads, artifact, baseline, measurements, baseline_measurements,
         development, comparison, profile, *, phase, source_passed=True):
    records = measurements or []
    passed = len(records) == 5 and all(row.get("status") == "passed" for row in records)
    values = [sample["inference_ms"] for record in records if record["status"] == "passed" for sample in record["samples"]]
    # Self-baselines are retained but never claimed as their own speed improvement.
    identical = artifact.get("onnx_sha256") == baseline.get("onnx_sha256") and threads == 1
    performance = ({"status": "baseline_identity", "speedup": 1.0, "ci95": None, "p95_ratio": 1.0,
                    "gate_passed": False, "candidate_stats": describe(values) if values else None,
                    "failure_reasons": ["Identity baseline; no claimed improvement"]} if identical else
                   paired_performance(baseline_measurements, records))
    validation = passed and all(row.get("validation", {}).get("passed") is True for row in records)
    error_rows = [{"status": record["status"], "error": record.get("error"), "job_name": record["job_name"]}
                  for record in records if record["status"] != "passed"]
    status = "passed" if passed and validation and source_passed and development.get("payload") is not None else "failed_runtime"
    if artifact.get("status") != "prepared":
        status = artifact.get("status", "unsupported")
        error_rows.append({"status": status, "error": artifact.get("reason", artifact.get("error"))})
    if not source_passed:
        status = "failed_correctness"
        error_rows.append({"status": status, "error": "Original M4 O1 configuration did not pass"})
    return {"id": f"{model}.{variant}.t{threads}", "model_id": model, "variant_id": variant,
            "threads": threads, "phase": phase, "status": status, "validation_passed": validation and source_passed,
            "artifact_path": artifact.get("path"), "artifact_sha256": artifact.get("onnx_sha256"),
            "baseline_artifact_path": baseline.get("path"), "baseline_artifact_sha256": baseline.get("onnx_sha256"),
            "baseline_threads": 1, "baseline_variant_id": "QB" if variant in ("Q1", "Q2", "Q3") else "O1",
            "size_bytes": artifact.get("size_bytes"), "graph_statistics": artifact.get("graph_statistics"),
            "performance": performance, "timing_stats": describe(values) if values else None,
            "development": development.get("payload"), "development_comparison": comparison,
            "runtime_profile": profile, "errors": error_rows,
            "attribution": ("INT8 plus selected threads jointly versus QB.t1; not a pure INT8 increment" if phase == "combination" else
                            "INT8 versus contemporaneous QB.t1 at matching t1") if variant in ("Q1", "Q2", "Q3") else
                           "FP32 graph/thread stack versus contemporaneous raw O1.t1",
            "measurement_jobs": [record["job_name"] for record in records]}


def _summary(dispatcher, audit):
    run = dispatcher.run
    receipts = []
    for path in sorted((dispatcher.directory / "receipts").glob("*.json")):
        receipts.append({"path": str(path), "sha256": sha256(path), **read_json(path)})
    return {"schema_version": 1, "stage": "B", "mode": "formal", "run_id": run["run_id"],
            "status": run["status"], "phase": run["phase"], "config": run["config"],
            "config_sha256": run["config_sha256"], "code_fingerprint": run["code_fingerprint"],
            "environment": run["environment"], "fp32_stage_a": audit["summary"],
            "fp32_source_sha256": audit["run_sha256"], "fp32_visible_failures": audit["visible_failures"],
            "dataset": {"manifest_sha256": run["manifest_sha256"], "source_sha256": run["source_sha256"],
                        "dataset_completion_sha256": run["dataset_completion_sha256"],
                        "scope": dispatcher.manifest["dataset_scope"], "source": dispatcher.manifest["source"],
                        "holdout_image_bytes_read_by_parent": False, "original_images_published": False},
            "rows": run.get("rows", []), "selections": run.get("selections", []),
            "frozen": run.get("frozen"), "final": run.get("final", {}),
            "observations": run.get("observations", {}), "drift_controls": run.get("drift_controls", {}),
            "evidence": {"artifact_sha256": run["artifacts"], "job_receipts": receipts,
                         "families": run.get("families", {}), "fp32_dependencies": audit["dependencies"]},
            "failures": run["failures"], "limitations": LIMITATIONS,
            "methodology": {"screening": SCREENING, "confirmation": CONFIRMATION, "observations": OBSERVATIONS,
                            "serial_workers": True, "parent_model_sessions": 0, "max_retries": 0,
                            "drift_group_max_attempts": 2, "drift_control": "QB.t1 before/after, separate 200 samples; max median drift factor 1.10",
                            "memory": "separate process; 20ms native RSS sampler, dependency/process-wide footprint; brief peaks can be missed",
                            "pipeline": "decoded RGB; direct pipeline_wall_ms, never sum stage P50; separate from microbenchmark",
                            "selection": "development labels + screening only; no holdout results are read before freezing",
                            "freeze": "all three models in one SHA-bound frozen.json before confirmation and holdout",
                            "holdout": "one baseline/candidate evaluation per model, no reselection on final failure",
                            "ratio": "geometric mean of paired block P50 baseline/candidate ratios",
                            "ci": "paired block (performance), paired image (accuracy), 2000 bootstrap resamples",
                            "prediction_space": "full1000-way without 20-class masking",
                            "report": "static/escaped; no remote assets, no original image copies, no aggregate score"}}


def run_stage_b(manifest_path, fp32_run, output_dir, *, resume=False, allow_fp32_failures=False,
                seed=20261001, calibration_timeout_s=900, worker_timeout_s=900, holdout_timeout_s=900):
    """Execute the approved workflow. Calling this API opts into real CPU jobs."""
    if type(resume) is not bool or type(allow_fp32_failures) is not bool:
        raise StageBError("resume/allow_fp32_failures must be bool")
    config = _config(seed, calibration_timeout_s, worker_timeout_s, holdout_timeout_s, allow_fp32_failures)
    manifest = load_stage_b_manifest(manifest_path)
    audit = audit_fp32_run(fp32_run, manifest, allow_failures=allow_fp32_failures)
    fingerprint, sources = code_fingerprint()
    environment = environment_info()
    directory = Path(output_dir).resolve()
    if directory == Path(audit["run_dir"]) or directory == Path(manifest["resolved_root"]):
        raise StageBError("Stage B output cannot overwrite FP32 inputs or the dataset")
    if not directory.parent.is_dir():
        raise StageBError("Output parent must already exist")
    if resume:
        run = validate_resume(read_json(directory / "run.json"), config=config, manifest=manifest,
                              audit=audit, fingerprint=fingerprint, environment=environment)
    else:
        if directory.exists():
            raise StageBError("Output directory exists; choose a new one or explicit --resume")
        directory.mkdir()
        for folder in ("jobs", "results", "receipts", "logs", "families", "profiles"):
            (directory / folder).mkdir()
        run = {"schema_version": 1, "stage": "B", "mode": "formal", "run_id": uuid.uuid4().hex,
               "run_dir": str(directory), "created_at": _now(), "phase": "auditing_reference", "status": "running",
               "config": config, "config_sha256": canonical_hash(config), "code_fingerprint": fingerprint,
               "source_hashes": sources, "environment": environment, "manifest_path": manifest["manifest_path"],
               "manifest_sha256": manifest["manifest_sha256"], "source_path": manifest["source_path"],
               "source_sha256": manifest["source_sha256"], "dataset_completion_sha256": manifest["dataset_completion_sha256"],
               "fp32_snapshot_sha256": audit["snapshot_sha256"], "fp32_prepared": audit["run"]["prepared_models"],
               "images": audit["run"]["images"], "artifacts": {}, "failures": [], "rows": [],
               "families": {}, "family_files": {}, "selections": [], "final": {}}
        atomic_json(directory / "run.json", run)
    dispatcher = Dispatcher(directory, run, manifest)
    try:
        raw = {model: _raw_artifact(run, model) for model in MODELS}
        reference_ready = {}
        for model in MODELS:
            job = make_job(run, directory, model, "O1", 1, kind="reference", phase="reference")
            result = dispatcher.execute(job, worker_timeout_s)
            reference_ready[model] = result["status"] == "passed"
        run["phase"] = "preparing_families"
        dispatcher.save()
        for model in MODELS:
            family_dir = directory / "families" / model
            family_path = family_dir / "family.json"
            if reference_ready[model]:
                job = make_job(run, directory, model, "QB", 1, kind="prepare", phase="prepare",
                               family_output_dir=str(family_dir))
                result = dispatcher.execute(job, calibration_timeout_s)
            else:
                result = {"status": "failed_correctness", "error": "Canonical source/reference audit failed", "payload": None}
            if model in run["family_files"]:
                # Reuse the exact SHA-bound family/failure marker; never rewrite history on resume.
                family_path = Path(run["family_files"][model])
                family = read_json(family_path)
                if family != run["families"][model] or (result["status"] == "passed" and family != result["payload"]):
                    raise StageBError("Resume family/status record changed")
            elif result["status"] == "passed":
                family = result["payload"]
            elif family_path.exists():
                # A timed-out/failed preparation may leave partial bytes, but they are not executable evidence.
                family = read_json(family_path)
                family = {"schema_version": 1, "model_id": model, "manifest_sha256": run["manifest_sha256"],
                          "status": result["status"], "input_model": {"path": raw[model]["path"], "sha256": raw[model]["onnx_sha256"]},
                          "variants": {variant: {"status": "unsupported", "reason": result.get("error", "Preparation failed")}
                                       for variant in FAMILY_VARIANTS}, "partial_family_sha256": sha256(family_path)}
                family_path = family_dir / "failed-family.json"
                if not family_path.exists():
                    atomic_json(family_path, family)
                elif read_json(family_path) != family:
                    raise StageBError("Failed family history changed on resume")
            else:
                family = {"schema_version": 1, "model_id": model, "manifest_sha256": run["manifest_sha256"],
                          "status": result["status"], "input_model": {"path": raw[model]["path"], "sha256": raw[model]["onnx_sha256"]},
                          "calibration": {"executed": False},
                          "variants": {variant: {"status": "unsupported", "reason": result.get("error", "Preparation failed")}
                                       for variant in FAMILY_VARIANTS}}
                family_dir.mkdir(exist_ok=True)
                atomic_json(family_path, family)
            run["families"][model] = family
            run["family_files"][model] = str(family_path)
            run["artifacts"][str(family_path)] = sha256(family_path)
            if family.get("status") not in ("passed", "prepared"):
                for variant_name, var_data in family.get("variants", {}).items():
                    if var_data.get("status") != "prepared":
                        failure_record = {
                            "job_name": f"prepare.{model}.{variant_name}",
                            "status": var_data.get("status", "failed_runtime"),
                            "error": var_data.get("reason", var_data.get("error", "Preparation failed")),
                            "no_retry": True
                        }
                        if not any(f.get("job_name") == failure_record["job_name"] for f in run["failures"]):
                            run["failures"].append(failure_record)
            dispatcher.save()
        configurations = []
        for model in MODELS:
            if reference_ready[model]:
                for threads in THREADS:
                    if _source_fp32_passed(audit, model, threads):
                        configurations.append({"id": f"{model}.O1.t{threads}", "model_id": model,
                                               "variant_id": "O1", "threads": threads, "artifact": raw[model]})
                for variant, artifact in run["families"][model]["variants"].items():
                    if artifact.get("status") == "prepared":
                        configurations.append({"id": f"{model}.{variant}.t1", "model_id": model,
                                               "variant_id": variant, "threads": 1, "artifact": artifact})
        run["phase"] = "screening"
        dispatcher.save()
        measurements = _measure_round(dispatcher, configurations, "screening", SCREENING)
        development, profiles = {}, {}
        run["phase"] = "development_only"
        dispatcher.save()
        for row in configurations:
            key = row["id"]
            development[key] = _eval(dispatcher, row["model_id"], row["variant_id"], row["threads"], row["artifact"], "development")
        run["phase"] = "runtime_diagnostic"
        dispatcher.save()
        for row in configurations:
            profiles[row["id"]] = _profile(dispatcher, row["model_id"], row["variant_id"], row["threads"], row["artifact"])
        rows = []
        missing_dev = {"payload": None}
        for model in MODELS:
            for variant, threads in [("O1", thread) for thread in THREADS] + [(name, 1) for name in FAMILY_VARIANTS]:
                key = f"{model}.{variant}.t{threads}"
                artifact = raw[model] if variant == "O1" else run["families"][model]["variants"][variant]
                baseline_variant = "QB" if variant in ("Q1", "Q2", "Q3") else "O1"
                baseline = raw[model] if baseline_variant == "O1" else run["families"][model]["variants"]["QB"]
                baseline_key = f"{model}.{baseline_variant}.t1"
                comparison = (_compare(dispatcher, model, variant, threads, development[baseline_key], development[key],
                                       "development_compare", "development") if key in development and baseline_key in development else None)
                row = _row(model, variant, threads, artifact, baseline, measurements.get(key, []),
                           measurements.get(baseline_key, []), development.get(key, missing_dev), comparison,
                           profiles.get(key, {}), phase="screening", source_passed=reference_ready[model] and (
                               _source_fp32_passed(audit, model, threads) if variant == "O1" else True))
                rows.append(row)
        run["rows"] = rows
        run["phase"] = "development_selection"
        selections = [select_candidate(model, [row for row in rows if row["model_id"] == model]) for model in MODELS]
        # At most one new combination: the best accepted Q plus the originally screened M4 O1 thread.
        for selection in selections:
            chosen = selection.get("candidate")
            if chosen is None or chosen["variant_id"] not in ("Q1", "Q2", "Q3"):
                continue
            model = selection["model_id"]
            m4_threads = [row for row in audit["summary"]["experiments"] if row["model_id"] == model
                          and row["variant_id"] == "O1" and row["status"] == "passed"]
            if not m4_threads:
                continue
            best = min(m4_threads, key=lambda row: (row["inference"]["stats"]["inference_ms"]["p50"], row["threads"]))
            threads = best["threads"]
            if threads == 1:
                continue
            variant = chosen["variant_id"]
            artifact = run["families"][model]["variants"][variant]
            combo_dev = _eval(dispatcher, model, variant, threads, artifact, "combination_development")
            baseline_key = f"{model}.QB.t1"
            comparison = _compare(dispatcher, model, variant, threads, development[baseline_key], combo_dev,
                                  "combination_compare", "development")
            if not accuracy_gate(comparison)["passed"]:
                # Rejected combinations remain visible, without paying a full measurement budget.
                combo_row = _row(model, variant, threads, artifact, run["families"][model]["variants"]["QB"], [], [],
                                 combo_dev, comparison, {}, phase="combination")
                rows.append(combo_row)
                continue
            profile = _profile(dispatcher, model, variant, threads, artifact, "combination_profile")
            pair = [{"id": baseline_key, "model_id": model, "variant_id": "QB", "threads": 1,
                     "artifact": run["families"][model]["variants"]["QB"], "role": "baseline"},
                    {"id": f"{model}.{variant}.t{threads}", "model_id": model, "variant_id": variant,
                     "threads": threads, "artifact": artifact}]
            combo_measurements = _measure_round(dispatcher, pair, "combination", SCREENING)
            combo_row = _row(model, variant, threads, artifact, pair[0]["artifact"], combo_measurements[pair[1]["id"]],
                             combo_measurements[baseline_key], combo_dev, comparison, profile, phase="combination")
            rows.append(combo_row)
            updated = select_candidate(model, [row for row in rows if row["model_id"] == model])
            selection.clear()
            selection.update(updated)
        run.update(rows=rows, selections=selections, phase="freezing")
        dispatcher.save()
        if any(row.get("candidate") is None for row in selections):
            run.update(status="blocked_correctness", phase="holdout_not_executed", completed_at=_now())
            dispatcher.save()
            summary = _summary(dispatcher, audit)
            write_report(directory, summary, overwrite=resume)
            return summary
        frozen_path = directory / "frozen.json"
        if frozen_path.exists():
            frozen_value = validate_frozen(read_json(frozen_path), manifest_sha256=run["manifest_sha256"],
                                           code=fingerprint, family_files=run["family_files"])
            expected = [{"model_id": row["model_id"], "variant_id": row["variant_id"],
                         "artifact_sha256": row["artifact_sha256"], "baseline_artifact_sha256": row["baseline_artifact_sha256"],
                         "threads": row["threads"], "baseline_threads": 1, "selection_basis": "development_only",
                         "family_sha256": sha256(run["family_files"][row["model_id"]]),
                         "artifact_path": row["artifact_path"], "baseline_artifact_path": row["baseline_artifact_path"]}
                        for row in (selection["candidate"] for selection in selections)]
            if frozen_value["candidates"] != expected:
                raise StageBError("Resume selection changed after freeze; holdout cannot be used for reselection")
            frozen_sha = sha256(frozen_path)
        else:
            frozen_value, frozen_sha = freeze_candidates(selections, run["manifest_sha256"], fingerprint,
                                                        run["family_files"], frozen_path)
        frozen = {"path": str(frozen_path), "sha256": frozen_sha, "value": frozen_value}
        run.update(frozen=frozen, phase="observations")
        run["artifacts"][str(frozen_path)] = frozen_sha
        dispatcher.save()
        _run_observations(dispatcher, selections)
        run["phase"] = "confirmation"
        dispatcher.save()
        confirm_configurations = []
        for selection in selections:
            chosen = selection["candidate"]
            model = selection["model_id"]
            confirm_configurations.extend([
                {"id": f"{model}.baseline", "model_id": model, "variant_id": chosen["baseline_variant_id"],
                 "threads": 1, "artifact": {"path": chosen["baseline_artifact_path"],
                                             "onnx_sha256": chosen["baseline_artifact_sha256"]}, "role": "baseline"},
                {"id": chosen["id"], "model_id": model, "variant_id": chosen["variant_id"], "threads": chosen["threads"],
                 "artifact": {"path": chosen["artifact_path"], "onnx_sha256": chosen["artifact_sha256"]}, "role": "candidate"}])
        confirmation = _measure_round(dispatcher, confirm_configurations, "confirmation", CONFIRMATION)
        run["phase"] = "frozen_holdout"
        dispatcher.save()
        for selection in selections:
            chosen = selection["candidate"]
            model = selection["model_id"]
            baseline_artifact = {"path": chosen["baseline_artifact_path"], "onnx_sha256": chosen["baseline_artifact_sha256"]}
            candidate_artifact = {"path": chosen["artifact_path"], "onnx_sha256": chosen["artifact_sha256"]}
            baseline_eval = _eval(dispatcher, model, chosen["baseline_variant_id"], 1, baseline_artifact, "holdout",
                                  split="holdout", role="baseline", frozen=frozen)
            candidate_eval = _eval(dispatcher, model, chosen["variant_id"], chosen["threads"], candidate_artifact, "holdout",
                                   split="holdout", frozen=frozen)
            comparison = _compare(dispatcher, model, chosen["variant_id"], chosen["threads"], baseline_eval, candidate_eval,
                                  "holdout_compare", "holdout")
            performance = paired_performance(confirmation[f"{model}.baseline"], confirmation[chosen["id"]])
            identical = chosen["artifact_sha256"] == chosen["baseline_artifact_sha256"] and chosen["threads"] == 1
            accuracy = accuracy_gate(comparison, split="holdout")
            adopted = not identical and accuracy["passed"] and performance["gate_passed"]
            quantized = chosen["variant_id"] in ("Q1", "Q2", "Q3")
            int8_status = selection.get("int8_evidence_status", "evaluated_no_benefit")
            int8_evaluated = int8_status == "evaluated_no_benefit"
            if identical and accuracy["passed"]:
                status = "retained_fp32_no_int8_benefit" if int8_evaluated else "retained_fp32_int8_not_evaluated"
            elif adopted:
                if quantized:
                    status = "adopt_int8"
                elif int8_evaluated or int8_status == "qualified_available":
                    status = "adopt_fp32_only_no_int8_benefit"
                else:
                    status = "fp32_confirmed_int8_not_evaluated"
            else:
                status = "not_adopted_accuracy_budget" if accuracy["status"] == "failed" else "not_adopted_insufficient_evidence"

            if not adopted:
                if int8_evaluated:
                    final_reason = "No retuning/reselection after holdout; no INT8 adoption benefit"
                elif quantized:
                    final_reason = "No retuning/reselection after holdout; candidate failed holdout gate"
                else:
                    final_reason = f"No retuning/reselection after holdout; INT8 not evaluated ({int8_status})"
            else:
                if quantized:
                    final_reason = "Independent confirmation and frozen holdout passed the disclosed budgets"
                elif int8_evaluated:
                    final_reason = "Independent confirmation and frozen holdout passed the disclosed budgets; no INT8 adoption benefit"
                elif int8_status == "qualified_available":
                    final_reason = "Independent confirmation and frozen holdout passed the disclosed budgets; FP32 candidate adopted over qualified INT8"
                else:
                    final_reason = f"Independent confirmation and frozen holdout passed the disclosed budgets; INT8 candidates not evaluated ({int8_status})"

            run["final"][model] = {"candidate_id": chosen["id"], "status": status, "adopted": adopted,
                                    "int8_adopted": adopted and quantized,
                                    "int8_evidence_status": int8_status,
                                    "frozen_sha256": frozen_sha,
                                    "confirmation": performance, "holdout_comparison": comparison,
                                    "holdout_gate": accuracy, "baseline_holdout": baseline_eval["payload"],
                                    "candidate_holdout": candidate_eval["payload"],
                                    "selection_changed_after_holdout": False,
                                    "reason": final_reason}
            dispatcher.save()
        # Lack of adoption is a valid completed negative result, not a fake success recommendation.
        has_failures = (
            bool(run["failures"])
            or any(row.get("status") not in ("passed", "baseline_identity") for row in rows)
            or any(family.get("status") not in ("passed", "prepared") for family in run.get("families", {}).values())
            or any(sel.get("int8_evidence_status") in ("failed_runtime", "unsupported", "partial_failure", "not_evaluated")
                   for sel in selections)
        )
        run.update(status="completed_with_failures" if has_failures else "completed",
                   phase="finished", completed_at=_now())
        dispatcher.save()
        summary = _summary(dispatcher, audit)
        write_report(directory, summary, overwrite=resume)
        return summary
    except BaseException as exc:
        run.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed_runtime",
                   error=f"{type(exc).__name__}: {exc}", phase="stopped", completed_at=_now())
        dispatcher.save()
        # Do not mask a provenance failure with a success report or change frozen selection.
        summary = _summary(dispatcher, audit)
        summary["failures"] = summary["failures"] + [{"status": run["status"], "error": run["error"]}]
        write_report(directory, summary, overwrite=resume)
        raise
