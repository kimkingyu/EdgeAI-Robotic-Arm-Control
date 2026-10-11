"""阶段 B 的离线 QDQ 量化、带标签精度与 CPU 执行证据。

导入期只用标准库。校准只读取 calibration；holdout 图片在冻结检查前
绝不打开。存在 QDQ 不代表运行时整数执行，更不代表精度或速度通过。
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import uuid
from typing import Any

__all__ = ["QuantizationError", "QUANT_VARIANTS", "CalibrationDataReader",
           "prepare_quantized_family", "evaluate_classifier", "compare_accuracies",
           "profile_quantized_execution"]

QUANT_VARIANTS = ("QB", "Q1", "Q2", "Q3")
MODEL_IDS = ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0")
_TARGET_OPS = ("Conv", "Gemm", "MatMul")
_MINIMUM_COUNTS = {"calibration": 256, "development": 128, "holdout": 1000}
_INTEGER_OPS = frozenset({"QLinearConv", "QGemm", "QLinearMatMul", "MatMulInteger",
                          "ConvInteger", "MatMulIntegerToFloat"})
_FLOAT_OPS = frozenset({"Conv", "Gemm", "MatMul", "FusedConv", "FusedGemm",
                        "FusedMatMul", "BatchNormalization", "Add", "Mul", "Div",
                        "Relu", "Sigmoid", "HardSigmoid", "HardSwish", "Clip",
                        "AveragePool", "GlobalAveragePool", "MaxPool"})
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_INPUT_SHAPE = (1, 3, 224, 224)
_OUTPUT_SHAPE = (1, 1000)


class QuantizationError(RuntimeError):
    """清单、冻结、模型契约或量化运行失败。"""


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise QuantizationError(f"缺少本地依赖 {name}；不会自动安装或联网: {error}") from error


def _core() -> Any:
    return _module("src.vision.classification")


def _benchmark() -> Any:
    return _module("src.vision.classification_benchmark")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise QuantizationError(f"{name} 必须为 >= {minimum} 的整数（不接受 bool）")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise QuantizationError(f"{name} 必须为 SHA256")
    return value.lower()


def _model_id(model_id: str) -> None:
    if model_id not in MODEL_IDS:
        raise QuantizationError(f"未知 model_id {model_id!r}；仅允许 {MODEL_IDS}")


def _hash_file(path: str | Path) -> str:
    try:
        return _core().compute_sha256(path)
    except (OSError, _core().ClassificationError) as error:
        raise QuantizationError(str(error)) from error


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: str | Path) -> tuple[dict, bytes]:
    try:
        return _benchmark()._read_json(Path(path))
    except _benchmark().BenchmarkError as error:
        raise QuantizationError(str(error)) from error


def _versions() -> dict:
    return _core().environment_metadata()


def _load_dataset(manifest_path: str | Path) -> dict:
    # 复用阶段 A schema/去重/source_id 防泄漏检查；此处不读取 holdout 字节。
    try:
        manifest = _benchmark().load_manifest(manifest_path, check_files=False)
        readiness = _benchmark().dataset_readiness(manifest)
    except _benchmark().BenchmarkError as error:
        raise QuantizationError(str(error)) from error
    root = Path(manifest["resolved_root"])
    if not root.is_dir():
        raise QuantizationError(f"数据根目录不存在: {root}")
    for entry in manifest["images"]:
        if entry["split"] == "smoke":
            continue
        if not any(isinstance(entry.get(key), str) and entry[key].strip()
                   for key in ("source", "source_url")):
            raise QuantizationError(f"缺少来源声明: {entry['id']}")
        if not isinstance(entry.get("license"), str) or not entry["license"].strip():
            raise QuantizationError(f"缺少许可声明: {entry['id']}")
        for key in ("width", "height"):
            _integer(entry.get(key), f"{entry['id']}.{key}", 1)
        try:
            path = Path(entry["resolved_path"])
            if not path.is_file() or path.stat().st_size <= 0:
                raise QuantizationError(f"图片缺失/为空: {entry['id']}")
        except OSError as error:
            raise QuantizationError(f"无法检查图片 {entry['id']}: {error}") from error
    manifest["dataset_readiness"] = readiness
    return manifest


def _require_stage_b_data(manifest: dict) -> None:
    ready = manifest["dataset_readiness"]
    if not ready["formal"]["ready"] or not ready["quantization"]["ready"]:
        raise QuantizationError(f"blocked_data: {ready['formal']['reason']}; "
                                f"{ready['quantization']['reason']}")
    # 不把用户写入的 ready 标记当作计数证据。
    for split, minimum in _MINIMUM_COUNTS.items():
        images = [entry for entry in manifest["images"] if entry["split"] == split]
        count = len(images) if split == "calibration" else sum(
            entry["label"] is not None for entry in images)
        if count < minimum:
            raise QuantizationError(f"blocked_data: {split} {count}/{minimum}")


def _check_dataset_bytes(manifest: dict, splits: tuple[str, ...]) -> None:
    for entry in manifest["images"]:
        if entry["split"] in splits and _hash_file(entry["resolved_path"]) != entry["sha256"].lower():
            raise QuantizationError(f"图片 SHA256 不匹配: {entry['id']}")


def _ordered_images(manifest: dict, split: str, seed: int | None = None) -> list[dict]:
    images = sorted((entry for entry in manifest["images"] if entry["split"] == split),
                    key=lambda entry: entry["id"])
    if seed is not None:
        random.Random(seed).shuffle(images)
    return images


def _prepare_array(entry: dict, model_id: str) -> Any:
    np, core = _module("numpy"), _core()
    if _hash_file(entry["resolved_path"]) != entry["sha256"].lower():
        raise QuantizationError(f"图片 SHA256 不匹配: {entry['id']}")
    try:
        image = core.open_rgb_image(entry["resolved_path"])
        try:
            if tuple(image.size) != (entry["width"], entry["height"]):
                raise QuantizationError(f"图片尺寸与清单不符: {entry['id']}")
            tensor = core.prepare_image(image, model_id=model_id)
            value = tensor.detach().cpu().numpy()
        finally:
            image.close()
    except core.ClassificationError as error:
        raise QuantizationError(f"{entry['id']}: {error}") from error
    return _validate_array(value, "预处理输入", np)


def _validate_array(value: Any, name: str, np: Any = None) -> Any:
    np = np or _module("numpy")
    if not isinstance(value, np.ndarray) or value.shape != _INPUT_SHAPE:
        raise QuantizationError(f"{name} 必须是 float32 ndarray [1,3,224,224]")
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise QuantizationError(f"{name} 必须是有限 FP32，禁止隐式转换/降级")
    return np.ascontiguousarray(value)


class CalibrationDataReader:
    """ORT CalibrationDataReader 协议：只保存轻量清单，一次解码一张。

    ORT 的 ABC 使用 get_next 的结构化 subclasshook，不需要在导入期
    导入 ORT。reset/rewind 保持同一图片顺序和区间，可反复遍历。
    """
    def __init__(self, model_id: str, images: list[dict], input_name: str):
        _model_id(model_id)
        if not images or any(entry.get("split") != "calibration" for entry in images):
            raise QuantizationError("校准 Reader 只接受非空 calibration 图片列表")
        if len({entry["id"] for entry in images}) != len(images):
            raise QuantizationError("校准 Reader 含重复图片 ID")
        self.model_id = model_id
        self.images = copy.deepcopy(images)
        self.input_name = input_name
        self.start, self.end, self.index = 0, len(images), 0

    def get_next(self) -> dict | None:
        if self.index >= self.end:
            return None
        entry = self.images[self.index]
        value = _prepare_array(entry, self.model_id)
        self.index += 1
        return {self.input_name: value}

    def rewind(self) -> None:
        self.index = self.start

    def reset(self) -> None:
        self.rewind()

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        value = self.get_next()
        if value is None:
            raise StopIteration
        return value

    def set_range(self, start_index: int, end_index: int) -> None:
        _integer(start_index, "start_index")
        _integer(end_index, "end_index")
        if not 0 <= start_index < end_index <= len(self.images):
            raise QuantizationError("校准 Reader 区间无效")
        self.start, self.end = start_index, end_index
        self.rewind()


class _OneImageReader:
    def __init__(self, value: dict):
        self.value = value

    def get_next(self) -> dict | None:
        value, self.value = self.value, None
        return value


def _session_config(threads: int) -> dict:
    return {"threads": threads, "inter_op_threads": 1,
            "providers": ["CPUExecutionProvider"], "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_ENABLE_ALL", "allow_spinning": "0",
            "session_config_entries": {"session.intra_op.allow_spinning": "0",
                                       "session.inter_op.allow_spinning": "0"}}


def _create_session(path: str | Path, *, threads: int = 1,
                    profile_prefix: str | None = None) -> Any:
    _integer(threads, "threads", 1)
    ort = _module("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in _session_config(threads)["session_config_entries"].items():
        options.add_session_config_entry(key, value)
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    try:
        session = ort.InferenceSession(str(path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise QuantizationError(f"禁止非 CPU provider: {session.get_providers()}")
        return session
    except QuantizationError:
        raise
    except Exception as error:
        raise QuantizationError(f"ORT CPU 会话失败 ({path}): {type(error).__name__}: {error}") from error


def _check_session_contract(session: Any) -> None:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1
            or tuple(inputs[0].shape) != _INPUT_SHAPE
            or tuple(outputs[0].shape) != _OUTPUT_SHAPE
            or inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)"):
        raise QuantizationError("ORT 输入/输出必须为静态 [1,3,224,224]/[1,1000] FP32")


def _run(session: Any, value: Any) -> Any:
    np = _module("numpy")
    try:
        outputs = session.run(None, {session.get_inputs()[0].name: value})
    except Exception as error:
        raise QuantizationError(f"ORT 推理失败: {type(error).__name__}: {error}") from error
    if (len(outputs) != 1 or not isinstance(outputs[0], np.ndarray)
            or outputs[0].shape != _OUTPUT_SHAPE or outputs[0].dtype != np.float32
            or not np.isfinite(outputs[0]).all()):
        raise QuantizationError("ORT 输出 shape/dtype/有限值契约失败")
    return outputs[0]


def _check_graph(path: str | Path, *, fp32_parent: bool = False) -> tuple[Any, dict]:
    onnx = _module("onnx")
    try:
        model = onnx.load(str(path), load_external_data=False)
        # 既有导出明确禁用外部权重；不遗漏体积/SHA，也不读取任意外部路径。
        for tensor in model.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
                raise QuantizationError("不支持 external_data；不会忽略权重文件或偷偷内嵌")
        onnx.checker.check_model(model, full_check=True)
    except QuantizationError:
        raise
    except Exception as error:
        raise QuantizationError(f"ONNX checker 失败: {type(error).__name__}: {error}") from error
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise QuantizationError("ONNX 必须有唯一输入和输出")
    for info, expected in ((graph.input[0], _INPUT_SHAPE), (graph.output[0], _OUTPUT_SHAPE)):
        tensor = info.type.tensor_type
        shape = tuple(d.dim_value if d.HasField("dim_value") else None for d in tensor.shape.dim)
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != expected:
            raise QuantizationError(f"ONNX {info.name} 必须为静态 FP32 {expected}")
    imports = {item.domain or "ai.onnx": item.version for item in model.opset_import}
    if imports.get("ai.onnx") != 17:
        raise QuantizationError(f"必须保留 opset17，禁止静默改版本: {imports}")
    counts = Counter(node.op_type for node in graph.node)
    if fp32_parent and any(counts[op] for op in _INTEGER_OPS | {"QuantizeLinear", "DequantizeLinear"}):
        raise QuantizationError("raw/QB 必须为未量化的 FP32 父图")
    stats = {"sha256": _hash_file(path), "size_bytes": Path(path).stat().st_size,
             "node_count": len(graph.node), "op_type_counts": dict(sorted(counts.items())),
             "opset_imports": imports,
             "inputs": {graph.input[0].name: {"shape": list(_INPUT_SHAPE), "elem_type": 1}},
             "outputs": {graph.output[0].name: {"shape": list(_OUTPUT_SHAPE), "elem_type": 1}},
             "producer_name": model.producer_name, "producer_version": model.producer_version}
    return model, stats


def _qdq_coverage(parent: Any, model: Any) -> dict:
    onnx = _module("onnx")
    producers = {output: node for node in model.graph.node for output in node.output}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    covered, uncovered = [], []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in _TARGET_OPS:
            continue
        details = []
        for input_index, name in enumerate(node.input[:2]):
            dq = producers.get(name)
            signed = False
            if dq is not None and dq.op_type == "DequantizeLinear" and len(dq.input) >= 3:
                zero = initializers.get(dq.input[2])
                integer_weight = initializers.get(dq.input[0])
                q = producers.get(dq.input[0])
                signed = (zero is not None and zero.data_type == onnx.TensorProto.INT8
                          and ((integer_weight is not None and integer_weight.data_type == onnx.TensorProto.INT8)
                               or (q is not None and q.op_type == "QuantizeLinear")))
            details.append({"input_index": input_index, "s8_dequantized": bool(signed)})
        record = {"name": node.name or f"{node.op_type}@{index}", "op_type": node.op_type,
                  "outputs": list(node.output), "inputs": details}
        (covered if len(details) == 2 and all(item["s8_dequantized"] for item in details)
         else uncovered).append(record)
    parent_targets = [{"name": node.name or f"{node.op_type}@{index}", "op_type": node.op_type,
                       "outputs": list(node.output)}
                      for index, node in enumerate(parent.graph.node) if node.op_type in _TARGET_OPS]
    qdq_counts = Counter(node.op_type for node in model.graph.node
                         if node.op_type in ("QuantizeLinear", "DequantizeLinear"))
    non_targets = Counter(node.op_type for node in model.graph.node
                          if node.op_type not in _TARGET_OPS + ("QuantizeLinear", "DequantizeLinear"))
    return {"target_ops": list(_TARGET_OPS), "parent_target_nodes": parent_targets,
            "parent_target_node_count": len(parent_targets), "qdq_node_counts": dict(qdq_counts),
            "qdq_target_node_count": len(covered), "covered_target_nodes": covered,
            "uncovered_target_nodes": uncovered, "non_target_operator_counts": dict(non_targets),
            "coverage_fraction": len(covered) / len(parent_targets) if parent_targets else 0.0,
            "precision_note": "仅统计 S8 激活/权重 DQ 输入覆盖；未覆盖及非目标算子保留浮点/原精度。"
                              "QDQ 节点不证明运行时整数执行，本产物不是全 INT8。"}


def _quantization_api() -> dict:
    quant = _module("onnxruntime.quantization")
    calibrate = _module("onnxruntime.quantization.calibrate")
    shape = _module("onnxruntime.quantization.shape_inference")
    signature = inspect.signature(quant.quantize_static)
    required = {"calibration_cache_path", "calibration_providers", "extra_options",
                "quant_format", "per_channel", "calibrate_method", "reduce_range",
                "op_types_to_quantize", "activation_type", "weight_type"}
    if not required <= set(signature.parameters):
        raise QuantizationError("本机 ORT 缺少已核实的静态量化/cache API；不降级")
    try:
        source = inspect.getsource(quant.quantize_static)
        factory_source = inspect.getsource(calibrate.create_calibrator)
    except (OSError, TypeError) as error:
        raise QuantizationError(f"无法核实本机 ORT 选项源码: {error}") from error
    for name in ("ActivationSymmetric", "WeightSymmetric", "CalibTensorRangeSymmetric", "CalibPercentile"):
        if name not in source:
            raise QuantizationError(f"本机 ORT 未确认选项 {name}；不猜测/忽略")
    if not re.search(r'\("CalibPercentile",\s*"percentile"\)', source) or 'get("percentile"' not in factory_source:
        raise QuantizationError("无法确认 CalibPercentile -> percentile 的官方映射")
    for name in ("skip_optimization", "skip_symbolic_shape", "skip_onnx_shape"):
        if name not in inspect.signature(shape.quant_pre_process).parameters:
            raise QuantizationError(f"quant_pre_process 缺少 {name}")
    evidence = {"quantize_static_signature": str(signature),
                "quant_pre_process_signature": str(inspect.signature(shape.quant_pre_process)),
                "quantize_source_sha256": _hash_file(inspect.getfile(quant.quantize_static)),
                "calibrate_source_sha256": _hash_file(inspect.getfile(calibrate.create_calibrator)),
                "verified_options": ["ActivationSymmetric", "WeightSymmetric",
                                     "CalibTensorRangeSymmetric", "CalibPercentile"],
                "percentile_option_mapping": "CalibPercentile -> percentile"}
    return {"quant": quant, "calibrate": calibrate, "shape": shape, "evidence": evidence}


def _variant_config(variant: str, parent_sha: str) -> dict:
    common = {"parent_variant": "QB", "parent_onnx_sha256": parent_sha,
              "session": _session_config(1), "quant_format": "QDQ", "activation_type": "QInt8",
              "weight_type": "QInt8", "activation_granularity": "per-tensor",
              "per_channel": variant in ("Q2", "Q3"), "reduce_range": False,
              "op_types_to_quantize": list(_TARGET_OPS), "nodes_to_quantize": None,
              "nodes_to_exclude": None, "use_external_data_format": False,
              "calibration_providers": ["CPUExecutionProvider"],
              "calibrate_method": "Percentile" if variant == "Q3" else "MinMax",
              "extra_options": {"ActivationSymmetric": False, "WeightSymmetric": True,
                                "CalibTensorRangeSymmetric": False},
              "calibration_collection": {"batch_size": 1, "streaming": True,
                                          "activation_per_channel": False,
                                          "moving_average": False}}
    if variant == "Q3":
        common["extra_options"]["CalibPercentile"] = 99.99
        common["calibration_collection"].update({"percentile": 99.99, "histogram_num_bins": 2048})
    return common


def _collect_ranges(qb_path: Path, reader: CalibrationDataReader, method: str,
                    directory: Path, binding: dict, api: dict) -> tuple[Path, dict]:
    calibration = api["calibrate"]
    augmented = directory / f"augmented-{method}.onnx"
    kwargs = {"model_path": qb_path, "op_types_to_calibrate": list(_TARGET_OPS),
              "augmented_model_path": str(augmented), "symmetric": False,
              "use_external_data_format": False}
    if method == "MinMax":
        # ORT1.30 的 max_intermediate_outputs=1 会提前 clear 而非计算范围。
        # 每次喂一张，collect_data 正常 compute+clear，内存仍为单图。
        calibrator = calibration.MinMaxCalibrater(**kwargs, moving_average=False,
                                                  max_intermediate_outputs=None, per_channel=False)
    else:
        calibrator = calibration.PercentileCalibrater(**kwargs, percentile=99.99, num_bins=2048)
    calibrator.augment_graph()
    calibrator.execution_providers = ["CPUExecutionProvider"]
    calibrator.infer_session = _create_session(augmented, threads=1)
    reader.rewind()
    count = 0
    while (value := reader.get_next()) is not None:
        calibrator.collect_data(_OneImageReader(value))
        count += 1
        # collect_data 的官方实现会清空 intermediate_outputs；不跨图积累 featuremap。
        if calibrator.intermediate_outputs:
            raise QuantizationError("校准器未释放单图中间输出；拒绝继续积累")
    if count != len(reader):
        raise QuantizationError("校准实际 collect_count 与固定图片列表不符")
    ranges = calibrator.compute_data()
    for name, data in ranges.items():
        np = _module("numpy")
        if not all(np.isfinite(value).all() for value in data.range_value):
            raise QuantizationError(f"校准范围含 NaN/Inf: {name}")
    cache = directory / f"calibration-{method}.json"
    calibration.save_tensors_data(ranges, cache, smooth_quant=False)
    metadata = {"schema_version": 1, **binding, "method": method,
                "parameters": {"symmetric": False, "activation_per_channel": False,
                               **({"percentile": 99.99, "num_bins": 2048} if method == "Percentile"
                                  else {"moving_average": False, "max_intermediate_outputs": None})},
                "collect_count": count, "streaming_batch_size": 1,
                "session": _session_config(1), "cache_sha256": _hash_file(cache),
                "cache_format": "official ORT TensorsData JSON; no pickle",
                "range_tensor_count": len(ranges.data), "cache_filename": cache.name}
    (directory / f"calibration-{method}.metadata.json").write_bytes(_json_bytes(metadata))
    del calibrator
    return cache, metadata


def _validation_images(raw_session: Any, session: Any, images: list[dict],
                       model_id: str, *, compare_raw: bool) -> dict:
    np = _module("numpy")
    _check_session_contract(session)
    records = []
    for entry in images:
        value = _prepare_array(entry, model_id)
        output = _run(session, value)
        record = {"id": entry["id"], "sha256": entry["sha256"],
                  "shape": list(output.shape), "dtype": str(output.dtype), "finite": True}
        if compare_raw:
            reference = _run(raw_session, value)
            difference = np.abs(reference - output)
            record["comparison"] = {"passed": bool(np.allclose(reference, output, rtol=1e-4, atol=1e-4)),
                                    "rtol": 1e-4, "atol": 1e-4,
                                    "max_abs_error": float(difference.max()),
                                    "mean_abs_error": float(difference.mean())}
            if not record["comparison"]["passed"]:
                raise QuantizationError(f"QB vs raw 数值比较失败 {entry['id']}: {record['comparison']}")
        records.append(record)
    return {"passed": True, "image_count": len(records), "split": "development",
            "provider": "CPUExecutionProvider", "results": records,
            "fp32_equivalence_required": compare_raw,
            "note": "QB 检查 raw 数值等价；Q 只检查 shape/dtype/有限值，不等于精度通过。"}


def _failure(error: Exception) -> tuple[str, str]:
    reason = f"{type(error).__name__}: {error}"
    lowered = reason.lower()
    unsupported = isinstance(error, NotImplementedError) or any(
        text in lowered for text in ("not_implemented", "not implemented", "unsupported", "no implementation"))
    return ("unsupported" if unsupported else "failed_runtime"), reason


def prepare_quantized_family(model_id, fp32_path, manifest_path, output_dir, *,
                             overwrite=False, seed=20261001) -> dict:
    """在临时目录生成同父图 QB/Q1/Q2/Q3；只发布已校验产物，完成标记最后写。"""
    _model_id(model_id)
    _integer(seed, "seed")
    if type(overwrite) is not bool:
        raise QuantizationError("overwrite 必须为 bool")
    source, output = Path(fp32_path).resolve(), Path(output_dir).resolve()
    if output.exists() and not output.is_dir():
        raise QuantizationError("output_dir 必须是目录")
    filenames = [f"{variant}.onnx" for variant in QUANT_VARIANTS] + ["family.json"]
    filenames += [f"calibration-{method}{suffix}.json" for method in ("MinMax", "Percentile")
                  for suffix in ("", ".metadata")]
    if not overwrite and any((output / name).exists() for name in filenames):
        raise QuantizationError("输出已存在；默认拒绝覆盖，请显式 overwrite=True")
    planned = {output / name for name in filenames}
    if source in planned:
        raise QuantizationError("禁止覆盖 raw/旧 ONNX")
    if Path(manifest_path).resolve() in planned:
        raise QuantizationError("禁止覆盖输入数据 manifest")
    manifest = _load_dataset(manifest_path)
    _require_stage_b_data(manifest)
    _check_dataset_bytes(manifest, ("calibration", "development"))
    calibration_images = _ordered_images(manifest, "calibration", seed)[:_MINIMUM_COUNTS["calibration"]]
    development_images = _ordered_images(manifest, "development", seed)[:4]
    original_model, original_stats = _check_graph(source, fp32_parent=True)
    api = _quantization_api()
    try:
        _core().configure_cpu_threads(1)
    except _core().ClassificationError as error:
        raise QuantizationError(str(error)) from error
    image_list = [{"id": entry["id"], "sha256": entry["sha256"],
                   "source_id": entry.get("source_id")} for entry in calibration_images]
    list_hash = hashlib.sha256(_json_bytes(image_list)).hexdigest()
    family = {"schema_version": 1, "model_id": model_id,
              "input_model": {"path": str(source), "sha256": original_stats["sha256"],
                              "graph_statistics": original_stats},
              "manifest_sha256": manifest["manifest_sha256"],
              "calibration": {"count": len(calibration_images),
                              "image_ids": [entry["id"] for entry in calibration_images],
                              "sha256": list_hash, "image_list_hash": list_hash,
                              "images": image_list, "split": "calibration", "seed": seed,
                              "sampling": f"sort by id; random.Random(seed).shuffle; first {len(calibration_images)}",
                              "source_policy": {"source_records": sorted({
                                  (entry.get("source") or entry.get("source_url"), entry["license"])
                                  for entry in calibration_images}),
                                  "upload": False, "holdout_pixels_read": False,
                                  "byte_verification": {"calibration": "all SHA256 checked",
                                                        "development": "all SHA256 checked",
                                                        "holdout": "metadata/existence only; bytes guarded until frozen"},
                                  "label_mapping": copy.deepcopy(manifest.get("label_mapping")),
                                  "note": "本地来源/许可声明，不认证版权或近重复；不上传图片。"}},
              "dataset_readiness": manifest["dataset_readiness"], "variants": {},
              "versions": _versions(), "api_evidence": api["evidence"],
              "limitations": ["20 类或其他子集不等于完整 ImageNet；只使用固定 V1 预处理。",
                              "SHA256/source_id 防泄漏不能排除未知近重复图片。",
                              "未读取 holdout 图片/预测；未做真实精度、性能或硬件采用判定。",
                              "未覆盖算子保留浮点；QDQ 存在不证明 runtime 整数执行。",
                              "逐张增量直方图的确定顺序已冻结；并非全样本 featuremap 堆叠。"],
              "output_dir": str(output),
              "io_policy": {"computation": "system temporary directory; ASCII native ONNX paths on Windows",
                            "publication": "verified copy into output filesystem; atomic os.replace; family.json last"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    # ONNX 的 C++ infer_shapes_path 在本机 Windows 中文路径下不生成 inferred 文件。
    # 只隔离文件 I/O，不更改图/量化参数；跨盘复制后仍在输出盘内原子发布。
    with tempfile.TemporaryDirectory(prefix=f"{model_id}-quant-") as temporary, \
            tempfile.TemporaryDirectory(prefix=f".{model_id}-publish-", dir=output.parent) as publication:
        staged = Path(temporary)
        if os.name == "nt" and not str(staged).isascii():
            raise QuantizationError("Windows ONNX 原生 I/O 需要 ASCII 临时路径；请将 TEMP/TMP 指向 ASCII 目录")
        native_source = staged / "raw.onnx"
        shutil.copyfile(source, native_source)
        if _hash_file(native_source) != original_stats["sha256"]:
            raise QuantizationError("原图临时副本 SHA256 不匹配；拒绝校准")
        publish = []
        qb = staged / "QB.onnx"
        qb_config = {"parent_variant": "raw", "parent_onnx_sha256": original_stats["sha256"],
                     "session": _session_config(1), "quant_pre_process": {
                         "skip_optimization": True, "skip_symbolic_shape": True,
                         "skip_onnx_shape": False, "auto_merge": False,
                         "int_max": 2147483647, "guess_output_rank": False, "verbose": 0,
                         "save_as_external_data": False, "all_tensors_to_one_file": False,
                         "external_data_location": None, "external_data_size_threshold": 1024},
                     "reason": "原导出已折叠 BN；不先融合为量化器不覆盖的 FusedConv。"}
        qb_record = {"status": "failed_runtime", "path": None, "onnx_sha256": None,
                     "size_bytes": None, "config": qb_config, "graph_statistics": None,
                     "validation": {"passed": False}}
        family["variants"]["QB"] = qb_record
        try:
            api["shape"].quant_pre_process(input_model=native_source, output_model_path=qb,
                                           **qb_config["quant_pre_process"])
            qb_model, qb_stats = _check_graph(qb, fp32_parent=True)
            raw_session = _create_session(source)
            _check_session_contract(raw_session)
            qb_session = _create_session(qb)
            validation = _validation_images(raw_session, qb_session, development_images,
                                             model_id, compare_raw=True)
            del raw_session, qb_session
            qb_record.update({"status": "prepared", "path": str(output / qb.name),
                              "onnx_sha256": qb_stats["sha256"], "size_bytes": qb_stats["size_bytes"],
                              "graph_statistics": qb_stats, "validation": validation})
            publish.append(qb)
        except Exception as error:
            qb_record["status"], qb_record["reason"] = _failure(error)
            qb_model, qb_stats = None, None
        caches: dict[str, tuple[Path, dict]] = {}
        cache_failures = {}
        for variant in QUANT_VARIANTS[1:]:
            config = _variant_config(variant, qb_record["onnx_sha256"])
            record = {"status": "failed_runtime", "path": None, "onnx_sha256": None,
                      "size_bytes": None, "config": config, "graph_statistics": None,
                      "validation": {"passed": False}}
            family["variants"][variant] = record
            if qb_record["status"] != "prepared":
                record.update({"status": "unsupported", "reason": "QB 未通过验证；不切换父图或浮点方案"})
                continue
            if not any(qb_stats["op_type_counts"].get(op, 0) for op in _TARGET_OPS):
                record.update({"status": "unsupported", "reason": "QB 没有 Conv/Gemm/MatMul 目标算子；不会用浮点冒充量化"})
                continue
            method = config["calibrate_method"]
            try:
                if method in cache_failures:
                    raise QuantizationError(f"校准范围生成失败，未重试: {cache_failures[method]}")
                if method not in caches:
                    reader = CalibrationDataReader(model_id, calibration_images, qb_model.graph.input[0].name)
                    binding = {"qb_sha256": qb_stats["sha256"],
                               "manifest_sha256": manifest["manifest_sha256"],
                               "calibration_image_list_sha256": list_hash,
                               "calibration_image_ids": family["calibration"]["image_ids"]}
                    try:
                        caches[method] = _collect_ranges(qb, reader, method, staged, binding, api)
                    except Exception as error:
                        cache_failures[method] = f"{type(error).__name__}: {error}"
                        raise
                    cache, _ = caches[method]
                    publish.extend([cache, staged / f"calibration-{method}.metadata.json"])
                cache, cache_metadata = caches[method]
                config["calibration_cache"] = copy.deepcopy(cache_metadata)
                quant = api["quant"]
                candidate = staged / f"{variant}.onnx"
                quant.quantize_static(
                    model_input=qb, model_output=candidate, calibration_data_reader=None,
                    calibration_cache_path=cache, quant_format=quant.QuantFormat.QDQ,
                    op_types_to_quantize=list(_TARGET_OPS), per_channel=config["per_channel"],
                    reduce_range=False, activation_type=quant.QuantType.QInt8,
                    weight_type=quant.QuantType.QInt8, nodes_to_quantize=None,
                    nodes_to_exclude=None, use_external_data_format=False,
                    calibrate_method=getattr(quant.CalibrationMethod, method),
                    calibration_providers=["CPUExecutionProvider"],
                    extra_options=copy.deepcopy(config["extra_options"]))
                candidate_model, stats = _check_graph(candidate)
                stats["quantization_coverage"] = _qdq_coverage(qb_model, candidate_model)
                session = _create_session(candidate)
                validation = _validation_images(None, session, development_images, model_id, compare_raw=False)
                del session
                if stats["quantization_coverage"]["qdq_target_node_count"] == 0:
                    raise QuantizationError("unsupported: 本候选无真实 S8 QDQ 目标算子覆盖；不发布浮点冒充量化")
                record.update({"status": "prepared", "path": str(output / candidate.name),
                               "onnx_sha256": stats["sha256"], "size_bytes": stats["size_bytes"],
                               "graph_statistics": stats, "validation": validation})
                publish.append(candidate)
            except Exception as error:
                record["status"], record["reason"] = _failure(error)
                candidate = staged / f"{variant}.onnx"
                if candidate.is_file():
                    record["attempted_onnx_sha256"] = _hash_file(candidate)
                    try:
                        _, stats = _check_graph(candidate)
                        record["graph_statistics"] = stats
                        model = _module("onnx").load(str(candidate), load_external_data=False)
                        record["graph_statistics"]["quantization_coverage"] = _qdq_coverage(qb_model, model)
                    except Exception:
                        pass
        if (_hash_file(source) != original_stats["sha256"]
                or _hash_file(manifest_path) != manifest["manifest_sha256"]):
            raise QuantizationError("raw ONNX 或 manifest 在准备中发生变化；拒绝发布")
        family["status"] = ("prepared" if all(item["status"] == "prepared" for item in family["variants"].values())
                            else "completed_with_failures")
        staged_family = staged / "family.json"
        staged_family.write_bytes(_json_bytes(family))
        # 所有候选已完成检查后才发布；family.json 是最后的完成标记。
        output.mkdir(parents=True, exist_ok=True)
        if not overwrite and any((output / name).exists() for name in filenames):
            raise QuantizationError("发布时输出已存在；拒绝覆盖")
        publication_path = Path(publication)
        # 原生计算临时目录可能位于其他盘。先校验全部副本，再开始发布，
        # 避免跨设备 rename 失败，也不把不完整 family 写成完成状态。
        copied = []
        for artifact in publish + [staged_family]:
            destination = publication_path / artifact.name
            shutil.copyfile(artifact, destination)
            if _hash_file(destination) != _hash_file(artifact):
                raise QuantizationError(f"发布副本 SHA256 不匹配: {artifact.name}")
            copied.append(destination)
        for artifact in copied:
            os.replace(artifact, output / artifact.name)
    return family


def _verify_frozen(frozen_path: str | Path | None, expected_sha: str | None,
                   manifest_sha: str, model_id: str, artifact_sha: str, threads: int) -> dict:
    if frozen_path is None or expected_sha is None:
        raise QuantizationError("holdout guard: 必须提供 frozen_path 和 expected_frozen_sha256")
    expected_sha = _digest(expected_sha, "expected_frozen_sha256")
    frozen, raw = _read_json(frozen_path)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise QuantizationError("holdout guard: frozen SHA256 不匹配")
    if type(frozen.get("schema_version")) is not int or frozen["schema_version"] != 1 or frozen.get("status") != "frozen":
        raise QuantizationError("holdout guard: schema_version=1/status=frozen 必须明确")
    if _digest(frozen.get("manifest_sha256"), "frozen.manifest_sha256") != manifest_sha:
        raise QuantizationError("holdout guard: frozen manifest_sha256 不匹配")
    candidates = frozen.get("candidates")
    if not isinstance(candidates, list) or not candidates or any(not isinstance(item, dict) for item in candidates):
        raise QuantizationError("holdout guard: candidates 必须为非空对象列表")
    matches = [item for item in candidates if item.get("model_id") == model_id]
    if len(matches) != 1:
        raise QuantizationError("holdout guard: 当前 model_id 必须恰有一个冻结候选")
    item = matches[0]
    selected_sha = _digest(item.get("artifact_sha256"), "candidate.artifact_sha256")
    selected_threads = _integer(item.get("threads"), "candidate.threads", 1)
    allowed = [(selected_sha, selected_threads, "candidate")]
    if item.get("baseline_artifact_sha256") is not None:
        allowed.append((_digest(item["baseline_artifact_sha256"], "candidate.baseline_artifact_sha256"),
                        _integer(item.get("baseline_threads"), "candidate.baseline_threads", 1), "baseline"))
    roles = [role for sha, count, role in allowed if sha == artifact_sha and count == threads]
    if not roles:
        raise QuantizationError("holdout guard: artifact SHA256 或 threads 不在该 model_id 冻结候选/基线中")
    return {"path": str(Path(frozen_path).resolve()), "sha256": actual, "roles": roles}


def evaluate_classifier(model_id, onnx_path, manifest_path, *, split='development',
                        threads=1, frozen_path=None, expected_frozen_sha256=None) -> dict:
    """全 1000 类排名；label 只来自清单，不用模型预测生成真值。"""
    _model_id(model_id)
    _integer(threads, "threads", 1)
    if split not in ("development", "holdout"):
        raise QuantizationError("split 只允许 development 或 holdout")
    if split == "holdout" and (frozen_path is None or expected_frozen_sha256 is None):
        raise QuantizationError("holdout guard: 尚未冻结，拒绝获取最终成绩")
    manifest = _load_dataset(manifest_path)
    images = _ordered_images(manifest, split)
    minimum = _MINIMUM_COUNTS[split]
    if len(images) < minimum or any(entry["label"] is None for entry in images):
        raise QuantizationError(f"blocked_data: {split} 至少 {minimum} 张且每张必须有真实 0..999 标签")
    artifact_sha = _hash_file(onnx_path)
    frozen = (_verify_frozen(frozen_path, expected_frozen_sha256, manifest["manifest_sha256"],
                             model_id, artifact_sha, threads) if split == "holdout" else None)
    _check_dataset_bytes(manifest, (split,))
    _, stats = _check_graph(onnx_path)
    np = _module("numpy")
    try:
        _core().configure_cpu_threads(1)
        session = _create_session(onnx_path, threads=threads)
        _check_session_contract(session)
        per_image = []
        for entry in images:
            logits = _run(session, _prepare_array(entry, model_id))[0]
            top5 = [int(index) for index in np.argsort(-logits, kind="stable")[:5]]
            label = entry["label"]
            per_image.append({"id": entry["id"], "label": label, "top1": top5[0], "top5": top5,
                              "top1_correct": top5[0] == label, "top5_correct": label in top5,
                              "sha256": entry["sha256"]})
    except QuantizationError:
        raise
    except Exception as error:
        raise QuantizationError(f"精度评估失败: {type(error).__name__}: {error}") from error
    if (_hash_file(manifest_path) != manifest["manifest_sha256"]
            or _hash_file(onnx_path) != artifact_sha
            or (frozen and _hash_file(frozen_path) != frozen["sha256"])):
        raise QuantizationError("manifest/artifact/frozen 在评估中变化，成绩无效")
    return {"schema_version": 1, "manifest_sha256": manifest["manifest_sha256"],
            "model_id": model_id, "artifact_sha256": artifact_sha, "split": split,
            "image_count": len(per_image), "top1_accuracy": sum(row["top1_correct"] for row in per_image) / len(per_image),
            "top5_accuracy": sum(row["top5_correct"] for row in per_image) / len(per_image),
            "per_image": per_image, "session": _session_config(threads), "frozen": frozen,
            "graph_statistics": stats, "versions": _versions(),
            "limitations": ["仅此带标签子集精度；不等于完整 ImageNet，也不验证标签来源的真实性。",
                            "所有排名覆盖 1000 类，未 mask 子集类别；并列分数按类索引稳定排序。",
                            "精度不等于 softmax 分数、预测一致率或速度。"]}


def _accuracy_rows(value: Any, name: str) -> list[dict]:
    if not isinstance(value, dict):
        raise QuantizationError(f"{name} 必须为评估对象")
    _model_id(value.get("model_id"))
    _digest(value.get("manifest_sha256"), f"{name}.manifest_sha256")
    _digest(value.get("artifact_sha256"), f"{name}.artifact_sha256")
    if value.get("split") not in ("development", "holdout"):
        raise QuantizationError(f"{name}.split 无效")
    rows = value.get("per_image")
    if not isinstance(rows, list) or not rows or _integer(value.get("image_count"), "image_count", 1) != len(rows):
        raise QuantizationError(f"{name} 必须有完整非空逐图片标签/预测")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise QuantizationError(f"{name} 图片 ID 无效/重复")
        seen.add(row["id"])
        _digest(row.get("sha256"), f"{name}.image.sha256")
        label = _integer(row.get("label"), "真实 label")
        top1 = _integer(row.get("top1"), "top1")
        top5 = row.get("top5")
        if (label > 999 or top1 > 999 or not isinstance(top5, list) or len(top5) != 5
                or any(type(index) is not int or not 0 <= index <= 999 for index in top5)
                or len(set(top5)) != 5 or top1 != top5[0]):
            raise QuantizationError(f"{name} 必须使用真实 0..999 标签和全 1000 类 Top-5 索引")
        for key, expected in (("top1_correct", top1 == label), ("top5_correct", label in top5)):
            if type(row.get(key)) is not bool or row[key] != expected:
                raise QuantizationError(f"{name}.{key} 与真实标签不符")
    for metric in ("top1", "top5"):
        accuracy = value.get(f"{metric}_accuracy")
        expected = sum(row[f"{metric}_correct"] for row in rows) / len(rows)
        if (isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
                or not math.isfinite(accuracy) or not 0 <= accuracy <= 1
                or not math.isclose(accuracy, expected, rel_tol=0, abs_tol=1e-12)):
            raise QuantizationError(f"{name}.{metric}_accuracy 不是逐图真值计算的准确率")
    return rows


def compare_accuracies(reference, candidate, *, seed=20261001, bootstrap_samples=2000) -> dict:
    """按同一图片配对重采样下降百分点；区间跨越 1pp 时保持证据不足。"""
    _integer(seed, "seed")
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    baseline_rows, candidate_rows = _accuracy_rows(reference, "reference"), _accuracy_rows(candidate, "candidate")
    for key in ("manifest_sha256", "split", "model_id"):
        if reference[key] != candidate[key]:
            raise QuantizationError(f"精度比较未配对: {key} 不同")
    if len(baseline_rows) != len(candidate_rows):
        raise QuantizationError("精度比较未配对: 图片数量不同")
    for baseline, row in zip(baseline_rows, candidate_rows):
        if any(baseline[key] != row[key] for key in ("id", "label", "sha256")):
            raise QuantizationError("精度比较未配对: 图片 ID 顺序/真值/字节 SHA 不同")
    np = _module("numpy")
    differences = np.asarray([[int(a[f"{metric}_correct"]) - int(b[f"{metric}_correct"])
                               for metric in ("top1", "top5")]
                              for a, b in zip(baseline_rows, candidate_rows)], dtype=np.float64)
    rng, count = np.random.default_rng(seed), len(baseline_rows)
    samples = np.empty((bootstrap_samples, 2), dtype=np.float64)
    for start in range(0, bootstrap_samples, 64):
        end = min(start + 64, bootstrap_samples)
        indices = rng.integers(0, count, size=(end - start, count))
        samples[start:end] = differences[indices].mean(axis=1) * 100
    result = {"schema_version": 1, "same_manifest": True, "paired": True,
              "manifest_sha256": reference["manifest_sha256"], "model_id": reference["model_id"],
              "split": reference["split"], "image_count": count,
              "reference_artifact_sha256": reference["artifact_sha256"],
              "candidate_artifact_sha256": candidate["artifact_sha256"],
              "seed": seed, "bootstrap_samples": bootstrap_samples, "confidence_level": 0.95,
              "resampling_unit": "paired image", "budget_pp": 1.0,
              "limitations": ["按图片配对 bootstrap，不认证真值质量或总体代表性。",
                              "下降=基线准确率-候选准确率，单位百分点；跨预算不强行通过。"]}
    statuses = []
    for index, metric in enumerate(("top1", "top5")):
        lower, upper = [float(item) for item in np.percentile(samples[:, index], [2.5, 97.5])]
        status = "passed" if upper <= 1.0 else "failed" if lower > 1.0 else "inconclusive"
        statuses.append(status)
        result[metric] = {"drop_pp": float(differences[:, index].mean() * 100),
                          "ci95_pp": [lower, upper], "budget_pp": 1.0, "budget_status": status,
                          "budget_passed": True if status == "passed" else False if status == "failed" else None}
        result[f"{metric}_drop_pp"] = result[metric]["drop_pp"]
        result[f"{metric}_ci95_pp"] = result[metric]["ci95_pp"]
    result["budget_status"] = "passed" if all(status == "passed" for status in statuses) else (
        "failed" if "failed" in statuses else "inconclusive")
    result["budget_passed"] = True if result["budget_status"] == "passed" else (
        False if result["budget_status"] == "failed" else None)
    return result


def _profile_events(events: Any) -> dict:
    if not isinstance(events, list):
        raise QuantizationError("ORT profile 必须为事件列表")
    operators, integers, floats, conversions, others = Counter(), Counter(), Counter(), Counter(), Counter()
    providers = set()
    integer_events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        args = event.get("args")
        if not isinstance(args, dict) or not isinstance(args.get("op_name"), str):
            continue
        op, provider = args["op_name"], args.get("provider")
        # fence/其他重复标记不含 provider，不能冒充一次 kernel 执行。
        if not isinstance(provider, str) or not provider:
            continue
        providers.add(provider)
        operators[op] += 1
        if provider == "CPUExecutionProvider" and op in _INTEGER_OPS:
            integers[op] += 1
            integer_events.append({"event_index": index, "name": event.get("name"),
                                   "op_name": op, "provider": provider, "duration_us": event.get("dur")})
        elif op in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"):
            conversions[op] += 1
        elif op in _FLOAT_OPS:
            floats[op] += 1
        else:
            others[op] += 1
    note = ("发现保守白名单中的 CPU 整数计算 kernel；仅说明这些算子实际执行整数路径，"
            "不代表全 INT8 或更快。" if integer_events else
            "未观察到已确认的 CPU 整数计算算子；QB 无整数正常，QDQ/转换事件不证明 INT8 执行或加速。")
    return {"operator_counts": dict(sorted(operators.items())), "int8_operator_events": integer_events,
            "int8_operator_counts": dict(sorted(integers.items())),
            "float_operator_counts": dict(sorted(floats.items())),
            "conversion_operator_counts": dict(sorted(conversions.items())),
            "other_operator_counts": dict(sorted(others.items())), "providers": sorted(providers),
            "precision_note": note, "integer_whitelist": sorted(_INTEGER_OPS),
            "classification_policy": "仅含 provider 的 Node kernel 事件；未认识的算子不猜测为整数。"}


def profile_quantized_execution(onnx_path, input_array, output_dir, *, threads=1) -> dict:
    """独立 CPU session，1 次预热+5 次诊断输入；原始 trace 不参与性能测量。"""
    _integer(threads, "threads", 1)
    value = _validate_array(input_array, "profile input_array")
    _, stats = _check_graph(onnx_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".quant-profile-", dir=output) as temporary:
        prefix = str(Path(temporary) / "ort")
        session = _create_session(onnx_path, threads=threads, profile_prefix=prefix)
        trace_path = None
        try:
            _check_session_contract(session)
            for _ in range(6):
                _run(session, value)
        finally:
            trace_path = Path(session.end_profiling())
        try:
            raw = trace_path.read_bytes()
            events = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
                QuantizationError(f"profile 非有限常量 {value}")))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise QuantizationError(f"无法解析 ORT 原始 trace: {error}") from error
        result = _profile_events(events)
        destination = output / f"profile-{uuid.uuid4().hex}.json"
        os.replace(trace_path, destination)
    return {"schema_version": 1, "artifact_sha256": stats["sha256"],
            "trace": {"path": str(destination), "sha256": hashlib.sha256(raw).hexdigest(),
                      "size_bytes": len(raw)}, "trace_path": str(destination),
            "trace_sha256": hashlib.sha256(raw).hexdigest(), "warmup": 1, "inputs": 5,
            "session": _session_config(threads), **result, "versions": _versions(),
            "limitations": ["独立诊断 profile 包含预热事件，不参与正式计时或速度判定。",
                            "浮点算子按保守名称统计，未知/转换算子分列；无整数证据不叫 INT8 加速。"]}
