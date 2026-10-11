"""三模型（ResNet18 / MobileNetV3-Small / ShuffleNetV2-x1.0）变体构建与 ORT 会话模块。

支持 PyTorch 算子融合 (F1)、内存布局优化 (L1/FL1)、ONNX Runtime 会话配置 (O0/O1) 以及图统计。
"""

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.vision.classification import (
    MODEL_IDS,
    ClassificationError,
    _require,
    compute_sha256,
)

__all__ = [
    "VARIANT_IDS",
    "TORCH_VARIANT_IDS",
    "ORT_VARIANT_IDS",
    "fuse_classifier",
    "build_torch_variant",
    "create_ort_session",
    "graph_statistics",
]

VARIANT_IDS: Tuple[str, ...] = ("B0", "F1", "L1", "FL1", "O0", "O1")
TORCH_VARIANT_IDS: Tuple[str, ...] = ("B0", "F1", "L1", "FL1")
ORT_VARIANT_IDS: Tuple[str, ...] = ("O0", "O1")


def _check_and_fuse_pair(conv: Any, bn: Any, tag: str) -> Any:
    """严格校验并融合单个 Conv2d 与 BatchNorm2d 对。"""
    torch = _require("torch")
    if not isinstance(conv, torch.nn.Conv2d) or not isinstance(bn, torch.nn.BatchNorm2d):
        raise ClassificationError(f"{tag} 算子类型异常: conv 须为 Conv2d，bn 须为 BatchNorm2d")
    if bn.running_mean is None or bn.running_var is None:
        raise ClassificationError(f"{tag} BatchNorm2d 缺少有效 running stats，无法融合")
    return torch.nn.utils.fuse_conv_bn_eval(conv, bn)


def _fuse_resnet18(model: Any) -> Tuple[Any, int, List[Tuple[str, str]], List[str]]:
    """对 ResNet18 执行显式属性 Conv-BN 融合。"""
    torch = _require("torch")
    if not hasattr(model, "conv1") or not hasattr(model, "bn1"):
        raise ClassificationError("模型缺少 ResNet18 stem 结构 (conv1/bn1)")

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        if not hasattr(model, layer_name):
            raise ClassificationError(f"模型缺少 ResNet18 阶段层: {layer_name}")
        layer = getattr(model, layer_name)
        if len(layer) != 2 or any(b.__class__.__name__ != "BasicBlock" for b in layer):
            raise ClassificationError(f"结构不符合 ResNet18: {layer_name} 须包含恰好 2 个 BasicBlock")

    fused = copy.deepcopy(model)
    fused_paths: List[Tuple[str, str]] = []

    fused.conv1 = _check_and_fuse_pair(fused.conv1, fused.bn1, "stem")
    fused.bn1 = torch.nn.Identity()
    fused_paths.append(("conv1", "bn1"))

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        for block_idx, block in enumerate(getattr(fused, layer_name)):
            tag = f"{layer_name}[{block_idx}]"
            block.conv1 = _check_and_fuse_pair(block.conv1, block.bn1, f"{tag}.conv1")
            block.bn1 = torch.nn.Identity()
            fused_paths.append((f"{tag}.conv1", f"{tag}.bn1"))

            block.conv2 = _check_and_fuse_pair(block.conv2, block.bn2, f"{tag}.conv2")
            block.bn2 = torch.nn.Identity()
            fused_paths.append((f"{tag}.conv2", f"{tag}.bn2"))

            if block.downsample is not None:
                if len(block.downsample) < 2:
                    raise ClassificationError(f"{tag}.downsample 结构不完整，期望含 [Conv2d, BatchNorm2d]")
                block.downsample[0] = _check_and_fuse_pair(
                    block.downsample[0], block.downsample[1], f"{tag}.downsample"
                )
                block.downsample[1] = torch.nn.Identity()
                fused_paths.append((f"{tag}.downsample[0]", f"{tag}.downsample[1]"))

    unfused_bns = [name for name, m in fused.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    return fused, len(fused_paths), fused_paths, unfused_bns


def _fuse_sequential_conv_bns(model: Any) -> Tuple[Any, int, List[Tuple[str, str]], List[str]]:
    """递归遍历模型所有 Sequential 容器，融合其中的相邻 Conv2d/BatchNorm2d 对。"""
    torch = _require("torch")
    fused = copy.deepcopy(model)
    fused_paths: List[Tuple[str, str]] = []

    for mod_path, module in fused.named_modules():
        if isinstance(module, torch.nn.Sequential):
            i = 0
            while i < len(module) - 1:
                sub_conv = module[i]
                sub_bn = module[i + 1]
                if isinstance(sub_conv, torch.nn.Conv2d) and isinstance(sub_bn, torch.nn.BatchNorm2d):
                    prefix = f"{mod_path}." if mod_path else ""
                    tag_conv = f"{prefix}{i}"
                    tag_bn = f"{prefix}{i + 1}"
                    module[i] = _check_and_fuse_pair(sub_conv, sub_bn, f"{tag_conv}->{tag_bn}")
                    module[i + 1] = torch.nn.Identity()
                    fused_paths.append((tag_conv, tag_bn))
                    i += 2
                else:
                    i += 1

    unfused_bns = [name for name, m in fused.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    return fused, len(fused_paths), fused_paths, unfused_bns


def fuse_classifier(model: Any, model_id: str = "resnet18") -> Tuple[Any, Dict[str, Any]]:
    """对给定模型执行 Conv-BN 算子融合，BN 替换为 Identity。

    返回深拷贝后的 (fused_model, metadata)。
    严格保留分支语义，仅融合 ResNet 显式属性与另两模型 Sequential 内紧邻的 Conv2d/BN 对。
    """
    if model_id not in MODEL_IDS:
        raise ClassificationError(f"未知模型 ID: '{model_id}'。支持列表: {MODEL_IDS}")

    torch = _require("torch")
    if getattr(model, "training", False):
        raise ClassificationError("模型必须在 eval() 模式下方可执行 Conv-BN 融合")

    if model_id == "resnet18":
        fused_model, pair_count, fused_paths, unfused_bns = _fuse_resnet18(model)
    else:
        fused_model, pair_count, fused_paths, unfused_bns = _fuse_sequential_conv_bns(model)

    fused_model.eval().to(dtype=torch.float32, device="cpu")

    metadata = {
        "model_id": model_id,
        "variant_id": "F1",
        "fused": True,
        "fused_pair_count": pair_count,
        "fused_paths": fused_paths,
        "unfused_bn_paths": unfused_bns,
        "memory_format": "contiguous",
    }
    return fused_model, metadata


def build_torch_variant(
    model: Any,
    model_id: str,
    variant_id: str,
) -> Tuple[Any, Dict[str, Any]]:
    """构建 PyTorch 实验变体 (B0 / F1 / L1 / FL1)。

    保证布局变换及算子融合绝不污染原始 baseline 模型对象。
    返回 (variant_model, metadata)，其中 metadata 固定含 memory_format ('contiguous'/'channels_last')
    以及 fused_pair_count（含 0）。
    """
    if variant_id not in TORCH_VARIANT_IDS:
        raise ClassificationError(f"不支持的 PyTorch 变体 ID: '{variant_id}'。支持列表: {TORCH_VARIANT_IDS}")
    if model_id not in MODEL_IDS:
        raise ClassificationError(f"未知模型 ID: '{model_id}'。支持列表: {MODEL_IDS}")

    torch = _require("torch")

    if variant_id == "B0":
        variant_model = copy.deepcopy(model).eval().to(dtype=torch.float32, device="cpu")
        unfused_bns = [name for name, m in variant_model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
        metadata = {
            "model_id": model_id,
            "variant_id": "B0",
            "fused": False,
            "fused_pair_count": 0,
            "fused_paths": [],
            "unfused_bn_paths": unfused_bns,
            "memory_format": "contiguous",
        }
        return variant_model, metadata

    elif variant_id == "F1":
        return fuse_classifier(model, model_id)

    elif variant_id == "L1":
        variant_model = copy.deepcopy(model).eval().to(
            dtype=torch.float32, device="cpu", memory_format=torch.channels_last
        )
        unfused_bns = [name for name, m in variant_model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
        metadata = {
            "model_id": model_id,
            "variant_id": "L1",
            "fused": False,
            "fused_pair_count": 0,
            "fused_paths": [],
            "unfused_bn_paths": unfused_bns,
            "memory_format": "channels_last",
        }
        return variant_model, metadata

    elif variant_id == "FL1":
        fused_model, f_meta = fuse_classifier(model, model_id)
        variant_model = fused_model.to(memory_format=torch.channels_last)
        metadata = copy.deepcopy(f_meta)
        metadata["variant_id"] = "FL1"
        metadata["memory_format"] = "channels_last"
        return variant_model, metadata


def create_ort_session(
    onnx_path: Union[str, Path],
    *,
    threads: int = 1,
    variant_id: str = "O1",
    profiling: bool = False,
    profile_file_prefix: Optional[str] = None,
) -> Any:
    """为 ONNX 模型创建严格受控的 CPU ONNX Runtime 推理会话。

    仅允许 CPUExecutionProvider；intra_op 显式指定，inter_op 固定为 1；
    执行模式固定为 SEQUENTIAL；
    O0 对应 ORT_DISABLE_ALL，O1 对应 ORT_ENABLE_ALL；
    固定 session.intra_op.allow_spinning="0" 并在会话元数据中完整记录。
    支持可选 profiling 与 profile_file_prefix 便于独立诊断 worker。
    """
    if variant_id not in ORT_VARIANT_IDS:
        raise ClassificationError(f"不支持的 ORT 变体 ID: '{variant_id}'。支持列表: {ORT_VARIANT_IDS}")
    if not isinstance(threads, int) or threads <= 0:
        raise ClassificationError(f"threads 必须为正整数，收到: {threads}")

    path = Path(onnx_path).resolve()
    if not path.is_file():
        raise ClassificationError(f"ONNX 模型文件不存在: {path}")

    ort = _require("onnxruntime")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if variant_id == "O0":
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opt_level_name = "ORT_DISABLE_ALL"
    else:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt_level_name = "ORT_ENABLE_ALL"

    # 固定禁止 spinning 避免空载吃满 CPU 资源
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

    if profiling:
        sess_options.enable_profiling = True
        if profile_file_prefix is not None:
            sess_options.profile_file_prefix = str(profile_file_prefix)

    session = ort.InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    metadata = {
        "variant_id": variant_id,
        "threads": threads,
        "inter_op_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization_level": opt_level_name,
        "allow_spinning": "0",
        "providers": ["CPUExecutionProvider"],
        "profiling_enabled": bool(profiling),
        "profile_file_prefix": str(profile_file_prefix) if profile_file_prefix is not None else None,
        "onnx_path": str(path),
    }
    setattr(session, "_benchmark_metadata", metadata)
    return session


def graph_statistics(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    """提取 ONNX 模型的算子分布、节点总数、输入输出形状、opset 版本及 SHA256。"""
    path = Path(onnx_path).resolve()
    if not path.is_file():
        raise ClassificationError(f"ONNX 模型文件不存在: {path}")

    onnx = _require("onnx")
    sha256_val = compute_sha256(path)
    size_bytes = path.stat().st_size

    model = onnx.load(str(path))
    graph = model.graph

    op_counts: Dict[str, int] = {}
    for node in graph.node:
        op = node.op_type
        op_counts[op] = op_counts.get(op, 0) + 1

    def _extract_shape_dtype(val_info: Any) -> Dict[str, Any]:
        t = val_info.type.tensor_type
        shape = []
        for dim in t.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append(None)
        return {"shape": shape, "elem_type": t.elem_type}

    inputs = {inp.name: _extract_shape_dtype(inp) for inp in graph.input}
    outputs = {out.name: _extract_shape_dtype(out) for out in graph.output}
    opset_imports = {item.domain or "ai.onnx": item.version for item in model.opset_import}

    return {
        "sha256": sha256_val,
        "size_bytes": size_bytes,
        "node_count": len(graph.node),
        "op_type_counts": op_counts,
        "inputs": inputs,
        "outputs": outputs,
        "opset_imports": opset_imports,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
    }
