"""三模型（ResNet18 / MobileNetV3-Small / ShuffleNetV2-x1.0）图像分类与优化基础模块。

提供模型规范查询、CPU 线程配置、权重安全加载、预处理、Top-K 预测与数值比对能力。
顶层仅依赖 Python 标准库，重量级依赖（torch / torchvision / PIL / onnx / onnxruntime）按需延迟导入。
"""

import copy
import hashlib
import json
import os
import platform
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "MODEL_IDS",
    "ClassificationError",
    "get_model_spec",
    "configure_cpu_threads",
    "load_classifier",
    "open_rgb_image",
    "prepare_image",
    "compute_sha256",
    "topk_predictions",
    "compare_logits",
    "environment_metadata",
]

MODEL_IDS: Tuple[str, ...] = ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0")

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "models" / "weights"

MODEL_SPECS_INFO: Dict[str, Dict[str, str]] = {
    "resnet18": {
        "architecture": "resnet18",
        "weights_id": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "checkpoint_filename": "resnet18-f37072fd.pth",
        "hash_prefix": "f37072fd",
        "torchvision_model": "resnet18",
        "torchvision_weights": "ResNet18_Weights",
    },
    "mobilenet_v3_small": {
        "architecture": "mobilenet_v3_small",
        "weights_id": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "weights_url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        "checkpoint_filename": "mobilenet_v3_small-047dcff4.pth",
        "hash_prefix": "047dcff4",
        "torchvision_model": "mobilenet_v3_small",
        "torchvision_weights": "MobileNet_V3_Small_Weights",
    },
    "shufflenet_v2_x1_0": {
        "architecture": "shufflenet_v2_x1_0",
        "weights_id": "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1",
        "weights_url": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
        "checkpoint_filename": "shufflenetv2_x1-5666bf0f80.pth",
        "hash_prefix": "5666bf0f80",
        "torchvision_model": "shufflenet_v2_x1_0",
        "torchvision_weights": "ShuffleNet_V2_X1_0_Weights",
    },
}


class ClassificationError(RuntimeError):
    """分类流水线统一异常。"""


def _require(pkg_name: str) -> Any:
    """延迟获取可选重依赖，缺失时返回友好提示。"""
    try:
        if pkg_name == "PIL":
            from PIL import Image, ImageOps
            return Image, ImageOps
        return __import__(pkg_name)
    except ImportError as exc:
        hints = {
            "torch": "pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "torchvision": "pip install torchvision --index-url https://download.pytorch.org/whl/cpu",
            "PIL": "pip install Pillow",
            "numpy": "pip install numpy",
            "onnx": "pip install onnx",
            "onnxruntime": "pip install onnxruntime",
        }
        raise ClassificationError(f"缺少必要依赖 '{pkg_name}'。请安装: {hints.get(pkg_name, pkg_name)}") from exc


def compute_sha256(path: Union[str, Path]) -> str:
    """计算本地文件的 SHA256 十六进制哈希。"""
    target = Path(path).resolve()
    if not target.is_file():
        raise ClassificationError(f"目标文件不存在，无法计算 SHA256: {target}")
    hasher = hashlib.sha256()
    with target.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def environment_metadata() -> Dict[str, Any]:
    """返回当前 Python、OS 平台以及核心依赖版本信息。

    使用 importlib.metadata 读取版本，不导入尚未需要的 heavy backend，避免污染 worker 启动开销与 RSS。
    """
    import importlib.metadata

    def _ver(pkg_name: str) -> Optional[str]:
        dist_map = {"pillow": "Pillow", "pil": "Pillow"}
        dist = dist_map.get(pkg_name.lower(), pkg_name)
        try:
            return importlib.metadata.version(dist)
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _ver("torch"),
        "torchvision": _ver("torchvision"),
        "pillow": _ver("pillow"),
        "numpy": _ver("numpy"),
        "onnx": _ver("onnx"),
        "onnxruntime": _ver("onnxruntime"),
    }


_configured_inter_op_threads: Optional[int] = None


def configure_cpu_threads(threads: int, inter_op_threads: int = 1) -> None:
    """配置 PyTorch CPU 线程数。

    必须对重复调用保持幂等，错误显式报告，严禁在运行后随意重置 inter_op_threads。
    """
    global _configured_inter_op_threads
    if not isinstance(threads, int) or threads <= 0:
        raise ClassificationError(f"线程数 threads 必须为正整数，收到: {threads}")
    if not isinstance(inter_op_threads, int) or inter_op_threads <= 0:
        raise ClassificationError(f"线程数 inter_op_threads 必须为正整数，收到: {inter_op_threads}")

    torch = _require("torch")

    if _configured_inter_op_threads is not None:
        if inter_op_threads != _configured_inter_op_threads:
            raise ClassificationError(
                f"不能运行后任意重置 inter_op_threads: 当前已设定为 {_configured_inter_op_threads}，"
                f"试图修改为 {inter_op_threads}。PyTorch 要求 inter_op 线程数仅允许在并行启动前设定一次。"
            )
    else:
        try:
            curr_inter = torch.get_num_interop_threads()
            if curr_inter != inter_op_threads:
                torch.set_num_interop_threads(inter_op_threads)
            _configured_inter_op_threads = inter_op_threads
        except RuntimeError as exc:
            curr_inter = torch.get_num_interop_threads()
            if curr_inter == inter_op_threads:
                _configured_inter_op_threads = inter_op_threads
            else:
                raise ClassificationError(
                    f"设定 inter_op_threads={inter_op_threads} 失败（当前为 {curr_inter}，可能已启动并行工作）: {exc}"
                ) from exc

    torch.set_num_threads(threads)


def get_model_spec(model_id: str) -> Dict[str, Any]:
    """查询指定模型的官方 V1 规范字典。"""
    if model_id not in MODEL_IDS:
        raise ClassificationError(f"未知模型 ID: '{model_id}'。支持的模型为: {MODEL_IDS}")

    info = MODEL_SPECS_INFO[model_id]
    torchvision = _require("torchvision")

    weights_cls = getattr(torchvision.models, info["torchvision_weights"])
    weights = weights_cls.IMAGENET1K_V1
    categories = list(weights.meta["categories"])

    preprocess = {
        "layout": "NCHW",
        "color_space": "RGB",
        "resize": 256,
        "center_crop": 224,
        "interpolation": "bilinear",
        "antialias": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_shape": [1, 3, 224, 224],
    }

    return {
        "model_id": model_id,
        "architecture": info["architecture"],
        "weights_id": info["weights_id"],
        "weights_url": info["weights_url"],
        "checkpoint_filename": info["checkpoint_filename"],
        "hash_prefix": info["hash_prefix"],
        "categories": categories,
        "preprocess": preprocess,
    }


def open_rgb_image(path: Union[str, Path]):
    """打开图像、校正 EXIF 朝向、返回完全解码并关闭文件句柄的独立 RGB PIL.Image 实例。"""
    Image, ImageOps = _require("PIL")
    image_path = Path(path).resolve()
    if not image_path.is_file():
        raise ClassificationError(f"图像文件不存在: {image_path}")
    try:
        with Image.open(image_path) as img:
            rgb_image = ImageOps.exif_transpose(img).convert("RGB")
            rgb_image.load()
            return rgb_image
    except Exception as exc:
        if isinstance(exc, ClassificationError):
            raise
        raise ClassificationError(f"读取并解码图像失败 ({image_path}): {exc}") from exc


def prepare_image(image: Any, model_id: str = "resnet18") -> Any:
    """使用 torchvision 官方 ImageNet-1K V1 权重变换预处理图像为 [1, 3, 224, 224] FP32 Tensor。"""
    torch, torchvision = _require("torch"), _require("torchvision")
    Image, _ = _require("PIL")
    if not isinstance(image, Image.Image):
        raise ClassificationError(f"prepare_image 输入必须为 PIL.Image.Image，收到: {type(image)}")
    if model_id not in MODEL_IDS:
        raise ClassificationError(f"未知模型 ID: '{model_id}'。支持的模型为: {MODEL_IDS}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    info = MODEL_SPECS_INFO[model_id]
    weights_cls = getattr(torchvision.models, info["torchvision_weights"])
    weights = weights_cls.IMAGENET1K_V1

    try:
        tensor = weights.transforms()(image)
    except Exception as exc:
        raise ClassificationError(f"图像预处理转换失败 ({model_id}): {exc}") from exc

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(dtype=torch.float32, device="cpu")
    if list(tensor.shape) != [1, 3, 224, 224]:
        raise ClassificationError(f"预处理输出维度异常，期望 [1, 3, 224, 224]，实际为 {list(tensor.shape)}")
    return tensor


def _download_official_weights(url: str, target_path: Path, expected_hash_prefix: str) -> float:
    """安全下载官方权重至临时文件，校验哈希前缀后原子发布。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}")
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EdgeAI-Robotic-Arm-Control/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, temp_path.open("wb") as out_f:
            while chunk := resp.read(1024 * 1024):
                out_f.write(chunk)
        downloaded_sha = compute_sha256(temp_path)
        if not downloaded_sha.startswith(expected_hash_prefix):
            raise ClassificationError(
                f"官方权重 SHA256 校验失败: 期望前缀 '{expected_hash_prefix}'，实际为 '{downloaded_sha}'"
            )
        if target_path.exists():
            temp_path.unlink(missing_ok=True)
        else:
            temp_path.replace(target_path)
        return (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        if isinstance(exc, ClassificationError):
            raise
        raise ClassificationError(f"下载官方权重失败 ({url}): {exc}") from exc


def load_classifier(
    model_id: str = "resnet18",
    weights_path: Optional[Union[str, Path]] = None,
    *,
    download_weights: bool = False,
    threads: int = 1,
) -> Tuple[Any, Dict[str, Any]]:
    """加载 CPU FP32 图像分类模型及完整元数据。"""
    if model_id not in MODEL_IDS:
        raise ClassificationError(f"未知模型 ID: '{model_id}'。支持的模型为: {MODEL_IDS}")

    configure_cpu_threads(threads)

    torch, torchvision = _require("torch"), _require("torchvision")
    spec = get_model_spec(model_id)
    info = MODEL_SPECS_INFO[model_id]

    default_path = DEFAULT_WEIGHTS_DIR / spec["checkpoint_filename"]
    target_path = Path(weights_path if weights_path is not None else default_path).resolve()

    download_ms = 0.0
    if not target_path.is_file():
        if not download_weights:
            raise ClassificationError(
                f"权重文件不存在: {target_path}。若需自动下载官方权重，请显式传入 download_weights=True。"
            )
        download_ms = _download_official_weights(spec["weights_url"], target_path, spec["hash_prefix"])

    sha256_val = compute_sha256(target_path)
    matches_official = sha256_val.startswith(spec["hash_prefix"])
    weights_source = "official_imagenet1k_v1" if matches_official else "local_custom"
    weights_id = spec["weights_id"] if matches_official else f"custom_{sha256_val[:8]}"

    try:
        state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ClassificationError(f"安全加载权重文件失败 ({target_path}): {exc}") from exc

    # 解包包装在 state_dict 字典内部的权重
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        # 如果顶层没有卷积/特征权重键，则展开
        if not any(k.startswith(("conv", "features", "layer")) for k in state_dict):
            state_dict = state_dict["state_dict"]

    model_builder = getattr(torchvision.models, info["torchvision_model"])
    model = model_builder()

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ClassificationError(f"模型结构权重加载失败 (strict=True): {exc}") from exc

    model.eval().to(dtype=torch.float32, device="cpu")

    metadata = {
        "model_id": model_id,
        "architecture": spec["architecture"],
        "weights_id": weights_id,
        "weights_source": weights_source,
        "weights_path": str(target_path),
        "weights_sha256": sha256_val,
        "matches_official_hash": matches_official,
        "categories": spec["categories"],
        "categories_note": "官方 ImageNet-1K 类别映射" if matches_official else "自定义权重: ImageNet-1K 映射仅为假设",
        "category_mapping_assumed": not matches_official,
        "preprocess": spec["preprocess"],
        "input_shape": [1, 3, 224, 224],
        "device": "cpu",
        "dtype": "float32",
        "threads": threads,
        "versions": environment_metadata(),
        "weights_download_ms": download_ms,
        "official_url": spec["weights_url"],
        "weights_url": spec["weights_url"] if matches_official else None,
    }
    return model, metadata


def _check_and_normalize_logits(logits: Any, name: str, expected_classes: Optional[int] = None) -> Any:
    """校验 logits 张量或数组的维度、类型和有限性，并返回一维浮点数组。"""
    np = _require("numpy")

    if hasattr(logits, "detach"):
        # torch.Tensor 处理
        if logits.device.type != "cpu":
            raise ClassificationError(f"{name} 必须位于 CPU 设备，收到: {logits.device}")
        if not logits.is_floating_point():
            raise ClassificationError(f"{name} 必须为浮点类型张量，收到: {logits.dtype}")
        arr = logits.detach().cpu().numpy()
    elif isinstance(logits, np.ndarray):
        if not np.issubdtype(logits.dtype, np.floating):
            raise ClassificationError(f"{name} 必须为浮点类型数组，收到: {logits.dtype}")
        arr = logits
    else:
        raise ClassificationError(f"{name} 必须为 torch.Tensor 或 numpy.ndarray，收到: {type(logits)}")

    if not np.isfinite(arr).all():
        raise ClassificationError(f"{name} 包含非法非有限浮点数 (NaN 或 Inf)")

    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ClassificationError(f"{name} 形状必须为 [1, C]，实际为 {list(arr.shape)}")

    if expected_classes is not None and arr.shape[1] != expected_classes:
        raise ClassificationError(f"{name} 形状必须为 [1, {expected_classes}]，实际为 {list(arr.shape)}")

    return arr


def topk_predictions(logits: Any, categories: Sequence[str], k: int = 5) -> List[Dict[str, Any]]:
    """针对模型输出 logits 计算稳定 Softmax 概率并返回 Top-K 预测列表。

    原生支持 numpy.ndarray 与 torch.Tensor，不强制 ORT worker 导入 torch。
    """
    np = _require("numpy")
    num_classes = len(categories)
    arr = _check_and_normalize_logits(logits, "logits", expected_classes=num_classes)

    if not isinstance(k, int) or k <= 0 or k > num_classes:
        raise ClassificationError(f"参数 k 必须为正整数且不超过类别数 [1, {num_classes}]，收到: {k}")

    row = arr[0]
    shift_x = row - np.max(row)
    exps = np.exp(shift_x)
    probs = exps / np.sum(exps)

    top_indices = np.argsort(probs)[::-1][:k]
    top_probs = probs[top_indices]

    result = [
        {"index": int(idx), "label": str(categories[idx]), "probability": float(prob)}
        for idx, prob in zip(top_indices, top_probs)
    ]
    # 确保可安全序列化
    json.dumps(result, allow_nan=False)
    return result


def compare_logits(
    reference: Any,
    candidate: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """比对基线与待测模型的输出 logits，检查数值误差与 Top-1/Top-5 一致性。"""
    np = _require("numpy")
    ref_arr = _check_and_normalize_logits(reference, "reference")
    cand_arr = _check_and_normalize_logits(candidate, "candidate")

    if ref_arr.shape != cand_arr.shape:
        raise ClassificationError(f"输出维度不一致，reference: {list(ref_arr.shape)}, candidate: {list(cand_arr.shape)}")

    diff = np.abs(ref_arr - cand_arr)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    is_allclose = bool(np.allclose(ref_arr, cand_arr, rtol=rtol, atol=atol))

    top1_ref = int(np.argmax(ref_arr, axis=-1).flat[0])
    top1_cand = int(np.argmax(cand_arr, axis=-1).flat[0])
    top1_equal = bool(top1_ref == top1_cand)

    k_top5 = min(5, ref_arr.shape[-1])
    top5_ref = np.argsort(ref_arr[0])[::-1][:k_top5].tolist()
    top5_cand = np.argsort(cand_arr[0])[::-1][:k_top5].tolist()
    top5_equal = bool(top5_ref == top5_cand)

    # passed 以数值为准并记录排名差异
    passed = bool(is_allclose and top1_equal)

    return {
        "allclose": is_allclose,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "top1_equal": top1_equal,
        "top5_equal": top5_equal,
        "top1_ref": top1_ref,
        "top1_cand": top1_cand,
        "top5_ref": top5_ref,
        "top5_cand": top5_cand,
        "rank_difference": not top5_equal,
        "rtol": float(rtol),
        "atol": float(atol),
        "passed": passed,
    }
