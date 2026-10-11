"""ResNet18 图像分类与优化对照模块。

提供模型加载、预处理、Conv-BN 融合优化、Top-K 预测与输出对齐比对功能。
延迟按需引入重依赖，保证 CLI --help 与环境探针在缺少 torch 时的可用性。
"""

import copy
import hashlib
import os
import platform
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "ResNetError",
    "load_classifier",
    "open_rgb_image",
    "prepare_image",
    "fuse_resnet18",
    "topk_predictions",
    "compare_logits",
    "environment_metadata",
]

OFFICIAL_WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
OFFICIAL_HASH_PREFIX = "f37072fd"
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "models" / "weights" / "resnet18-f37072fd.pth"


from src.vision.classification import ClassificationError


class ResNetError(ClassificationError):
    """ResNet 分类流水线专用异常。"""


def _require(pkg_name: str):
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
        }
        raise ResNetError(f"缺少必要依赖 '{pkg_name}'。请安装: {hints.get(pkg_name, pkg_name)}") from exc


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def environment_metadata() -> Dict[str, Any]:
    """返回当前 Python、OS 平台以及核心依赖版本信息。"""
    def _ver(mod: str) -> Optional[str]:
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _ver("torch"),
        "torchvision": _ver("torchvision"),
        "pillow": _ver("PIL"),
    }


def open_rgb_image(path: Union[str, Path]):
    """打开图像、校正 EXIF 朝向、返回完全解码的独立 RGB PIL.Image 实例。"""
    Image, ImageOps = _require("PIL")
    image_path = Path(path).resolve()
    if not image_path.is_file():
        raise ResNetError(f"图像文件不存在: {image_path}")
    try:
        with Image.open(image_path) as img:
            rgb_image = ImageOps.exif_transpose(img).convert("RGB")
            rgb_image.load()
            return rgb_image
    except Exception as exc:
        if isinstance(exc, ResNetError):
            raise
        raise ResNetError(f"读取并解码图像失败 ({image_path}): {exc}") from exc


def prepare_image(image: Any):
    """使用 torchvision 官方 ResNet18_Weights 变换预处理图像为 [1, 3, 224, 224] Tensor。"""
    torch, torchvision = _require("torch"), _require("torchvision")
    Image, _ = _require("PIL")
    if not isinstance(image, Image.Image):
        raise ResNetError(f"prepare_image 输入必须为 PIL.Image.Image，收到: {type(image)}")
    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        tensor = weights.transforms()(image)
    except Exception as exc:
        raise ResNetError(f"图像预处理转换失败: {exc}") from exc

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(dtype=torch.float32, device="cpu")
    if list(tensor.shape) != [1, 3, 224, 224]:
        raise ResNetError(f"预处理输出维度异常，期望 [1, 3, 224, 224]，实际为 {list(tensor.shape)}")
    return tensor


def _download_official_weights(target_path: Path) -> float:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}")
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(OFFICIAL_WEIGHTS_URL, headers={"User-Agent": "EdgeAI-Robotic-Arm-Control/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, temp_path.open("wb") as out_f:
            while chunk := resp.read(1024 * 1024):
                out_f.write(chunk)
        if not (downloaded_sha := _compute_sha256(temp_path)).startswith(OFFICIAL_HASH_PREFIX):
            raise ResNetError(f"官方权重 SHA256 校验失败: 期望前缀 '{OFFICIAL_HASH_PREFIX}'，实际为 '{downloaded_sha}'")
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
        if isinstance(exc, ResNetError):
            raise
        raise ResNetError(f"下载官方权重失败 ({OFFICIAL_WEIGHTS_URL}): {exc}") from exc


def load_classifier(
    weights_path: Optional[Union[str, Path]] = None,
    *,
    download_weights: bool = False,
    threads: int = 1,
) -> Tuple[Any, Dict[str, Any]]:
    """加载 CPU FP32 ResNet18 模型及相关元数据。"""
    torch, torchvision = _require("torch"), _require("torchvision")
    if not isinstance(threads, int) or threads <= 0:
        raise ResNetError(f"线程数 threads 必须为正整数，收到: {threads}")
    torch.set_num_threads(threads)

    target_path = Path(weights_path if weights_path is not None else DEFAULT_WEIGHTS_PATH).resolve()
    download_ms = 0.0
    if not target_path.is_file():
        if not download_weights:
            raise ResNetError(f"权重文件不存在: {target_path}。若需自动下载官方权重，请显式传入 download_weights=True。")
        download_ms = _download_official_weights(target_path)

    sha256_val = _compute_sha256(target_path)
    matches_official = sha256_val.startswith(OFFICIAL_HASH_PREFIX)
    weights_source = "official_imagenet1k_v1" if matches_official else "local_custom"
    weights_id = "ResNet18_Weights.IMAGENET1K_V1" if matches_official else f"custom_{sha256_val[:8]}"

    try:
        state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ResNetError(f"安全加载权重文件失败 ({target_path}): {exc}") from exc

    if isinstance(state_dict, dict) and "state_dict" in state_dict and not any(k.startswith("conv1") for k in state_dict):
        state_dict = state_dict["state_dict"]

    model = torchvision.models.resnet18()
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ResNetError(f"模型结构权重加载失败 (strict=True): {exc}") from exc

    model.eval().to(dtype=torch.float32, device="cpu")
    weights_spec = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    categories = list(weights_spec.meta["categories"])

    metadata = {
        "architecture": "resnet18", "weights_id": weights_id, "weights_source": weights_source,
        "official_url": OFFICIAL_WEIGHTS_URL, "weights_url": OFFICIAL_WEIGHTS_URL if matches_official else None,
        "weights_path": str(target_path), "weights_sha256": sha256_val,
        "matches_official_hash": matches_official, "categories": categories,
        "categories_note": "官方 ImageNet-1K 类别映射" if matches_official else "自定义权重: ImageNet-1K 映射仅为假设",
        "category_mapping_assumed": not matches_official,
        "preprocess": {
            "layout": "NCHW", "color_space": "RGB", "resize": 256, "center_crop": 224,
            "interpolation": "bilinear", "antialias": True,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "input_shape": [1, 3, 224, 224],
        },
        "device": "cpu", "dtype": "float32", "threads": threads,
        "versions": environment_metadata(), "weights_download_ms": download_ms,
    }
    return model, metadata


def _check_and_fuse_pair(conv: Any, bn: Any, tag: str):
    torch = _require("torch")
    if not isinstance(conv, torch.nn.Conv2d) or not isinstance(bn, torch.nn.BatchNorm2d):
        raise ResNetError(f"{tag} 算子类型异常: conv 须为 Conv2d，bn 须为 BatchNorm2d")
    if bn.running_mean is None or bn.running_var is None:
        raise ResNetError(f"{tag} BatchNorm2d 缺少有效 running stats，无法融合")
    return torch.nn.utils.fuse_conv_bn_eval(conv, bn)


def fuse_resnet18(model: Any) -> Tuple[Any, int]:
    """对 ResNet18 执行 Conv-BN 算子融合，BN 替换为 Identity。返回深拷贝 (fused_model, pair_count)。"""
    torch = _require("torch")
    if getattr(model, "training", False):
        raise ResNetError("模型必须在 eval() 模式下方可执行 Conv-BN 融合")

    if not hasattr(model, "conv1") or not hasattr(model, "bn1"):
        raise ResNetError("模型缺少 ResNet18 stem 结构 (conv1/bn1)")

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        if not hasattr(model, layer_name):
            raise ResNetError(f"模型缺少 ResNet18 阶段层: {layer_name}")
        layer = getattr(model, layer_name)
        if len(layer) != 2 or any(b.__class__.__name__ != "BasicBlock" for b in layer):
            raise ResNetError(f"结构不符合 ResNet18: {layer_name} 须包含恰好 2 个 BasicBlock")

    fused = copy.deepcopy(model)
    fused.conv1 = _check_and_fuse_pair(fused.conv1, fused.bn1, "stem")
    fused.bn1 = torch.nn.Identity()
    pair_count = 1

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        for block_idx, block in enumerate(getattr(fused, layer_name)):
            tag = f"{layer_name}.b{block_idx}"
            block.conv1 = _check_and_fuse_pair(block.conv1, block.bn1, f"{tag}.conv1")
            block.bn1 = torch.nn.Identity()
            block.conv2 = _check_and_fuse_pair(block.conv2, block.bn2, f"{tag}.conv2")
            block.bn2 = torch.nn.Identity()
            pair_count += 2

            if block.downsample is not None:
                if len(block.downsample) < 2:
                    raise ResNetError(f"{tag}.downsample 结构不完整，期望含 [Conv2d, BatchNorm2d]")
                block.downsample[0] = _check_and_fuse_pair(block.downsample[0], block.downsample[1], f"{tag}.down")
                block.downsample[1] = torch.nn.Identity()
                pair_count += 1

    fused.eval().to(dtype=torch.float32, device="cpu")
    return fused, pair_count


def _check_logits_tensor(tensor: Any, name: str, expected_classes: Optional[int] = None) -> None:
    torch = _require("torch")
    if not isinstance(tensor, torch.Tensor):
        raise ResNetError(f"{name} 必须为 torch.Tensor 实例，收到: {type(tensor)}")
    if not torch.is_floating_point(tensor):
        raise ResNetError(f"{name} 必须为浮点类型张量 (float32/float64)，收到: {tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ResNetError(f"{name} 必须位于 CPU 设备，收到: {tensor.device}")
    if not torch.isfinite(tensor).all():
        raise ResNetError(f"{name} 包含非法非有限浮点数 (NaN 或 Inf)")
    if expected_classes is not None and (
        tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] != expected_classes
    ):
        raise ResNetError(f"{name} 形状必须为 [1, {expected_classes}]，实际为 {list(tensor.shape)}")


def topk_predictions(logits: Any, categories: List[str], k: int = 5) -> List[Dict[str, Any]]:
    """针对模型输出 logits 计算稳定 Softmax 概率并返回 Top-K 预测字典列表。"""
    torch = _require("torch")
    num_classes = len(categories)
    _check_logits_tensor(logits, "logits", expected_classes=num_classes)

    if not isinstance(k, int) or k <= 0 or k > num_classes:
        raise ResNetError(f"参数 k 必须为正整数且不超过类别数 [1, {num_classes}]，收到: {k}")

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs[0], k=k)
    return [
        {"index": int(idx), "label": str(categories[idx]), "probability": float(prob)}
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
    ]


def compare_logits(
    reference: Any,
    candidate: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """比对基线与待测模型的输出 logits，检查数值误差与 Top-1/Top-5 一致性。"""
    torch = _require("torch")
    _check_logits_tensor(reference, "reference")
    _check_logits_tensor(candidate, "candidate")

    if reference.shape != candidate.shape:
        raise ResNetError(f"输出维度不一致，reference: {list(reference.shape)}, candidate: {list(candidate.shape)}")

    diff = torch.abs(reference - candidate)
    is_allclose = bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol))
    top1_equal = bool(int(torch.argmax(reference, dim=-1).flatten()[0].item()) == int(torch.argmax(candidate, dim=-1).flatten()[0].item()))

    k_top5 = min(5, reference.shape[-1])
    top5_ref = torch.topk(reference, k=k_top5, dim=-1).indices.flatten().tolist()
    top5_cand = torch.topk(candidate, k=k_top5, dim=-1).indices.flatten().tolist()

    return {
        "allclose": is_allclose,
        "max_abs_error": float(torch.max(diff).item()),
        "mean_abs_error": float(torch.mean(diff).item()),
        "top1_equal": top1_equal,
        "top5_equal": bool(top5_ref == top5_cand),
        "rtol": float(rtol),
        "atol": float(atol),
        "passed": bool(is_allclose and top1_equal),
    }
