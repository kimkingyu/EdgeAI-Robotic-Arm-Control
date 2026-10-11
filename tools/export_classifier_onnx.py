#!/usr/bin/env python3
"""三模型（ResNet18 / MobileNetV3-Small / ShuffleNetV2-x1.0）通用 ONNX FP32 导出与验证工具。

ONNX 输入为已归一化的 NCHW float32 [1, 3, 224, 224]，输出为 [1, 1000] float32。
静态 opset 17，禁用 dynamo，禁用 external_data。
对传入的所有真实图片，先执行 PyTorch 参考推理，再用 ONNX Runtime CPUExecutionProvider 逐图比对校验。
全部校验通过后原子发布模型与 JSON 元数据，默认拒绝覆盖已有输出。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def positive_int(text: str) -> int:
    """验证输入为正整数。"""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def export_and_verify(
    model_id: str,
    image_paths: Union[str, Path, Sequence[Union[str, Path]]],
    output_path: Union[str, Path],
    *,
    weights_path: Optional[Union[str, Path]] = None,
    download_weights: bool = False,
    threads: int = 1,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """导出指定模型并用一张或多张真实图片严格校验。

    临时导出校验通过后才发布到目标路径；默认拒绝覆盖。
    """
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_suffix(".json")

    if output_path.suffix.lower() != ".onnx":
        raise ValueError("--output must end in .onnx")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise ValueError("Output exists; choose another --output or explicitly --overwrite")

    if isinstance(image_paths, (str, Path)):
        images_list = [Path(image_paths).resolve()]
    else:
        images_list = [Path(p).resolve() for p in image_paths]

    if not images_list:
        raise ValueError("至少需要提供一张用于校验的图片")

    for p in images_list:
        if not p.is_file():
            raise ValueError("Image file not found: %s" % p)

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional requirements-resnet.txt in the demo environment") from exc

    from src.vision.classification import (
        MODEL_IDS,
        ClassificationError,
        load_classifier,
        open_rgb_image,
        prepare_image,
        compare_logits,
        topk_predictions,
        compute_sha256,
    )
    from src.vision.classification_variants import (
        create_ort_session,
        graph_statistics,
    )

    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model_id: '{model_id}', expected one of {MODEL_IDS}")

    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
        raise RuntimeError("This PyTorch exporter cannot explicitly select dynamo=False")

    model, model_metadata = load_classifier(
        model_id=model_id,
        weights_path=weights_path,
        download_weights=download_weights,
        threads=threads,
    )

    # 预先处理所有测试图片并获取 PyTorch 参考输出
    prepared_samples = []
    for img_p in images_list:
        image = open_rgb_image(img_p)
        tensor = prepare_image(image, model_id=model_id)
        with torch.inference_mode():
            reference = model(tensor)
        prepared_samples.append((img_p, tensor, reference))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{model_id}-export-", dir=output_path.parent) as temporary:
        staged = Path(temporary) / "model.onnx"
        first_tensor = prepared_samples[0][1]

        started = time.perf_counter_ns()
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (first_tensor,),
                str(staged),
                opset_version=17,
                dynamo=False,
                external_data=False,
                export_params=True,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes=None,
                do_constant_folding=True,
            )
        export_ms = (time.perf_counter_ns() - started) / 1e6

        graph = onnx.load(str(staged))
        onnx.checker.check_model(graph, full_check=True)

        if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
            raise RuntimeError("Expected one input and one output")

        input_info = graph.graph.input[0].type.tensor_type
        output_info = graph.graph.output[0].type.tensor_type
        input_shape = [d.dim_value for d in input_info.shape.dim]
        output_shape = [d.dim_value for d in output_info.shape.dim]

        if (
            input_shape != [1, 3, 224, 224]
            or output_shape != [1, 1000]
            or input_info.elem_type != onnx.TensorProto.FLOAT
            or output_info.elem_type != onnx.TensorProto.FLOAT
        ):
            raise RuntimeError("Unexpected exported shape or dtype")

        imports = {item.domain: item.version for item in graph.opset_import}
        if imports.get("") != 17:
            raise RuntimeError("Exporter did not retain opset 17: %s" % imports)

        # 收集图统计结构
        graph_stats = graph_statistics(staged)

        # 创建受控 ORT 会话验证
        session = create_ort_session(staged, threads=threads, variant_id="O1")

        validations = []
        for img_p, tensor, reference in prepared_samples:
            result = session.run(["logits"], {"images": tensor.numpy()})[0]
            if result.dtype != np.float32:
                raise RuntimeError("ONNX Runtime returned a non-FP32 output")

            validation = compare_logits(reference, result, rtol=1e-4, atol=1e-4)
            if not validation["passed"]:
                raise RuntimeError(
                    f"ONNX consistency failed for {img_p}: {json.dumps(validation)}"
                )

            validations.append({
                "image_path": str(img_p),
                "image_sha256": compute_sha256(img_p),
                "comparison": validation,
                "pytorch_top5": topk_predictions(reference, model_metadata["categories"]),
                "onnx_top5": topk_predictions(result, model_metadata["categories"]),
            })

        staged_sha256 = compute_sha256(staged)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "verified",
            "model_id": model_id,
            "model": model_metadata,
            "onnx": {
                "path": str(output_path),
                "opset": 17,
                "dynamo": False,
                "external_data": False,
                "sha256": staged_sha256,
                "size_bytes": staged.stat().st_size,
                "export_ms": export_ms,
                "input_name": "images",
                "input_shape": input_shape,
                "input_dtype": "float32",
                "output_name": "logits",
                "output_shape": output_shape,
                "output_dtype": "float32",
                "normalization_in_graph": False,
                "graph_statistics": graph_stats,
            },
            "validation": {
                "verified_images_count": len(validations),
                "provider": "CPUExecutionProvider",
                "onnx_version": onnx.__version__,
                "onnxruntime_version": ort.__version__,
                "results": validations,
            },
            "limitations": [
                "Numerical checks on verified images are not a substitute for dataset accuracy evaluation.",
                "No RKNN conversion, INT8 calibration, or NPU benchmark performed.",
                "Graph input is normalized; future RKNN conversion must avoid double normalization.",
            ],
        }

        staged_manifest = Path(temporary) / "model.json"
        staged_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        # 释放 ORT session 对 staged 文件的句柄占用
        del session

        # 再次检查，防止导出期间被外部写入覆盖
        if not overwrite and (output_path.exists() or manifest_path.exists()):
            raise ValueError("Output exists; choose another --output or explicitly --overwrite")

        # 使用 os.replace 执行原子发布，不执行破坏原子性的预先 unlink
        os.replace(staged, output_path)
        os.replace(staged_manifest, manifest_path)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。不在此处引入第三方重库，保证 --help 零依赖。"""
    parser = argparse.ArgumentParser(
        description="Export a verified FP32 ONNX classification model on CPU (ResNet18, MobileNetV3-Small, ShuffleNetV2)."
    )
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"],
        help="Model architecture to export (default: resnet18).",
    )
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        required=False,
        help="Path to verification image file. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for .onnx output (also writes matching .json manifest).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional local weights file path.",
    )
    parser.add_argument(
        "--threads",
        type=positive_int,
        default=1,
        help="CPU thread count (positive integer, default: 1).",
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Explicitly permit downloading official pretrained weights if not found locally.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output .onnx and .json files if present.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.images:
        print("错误: 必须通过 --image 指定至少一张测试图片", file=sys.stderr)
        sys.exit(1)

    try:
        manifest = export_and_verify(
            model_id=args.model,
            image_paths=args.images,
            output_path=args.output,
            weights_path=args.weights,
            download_weights=args.download_weights,
            threads=args.threads,
            overwrite=args.overwrite,
        )
        print("ONNX 导出与验证成功:")
        print(f"  模型: {manifest['model_id']}")
        print(f"  路径: {manifest['onnx']['path']}")
        print(f"  SHA256: {manifest['onnx']['sha256']}")
        print(f"  验证图片数: {manifest['validation']['verified_images_count']}")
    except Exception as exc:
        print(f"导出失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
