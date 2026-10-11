#!/usr/bin/env python3
"""Export ResNet18 FP32 and verify the actual ONNX artifact on a real image.

The ONNX input is already-normalized NCHW float32. Normalization is NOT embedded.
This does not build an RKNN model and does not measure NPU performance.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def export_and_verify(image_path, output_path, *, weights_path=None, download_weights=False,
                      threads=1, overwrite=False):
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(".json")
    image_path = Path(image_path)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("--output must end in .onnx")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise ValueError("Output exists; choose another --output or explicitly --overwrite")
    if not image_path.is_file():
        raise ValueError("Image file not found: %s" % image_path)
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional requirements-resnet.txt in the demo environment") from exc
    from src.vision.resnet_classifier import (
        load_classifier, open_rgb_image, prepare_image, compare_logits, topk_predictions,
    )
    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
        raise RuntimeError("This PyTorch exporter cannot explicitly select dynamo=False")
    model, model_metadata = load_classifier(weights_path, download_weights=download_weights,
                                            threads=threads)
    image = open_rgb_image(image_path)
    tensor = prepare_image(image)
    with torch.inference_mode():
        reference = model(tensor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Validate the staged artifact before replacing either public output.
    with tempfile.TemporaryDirectory(prefix=".resnet-export-", dir=output_path.parent) as temporary:
        staged = Path(temporary) / "model.onnx"
        started = time.perf_counter_ns()
        with torch.inference_mode():
            torch.onnx.export(model, (tensor,), str(staged), opset_version=17, dynamo=False,
                              external_data=False, export_params=True,
                              input_names=["images"], output_names=["logits"],
                              dynamic_axes=None, do_constant_folding=True)
        export_ms = (time.perf_counter_ns() - started) / 1e6
        graph = onnx.load(str(staged))
        onnx.checker.check_model(graph, full_check=True)
        if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
            raise RuntimeError("Expected one input and one output")
        input_info = graph.graph.input[0].type.tensor_type
        output_info = graph.graph.output[0].type.tensor_type
        input_shape = [d.dim_value for d in input_info.shape.dim]
        output_shape = [d.dim_value for d in output_info.shape.dim]
        if (input_shape != [1, 3, 224, 224] or output_shape != [1, 1000]
                or input_info.elem_type != onnx.TensorProto.FLOAT
                or output_info.elem_type != onnx.TensorProto.FLOAT):
            raise RuntimeError("Unexpected exported shape or dtype")
        imports = {item.domain: item.version for item in graph.opset_import}
        if imports.get("") != 17:
            raise RuntimeError("Exporter did not retain opset 17: %s" % imports)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(staged), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        result = session.run(["logits"], {"images": tensor.numpy()})[0]
        if result.dtype != np.float32:
            raise RuntimeError("ONNX Runtime returned a non-FP32 output")
        actual = torch.from_numpy(result)
        validation = compare_logits(reference, actual, rtol=1e-4, atol=1e-4)
        if not validation["passed"]:
            raise RuntimeError("ONNX consistency failed: %s" % json.dumps(validation))
        manifest = {
            "schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
            "status": "verified_on_one_image", "model": model_metadata,
            "onnx": {"path": str(output_path.resolve()), "opset": 17, "dynamo": False,
                     "external_data": False, "sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                     "size_bytes": staged.stat().st_size, "export_ms": export_ms,
                     "input_name": "images", "input_shape": input_shape, "input_dtype": "float32",
                     "output_name": "logits", "output_shape": output_shape, "output_dtype": "float32",
                     "normalization_in_graph": False},
            "validation": {**validation, "provider": "CPUExecutionProvider",
                           "onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__,
                           "image_path": str(image_path.resolve()),
                           "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                           "pytorch_top5": topk_predictions(reference, model_metadata["categories"]),
                           "onnx_top5": topk_predictions(actual, model_metadata["categories"])},
            "limitations": ["One image numerical check is not dataset accuracy evaluation.",
                            "No RKNN conversion, INT8 calibration, or NPU benchmark performed.",
                            "Graph input is normalized; future RKNN conversion must avoid double normalization."],
        }
        staged_manifest = Path(temporary) / "model.json"
        staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                              allow_nan=False) + "\n", encoding="utf-8")
        # Recheck after the potentially long export so another run is not silently overwritten.
        if not overwrite and (output_path.exists() or manifest_path.exists()):
            raise ValueError("Output appeared during export; refusing to overwrite it")
        del session
        os.replace(staged, output_path)
        os.replace(staged_manifest, manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="实际用于 PyTorch/ONNX 对照的图片")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "models/weights/resnet18-fp32.onnx")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = export_and_verify(args.image, args.output, weights_path=args.weights,
                                     download_weights=args.download_weights, threads=args.threads,
                                     overwrite=args.overwrite)
        print("ONNX: %s" % args.output)
        print("Metadata: %s" % args.output.with_suffix(".json"))
        print("Verified CPU output: max_abs_error=%.8g, top1_equal=%s" %
              (manifest["validation"]["max_abs_error"], manifest["validation"]["top1_equal"]))
        return 0
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print("[ResNet export error] %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
