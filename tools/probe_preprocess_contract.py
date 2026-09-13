#!/usr/bin/env python3
"""Verify real RKNN assets with unchanged uint8 NHWC application inputs.

Loads models and runs a single input-acceptance inference; never initializes a
camera or controller. Metadata describes model tensors, not the application's
input buffer layout or the color/normalization semantics of a vendor model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

MODELS = (
    ("yolo640", "yolov8n_int8.rknn", 640),
    ("qwen448", "qwen3vl4b_vision.rknn", 448),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--profile", choices=[m[0] for m in MODELS], action="append")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {
        "schema_version": 1, "status": "failed", "mock": False,
        "purpose": "application_input_acceptance_not_model_accuracy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(__file__), "python": sys.version,
        "python_executable": sys.executable, "machine": platform.machine(),
        "profiles": [],
    }
    try:
        import cv2
        import numpy as np
        from rknnlite.api import RKNNLite
        if not args.inspector.is_file():
            raise FileNotFoundError("Missing compiled metadata inspector")
        result["header_sha256"] = sha256(args.header)
        result["numpy"] = np.__version__
        result["opencv"] = cv2.__version__
        result["image_sha256"] = sha256(args.image)
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Input image could not be decoded")
        result["image_path"] = str(args.image.resolve())
        result["image_shape"] = list(frame.shape)
        for name, filename, size in MODELS:
            if args.profile and name not in args.profile:
                continue
            model = (args.asset_root / "models" / "weights" / filename).resolve()
            profile = {"name": name, "model_path": str(model), "model_sha256": sha256(model)}
            result["profiles"].append(profile)
            with tempfile.TemporaryDirectory(prefix="edgeai-contract-") as directory:
                metadata_path = Path(directory) / "metadata.json"
                completed = subprocess.run([str(args.inspector.resolve()), str(model), str(metadata_path)],
                                           capture_output=True, text=True, timeout=180)
                profile["metadata_process"] = {"returncode": completed.returncode,
                                               "stdout": completed.stdout, "stderr": completed.stderr}
                if completed.returncode:
                    raise RuntimeError("Metadata query failed for " + name)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                profile["metadata"] = metadata
            inputs = metadata["inputs"]
            if len(inputs) != 1 or inputs[0]["dims"] not in ([1, 3, size, size], [1, size, size, 3]):
                raise ValueError("Unexpected model input dimensions for " + name)
            rgb = cv2.cvtColor(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR),
                               cv2.COLOR_BGR2RGB)
            batch = np.expand_dims(rgb, 0)
            if batch.dtype != np.uint8 or not batch.flags.c_contiguous or not np.shares_memory(rgb, batch):
                raise ValueError("Application input contract not satisfied")
            profile["application_input"] = {
                "dtype": "uint8", "layout": "NHWC", "color": "RGB",
                "shape": list(batch.shape), "batch_is_view": True,
                "external_normalization": "none", "external_transpose": False,
                "interpolation": "opencv.INTER_LINEAR", "byte_sha256": hashlib.sha256(batch.tobytes()).hexdigest(),
            }
            runtime = RKNNLite()
            try:
                if runtime.load_rknn(str(model)) != 0:
                    raise RuntimeError("RKNN model load failed")
                if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
                    raise RuntimeError("Real RKNN initialization failed")
                outputs = runtime.inference(inputs=[batch], data_format=["nhwc"])
                if outputs is None or len(outputs) != len(metadata["outputs"]):
                    raise ValueError("Unexpected output count")
                profile["outputs"] = []
                for output in outputs:
                    value = np.asarray(output)
                    if value.size == 0 or not np.all(np.isfinite(value)):
                        raise ValueError("Empty or nonfinite model output")
                    profile["outputs"].append({
                        "shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                        "min": float(value.min()), "max": float(value.max()),
                    })
                profile["input_accepted_by_real_runtime"] = True
                # Acceptance is not proof of training preprocessing or mAP.
                profile["normalization_semantics"] = (
                    "Project YOLO conversion sets mean=0,std=255; no new external normalization."
                    if name == "yolo640" else
                    "Vendor visual encoder: preserve existing application buffer; internal normalization not independently verified."
                )
                print(name + ": real uint8 NHWC input accepted; outputs=" + str(len(outputs)))
            finally:
                runtime.release()
        if not result["profiles"]:
            raise ValueError("No profiles evaluated")
        result["status"] = "passed"
        write_report(args.report, result)
        return 0
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        write_report(args.report, result)
        print(result["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
