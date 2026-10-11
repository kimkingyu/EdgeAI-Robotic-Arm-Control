#!/usr/bin/env python3
"""阶段 B 本地静态量化或带标签评估；不下载权重、不访问网络。

准备 QB/Q1/Q2/Q3 必须达到批准的完整数据门槛。holdout 评估必须提供
已冻结候选文件及预先确定的 SHA256；--help 不导入 torch/ORT 等重依赖。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def _seed(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("seed 必须非负")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"))
    parser.add_argument("--onnx", required=True, type=Path, help="准备时传入原始 FP32 opset17 图")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=_seed, default=20261001)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=("prepare", "evaluate"), default="prepare")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--threads", type=_positive_integer, default=1,
                        help="只供 evaluate；准备阶段固定单线程")
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--expected-frozen-sha256")
    args = parser.parse_args(argv)

    from src.vision.classification_quantization import (
        QuantizationError, evaluate_classifier, prepare_quantized_family,
    )
    try:
        if args.mode == "prepare":
            if (args.threads != 1 or args.split != "development" or args.frozen is not None
                    or args.expected_frozen_sha256 is not None):
                raise QuantizationError("prepare 固定线程1，不允许 holdout/冻结评估选项")
            result = prepare_quantized_family(args.model, args.onnx, args.manifest,
                                               args.output_dir, overwrite=args.overwrite,
                                               seed=args.seed)
        else:
            destination = args.output_dir.resolve() / f"accuracy-{args.split}.json"
            inputs = {args.onnx.resolve(), args.manifest.resolve()}
            if args.frozen is not None:
                inputs.add(args.frozen.resolve())
            if destination in inputs:
                raise QuantizationError("禁止用精度结果覆盖输入模型/数据清单/冻结文件")
            if destination.exists() and not args.overwrite:
                raise QuantizationError("精度输出已存在；默认拒绝覆盖")
            result = evaluate_classifier(args.model, args.onnx, args.manifest,
                                           split=args.split, threads=args.threads,
                                           frozen_path=args.frozen,
                                           expected_frozen_sha256=args.expected_frozen_sha256)
            import os
            import tempfile
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".accuracy-", dir=destination.parent) as temporary:
                staged = Path(temporary) / destination.name
                staged.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                               allow_nan=False) + "\n", encoding="utf-8")
                if destination.exists() and not args.overwrite:
                    raise QuantizationError("发布时精度输出已存在；拒绝覆盖")
                os.replace(staged, destination)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.mode == "prepare" and result["status"] != "prepared":
            return 1
        return 0
    except (QuantizationError, OSError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
