#!/usr/bin/env python3
"""
Model Quantization & Conversion Tool for RK3588 (rknn-toolkit2)
Author: kimkingyu
"""

import sys
import argparse
from rknn.api import RKNN

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX model to RKNN with INT8 Quantization")
    parser.add_argument("--onnx", type=str, default="yolov8n.onnx", help="Path to input ONNX model")
    parser.add_argument("--output", type=str, default="yolov8n_int8.rknn", help="Path to output RKNN model")
    parser.add_argument("--target", type=str, default="rk3588", help="Target platform (rk3588, rk3566, etc.)")
    parser.add_argument("--dataset", type=str, default="dataset.txt", help="Calibration dataset for INT8 PTQ")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[RKNN] Starting conversion for {args.onnx} -> {args.output} on target {args.target}")

    rknn = RKNN(verbose=True)

    # 1. Config target platform and quantization algorithm
    print("[RKNN] Configuring target and mean/std...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
        quantized_algorithm='normal',
        optimization_level=3
    )

    # 2. Load ONNX model
    print(f"[RKNN] Loading ONNX model from {args.onnx}...")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("[RKNN] Load ONNX failed!")
        sys.exit(ret)

    # 3. Build RKNN Model with INT8 Post-Training Quantization (PTQ)
    print(f"[RKNN] Building RKNN model (do_quantization=True, dataset={args.dataset})...")
    ret = rknn.build(do_quantization=True, dataset=args.dataset)
    if ret != 0:
        print("[RKNN] Build model failed!")
        sys.exit(ret)

    # 4. Export RKNN Model
    print(f"[RKNN] Exporting to {args.output}...")
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print("[RKNN] Export RKNN failed!")
        sys.exit(ret)

    print("[RKNN] Model successfully exported and optimized for RK3588 NPU!")
    rknn.release()

if __name__ == "__main__":
    main()
