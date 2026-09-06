#!/usr/bin/env python3
"""
面向 RK3588 工业视觉检测模型的 INT8 训练后量化 (PTQ) 导出工具
"""
import argparse
import sys
import os

try:
    from rknn.api import RKNN
    HAS_RKNN_TOOLKIT = True
except ImportError:
    HAS_RKNN_TOOLKIT = False


def main():
    parser = argparse.ArgumentParser(description="视觉模型 RKNN INT8 量化与导出工具")
    parser.add_argument("--onnx", default="models/weights/yolov8n.onnx", help="原始 ONNX 静态计算图路径")
    parser.add_argument("--output", default="models/weights/yolov8n_int8.rknn", help="导出的 .rknn 模型路径")
    parser.add_argument("--calib-dir", default="data/calibration/images", help="量化校准图片目录 (50~100张)")
    parser.add_argument("--target-platform", default="rk3588", help="目标硬件平台 (rk3588)")
    args = parser.parse_args()

    print("=" * 60)
    print("  工业视觉检测模型: RK3588 INT8 非对称量化导出")
    print("=" * 60)

    if not HAS_RKNN_TOOLKIT:
        print("[提示] 本机未安装 rknn-toolkit2（通常在 x86 PC Linux 上进行转换）。")
        print("      pip3 install rknn-toolkit2")
        return

    rknn = RKNN(verbose=False)
    # 1. 配置芯片目标与量化预处理参数
    print("[1/4] 配置芯片目标与均值方差归一化...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target_platform,
        quantized_algorithm="kl_divergence", # 采用 KL 散度精细校准
        quantized_method="asymmetric_quantized-8"
    )

    # 2. 载入 ONNX
    print(f"[2/4] 加载 ONNX 模型: {args.onnx} ...")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("[错误] 加载 ONNX 失败")
        sys.exit(1)

    # 3. 构建并量化
    print("[3/4] 执行 INT8 离线校准与算子融合...")
    dataset_txt = "data/calibration/dataset.txt" if os.path.exists("data/calibration/dataset.txt") else None
    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    if ret != 0:
        print("[错误] 量化构建失败")
        sys.exit(1)

    # 4. 导出
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ret = rknn.export_rknn(args.output)
    if ret == 0:
        print(f"[4/4] 导出成功 -> {args.output}")
    rknn.release()


if __name__ == "__main__":
    main()
