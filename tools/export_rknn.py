#!/usr/bin/env python3
"""
面向 RK3588 工业视觉检测模型的 INT8 训练后量化 (PTQ) 导出工具

支持 YOLOv8n / YOLOv11n 双模型对比量化，产出：
  1. .rknn 量化模型
  2. 算子硬件直通率报告（NPU / CPU 回退算子统计）
"""
import argparse
import json
import os
import re
import sys
import time

try:
    from rknn.api import RKNN
    HAS_RKNN_TOOLKIT = True
except ImportError:
    HAS_RKNN_TOOLKIT = False


def parse_op_distribution(log_text: str) -> dict:
    """从 rknn build 日志中解析算子在 NPU / CPU 上的分布情况"""
    npu_ops, cpu_ops = [], []
    for line in log_text.splitlines():
        low = line.lower()
        m = re.search(r"\b(cpu|npu)\b", low)
        if not m:
            continue
        if "target: cpu" in low or "run on cpu" in low:
            cpu_ops.append(line.strip())
        elif "target: npu" in low:
            npu_ops.append(line.strip())
    total = len(npu_ops) + len(cpu_ops)
    return {
        "npu_op_count": len(npu_ops),
        "cpu_fallback_count": len(cpu_ops),
        "npu_passthrough_rate": round(len(npu_ops) / total * 100, 2) if total else None,
        "cpu_fallback_ops": cpu_ops[:20],
    }


def convert(onnx_path: str, output_path: str, dataset_txt: str,
            target_platform: str, do_quant: bool, algo: str) -> dict:
    """执行单个模型的量化转换，返回结构化结果"""
    tag = os.path.basename(onnx_path)
    print("=" * 68)
    print(f"  模型转换: {tag}  |  量化: {'INT8-PTQ' if do_quant else 'FP16'}")
    print("=" * 68)

    result = {"onnx": onnx_path, "output": output_path,
              "quantized": do_quant, "algorithm": algo if do_quant else "none"}

    rknn = RKNN(verbose=True)

    # 1. 芯片目标与归一化参数（YOLO 输入为 0~1 归一化的 RGB）
    print("[1/4] 配置芯片目标与均值方差归一化 ...")
    cfg = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target_platform,
    )
    if do_quant:
        cfg["quantized_algorithm"] = algo          # normal / mmse / kl_divergence
        cfg["quantized_method"] = "channel"        # 逐通道非对称量化，精度优于逐层
    rknn.config(**cfg)

    # 2. 载入 ONNX 静态计算图
    print(f"[2/4] 加载 ONNX 模型 ...")
    if rknn.load_onnx(model=onnx_path) != 0:
        result["error"] = "load_onnx failed"
        rknn.release()
        return result

    # 3. 离线校准 + 算子融合构建
    print("[3/4] 执行离线校准与算子融合（耗时较长，请耐心等待）...")
    t0 = time.time()
    ds = dataset_txt if (do_quant and os.path.exists(dataset_txt)) else None
    if do_quant and ds is None:
        print(f"[警告] 校准集清单不存在: {dataset_txt}，回退为非量化构建")
        do_quant = False
    if rknn.build(do_quantization=do_quant, dataset=ds) != 0:
        result["error"] = "build failed"
        rknn.release()
        return result
    result["build_seconds"] = round(time.time() - t0, 2)

    # 4. 导出 rknn
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if rknn.export_rknn(output_path) == 0:
        size_mb = round(os.path.getsize(output_path) / 1024 / 1024, 2)
        result["size_mb"] = size_mb
        result["success"] = True
        print(f"[4/4] 导出成功 -> {output_path}  ({size_mb} MB, 耗时 {result['build_seconds']}s)")
    else:
        result["error"] = "export failed"
    rknn.release()
    return result


def main():
    parser = argparse.ArgumentParser(description="视觉模型 RKNN INT8 量化与导出工具")
    parser.add_argument("--onnx", default="models/weights/yolov8n.onnx", help="ONNX 静态计算图路径")
    parser.add_argument("--output", default="", help="导出的 .rknn 路径（留空则自动命名）")
    parser.add_argument("--dataset", default="data/calibration/dataset.txt", help="量化校准集清单")
    parser.add_argument("--target-platform", default="rk3588", help="目标硬件平台")
    parser.add_argument("--algorithm", default="normal",
                        choices=["normal", "mmse", "kl_divergence"], help="量化校准算法")
    parser.add_argument("--fp16", action="store_true", help="不量化，导出 FP16 基线模型")
    parser.add_argument("--report", default="", help="转换结果 JSON 报告输出路径")
    args = parser.parse_args()

    if not HAS_RKNN_TOOLKIT:
        print("[提示] 当前环境未安装 rknn-toolkit2。")
        print("      板端安装：pip install rknn-toolkit2==2.3.2")
        return

    if not os.path.exists(args.onnx):
        print(f"[错误] ONNX 模型不存在: {args.onnx}")
        sys.exit(1)

    out = args.output
    if not out:
        stem = os.path.splitext(os.path.basename(args.onnx))[0].replace("_static", "")
        suffix = "fp16" if args.fp16 else f"int8_{args.algorithm}"
        out = f"models/weights/{stem}_{suffix}.rknn"

    res = convert(args.onnx, out, args.dataset, args.target_platform,
                  do_quant=not args.fp16, algo=args.algorithm)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[报告] 已写入 {args.report}")

    sys.exit(0 if res.get("success") else 1)


if __name__ == "__main__":
    main()
