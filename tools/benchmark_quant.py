#!/usr/bin/env python3
"""
RK3588 端侧 AI 系统性能汇总报告

本工具只做一件事：汇总**已实测产出**的基准报告。
它不自己造数、不估算、不填充占位值 —— 没有实测报告就明确标注"未实测"，
而不是打印一个看起来很漂亮的数字。

数据来源：
  视觉  ← docs/benchmarks/npu_*.json        (由 tools/benchmark_npu*.py 产出)
  大模型 ← docs/benchmarks/llm_*.json        (由 tools/benchmark_rkllm.py 产出，尚未有真实数据)
"""
import argparse
import json
import os
from typing import Any, Dict, Optional


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def best_latency(bench: Optional[dict]) -> Optional[dict]:
    if not bench:
        return None
    ok = [r for r in bench.get("results", []) if "error" not in r and "avg_ms" in r]
    return min(ok, key=lambda x: x["avg_ms"]) if ok else None


def report_vision(d: str):
    print("\n### 1. 视觉检测引擎 (RKNN INT8)")

    v8_conv = load_json(os.path.join(d, "convert_yolov8n_int8.json"))
    v8_best = best_latency(load_json(os.path.join(d, "npu_yolov8n_int8.json")))
    par = load_json(os.path.join(d, "npu_parallel_yolov8n.json"))

    if not (v8_conv or v8_best or par):
        print("  未找到实测报告。请先运行：")
        print("    tools/export_rknn.py  →  tools/benchmark_npu.py  →  tools/benchmark_npu_parallel.py")
        return

    print("| 评估维度 | 实测值 |")
    print("| :--- | :--- |")
    if v8_conv and v8_conv.get("success"):
        print(f"| 模型 / 量化 | YOLOv8n / INT8 逐通道非对称 PTQ |")
        print(f"| 量化后体积 | **{v8_conv['size_mb']} MB** |")
    if v8_best:
        print(f"| 单帧推理耗时 | **{v8_best['avg_ms']} ms** ({v8_best['desc']}) |")
        print(f"| P99 时延 | {v8_best['p99_ms']} ms |")
        print(f"| 时延抖动 (σ) | ±{v8_best['std_ms']} ms |")
        print(f"| 单路吞吐 | **{v8_best['fps']} FPS** |")
    if par:
        ok = [r for r in par.get("results", []) if "error" not in r]
        if ok:
            best = max(ok, key=lambda x: x["system_fps"])
            print(f"| 三核并发系统吞吐 | **{best['system_fps']} FPS** ({best['desc']}) |")
            if len(ok) >= 2:
                print(f"| 并发加速比 | **{best['system_fps'] / ok[0]['system_fps']:.2f}x** |")
    print("\n> mAP@0.5 精度损失：**尚未实测**（需真实工件数据集，待相机到货后补齐）")


def report_llm(d: str):
    print("\n### 2. 端侧大模型 (Qwen2.5 / RKLLM)")

    llm = load_json(os.path.join(d, "llm_qwen_benchmark.json"))
    if not llm:
        print("  **尚未实测**。")
        print("  受阻原因：rkllm-toolkit 官方仅发布 linux_x86_64 wheel，无 aarch64 版本，")
        print("            香橙派无法在板端完成量化；且板端运行时为 v1.0.1，早于 Qwen2.5 发布。")
        print("  详见 docs/QWEN_DEPLOYMENT_BLOCKER.md")
        return

    print("| 评估维度 | 实测值 |")
    print("| :--- | :--- |")
    for k, label in (("model", "模型"), ("precision", "量化规格"),
                     ("ttft_ms", "首字时延 TTFT (ms)"), ("decode_tps", "生成吞吐 (Tokens/s)"),
                     ("mem_footprint_mb", "内存驻留 (MB)")):
        if llm.get(k) is not None:
            print(f"| {label} | **{llm[k]}** |")


def main():
    ap = argparse.ArgumentParser(description="RK3588 端侧 AI 性能汇总报告")
    ap.add_argument("--report-dir", default="docs/benchmarks", help="实测报告目录")
    args = ap.parse_args()

    print("=" * 70)
    print("  工业端侧 AI 系统: RK3588 (6 TOPS NPU) 性能汇总")
    print("=" * 70)
    print("  说明：本报告仅汇总实测数据，未实测项会明确标注，不做任何估算填充。")

    report_vision(args.report_dir)
    report_llm(args.report_dir)

    print("\n" + "=" * 70)
    print("完整分析见 docs/benchmarks/NPU_BENCHMARK_REPORT.md")
    print("复现步骤见 docs/NPU_REPRODUCTION_GUIDE.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
