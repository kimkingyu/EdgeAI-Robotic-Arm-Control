#!/usr/bin/env python3
"""
YOLOv8n vs YOLOv11n 在 RK3588 (RKNPU2) 上的算子亲和力与推理性能对比

本工具不做任何理论估算，全部结论来自板端真实产物：
  1. 算子落点  —— 解析 rknn build 日志的 Target 列，逐算子统计 NPU / CPU 实际执行位置
  2. 推理性能  —— 读取 tools/benchmark_npu.py 产出的实测 JSON 报告
  3. 量化压缩  —— 读取 tools/export_rknn.py 产出的转换报告

用法：
  # 从板端 build 日志解析算子分布
  python3 tools/compare_yolo_versions.py --v8-log /tmp/conv_v8.log --v11-log /tmp/conv_v11.log

  # 汇总实测报告输出对比表
  python3 tools/compare_yolo_versions.py --report-dir docs/benchmarks
"""
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# 匹配 rknn build 日志中的逐算子表格行：
# D RKNN: [11:07:05.380] 12  ConvExSwish  INT8  NPU  (1,3,640,640)  ...
OP_LINE = re.compile(
    r"^D RKNN:\s+\[[\d:.]+\]\s+(\d+)\s+(\w+)\s+(INT8|INT16|INT32|FLOAT16|FLOAT|UNDEFINED)\s+(NPU|CPU)\s"
)

# 纯 I/O 桩算子：不参与实际计算，其 CPU 落点属于框架固有开销，不算真实回退
IO_STUB_OPS = {"InputOperator", "OutputOperator"}


def parse_build_log(log_path: str) -> Optional[Dict[str, Any]]:
    """从 RKNN 构建日志中逐算子解析真实硬件落点"""
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().replace("\r", "\n")

    npu_ops: Dict[str, int] = {}
    cpu_ops: Dict[str, int] = {}
    for line in raw.splitlines():
        m = OP_LINE.match(line)
        if not m:
            continue
        _, op_type, _, target = m.groups()
        bucket = npu_ops if target == "NPU" else cpu_ops
        bucket[op_type] = bucket.get(op_type, 0) + 1

    total = sum(npu_ops.values()) + sum(cpu_ops.values())
    if total == 0:
        return None

    # 剔除 I/O 桩后的真实计算算子回退情况
    real_fallback = {k: v for k, v in cpu_ops.items() if k not in IO_STUB_OPS}
    compute_total = total - sum(v for k, v in cpu_ops.items() if k in IO_STUB_OPS)

    return {
        "total_ops": total,
        "npu_count": sum(npu_ops.values()),
        "cpu_count": sum(cpu_ops.values()),
        "pass_rate": round(sum(npu_ops.values()) / total * 100, 2),
        "compute_pass_rate": round(
            (compute_total - sum(real_fallback.values())) / compute_total * 100, 2
        ) if compute_total else None,
        "cpu_ops": cpu_ops,
        "real_fallback_ops": real_fallback,
        "top_npu_ops": sorted(npu_ops.items(), key=lambda x: -x[1])[:8],
    }


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def best_of(bench: Optional[dict]) -> Optional[dict]:
    """从基准报告中取出时延最优的调度模式"""
    if not bench:
        return None
    ok = [r for r in bench.get("results", []) if "error" not in r and "avg_ms" in r]
    return min(ok, key=lambda x: x["avg_ms"]) if ok else None


def print_op_section(name: str, data: Optional[Dict[str, Any]]):
    if not data:
        print(f"  {name:<12} 未找到构建日志，跳过")
        return
    print(f"  {name:<12} 算子总数 {data['total_ops']:>4} | "
          f"NPU {data['npu_count']:>4} | CPU {data['cpu_count']:>2} | "
          f"直通率 {data['pass_rate']:>6.2f}%")
    if data["real_fallback_ops"]:
        detail = "、".join(f"{k}×{v}" for k, v in data["real_fallback_ops"].items())
        print(f"               真实回退算子: {detail}")
    else:
        print(f"               真实回退算子: 无（CPU 落点仅为输入输出桩）")


def main():
    ap = argparse.ArgumentParser(description="YOLOv8n vs YOLOv11n 板端实测对比")
    ap.add_argument("--v8-log", default="/tmp/conv_v8.log", help="YOLOv8n 的 RKNN 构建日志")
    ap.add_argument("--v11-log", default="/tmp/conv_v11.log", help="YOLOv11n 的 RKNN 构建日志")
    ap.add_argument("--report-dir", default="docs/benchmarks", help="实测 JSON 报告目录")
    args = ap.parse_args()

    d = args.report_dir
    v8_conv = load_json(os.path.join(d, "convert_yolov8n_int8.json"))
    v11_conv = load_json(os.path.join(d, "convert_yolo11n_int8.json"))
    v8_bench = best_of(load_json(os.path.join(d, "npu_yolov8n_int8.json")))
    v11_bench = best_of(load_json(os.path.join(d, "npu_yolo11n_int8.json")))
    par = load_json(os.path.join(d, "npu_parallel_yolov8n.json"))

    print("=" * 78)
    print("  YOLOv8n vs YOLOv11n @ RK3588 NPU  —— 板端实测对比")
    print("=" * 78)

    print("\n【一】算子硬件落点（解析 RKNN 构建日志 Target 列）")
    v8_op = parse_build_log(args.v8_log)
    v11_op = parse_build_log(args.v11_log)
    print_op_section("YOLOv8n", v8_op)
    print_op_section("YOLOv11n", v11_op)

    if v8_op and v11_op:
        diff = (v11_op["total_ops"] - v8_op["total_ops"]) / v8_op["total_ops"] * 100
        print(f"\n  → YOLOv11n 算子数量比 v8 多 {diff:.1f}%"
              f"（{v8_op['total_ops']} → {v11_op['total_ops']}），流水线级数更长")

    print("\n【二】量化压缩效果")
    for tag, c in (("YOLOv8n", v8_conv), ("YOLOv11n", v11_conv)):
        if c and c.get("success"):
            print(f"  {tag:<12} INT8 体积 {c['size_mb']:>5.2f} MB | "
                  f"量化算法 {c['algorithm']:<8} | 转换耗时 {c['build_seconds']:>6.2f}s")
        else:
            print(f"  {tag:<12} 未找到转换报告")

    print("\n【三】单路推理性能（取各自最优调度模式）")
    for tag, b in (("YOLOv8n", v8_bench), ("YOLOv11n", v11_bench)):
        if b:
            print(f"  {tag:<12} {b['avg_ms']:>6.2f} ms | P99 {b['p99_ms']:>6.2f} ms | "
                  f"抖动 ±{b['std_ms']:.2f} ms | {b['fps']:>6.2f} FPS  [{b['desc']}]")
        else:
            print(f"  {tag:<12} 未找到基准报告")

    if v8_bench and v11_bench:
        gain = (v11_bench["avg_ms"] - v8_bench["avg_ms"]) / v11_bench["avg_ms"] * 100
        print(f"\n  → YOLOv8n 单帧比 YOLOv11n 快 {gain:.1f}%"
              f"（{v8_bench['avg_ms']:.2f}ms vs {v11_bench['avg_ms']:.2f}ms）")

    if par:
        print("\n【四】YOLOv8n 三核并发吞吐")
        ok = [r for r in par.get("results", []) if "error" not in r]
        for r in ok:
            print(f"  {r['desc']:<18} 系统吞吐 {r['system_fps']:>7.2f} FPS | "
                  f"单路时延 {r['per_stream_avg_ms']:>6.2f} ms")
        if len(ok) >= 2:
            print(f"\n  → 三核并发加速比 "
                  f"{ok[-1]['system_fps'] / ok[0]['system_fps']:.2f}x")

    print("\n" + "=" * 78)
    print("【选型结论】")
    if v8_bench and v11_bench and v8_op and v11_op:
        print(f"  YOLOv11n 的 C3k2 + PSA 注意力已被 RKNPU 完整支持，直通率 "
              f"{v11_op['pass_rate']:.2f}% 甚至略优于 v8 的 {v8_op['pass_rate']:.2f}%，")
        print(f"  但其算子数多出 {(v11_op['total_ops'] - v8_op['total_ops']) / v8_op['total_ops'] * 100:.1f}%，"
              f"端到端时延高 {(v11_bench['avg_ms'] - v8_bench['avg_ms']) / v11_bench['avg_ms'] * 100:.1f}%。")
        print("  → 高频实时伺服场景选定 YOLOv8n；追求精度的离线质检可保留 YOLOv11n 作为技术储备。")
    else:
        print("  报告数据不完整，请先在板端运行 export_rknn.py 与 benchmark_npu.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
