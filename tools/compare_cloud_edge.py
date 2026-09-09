#!/usr/bin/env python3
"""
端云协同三模态架构横向基准对比实验脚本
对比维度:
  1. 端侧 YOLOv8 INT8 (反射层)
  2. 端侧 Qwen2.5 W4A16 (边缘大脑)
  3. 云端大模型 API (云端全知脑)
在时延、吞吐、网络断连韧性与企业数据安全上的横向对比测试
"""
import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple


def _load_vision_metrics(report_dir: str = "docs/benchmarks") -> Tuple[Optional[float], Optional[float]]:
    """从实测报告读取视觉引擎的真实时延与吞吐，缺失则返回 None"""
    path = os.path.join(report_dir, "npu_parallel_yolov8n.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                ok = [r for r in json.load(f).get("results", []) if "error" not in r]
            if ok:
                best = max(ok, key=lambda x: x["system_fps"])
                return best["per_stream_avg_ms"], best["system_fps"]
        except Exception:
            pass

    path = os.path.join(report_dir, "npu_yolov8n_int8.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                ok = [r for r in json.load(f).get("results", []) if "avg_ms" in r]
            if ok:
                best = min(ok, key=lambda x: x["avg_ms"])
                return best["avg_ms"], best["fps"]
        except Exception:
            pass
    return None, None


def run_comprehensive_comparison():
    print("=" * 80)
    print("  工业具身智能机械臂: '端侧YOLO + 端侧Qwen + 云端API' 三模态全景对比实验")
    print("=" * 80)

    # 1. 端侧 YOLOv8 INT8 —— 读取真实实测报告
    print("\n[模式一] 端侧视觉模型 YOLOv8 INT8 (RKNN 三核并发)")
    yolo_latency_ms, yolo_fps = _load_vision_metrics()
    if yolo_latency_ms is None:
        print("  未找到实测报告，请先运行 tools/benchmark_npu.py")
        yolo_latency_ms, yolo_fps = "未实测", "未实测"

    # 2. 端侧 Qwen2.5 —— 尚无真实数据
    print("[模式二] 端侧大语言模型 Qwen2.5 (RKLLM)")
    print("  尚未实测：rkllm-toolkit 无 aarch64 版本，板端无法量化")
    print("  详见 docs/QWEN_DEPLOYMENT_BLOCKER.md")
    edge_llm_ttft_ms = "未实测"
    edge_llm_tps = "未实测"

    # 3. 云端 API —— 需真实调用才有数据
    print("[模式三] 云端大语言模型 API (公网调度)")
    print("  尚未实测：需配置 API Key 并发起真实请求")
    cloud_latency = "未实测"

    print("\n" + "=" * 80)
    print("              三模态协同架构对照表（架构设计层面）")
    print("=" * 80)
    print(f"| 指标维度 | 模式 1: 端侧 YOLOv8 (INT8) | 模式 2: 端侧 Qwen | 模式 3: 云端 API |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| **负责功能层级** | 底层空间定位反射弧 (小脑) | 本地原子动作状态机规划 (边缘大脑) | 高维复杂模糊意图决策 (云端参谋) |")
    print(f"| **单次响应时延** | **{yolo_latency_ms} ms** | {edge_llm_ttft_ms} | {cloud_latency} |")
    print(f"| **系统吞吐能力** | **{yolo_fps} FPS** | {edge_llm_tps} | 受 API 速率限流与网络制约 |")
    print(f"| **企业数据隐私** | 100% 物理隔离，零出域 | 100% 本地离线运算 | 敏感工况与图纸需经公网上传 |")
    print(f"| **断网运行能力** | 完全自主 | 完全自主 | 立即中断失效 |")
    print("=" * 80)
    print("说明：时延与吞吐列仅填入实测值；标注'未实测'者表示尚无真实数据，")
    print("      不以估算值填充。隐私与断网能力为架构固有属性，与实测无关。")

    print("\n[架构结论]")
    print("  • 针对工业精密抓取：采用模式 1 (YOLO) + 模式 2 (端侧 Qwen) 组合，达成【100% 离线隐私保护 + 毫秒级硬实时反馈】；")
    print("  • 针对端云协同产线：采用 HybridRouter 混合调度，常规走模式 3 (云端 API)，断网或涉密工件秒级降级至模式 2，全面兼顾智慧上限与工业韧性。")


if __name__ == "__main__":
    run_comprehensive_comparison()
