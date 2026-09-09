#!/usr/bin/env python3
"""
RK3588 NPU 板端推理性能基准测试

实测同一 .rknn 模型在单核 / 双核 / 三核负载均衡下的
单帧推理耗时、吞吐 FPS 与耗时抖动，输出结构化 Benchmark 报告。
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False


CORE_MODES = {
    "single": ("NPU_CORE_0", "单核基线"),
    "dual": ("NPU_CORE_0_1", "双核并发"),
    "triple": ("NPU_CORE_0_1_2", "三核满血负载均衡"),
    "auto": ("NPU_CORE_AUTO", "驱动自动调度"),
}


def load_input(img_path: str, size: int = 640) -> np.ndarray:
    """载入一张真实图片作为推理输入，失败时回退随机张量"""
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            return np.expand_dims(img, 0).astype(np.uint8)
    except Exception:
        pass
    return np.random.randint(0, 255, (1, size, size, 3), dtype=np.uint8)


def bench_one(model_path: str, mode: str, data: np.ndarray,
              warmup: int, rounds: int) -> dict:
    """对单一核心调度模式跑一轮基准测试"""
    attr, desc = CORE_MODES[mode]
    core_mask = getattr(RKNNLite, attr)

    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        return {"mode": mode, "error": "load_rknn failed"}
    if rknn.init_runtime(core_mask=core_mask) != 0:
        rknn.release()
        return {"mode": mode, "error": "init_runtime failed"}

    # 预热：让 NPU 频率爬坡到稳态，避免首帧冷启动污染数据
    for _ in range(warmup):
        rknn.inference(inputs=[data])

    costs = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        rknn.inference(inputs=[data])
        costs.append((time.perf_counter() - t0) * 1000.0)
    rknn.release()

    costs.sort()
    avg = statistics.mean(costs)
    return {
        "mode": mode,
        "desc": desc,
        "core_mask": attr,
        "avg_ms": round(avg, 3),
        "min_ms": round(costs[0], 3),
        "max_ms": round(costs[-1], 3),
        "p50_ms": round(costs[len(costs) // 2], 3),
        "p99_ms": round(costs[int(len(costs) * 0.99) - 1], 3),
        "std_ms": round(statistics.pstdev(costs), 3),
        "fps": round(1000.0 / avg, 2),
        "rounds": rounds,
    }


def main():
    ap = argparse.ArgumentParser(description="RK3588 NPU 多核推理基准测试")
    ap.add_argument("--model", default="models/weights/yolov8n_int8.rknn", help=".rknn 模型路径")
    ap.add_argument("--image", default="data/calibration/images/000000000009.jpg", help="测试输入图片")
    ap.add_argument("--rounds", type=int, default=200, help="正式测试轮数")
    ap.add_argument("--warmup", type=int, default=20, help="预热轮数")
    ap.add_argument("--modes", default="single,dual,triple,auto", help="要测试的核心调度模式")
    ap.add_argument("--report", default="", help="JSON 报告输出路径")
    args = ap.parse_args()

    if not HAS_RKNN:
        print("[错误] 未安装 rknn-toolkit-lite2，无法在板端推理")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[错误] 模型不存在: {args.model}")
        sys.exit(1)

    data = load_input(args.image)
    print("=" * 74)
    print(f"  RK3588 NPU 推理基准测试 | 模型: {os.path.basename(args.model)}")
    print(f"  输入: {data.shape} | 预热 {args.warmup} 轮 | 正式 {args.rounds} 轮")
    print("=" * 74)

    results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip() in CORE_MODES]:
        print(f"\n>>> 正在测试 [{CORE_MODES[mode][1]}] ...")
        r = bench_one(args.model, mode, data, args.warmup, args.rounds)
        results.append(r)
        if "error" in r:
            print(f"    失败: {r['error']}")
        else:
            print(f"    平均 {r['avg_ms']}ms | P99 {r['p99_ms']}ms | 抖动 ±{r['std_ms']}ms | {r['fps']} FPS")

    ok = [r for r in results if "error" not in r]
    print("\n" + "=" * 74)
    print(f"{'调度模式':<22}{'平均耗时':>12}{'P99':>10}{'抖动':>10}{'吞吐 FPS':>12}")
    print("-" * 74)
    for r in ok:
        print(f"{r['desc']:<20}{r['avg_ms']:>12.3f}ms{r['p99_ms']:>9.2f}{r['std_ms']:>9.2f}{r['fps']:>12.2f}")
    print("=" * 74)

    if len(ok) >= 2:
        base = next((r for r in ok if r["mode"] == "single"), ok[0])
        best = min(ok, key=lambda x: x["avg_ms"])
        if best["mode"] != base["mode"]:
            print(f"\n[加速比] {best['desc']} 相较 {base['desc']}: "
                  f"{base['avg_ms'] / best['avg_ms']:.2f}x  "
                  f"(耗时 {base['avg_ms']:.2f}ms -> {best['avg_ms']:.2f}ms)")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        payload = {"model": args.model, "input_shape": list(data.shape),
                   "rounds": args.rounds, "results": results}
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[报告] 已写入 {args.report}")


if __name__ == "__main__":
    main()
