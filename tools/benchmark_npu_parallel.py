#!/usr/bin/env python3
"""
RK3588 NPU 三核真并发吞吐验证

背景：RKNNLite.inference() 是同步阻塞调用，单线程下即便绑定 NPU_CORE_0_1_2，
      同一时刻也只有一路任务在跑，三核无法体现加速。
方案：为每个 NPU 核心独立创建一个 RKNNLite 实例并绑定到专属核心，
      用多线程并发喂帧，实测系统级真实吞吐上限（工业场景的多路相机/流水线并行）。
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False


def load_input(img_path: str, size: int = 640) -> np.ndarray:
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


class CoreWorker(threading.Thread):
    """绑定到单一 NPU 物理核心的独立推理工作线程"""

    def __init__(self, model_path, core_attr, data, rounds, barrier):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.core_attr = core_attr
        self.data = data
        self.rounds = rounds
        self.barrier = barrier
        self.costs = []
        self.error = None

    def run(self):
        rknn = RKNNLite()
        try:
            if rknn.load_rknn(self.model_path) != 0:
                self.error = "load_rknn failed"
                self.barrier.wait()
                return
            if rknn.init_runtime(core_mask=getattr(RKNNLite, self.core_attr)) != 0:
                self.error = "init_runtime failed"
                self.barrier.wait()
                return
            for _ in range(10):          # 各自预热
                rknn.inference(inputs=[self.data])
            self.barrier.wait()          # 所有核心就绪后同时开闸，保证真并发
            for _ in range(self.rounds):
                t0 = time.perf_counter()
                rknn.inference(inputs=[self.data])
                self.costs.append((time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.error = str(e)[:120]
        finally:
            try:
                rknn.release()
            except Exception:
                pass


def run_parallel(model_path, cores, data, rounds) -> dict:
    barrier = threading.Barrier(len(cores))
    workers = [CoreWorker(model_path, c, data, rounds, barrier) for c in cores]

    t0 = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    wall = time.perf_counter() - t0

    errs = [w.error for w in workers if w.error]
    if errs:
        return {"parallel_workers": len(cores), "error": errs[0]}

    all_costs = [c for w in workers for c in w.costs]
    total_frames = len(all_costs)
    return {
        "parallel_workers": len(cores),
        "cores": cores,
        "total_frames": total_frames,
        "wall_seconds": round(wall, 3),
        "system_fps": round(total_frames / wall, 2),
        "per_stream_avg_ms": round(statistics.mean(all_costs), 3),
        "per_stream_fps": round(1000.0 / statistics.mean(all_costs), 2),
        "p99_ms": round(sorted(all_costs)[int(total_frames * 0.99) - 1], 3),
    }


def main():
    ap = argparse.ArgumentParser(description="RK3588 NPU 三核真并发吞吐测试")
    ap.add_argument("--model", default="models/weights/yolov8n_int8.rknn")
    ap.add_argument("--image", default="data/calibration/images/000000000009.jpg")
    ap.add_argument("--rounds", type=int, default=150, help="每个工作线程的推理轮数")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    if not HAS_RKNN:
        print("[错误] 未安装 rknn-toolkit-lite2")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[错误] 模型不存在: {args.model}")
        sys.exit(1)

    data = load_input(args.image)
    print("=" * 78)
    print(f"  RK3588 NPU 真并发吞吐测试 | 模型: {os.path.basename(args.model)}")
    print(f"  策略: 每核心独立 RKNNLite 实例 + 多线程同步开闸")
    print("=" * 78)

    plans = [
        (["NPU_CORE_0"], "1 路 (单核独占)"),
        (["NPU_CORE_0", "NPU_CORE_1"], "2 路 (双核并行)"),
        (["NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2"], "3 路 (三核全开)"),
    ]

    results = []
    for cores, desc in plans:
        print(f"\n>>> 正在测试 [{desc}] ...")
        r = run_parallel(args.model, cores, data, args.rounds)
        r["desc"] = desc
        results.append(r)
        if "error" in r:
            print(f"    失败: {r['error']}")
        else:
            print(f"    系统吞吐 {r['system_fps']} FPS | 单路时延 {r['per_stream_avg_ms']}ms "
                  f"| P99 {r['p99_ms']}ms | 总帧数 {r['total_frames']}")

    ok = [r for r in results if "error" not in r]
    print("\n" + "=" * 78)
    print(f"{'并发方案':<24}{'系统吞吐':>14}{'单路时延':>14}{'单路FPS':>12}{'P99':>10}")
    print("-" * 78)
    for r in ok:
        print(f"{r['desc']:<22}{r['system_fps']:>12.2f}FPS{r['per_stream_avg_ms']:>12.2f}ms"
              f"{r['per_stream_fps']:>12.2f}{r['p99_ms']:>10.2f}")
    print("=" * 78)

    if len(ok) >= 2:
        base, best = ok[0], max(ok, key=lambda x: x["system_fps"])
        print(f"\n[并发加速比] {best['desc']} 相较 {base['desc']}: "
              f"{best['system_fps'] / base['system_fps']:.2f}x  "
              f"({base['system_fps']:.2f} FPS -> {best['system_fps']:.2f} FPS)")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "rounds_per_worker": args.rounds,
                       "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\n[报告] 已写入 {args.report}")


if __name__ == "__main__":
    main()
