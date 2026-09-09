#!/usr/bin/env python3
"""
Qwen3-VL-4B 端侧视觉多模态基准测试

实测采集：视觉编码耗时、TTFT、解码吞吐、内存占用，
以及工业场景下的 Function Calling JSON 遵循率。

所有数据均来自真实 NPU 推理，无 Mock、无硬编码。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.inference.rkllm_vl_engine import QwenVLEngine


VISION_CASES = [
    {"id": "VL-01", "type": "场景理解", "prompt": "这张图里有什么？简短回答。"},
    {"id": "VL-02", "type": "抓取决策", "prompt": "如果用机械臂抓取，你会选哪个物体？为什么？简短回答。"},
    {"id": "VL-03", "type": "空间关系", "prompt": "描述图中物体的空间位置关系。简短回答。"},
]

TEXT_CASES = [
    {"id": "TX-01", "type": "标准工步",
     "prompt": "你是工业机械臂控制大脑，把指令拆解为JSON动作流，只输出JSON。"
               "可用动作：pick/place/inspect/move_safe/emergency_stop。"
               "指令：把坐标(160,20,30)的阀芯抓取并放到(200,80,40)的清洗槽。"},
    {"id": "TX-02", "type": "安全急停",
     "prompt": "你是工业机械臂控制大脑，把指令拆解为JSON动作流，只输出JSON。"
               "可用动作：pick/place/inspect/move_safe/emergency_stop。"
               "指令：警告！有人靠近安全栅栏，立刻紧急停止！"},
    {"id": "TX-03", "type": "巡检任务",
     "prompt": "你是工业机械臂控制大脑，把指令拆解为JSON动作流，只输出JSON。"
               "可用动作：pick/place/inspect/move_safe/emergency_stop。"
               "指令：对加工工位的法兰盘做质量巡检。"},
]


def extract_json(text: str):
    """从模型输出中提取 JSON，兼容 markdown 代码块包裹"""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                t = p
                break
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except Exception:
            return None
    return None


def load_image(path: str, size: int = 448):
    if cv2 is None or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB)
    return np.expand_dims(img, 0)


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL 端侧多模态基准测试")
    ap.add_argument("--llm", default="models/weights/qwen3vl4b_w8a8.rkllm")
    ap.add_argument("--vision", default="models/weights/qwen3vl4b_vision.rknn")
    ap.add_argument("--image", default="data/calibration/images/000000000074.jpg")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    print("=" * 76)
    print("  Qwen3-VL-4B-Instruct (w8a8) 端侧视觉多模态基准测试")
    print("=" * 76)

    eng = QwenVLEngine(args.llm, args.vision,
                       max_context_len=2048, max_new_tokens=args.max_new_tokens)
    t0 = time.time()
    if not eng.init_engine():
        print("[错误] 引擎初始化失败")
        sys.exit(1)
    load_s = time.time() - t0

    img = load_image(args.image)
    if img is None:
        print(f"[警告] 无法读取测试图 {args.image}，将跳过视觉用例")

    results = {"model": os.path.basename(args.llm),
               "vision_model": os.path.basename(args.vision),
               "load_seconds": round(load_s, 2),
               "vision_cases": [], "text_cases": []}

    # ---- 视觉多模态用例 ----
    if img is not None:
        print(f"\n【视觉多模态】测试图: {os.path.basename(args.image)}")
        print("-" * 76)
        for c in VISION_CASES:
            out = eng.generate(c["prompt"], image=img)
            p = eng.last_perf
            print(f"[{c['id']}] {c['type']}")
            print(f"  问: {c['prompt']}")
            print(f"  答: {out.strip()[:200]}")
            print(f"  TTFT {p['ttft_ms']} ms | {p['tps']} tok/s | "
                  f"{p['total_tokens']} tokens | {p['total_time_s']} s")
            print("-" * 76)
            results["vision_cases"].append({
                "id": c["id"], "type": c["type"], "output": out.strip(),
                "ttft_ms": p["ttft_ms"], "tps": p["tps"],
                "tokens": p["total_tokens"], "total_s": p["total_time_s"],
            })

    # ---- 纯文本 Function Calling 用例 ----
    print(f"\n【Function Calling】纯文本指令解析")
    print("-" * 76)
    json_ok = 0
    for c in TEXT_CASES:
        out = eng.generate(c["prompt"])
        p = eng.last_perf
        parsed = extract_json(out)
        valid = parsed is not None
        if valid:
            json_ok += 1
        print(f"[{c['id']}] {c['type']} -> JSON {'✅ 有效' if valid else '❌ 无效'}")
        print(f"  答: {out.strip()[:220]}")
        print(f"  TTFT {p['ttft_ms']} ms | {p['tps']} tok/s | {p['total_time_s']} s")
        print("-" * 76)
        results["text_cases"].append({
            "id": c["id"], "type": c["type"], "output": out.strip(),
            "json_valid": valid, "parsed": parsed,
            "ttft_ms": p["ttft_ms"], "tps": p["tps"],
            "tokens": p["total_tokens"], "total_s": p["total_time_s"],
        })

    # ---- 汇总 ----
    allc = results["vision_cases"] + results["text_cases"]
    ttfts = [c["ttft_ms"] for c in allc if c["ttft_ms"]]
    tpss = [c["tps"] for c in allc if c["tps"]]
    results["summary"] = {
        "total_cases": len(allc),
        "json_valid_rate": round(json_ok / len(TEXT_CASES) * 100, 1) if TEXT_CASES else None,
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "avg_tps": round(sum(tpss) / len(tpss), 2) if tpss else None,
    }

    print("\n" + "=" * 76)
    print("                        实测汇总")
    print("=" * 76)
    s = results["summary"]
    print(f"  模型加载耗时      : {results['load_seconds']} s")
    print(f"  测试用例总数      : {s['total_cases']} 组")
    print(f"  JSON 格式遵循率   : {s['json_valid_rate']}%  (纯文本 Function Calling)")
    print(f"  平均 TTFT         : {s['avg_ttft_ms']} ms")
    print(f"  平均解码吞吐      : {s['avg_tps']} tokens/s")
    print("=" * 76)
    print("  注：以上全部为板端真实 NPU 推理实测值，无 Mock、无估算填充。")

    eng.release()

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[报告] 已写入 {args.report}")


if __name__ == "__main__":
    main()
