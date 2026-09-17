#!/usr/bin/env python3
"""
VLM 驱动的机械臂抓取闭环 —— 运行入口

用法：
  # 单条指令
  python3 tools/run_vlm_grasp.py --cmd "看看画面里有什么可以抓的，抓起来"

  # 交互模式
  python3 tools/run_vlm_grasp.py --interactive

  # 硬件未接入时的离线验证（相机/舵机自动降级 Mock）
  python3 tools/run_vlm_grasp.py --cmd "复位" --mock
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.vlm_grasp_pipeline import VLMGraspPipeline


DEFAULT_CONFIG = {
    "camera": {
        "device_id": 0, "width": 640, "height": 480,
        "mock_image": "data/calibration/images/000000000074.jpg",
    },
    # FP16：INT8 的分类分支已被量化破坏（分数恒为 0，检测不可用），
    # 根因见 docs/benchmarks/yolo_int8_classification_failure.json
    "vision": {"model_path": "models/weights/yolov8n_fp16.rknn", "conf_thresh": 0.25},
    "vlm": {
        "model_path": "models/weights/qwen3vl4b_w8a8.rkllm",
        "vision_path": "models/weights/qwen3vl4b_vision.rknn",
        "max_context_len": 2048,
        "max_new_tokens": 256,
    },
    "arm": {"i2c": {"bus": 7, "address": 0x40, "channels": [0, 1, 2, 3]}},
    "hand_eye": {"mm_per_pixel": 0.5, "center_u": 320, "center_v": 240, "base_x": 150.0},
    "pipeline": {"safe_z_height": 150.0, "grasp_z_height": 30.0},
    # 默认 OpenCV。MLIR 是显式实验后端，实测比 OpenCV 慢，别当默认用。
    # AOT 内核输出尺寸编译期固定，所以 640/448 两路各自指定库。
    "preprocess": {"backend": "opencv", "libraries": {}, "manifest": None},
}


def main():
    ap = argparse.ArgumentParser(description="VLM 驱动的机械臂抓取闭环")
    ap.add_argument("--cmd", default="", help="单条自然语言指令")
    ap.add_argument("--interactive", action="store_true", help="进入交互模式")
    ap.add_argument("--mock", action="store_true", help="强制 Mock（不下发真实动作）")
    ap.add_argument("--no-vlm", action="store_true", help="跳过 VLM，仅用规则引擎")
    ap.add_argument("--preprocess-backend", choices=("opencv", "mlir"), default="opencv",
                    help="预处理后端，默认 opencv；mlir 为实验后端，实测更慢")
    ap.add_argument("--preprocess-library", action="append", metavar="HxW=PATH", default=[],
                    help="mlir 后端按 profile 指定库，如 640x640=/path/libedgeai_preprocess.so；可重复")
    ap.add_argument("--preprocess-manifest", help="mlir 后端的构建 manifest.json 路径")
    args = ap.parse_args()

    libraries = {}
    for item in args.preprocess_library:
        key, separator, path = item.partition("=")
        if not separator or not path:
            ap.error("--preprocess-library 需要 HxW=PATH 形式，收到 " + item)
        libraries[key] = path
    if args.preprocess_backend == "mlir" and not libraries:
        ap.error("--preprocess-backend mlir 必须给出至少一个 --preprocess-library HxW=PATH")

    cfg = dict(DEFAULT_CONFIG)
    cfg["preprocess"] = {"backend": args.preprocess_backend,
                         "libraries": libraries,
                         "manifest": args.preprocess_manifest}
    if args.no_vlm:
        cfg["vlm"] = {"model_path": "__disabled__", "vision_path": "__disabled__"}

    pipe = VLMGraspPipeline(cfg, mock_mode=args.mock)
    pipe.setup()

    try:
        if args.interactive:
            print("\n输入指令（exit 退出）：")
            while True:
                try:
                    line = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if line.lower() in ("exit", "quit", "q"):
                    break
                if line:
                    pipe.run_once(line)
        else:
            cmd = args.cmd or "观察画面，如果有可抓取的物体就抓起来"
            pipe.run_once(cmd)
    finally:
        pipe.print_stats()
        pipe.stop()


if __name__ == "__main__":
    main()
