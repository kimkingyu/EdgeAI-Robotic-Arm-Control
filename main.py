#!/usr/bin/env python3
"""
面向工业场景的端侧 AI 系统与模型推理加速研究 (EdgeAI-Robotic-Arm-Control)
主控入口程序
"""
import argparse
import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from src.pipeline import GraspPipeline


def load_config(config_path: str):
    p = Path(config_path)
    if not p.exists() or not HAS_YAML:
        return {
            "camera": {"device_id": 0, "width": 640, "height": 480},
            "vision": {"model_path": "models/weights/yolov8n_int8.rknn"},
            "arm": {"controller_type": "i2c", "i2c": {"bus": 7, "address": 0x40, "channels": [0,1,2,3,4,5], "gripper_channel": 6}},
            "llm": {"enabled": True, "model_path": "models/weights/qwen2.5_0.5b_w4a16.rkllm"},
            "pipeline": {"safe_z_height": 150.0, "grasp_z_height": 30.0}
        }
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="端侧 AI 工业机械臂控制系统")
    parser.add_argument("-c", "--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--mock", action="store_true", help="启用全真模拟模式 (脱机/无实机环境自测)")
    parser.add_argument("--cmd", type=str, default="", help="直接执行单条工业自然语言指令")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 70)
    print("  面向工业场景的端侧 AI 系统与模型推理加速研究 (EdgeAI-Robotic-Arm-Control)")
    print(f"  模式: {'模拟运行 (Mock)' if args.mock else '实机连线 (Hardware)'}")
    print("=" * 70)

    pipeline = GraspPipeline(config=cfg, mock_mode=args.mock)
    if not pipeline.setup():
        print("[Main] 初始化失败")
        sys.exit(1)

    try:
        if args.cmd:
            # 单次命令行执行
            pipeline.execute_command(args.cmd)
        elif args.mock:
            # 模拟执行典型工业工件分拣指令
            pipeline.execute_command("把传送带上的法兰工件抓取并放到暂存区")
        else:
            # 交互式控制台
            print("\n进入工业操作员自然语言交互终端 (输入 'exit' 或 'quit' 退出):")
            while True:
                user_input = input("\n[工控终端] 请输入作业指令 >> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                pipeline.execute_command(user_input)
    except KeyboardInterrupt:
        print("\n[Main] 操作员中断停机")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
