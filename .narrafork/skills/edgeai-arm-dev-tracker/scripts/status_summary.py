#!/usr/bin/env python3
"""
WorkBuddy / Codex 风格的项目研发进度与里程碑自动生成器
用法:
  python status_summary.py
"""
import json
import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "EdgeAI-Robotic-Arm-Control").is_dir():
            return p / "EdgeAI-Robotic-Arm-Control"
        if (p / "CMakeLists.txt").exists() and (p / "include").is_dir():
            return p
        if (p / ".git").exists() and "EdgeAI" in p.name:
            return p
    return cwd


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def main():
    root = find_project_root()
    json_path = root / "docs" / "dev_logs" / "steps_history.json"
    if not json_path.exists():
        safe_print(f"[Tracker] 未找到进度数据: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    safe_print("=" * 70)
    safe_print(f"  EdgeAI 机械臂项目研发进度台账 (共完成 {len(history)} 个核心步骤)")
    safe_print("=" * 70)

    for item in history:
        safe_print(f"[Step {item.get('step', '?')}] {item.get('title')}")
        safe_print(f"   [模块] {item.get('module')}  |  [时间] {item.get('timestamp')}")
        safe_print(f"   [How]  {item.get('how')[:80]}...")
        safe_print(f"   [Why]  {item.get('why')[:80]}...")
        safe_print(f"   [亮点] {item.get('value')}")
        safe_print("-" * 70)


if __name__ == "__main__":
    main()
