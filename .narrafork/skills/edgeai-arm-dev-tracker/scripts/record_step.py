#!/usr/bin/env python3
"""
EdgeAI-Robotic-Arm-Control 研发全过程精准记录器
用法示例:
  python record_step.py \
    --step 1 \
    --title "板载 NPU 驱动环境摸底与 Ubuntu 22.04 内核刷写" \
    --module "BSP & OS" \
    --how "下载 Jammy 1.0.6 镜像并解压为 .img，通过 balenaEtcher 烧写进入系统，验证 rknpu 0.9.6 驱动" \
    --why "RKNN 工具链与 Python 3.10 在 Ubuntu 22.04 下稳定性最佳，避免 Ubuntu 24.04 因缺少 aarch64 prebuilt 包导致的驱动断链" \
    --evidence "dmesg 输出 [drm] Initialized rknpu 0.9.6，板载三核 NPU 就绪" \
    --value "简历：深入掌握 RK3588 底层 BSP 架构、驱动匹配与内核调试"
"""
import argparse
import datetime
import json
import os
from pathlib import Path


def find_project_root() -> Path:
    # 优先使用当前工作目录，或者向上查找包含 EdgeAI-Robotic-Arm-Control 或 .git 的目录
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "EdgeAI-Robotic-Arm-Control").is_dir():
            return p / "EdgeAI-Robotic-Arm-Control"
        if (p / "CMakeLists.txt").exists() and (p / "include").is_dir():
            return p
        if (p / ".git").exists() and "EdgeAI" in p.name:
            return p
    return cwd


def main():
    parser = argparse.ArgumentParser(description="EdgeAI 机械臂开发步骤与技术决策追踪器")
    parser.add_argument("--step", type=str, default="", help="步骤编号 (如 1, 2.1)")
    parser.add_argument("--title", required=True, help="本步完成的核心事项")
    parser.add_argument("--module", default="Core", help="所属模块: BSP/NPU/Vision/Kinematics/I2C/LLM/Build")
    parser.add_argument("--how", required=True, help="怎么做的: 涉及的具体命令、代码文件修改、参数调整")
    parser.add_argument("--why", required=True, help="是为了什么的: 研发意图、解决的技术痛点、架构决策依据")
    parser.add_argument("--evidence", default="运行验证通过", help="实测证据: 终端关键日志、性能指标、报错修复记录")
    parser.add_argument("--value", default="", help="价值转化: 对应简历/毕业课题的技术亮点或面试得分点")
    parser.add_argument("--project-dir", default="", help="可选显式指定工程根目录")
    args = parser.parse_args()

    project_root = Path(args.project_dir) if args.project_dir else find_project_root()
    log_dir = project_root / "docs" / "dev_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "DEV_PROGRESS_JOURNAL.md"
    json_file = log_dir / "steps_history.json"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. 维护 JSON 索引数据
    history = []
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    step_num = args.step if args.step else str(len(history) + 1)
    record = {
        "step": step_num,
        "timestamp": now,
        "title": args.title,
        "module": args.module,
        "how": args.how,
        "why": args.why,
        "evidence": args.evidence,
        "value": args.value
    }
    history.append(record)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 2. 写入/追加 Markdown 日志
    if not log_file.exists():
        header = """# EdgeAI-Robotic-Arm-Control 研发全过程与技术决策追踪日志

> 本日志由 `edgeai-arm-dev-tracker` 自动化插件维护。详细记录每一步【做了什么】、【怎么做的（命令/代码）】、【为了什么（决策考量）】、【实测证据】与【简历技术点沉淀】，确保全流程可复现、可溯源、可向面试官展开深入答辩。

---
"""
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(header)

    entry = f"""
## 📍 第 {step_num} 步：{args.title}
* **记录时间**：`{now}` ｜ **技术模块**：`[{args.module}]`

### 1. 怎么做的（How - 技术实现与具体操作）
{args.how}

### 2. 是为了什么（Why - 决策依据与解决痛点）
{args.why}

### 3. 验证证据（Evidence - 实测结果与日志支撑）
```text
{args.evidence}
```

"""
    if args.value:
        entry += f"""### 4. 简历与课题价值（Value - 面试问答与技术亮点映射）
> 💡 **亮点提炼**：{args.value}

"""

    entry += "---\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"[Tracker] 成功记录第 {step_num} 步: {args.title}")
    print(f"          落盘至: {log_file}")


if __name__ == "__main__":
    main()
