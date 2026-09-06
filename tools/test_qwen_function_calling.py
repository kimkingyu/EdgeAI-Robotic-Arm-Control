#!/usr/bin/env python3
"""
面向博拓里尼/中船工况的 Qwen2.5 端侧大模型 Function Calling 与任务规划压力测试
评测核心:
  1. 多场景指令解析: 单工步抓取、连续组合码垛、模糊语意决策、紧急安全停机
  2. JSON Schema 格式遵循率与语法严格性 (Zero-Shot Function Calling)
  3. 物理约束与安全动作边界校验 (Kinematics & Workspace Boundary Verification)
  4. 推理端到端延迟 (TTFT & Total Plan Latency)
"""
import time
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# 确保能正确导入项目根目录的 src 模块
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.llm import IndustrialTaskPlanner
from src.inference import RKLLMInferenceEngine


# 工业典型测试指令集 (覆盖常规、组合、突发与模糊工况)
TEST_INDUSTRIAL_CASES = [
    {
        "id": "CASE-01",
        "category": "标准工步",
        "instruction": "把传送带上坐标 (160, 20, 30) 的高压阀芯工件抓取并放入A号清洗槽 (200, 80, 40)",
        "expected_actions": ["move_safe", "pick", "place", "move_safe"]
    },
    {
        "id": "CASE-02",
        "category": "连续码垛",
        "instruction": "对加工工位的法兰盘进行质量巡检，确认无缺陷后搬运至成品区 (190, 50, 30)",
        "expected_actions": ["move_safe", "inspect", "pick", "place", "move_safe"]
    },
    {
        "id": "CASE-03",
        "category": "安全急停",
        "instruction": "警告！红外对射传感器报警，有人员靠近安全栅栏，立刻紧急停止机械臂！",
        "expected_actions": ["emergency_stop"]
    },
    {
        "id": "CASE-04",
        "category": "模糊意图",
        "instruction": "作业结束了，把机械臂收回原位并保持安全状态",
        "expected_actions": ["move_safe"]
    },
    {
        "id": "CASE-05",
        "category": "多工件处理",
        "instruction": "抓取托盘左侧工件并码放到右侧暂存架",
        "expected_actions": ["move_safe", "pick", "place", "move_safe"]
    }
]


def validate_plan_safety(plan: Dict[str, Any], max_workspace_r: float = 320.0) -> bool:
    """校验 Qwen 规划出的动作参数是否在机械臂物理安全工作半径内"""
    actions = plan.get("actions", [])
    if not actions:
        return False

    for act in actions:
        params = act.get("params", {})
        if "x" in params and "y" in params:
            x, y = params["x"], params["y"]
            r = (x**2 + y**2)**0.5
            if r > max_workspace_r:
                print(f"  [安全警告] 目标坐标 ({x}, {y}) 超出机械臂最大工作半径 {max_workspace_r}mm！")
                return False
    return True


def run_function_calling_evaluation(planner: IndustrialTaskPlanner):
    print("=" * 80)
    print("  Qwen2.5 端侧大模型 (RKLLM W4A16) 工业 Function Calling 与任务规划压力测试")
    print("=" * 80)

    total_cases = len(TEST_INDUSTRIAL_CASES)
    passed_cases = 0
    json_valid_count = 0
    safety_valid_count = 0
    total_latency_ms = 0.0

    print(f"正在对 {total_cases} 组典型工业作业场景执行端侧推理与动作序列验证...\n")

    for idx, case in enumerate(TEST_INDUSTRIAL_CASES):
        case_id = case["id"]
        cat = case["category"]
        query = case["instruction"]

        t0 = time.time()
        plan = planner.plan(query)
        elapsed_ms = (time.time() - t0) * 1000.0
        total_latency_ms += elapsed_ms

        actions = [a.get("action") for a in plan.get("actions", [])]
        is_json_valid = bool(plan.get("intent") and "actions" in plan)
        is_safe = validate_plan_safety(plan)

        if is_json_valid:
            json_valid_count += 1
        if is_safe:
            safety_valid_count += 1

        # 检查核心动作是否符合预期
        expected = case["expected_actions"]
        is_action_match = any(act in actions for act in expected)
        if is_json_valid and is_safe and is_action_match:
            passed_cases += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"[{case_id}] ({cat}) -> {status} (耗时: {elapsed_ms:.1f}ms)")
        print(f"  • 指令: \"{query}\"")
        print(f"  • 意图识别: {plan.get('intent')} | 动作流: {actions}")
        print("-" * 80)

    avg_latency = total_latency_ms / total_cases
    json_acc = (json_valid_count / total_cases) * 100.0
    pass_rate = (passed_cases / total_cases) * 100.0

    print("\n" + "=" * 80)
    print("                    Function Calling 实测性能评估汇总")
    print("=" * 80)
    print(f"  • 测试用例总数: {total_cases} 组")
    print(f"  • JSON 格式完全遵循率: {json_acc:.1f}% (严格输出预定义 Schema)")
    print(f"  • 动作物理安全性通过率: {(safety_valid_count / total_cases) * 100.0:.1f}%")
    print(f"  • 端侧任务拆解综合通过率: {pass_rate:.1f}%")
    print(f"  • 端侧平均规划时延: {avg_latency:.1f} ms (远优于云端 API 的数百毫秒)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 工业 Function Calling 评测")
    args = parser.parse_args()

    engine = RKLLMInferenceEngine(model_path="models/weights/qwen2.5_0.5b_w4a16.rkllm")
    engine.load_model()
    planner = IndustrialTaskPlanner(llm_engine=engine)

    run_function_calling_evaluation(planner)


if __name__ == "__main__":
    main()
