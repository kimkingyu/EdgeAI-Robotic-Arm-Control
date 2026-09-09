#!/usr/bin/env python3
"""
C++ 与 Python 两套逆运动学实现的交叉比对。

为什么需要：项目里同时存在 src/kinematics.cpp 与 src/kinematics/kinematics_base.py
两套独立实现。C++ 侧驱动实时控制线程，Python 侧驱动 VLM 抓取编排器。
两者若在角度符号、可达判定或限位上不一致，实机会表现为"仿真能抓、实机抓偏"，
且这类偏差在单侧测试中永远暴露不出来。

比对方法：以相同连杆参数(底座100/大臂150/小臂150)在同一批目标点上求解，
逐点比较可达判定与各关节角度。C++ 侧通过一个临时构建的导出程序输出解，
Python 侧直接调用。任何一侧拒绝而另一侧接受，都算不一致。

本脚本不访问任何硬件。

用法：
  python3 tools/test_ik_parity.py --cpp-solver ./bin/ik_dump
"""
import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kinematics import SimpleArmKinematics

ANGLE_TOL_DEG = 1e-6


def build_targets():
    """构造覆盖可达域、边界与不可达区的目标点。"""
    targets = []
    for r in range(40, 320, 10):
        for z in range(-40, 320, 10):
            targets.append((float(r), 0.0, float(z)))
    for x, y, z in [(150, 50, 120), (180, -60, 80), (0, 200, 100),
                    (0, -200, 100), (-150, 0, 100), (200, 0, 100),
                    (5, 0, 100), (295, 0, 100), (0, 0, 100), (0, 0, 250)]:
        targets.append((float(x), float(y), float(z)))
    return targets


def run_cpp(solver_path, targets):
    """调用 C++ 导出程序，每行输入 x y z，输出 JSON。"""
    payload = "\n".join(f"{x} {y} {z}" for x, y, z in targets)
    proc = subprocess.run([solver_path], input=payload, capture_output=True,
                          text=True, timeout=120)
    if proc.returncode != 0:
        print(f"C++ 求解器退出码 {proc.returncode}")
        print(proc.stderr[:2000])
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"无法解析 C++ 输出: {exc}")
        print(proc.stdout[:2000])
        return None


def main():
    parser = argparse.ArgumentParser(description="C++/Python 逆运动学交叉比对")
    parser.add_argument("--cpp-solver", required=True, help="C++ 导出程序路径")
    args = parser.parse_args()

    if not os.path.exists(args.cpp_solver):
        print(f"未找到 C++ 求解器: {args.cpp_solver}")
        print("这是构建缺失，不是比对结论。")
        return 2

    targets = build_targets()
    cpp_results = run_cpp(args.cpp_solver, targets)
    if cpp_results is None:
        return 2
    if len(cpp_results) != len(targets):
        print(f"C++ 返回 {len(cpp_results)} 条，期望 {len(targets)} 条")
        return 2

    kin = SimpleArmKinematics()

    print("=" * 74)
    print("  C++ 与 Python 逆运动学交叉比对")
    print("=" * 74)
    print(f"  连杆参数: 底座 {kin.link_lengths[0]} | 大臂 {kin.link_lengths[1]} "
          f"| 小臂 {kin.link_lengths[2]} mm")
    print(f"  测试点数: {len(targets)}")

    both_ok = both_reject = 0
    verdict_mismatch = []
    angle_mismatch = []
    max_angle_diff = 0.0

    for (x, y, z), cpp in zip(targets, cpp_results):
        py = kin.inverse_kinematics({"x": x, "y": y, "z": z})
        cpp_ok = cpp["reachable"]
        py_ok = py is not None

        if cpp_ok != py_ok:
            verdict_mismatch.append((x, y, z, cpp_ok, py_ok,
                                     cpp.get("reason", ""),
                                     kin.last_reject_reason or ""))
            continue

        if not cpp_ok:
            both_reject += 1
            continue

        both_ok += 1
        # Python 侧输出保留 1 位小数，比对时按该精度对齐
        for idx, (ca, pa) in enumerate(zip(cpp["angles"], py)):
            diff = abs(round(ca, 1) - pa)
            max_angle_diff = max(max_angle_diff, diff)
            if diff > 0.05 + ANGLE_TOL_DEG:
                angle_mismatch.append((x, y, z, idx, ca, pa, diff))

    print("\n" + "-" * 74)
    print("【结果】")
    print("-" * 74)
    print(f"  双方均可解      : {both_ok}")
    print(f"  双方均拒绝      : {both_reject}")
    print(f"  可达判定不一致  : {len(verdict_mismatch)}")
    print(f"  关节角不一致    : {len(angle_mismatch)}")
    print(f"  最大关节角偏差  : {max_angle_diff:.6f}°")

    if verdict_mismatch:
        print("\n  ❌ 可达判定分歧（最多显示 10 条）:")
        for x, y, z, c, p, cr, pr in verdict_mismatch[:10]:
            print(f"     ({x:>6.1f},{y:>6.1f},{z:>6.1f}) "
                  f"C++={'可解' if c else '拒绝'} Python={'可解' if p else '拒绝'}")
            if not c and cr:
                print(f"        C++ 理由: {cr}")
            if not p and pr:
                print(f"        Py  理由: {pr}")

    if angle_mismatch:
        print("\n  ❌ 关节角分歧（最多显示 10 条）:")
        for x, y, z, idx, ca, pa, diff in angle_mismatch[:10]:
            print(f"     ({x:>6.1f},{y:>6.1f},{z:>6.1f}) 关节{idx}: "
                  f"C++={ca:.4f}° Python={pa:.4f}° 差 {diff:.4f}°")

    print("\n" + "=" * 74)
    ok = not verdict_mismatch and not angle_mismatch and both_ok > 0
    if ok:
        print("  ✅ 两套实现在测试点上完全一致")
        print("     注意：一致不等于正确，仅说明两者不会互相矛盾；")
        print("     绝对精度仍需实机手眼标定验证。")
    else:
        print("  ❌ 存在分歧，实机会出现仿真与实测不符")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
