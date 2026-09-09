#!/usr/bin/env python3
"""
C++ 与 Python 两套 PCA9685 角度→tick 换算的交叉比对。

为什么需要：C++ 侧 include/pca9685.hpp 与 Python 侧 src/controller/i2c_arm.py
各自实现了角度→脉宽→tick 的换算。两者若舍入方式或时基不同，同一个角度
会下发不同的脉宽，实机表现为"换条控制路径就抓偏一点"，而单侧测试永远
测不出来。本轮已发现并修复过一次这类分歧(截断 vs 四舍五入，差 1 tick)。

C++ 侧的 tick 由 tools/test_cpp_control.cpp 所用的同一份驱动算出，
经由 pwm_dump 导出程序输出；Python 侧直接调用 PCA9685。

本脚本不访问任何硬件。

用法：
  python3 tools/test_pwm_parity.py --cpp-dump ./pwm_dump
"""
import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.i2c_arm import PCA9685


def main():
    parser = argparse.ArgumentParser(description="C++/Python PWM 换算交叉比对")
    parser.add_argument("--cpp-dump", required=True, help="C++ 导出程序路径")
    args = parser.parse_args()

    if not os.path.exists(args.cpp_dump):
        print(f"未找到 C++ 导出程序: {args.cpp_dump}（构建缺失，非比对结论）")
        return 2

    proc = subprocess.run([args.cpp_dump], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        print(f"C++ 导出程序退出码 {proc.returncode}\n{proc.stderr[:2000]}")
        return 2
    try:
        cpp = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"无法解析 C++ 输出: {exc}\n{proc.stdout[:2000]}")
        return 2

    print("=" * 74)
    print("  C++ 与 Python PCA9685 角度→tick 换算交叉比对")
    print("=" * 74)

    pca = PCA9685()
    pca.set_pwm_freq(50.0)
    print(f"  Python period_us = {pca.period_us:.4f}")
    print(f"  C++    period_us = {cpp['period_us']:.4f}")

    period_ok = abs(pca.period_us - cpp["period_us"]) < 1e-6
    print(f"  {'✅' if period_ok else '❌'} 时基一致")

    # 独立复算：用 floor(x+0.5) 复现 C++ std::round 语义，
    # 不能用 Python 内置 round（银行家舍入，在 .5 处会与 C++ 分歧）。
    mismatches = []
    half_cases = 0
    for entry in cpp["samples"]:
        angle = entry["angle"]
        pulse = 500.0 + (angle / 180.0) * 2000.0
        exact = pulse * 4096.0 / pca.period_us
        if abs(exact - math.floor(exact) - 0.5) < 1e-9:
            half_cases += 1
        py_ticks = int(math.floor(exact + 0.5))
        if py_ticks != entry["ticks"]:
            mismatches.append((angle, entry["ticks"], py_ticks))
    print(f"  其中恰好落在 .5 边界的角度: {half_cases} 个（舍入分歧高发点）")

    print(f"  比对角度数: {len(cpp['samples'])}")
    if mismatches:
        print(f"\n  ❌ {len(mismatches)} 个角度换算不一致（最多显示 10 条）:")
        for angle, c, p in mismatches[:10]:
            print(f"     {angle:6.2f}° → C++ {c} ticks / Python {p} ticks")
    else:
        print("  ✅ 全部角度换算一致")

    print("\n" + "=" * 74)
    ok = period_ok and not mismatches
    if ok:
        print("  ✅ 两套 PWM 换算一致")
        print("     注意：一致不等于脉宽正确，实际波形仍需示波器校准。")
    else:
        print("  ❌ 存在分歧，两条控制路径会下发不同脉宽")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
