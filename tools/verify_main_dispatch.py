#!/usr/bin/env python3
"""
校验主程序在假总线上实际写入的寄存器，是否与独立复算的关节角一致。

为什么需要：主程序的 setJointAngles 下发分支在硬件接线前从未被执行过。
单元测试验证的是驱动类本身，而"主程序在舵机连通时究竟写了什么寄存器、
对应什么角度、是否与手眼变换和逆解的结果吻合"是另一回事。这条路径若
等接线后才第一次跑，任何错误都会直接作用在实物舵机上。

做法：解析 fake_i2c_preload.c 捕获的写入日志，从 tick 反解出舵机角度与
IK 几何角，再用 Python 侧手眼变换独立走一遍全链路，比对两者是否吻合。

本脚本不访问任何硬件。

用法：
  python3 tools/verify_main_dispatch.py --log /tmp/i2c_writes.log \
      --calib configs/hand_eye_calib.json --pixel 340 240
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.hand_eye import HandEyeCalibrator
from src.kinematics import SimpleArmKinematics

# 与 C++ 侧 ArmKinematics::SERVO_OFFSETS 一致
SERVO_OFFSETS = [90.0, 90.0, 180.0, 90.0]
OSCILLATOR_HZ = 25_000_000.0

failures = []


def check(condition, label):
    if condition:
        print(f"  ✅ {label}")
    else:
        print(f"  ❌ {label}")
        failures.append(label)


def parse_log(path):
    """解析假总线日志，返回 (prescale, 关节批量写列表)"""
    prescale = None
    joint_writes = []
    pattern = re.compile(r"WRITE reg=0x([0-9A-F]{2}) len=(\d+) data=(.*)")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            reg, _, data = int(m.group(1), 16), int(m.group(2)), m.group(3).split()
            values = [int(x, 16) for x in data]
            if reg == 0xFE and values:
                prescale = values[0]
            elif reg == 0x06 and len(values) in (12, 16):
                joint_writes.append(values)
    return prescale, joint_writes


def main():
    parser = argparse.ArgumentParser(description="校验主程序实际下发的寄存器")
    parser.add_argument("--log", required=True, help="假总线写入日志")
    parser.add_argument("--calib", required=True, help="主程序所用标定文件")
    parser.add_argument("--pixel", nargs=2, type=float, required=True,
                        help="主程序检测桩的像素坐标 u v")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"未找到日志 {args.log}（主程序未运行或未启用假总线，非校验结论）")
        return 2

    print("=" * 74)
    print("  主程序下发路径校验（假总线捕获 vs 独立复算）")
    print("=" * 74)

    prescale, joint_writes = parse_log(args.log)
    check(prescale is not None, f"捕获到 PRESCALE 写入 (值 {prescale})")
    if prescale is None:
        return 1
    check(prescale == 135, f"PRESCALE=135 对应 50Hz×0.9 修正（实际 {prescale}）")
    check(len(joint_writes) > 0, f"捕获到 {len(joint_writes)} 次关节批量写入")
    if not joint_writes:
        return 1

    # 所有下发内容应完全一致：检测桩输出固定像素，标定与逆解均为确定性计算
    unique = {tuple(w) for w in joint_writes}
    check(len(unique) == 1, f"所有下发内容一致（{len(unique)} 种）")

    values = joint_writes[0]
    n_joints = len(values) // 4
    check(n_joints == 4, f"一次写入覆盖 {n_joints} 个关节")

    # 从寄存器反解 tick
    ticks = []
    on_ok = True
    for i in range(n_joints):
        on = values[4 * i] | (values[4 * i + 1] << 8)
        off = values[4 * i + 2] | (values[4 * i + 3] << 8)
        if on != 0:
            on_ok = False
        ticks.append(off)
    check(on_ok, "所有通道 ON 计数为 0")
    check(all(1 <= t <= 4095 for t in ticks), f"tick 均在有效范围内 {ticks}")

    period_us = 4096.0 * (prescale + 1) * 1_000_000.0 / OSCILLATOR_HZ
    servo_angles = [(t * period_us / 4096.0 - 500.0) / 2000.0 * 180.0 for t in ticks]
    ik_angles = [a - o for a, o in zip(servo_angles, SERVO_OFFSETS)]
    print(f"\n  周期 {period_us:.2f} us")
    print(f"  捕获 ticks    : {ticks}")
    print(f"  反解舵机角    : {[round(a, 3) for a in servo_angles]}")
    print(f"  反解 IK 几何角 : {[round(a, 3) for a in ik_angles]}")

    # 独立复算：手眼变换 → 逆解
    calib = HandEyeCalibrator()
    if not calib.load(args.calib):
        print(f"\n  无法载入标定 {args.calib}")
        return 2
    u, v = args.pixel
    pose = calib.pixel_to_robot(u, v)
    check(pose is not None, f"独立手眼变换成功 ({u},{v}) -> {pose}")
    if pose is None:
        return 1

    # 用与 C++ 相同的连杆参数（底座 105mm）
    kin = SimpleArmKinematics([105.0, 150.0, 150.0, 80.0])
    expected = kin.inverse_kinematics(pose)
    check(expected is not None,
          f"独立逆解成功 {expected}" if expected else
          f"独立逆解拒绝: {kin.last_reject_reason}")
    if expected is None:
        return 1

    print(f"\n  独立复算 IK 角 : {expected}")
    max_diff = max(abs(a - b) for a, b in zip(ik_angles, expected))
    # 容差来自 tick 量化：一个 tick 约 period/4096 us，折合约 0.49°
    tolerance = period_us / 4096.0 / 2000.0 * 180.0
    check(max_diff <= tolerance + 1e-9,
          f"下发角度与独立复算一致（最大差 {max_diff:.4f}°，量化容差 {tolerance:.4f}°）")

    print("\n" + "=" * 74)
    if failures:
        print(f"  ❌ {len(failures)} 项未通过：")
        for f in failures:
            print(f"     - {f}")
    else:
        print("  ✅ 主程序下发路径正确（假总线捕获，非实物舵机行为）")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
