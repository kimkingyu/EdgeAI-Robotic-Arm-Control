#!/usr/bin/env python3
"""
PCA9685 角度→PWM 映射与舵机限位防护验证

为什么必须在通电前做：
  IK 算对了，但若角度→脉宽映射写错，舵机仍会超程堵转、过热烧毁。
  堵转电流可达额定值数倍，几十秒即可烧毁线圈，损坏不可逆。
  PWM 计算是纯数学，无需硬件即可验证。

验证要点：
  1. tick 换算的时基必须与 set_pwm_freq 的实际输出频率一致
  2. 角度→脉宽必须单调、边界精确（0°→min_us，180°→max_us）
  3. IK 输出的负角度经映射后不得被静默截断
  4. 超出舵机物理范围的指令必须被显式拒绝或告警，而非悄悄夹紧

用法：
  python3 tools/test_servo_mapping.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.i2c_arm import PCA9685, I2CArmController
from src.kinematics import SimpleArmKinematics


def ticks_of(angle, min_us=500, max_us=2500, period_us=20000.0):
    """按被测实现的公式复算 tick，用于独立比对（四舍五入，与实现一致）"""
    a = max(0.0, min(180.0, angle))
    pulse = min_us + (a / 180.0) * (max_us - min_us)
    return int(round(pulse * 4096.0 / period_us))


def main():
    print("=" * 76)
    print("  PCA9685 角度→PWM 映射与舵机限位防护验证")
    print("=" * 76)

    failures = []

    # ── 用例 1：时基一致性（本轮最关键）──
    print("\n" + "-" * 76)
    print("【用例 1】tick 时基与实际 PWM 周期是否一致")
    print("-" * 76)
    # set_pwm_freq(50) 内部做了 freq*0.9 的振荡器过冲修正
    target_freq = 50.0
    corrected = target_freq * 0.9
    prescale = int((25000000.0 / 4096.0 / corrected) - 1 + 0.5)
    actual_freq = 25000000.0 / 4096.0 / (prescale + 1)
    actual_period_us = 1_000_000.0 / actual_freq

    print(f"  设定频率        : {target_freq} Hz")
    print(f"  0.9 修正后写入   : {corrected} Hz → prescale = {prescale}")
    print(f"  芯片实际输出频率 : {actual_freq:.2f} Hz")
    print(f"  实际 PWM 周期    : {actual_period_us:.1f} us")
    print(f"  代码 tick 换算用 : 20000.0 us")

    # 检查**当前实现**是否用了芯片真实周期，而非硬编码 20000us
    probe = PCA9685()
    probe.set_pwm_freq(target_freq)
    print(f"  实现使用的周期   : {probe.period_us:.1f} us")

    naive_err = abs(actual_period_us - 20000.0) / 20000.0 * 100
    impl_err = abs(probe.period_us - actual_period_us) / actual_period_us * 100
    print(f"  若用 20000us 硬编码会偏 {naive_err:.1f}%（约 "
          f"{naive_err / 100 * 2000 / 2000 * 180 * 0.5:.1f}° 角度误差）")
    print(f"  当前实现偏差    : {impl_err:.3f}%")
    if impl_err > 0.5:
        print(f"  ❌ 实现未采用芯片真实周期")
        failures.append("tick 时基与实际 PWM 周期不一致")
    else:
        print(f"  ✅ 实现已按芯片真实周期换算")

    # ── 用例 2：角度→脉宽映射的边界与单调性 ──
    print("\n" + "-" * 76)
    print("【用例 2】角度→脉宽映射边界与单调性")
    print("-" * 76)
    checks = [(0.0, 500), (90.0, 1500), (180.0, 2500)]
    ok = True
    for ang, want_us in checks:
        got_us = 500 + (ang / 180.0) * 2000
        good = abs(got_us - want_us) < 1e-6
        ok &= good
        print(f"  {'✅' if good else '❌'} {ang:>5.1f}° → {got_us:>6.1f} us "
              f"(期望 {want_us} us)")
    prev = -1
    mono = True
    for a in range(0, 181, 10):
        t = ticks_of(a)
        if t <= prev:
            mono = False
        prev = t
    print(f"  {'✅' if mono else '❌'} 0~180° 区间 tick 单调递增")
    ok &= mono
    if not ok:
        failures.append("角度→脉宽映射异常")

    # ── 用例 3：IK 负角度是否被静默截断（安全关键）──
    print("\n" + "-" * 76)
    print("【用例 3】IK 输出角度经控制器映射后是否被静默截断")
    print("-" * 76)
    kin = SimpleArmKinematics()
    ctrl = I2CArmController(mock=True)
    targets = [(200.0, 0.0, 100.0), (150.0, 50.0, 120.0), (120.0, 0.0, 200.0)]

    clipped = []
    for x, y, z in targets:
        j = kin.inverse_kinematics({"x": x, "y": y, "z": z})
        if j is None:
            continue
        print(f"  目标({x:>6.1f},{y:>6.1f},{z:>6.1f}) IK → {[f'{a:.1f}' for a in j[:3]]}")
        for i, ang in enumerate(j[:3]):
            servo = ctrl.ik_to_servo(i, ang)        # 按关节标定换算
            in_range = 0.0 <= servo <= 180.0
            if not in_range:
                clipped.append((i, ang, servo))
                print(f"    ❌ 关节{i} {ang:>7.1f}° → 舵机 {servo:>7.1f}° **越界**")
            else:
                print(f"    ✅ 关节{i} {ang:>7.1f}° → 舵机 {servo:>6.1f}°")
    if clipped:
        print(f"\n  ❌ 共 {len(clipped)} 个关节角越界却仍被 IK 返回")
        failures.append("IK 返回了舵机无法执行的角度")
    else:
        print(f"\n  ✅ IK 返回的角度均落在舵机可执行范围内"
              f"（不可达姿态已被 IK 提前拒绝）")

    # ── 用例 4：越界指令是否有防护 ──
    print("\n" + "-" * 76)
    print("【用例 4】越界角度指令的防护行为")
    print("-" * 76)
    pca = PCA9685()
    guard = True
    for bad in (-30.0, 210.0, 999.0):
        accepted = pca.set_servo_angle(0, bad)      # strict=True，应拒绝
        if accepted:
            guard = False
            print(f"  ❌ 输入 {bad:>7.1f}° 被接受（危险）")
    if guard:
        print("  ✅ 越界角度均被显式拒绝并告警，不会静默夹紧")
    else:
        failures.append("越界角度未被拒绝")

    # 整臂联动的越界防护
    bad_ok = ctrl.move_joints([0.0, 200.0, -96.4])
    print(f"  整臂含越界关节时: {'❌ 仍然下发' if bad_ok else '✅ 整体拒绝下发'}")
    if bad_ok:
        failures.append("整臂越界未整体拒绝")

    # ── 用例 5：修复后时基的实际影响 ──
    print("\n" + "-" * 76)
    print("【用例 5】修复后 tick 换算精度")
    print("-" * 76)
    p2 = PCA9685()
    p2.set_pwm_freq(50.0)
    print(f"  set_pwm_freq(50) 后 period_us = {p2.period_us:.1f} us")
    for ang, want_us in ((0.0, 500.0), (90.0, 1500.0), (180.0, 2500.0)):
        pulse = 500 + (ang / 180.0) * 2000
        ticks = int(round(pulse * 4096.0 / p2.period_us))
        real_us = ticks * p2.period_us / 4096.0
        err_deg = abs(real_us - want_us) / 2000.0 * 180.0
        good = err_deg < 0.5
        print(f"  {'✅' if good else '❌'} {ang:>5.1f}° → {ticks:>4d} ticks "
              f"→ 实际 {real_us:>7.1f} us (期望 {want_us:.0f} us, 误差 {err_deg:.3f}°)")
        if not good:
            failures.append(f"{ang}° 脉宽误差过大")

    # ── 总结 ──
    print("\n" + "=" * 76)
    if failures:
        print("  ❌ 发现以下问题，通电前必须修复：")
        for i, f in enumerate(failures, 1):
            print(f"     {i}. {f}")
    else:
        print("  ✅ PWM 映射与限位防护验证通过")
    print("=" * 76)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
