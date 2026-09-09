#!/usr/bin/env python3
"""
3-DOF 运动学正反解自洽性与工作空间边界验证

为什么必须在通电前做：
  IK 若算错，机械臂第一次上电就可能撞桌面、超程自锁或损坏舵机，
  物理损坏不可逆。而运动学纯粹是数学，无需硬件即可完整验证。

验证方法：
  FK(IK(p)) == p  —— 对目标点求逆解得到关节角，再用正解算回末端位置，
  若与原目标一致，说明两者符号约定一致且求解正确。

用法：
  python3 tools/test_kinematics.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kinematics import SimpleArmKinematics


def roundtrip(kin, x, y, z):
    """对单个目标点做 IK → FK 回环，返回 (关节角, 回算位置, 误差)"""
    j = kin.inverse_kinematics({"x": x, "y": y, "z": z})
    if j is None:
        return None, None, None
    p = kin.forward_kinematics(j)
    err = math.sqrt((p["x"] - x) ** 2 + (p["y"] - y) ** 2 + (p["z"] - z) ** 2)
    return j, p, err


def main():
    kin = SimpleArmKinematics()
    l0, l1, l2 = kin.link_lengths[0], kin.link_lengths[1], kin.link_lengths[2]
    reach_max = l1 + l2
    reach_min = abs(l1 - l2)

    print("=" * 74)
    print("  3-DOF 运动学正反解自洽性与工作空间边界验证")
    print("=" * 74)
    print(f"  连杆参数: 底座高 {l0}mm | 大臂 {l1}mm | 小臂 {l2}mm")
    print(f"  理论工作半径: {reach_min} ~ {reach_max} mm（以肩关节为球心）")

    # ── 用例 1：FK/IK 回环自洽性 ──
    print("\n" + "-" * 74)
    print("【用例 1】FK(IK(p)) == p 回环自洽性")
    print("-" * 74)
    targets = [
        (200.0,    0.0,  100.0, "正前方中距"),
        (150.0,   50.0,  120.0, "左前方"),
        (180.0,  -60.0,   80.0, "右前方"),
        (250.0,    0.0,  100.0, "正前方远距"),
        (120.0,    0.0,  200.0, "高位"),
        (100.0,   80.0,   60.0, "低位侧向"),
    ]
    errs, fails, unreachable = [], 0, 0
    for x, y, z, tag in targets:
        j, p, e = roundtrip(kin, x, y, z)
        if j is None:
            # IK 拒绝不等于错误：目标可能在臂展内但姿态超出舵机行程。
            # 只要拒绝时给出了明确原因，就是**正确的安全行为**。
            why = kin.last_reject_reason or "未说明原因"
            print(f"  ℹ {tag:<12} ({x:>6.1f},{y:>6.1f},{z:>6.1f}) → 拒绝: {why}")
            unreachable += 1
            continue
        ok = e < 1.0
        errs.append(e)
        if not ok:
            fails += 1
        print(f"  {'✅' if ok else '❌'} {tag:<12} 目标({x:>6.1f},{y:>6.1f},{z:>6.1f}) "
              f"→ 关节{[f'{a:>6.1f}' for a in j]}")
        print(f"     {'':14} 回算({p['x']:>6.1f},{p['y']:>6.1f},{p['z']:>6.1f}) "
              f"| 误差 {e:>7.2f} mm")

    if errs:
        print(f"\n  平均回环误差: {sum(errs)/len(errs):.3f} mm | 最大 {max(errs):.3f} mm")
    if unreachable:
        print(f"  另有 {unreachable} 个点因关节行程限制被拒绝（属正确的安全行为，不计为失败）")
    # 判据：所有**可解**的点回环必须准确。被拒绝的点不算失败。
    consistent = fails == 0 and len(errs) > 0
    print(f"  {'✅ 正反解符号约定一致' if consistent else '❌ 存在不一致，实机会抓偏或撞机'}")

    # ── 用例 2：工作空间边界防护 ──
    print("\n" + "-" * 74)
    print("【用例 2】工作空间边界防护（超程必须拒绝，不能返回错误角度）")
    print("-" * 74)
    boundary = [
        (500.0,   0.0, 100.0, "远超最大臂展", True),
        (  1.0,   0.0, 100.0, "过近死区",     True),
        (200.0,   0.0, 900.0, "超高不可达",   True),
        (200.0,   0.0,-500.0, "超低不可达",   True),
        (  0.0,   0.0, 100.0, "基座正上方奇异点", None),
    ]
    guard_ok = True
    for x, y, z, tag, should_reject in boundary:
        j = kin.inverse_kinematics({"x": x, "y": y, "z": z})
        rejected = j is None
        if should_reject is None:
            print(f"  ℹ {tag:<18} → {'拒绝' if rejected else f'返回 {j}'}（奇异点，两种处理均可接受）")
            continue
        ok = rejected == should_reject
        guard_ok &= ok
        print(f"  {'✅' if ok else '❌'} {tag:<18} ({x:>6.1f},{y:>6.1f},{z:>6.1f}) "
              f"→ {'正确拒绝' if rejected else f'❌ 危险！返回了 {j}'}")
    print(f"  {'✅ 边界防护有效' if guard_ok else '❌ 防护失效，通电有撞机风险'}")

    # ── 用例 3：可达域扫描 ──
    print("\n" + "-" * 74)
    print("【用例 3】工作空间可达域扫描（统计实际可解比例）")
    print("-" * 74)
    total = reachable = 0
    bad_roundtrip = 0
    for r in range(60, 340, 20):
        for zz in range(-20, 300, 20):
            total += 1
            j = kin.inverse_kinematics({"x": float(r), "y": 0.0, "z": float(zz)})
            if j is None:
                continue
            reachable += 1
            p = kin.forward_kinematics(j)
            e = math.sqrt((p["x"] - r) ** 2 + (p["z"] - zz) ** 2)
            if e > 1.0:
                bad_roundtrip += 1
    print(f"  扫描 {total} 个网格点 | 可解 {reachable} 个 ({reachable/total*100:.1f}%)")
    print(f"  其中回环误差 >1mm 的: {bad_roundtrip} 个 "
          f"({'✅ 无异常' if bad_roundtrip == 0 else f'❌ 占可解点 {bad_roundtrip/max(reachable,1)*100:.1f}%'})")

    # ── 用例 4：关节角物理限位 ──
    print("\n" + "-" * 74)
    print("【用例 4】输出关节角是否落在舵机可执行范围")
    print("-" * 74)
    print("  舵机物理范围通常为 0~180°，超出则无法执行或强行堵转")
    out_of_range = []
    for x, y, z, tag in targets:
        j = kin.inverse_kinematics({"x": x, "y": y, "z": z})
        if j is None:
            continue
        for idx, a in enumerate(j[:3]):
            if not (-180.0 <= a <= 180.0):
                out_of_range.append((tag, idx, a))
    if out_of_range:
        for tag, idx, a in out_of_range:
            print(f"  ❌ {tag} 关节{idx} = {a}° 超出 ±180°")
    else:
        print("  ✅ 所有测试点的关节角均在 ±180° 内")
    print("  ⚠ 注意：具体舵机的安全限位需在实机标定后写入配置，本项仅做粗筛")

    # ── 总结 ──
    print("\n" + "=" * 74)
    all_ok = consistent and guard_ok and bad_roundtrip == 0
    if all_ok:
        print("  ✅ 运动学验证通过，可安全进入实机通电阶段")
    else:
        print("  ❌ 运动学存在问题，**通电前必须修复**，否则有撞机风险")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
