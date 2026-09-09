#!/usr/bin/env python3
"""
手眼标定的退化与异常输入健壮性验证。

为什么需要：tools/test_hand_eye.py 验证的是「输入良好时算得对不对」，
但实机标定是人工示教的，真实故障恰恰来自坏输入：
标记点被摆成一条直线、两个点记重了、示教时读数漏填成 NaN、
标定文件被手工编辑坏了。这些情况若直接崩溃或悄悄返回垃圾坐标，
后果是机械臂朝一个错误位置扎下去。

本脚本不访问任何硬件。

用法：
  python3 tools/test_hand_eye_robust.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.vision.hand_eye import HandEyeCalibrator

failures = []


def check(condition, label):
    if condition:
        print(f"  ✅ {label}")
    else:
        print(f"  ❌ {label}")
        failures.append(label)


def guarded(label, fn):
    """执行 fn，把未捕获异常记为失败而不是让整个脚本崩掉。"""
    try:
        return True, fn()
    except Exception as exc:
        print(f"  ❌ {label} 抛出未捕获异常: {type(exc).__name__}: {str(exc)[:120]}")
        failures.append(f"{label} 抛出 {type(exc).__name__}")
        return False, None


def main():
    print("=" * 74)
    print("  手眼标定退化与异常输入健壮性验证")
    print("=" * 74)

    # ── 用例1：共线点（标记点摆成一条直线）──
    print("\n【用例1】共线标记点必须被拒绝，不能返回退化矩阵")
    collinear_px = [[100, 100], [200, 200], [300, 300], [400, 400], [500, 500]]
    collinear_rb = [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]
    c = HandEyeCalibrator()
    ok, result = guarded("共线标定", lambda: c.calibrate(collinear_px, collinear_rb))
    if ok:
        if result:
            # 即便求出矩阵，也必须能安全变换而非产生 NaN/inf
            _, p = guarded("共线后变换", lambda: c.pixel_to_robot(250, 250))
            finite = p is None or (np.isfinite(p["x"]) and np.isfinite(p["y"]))
            check(finite, "共线求解后变换不产生 NaN/inf（或明确返回 None）")
        else:
            check(True, "共线点被拒绝")
        check(True, "共线输入未导致崩溃")

    # ── 用例2：重复点 ──
    print("\n【用例2】全部重复的点")
    dup_px = [[100, 100]] * 5
    dup_rb = [[10, 10]] * 5
    c = HandEyeCalibrator()
    ok, result = guarded("重复点标定", lambda: c.calibrate(dup_px, dup_rb))
    if ok:
        check(True, "重复点输入未导致崩溃")
        if result:
            _, p = guarded("重复点后变换", lambda: c.pixel_to_robot(150, 150))
            finite = p is None or (np.isfinite(p["x"]) and np.isfinite(p["y"]))
            check(finite, "重复点求解后变换不产生 NaN/inf（或返回 None）")

    # ── 用例3：点数不匹配 ──
    print("\n【用例3】像素点与物理点数量不匹配")
    c = HandEyeCalibrator()
    ok, result = guarded("数量不匹配", lambda: c.calibrate(
        [[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]], [[0, 0], [1, 0], [1, 1], [0, 1]]))
    if ok:
        check(result is False, "数量不匹配被拒绝")
        check(not c.is_calibrated, "拒绝后未置为已标定状态")

    # ── 用例4：含 NaN / inf 的输入 ──
    print("\n【用例4】含 NaN / inf 的示教数据")
    for tag, bad in [("NaN", float("nan")), ("inf", float("inf"))]:
        c = HandEyeCalibrator()
        px = [[100, 100], [400, 100], [400, 300], [100, 300]]
        rb = [[0, 0], [120, 0], [120, 80], [bad, 80]]
        ok, result = guarded(f"含{tag}标定", lambda: c.calibrate(px, rb))
        if not ok:
            continue
        if result:
            _, p = guarded(f"含{tag}后变换", lambda: c.pixel_to_robot(250, 200))
            bad_output = p is not None and not (np.isfinite(p["x"]) and np.isfinite(p["y"]))
            check(not bad_output,
                  f"含{tag}输入不会产出非有限坐标（当前{'返回None' if p is None else '返回有限值'}）")
        else:
            check(True, f"含{tag}输入被拒绝")

    # ── 用例5：正常标定作为对照 ──
    print("\n【用例5】正常输入对照组")
    px = [[100, 100], [400, 100], [400, 300], [100, 300], [250, 200]]
    rb = [[0, 0], [120, 0], [120, 80], [0, 80], [60, 40]]
    c_good = HandEyeCalibrator()
    ok, result = guarded("正常标定", lambda: c_good.calibrate(px, rb, z_plane=30.0))
    check(ok and result, "正常输入标定成功")
    if result:
        p = c_good.pixel_to_robot(250, 200)
        err = ((p["x"] - 60) ** 2 + (p["y"] - 40) ** 2) ** 0.5
        check(err < 0.05, f"中心点定位误差 {err:.4f} mm")
        check(abs(p["z"] - 30.0) < 1e-9, "z 取标定平面高度")

    # ── 用例6：损坏的标定文件 ──
    print("\n【用例6】损坏 / 退化的标定文件")
    tmpdir = tempfile.mkdtemp()
    cases = {
        "非法JSON": "{not json",
        "缺homography字段": json.dumps({"z_plane": 30.0}),
        "矩阵尺寸错误": json.dumps({"homography": [[1, 0], [0, 1]], "z_plane": 30.0}),
        # 秩为1：numpy 对它不抛异常，而是返回元素达 1e16 的数值垃圾
        "奇异矩阵(不可逆)": json.dumps({"homography": [[1, 2, 3], [2, 4, 6], [3, 6, 9]],
                                        "z_plane": 30.0}),
        # 末行全零：射影除法的分母恒为 0
        "末行全零": json.dumps({"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                                 "z_plane": 30.0}),
        "含NaN矩阵": '{"homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, NaN]], "z_plane": 30.0}',
        "良态矩阵(应载入成功)": json.dumps({"homography": [[0.42, 0.0, 120.0],
                                                          [0.0, 0.42, -65.0],
                                                          [0.0, 0.0, 1.0]],
                                            "z_plane": 30.0}),
    }
    for tag, content in cases.items():
        path = os.path.join(tmpdir, f"{abs(hash(tag))}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        c = HandEyeCalibrator()
        ok, loaded = guarded(f"载入{tag}", lambda: c.load(path))
        if not ok:
            continue
        if loaded:
            # 载入成功时不能只查"输出是否有限"——奇异矩阵会给出有限但
            # 完全错误的坐标。必须验证正反变换自洽，否则等于没有校验。
            _, p = guarded(f"{tag}后变换", lambda: c.pixel_to_robot(250, 200))
            if p is None:
                check(True, f"{tag}: 载入后变换安全返回 None")
            else:
                finite = np.isfinite(p["x"]) and np.isfinite(p["y"])
                _, back = guarded(f"{tag}反投影", lambda: c.robot_to_pixel(p["x"], p["y"]))
                consistent = (back is not None
                              and abs(back[0] - 250) < 1.0 and abs(back[1] - 200) < 1.0)
                check(finite and consistent,
                      f"{tag}: 载入成功则正反变换必须自洽"
                      f"（有限={finite}, 回环={back}）")
        else:
            check(True, f"{tag}: 被拒绝载入")
        os.remove(path)
    os.rmdir(tmpdir)

    # ── 用例7：未标定状态下的所有接口 ──
    print("\n【用例7】未标定状态下各接口均安全返回")
    c = HandEyeCalibrator()
    check(c.pixel_to_robot(100, 100) is None, "pixel_to_robot 返回 None")
    check(c.robot_to_pixel(50, 50) is None, "robot_to_pixel 返回 None")
    check(not c.is_calibrated, "is_calibrated 为 False")
    ok, saved = guarded("未标定时保存", lambda: c.save(os.path.join(tempfile.gettempdir(),
                                                                   "should_not_exist.json")))
    if ok:
        check(saved is False, "未标定时保存被拒绝")

    print("\n" + "=" * 74)
    if failures:
        print(f"  ❌ {len(failures)} 项未通过：")
        for f in failures:
            print(f"     - {f}")
    else:
        print("  ✅ 全部通过（合成数据，非实机标定精度）")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
