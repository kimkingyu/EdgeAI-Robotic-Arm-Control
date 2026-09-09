#!/usr/bin/env python3
"""
C++ 与 Python 两套手眼变换的交叉比对。

为什么需要：C++ 侧 HandEyeTransformer 与 Python 侧 HandEyeCalibrator 各自
实现了像素→基座的射影变换与标定文件解析。两者若在矩阵解析顺序、齐次归一化
或退化判定上不一致，实机会出现"Python 仿真抓得准、C++ 实时线程抓偏"。
本项目此前已在 PWM 换算上栽过同类跟头（截断 vs 四舍五入差 1 tick），
因此新增实现必须先做交叉比对再接入控制路径。

本脚本不访问相机或 I2C 设备。

用法：
  python3 tools/test_handeye_parity.py --cpp-dump ./handeye_dump
"""
import argparse
import json
import os
import subprocess
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


def run_cpp(dump, calib_path, points):
    payload = "\n".join(f"{u} {v}" for u, v in points)
    proc = subprocess.run([dump, calib_path], input=payload,
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        print(f"C++ 导出程序退出码 {proc.returncode}\n{proc.stderr[:1500]}")
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"无法解析 C++ 输出: {exc}\n{proc.stdout[:1500]}")
        return None


def main():
    parser = argparse.ArgumentParser(description="C++/Python 手眼变换交叉比对")
    parser.add_argument("--cpp-dump", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.cpp_dump):
        print(f"未找到 {args.cpp_dump}（构建缺失，非比对结论）")
        return 2

    print("=" * 74)
    print("  C++ 与 Python 手眼变换交叉比对")
    print("=" * 74)

    tmpdir = tempfile.mkdtemp()
    calib_path = os.path.join(tmpdir, "hand_eye_calib.json")

    # 用与 test_hand_eye.py 同源的真值构造标定：约 0.42mm/px、3° 旋转、含透视项
    theta = np.deg2rad(3.0)
    s = 0.42
    H_gt = np.array([
        [s * np.cos(theta), -s * np.sin(theta), 120.0],
        [s * np.sin(theta), s * np.cos(theta), -65.0],
        [1.2e-5, 3.0e-5, 1.0],
    ])
    us = np.linspace(120, 520, 3)
    vs = np.linspace(100, 380, 3)
    px = np.array([[u, v] for v in vs for u in us], dtype=np.float64)
    h = np.hstack([px, np.ones((len(px), 1))])
    out = (H_gt @ h.T).T
    rb = out[:, :2] / out[:, 2:3]

    calib = HandEyeCalibrator()
    if not calib.calibrate(px, rb, z_plane=30.0):
        print("Python 侧标定失败，无法比对")
        return 2
    calib.save(calib_path)

    # 覆盖画面内外与边界的测试点
    points = [(u, v) for u in range(0, 660, 30) for v in range(0, 500, 30)]
    points += [(320.5, 240.5), (0.0, 0.0), (639.0, 479.0), (-50.0, -50.0)]

    cpp = run_cpp(args.cpp_dump, calib_path, points)
    if cpp is None:
        return 2

    check(cpp["loaded"], "C++ 成功载入 Python 产出的标定文件")
    if not cpp["loaded"]:
        print("  载入失败则后续比对无意义，提前结束")
        return 1
    check(abs(cpp["z_plane"] - calib.z_plane) < 1e-9,
          f"z_plane 一致 (C++ {cpp['z_plane']} / Python {calib.z_plane})")
    check(len(cpp["points"]) == len(points),
          f"返回点数一致 ({len(cpp['points'])}/{len(points)})")

    max_diff = 0.0
    mismatch = []
    invalid_diff = 0
    for (u, v), c in zip(points, cpp["points"]):
        p = calib.pixel_to_robot(u, v)
        py_valid = p is not None
        if c["valid"] != py_valid:
            invalid_diff += 1
            continue
        if not py_valid:
            continue
        # Python 侧输出按 0.01mm 量化(round 2 位)，比对时按该精度对齐
        dx = abs(round(c["x"], 2) - p["x"])
        dy = abs(round(c["y"], 2) - p["y"])
        dz = abs(c["z"] - p["z"])
        d = max(dx, dy, dz)
        max_diff = max(max_diff, d)
        if d > 1e-9:
            mismatch.append((u, v, c["x"], c["y"], p["x"], p["y"], d))

    check(invalid_diff == 0, f"有效性判定一致（分歧 {invalid_diff} 个）")
    check(not mismatch, f"坐标解算一致（最大偏差 {max_diff:.2e} mm）")
    if mismatch:
        print("\n  分歧样例（最多 5 条）:")
        for u, v, cx, cy, pxx, pyy, d in mismatch[:5]:
            print(f"     ({u},{v}) C++=({cx:.4f},{cy:.4f}) Python=({pxx:.4f},{pyy:.4f}) 差 {d:.4e}")

    # 反投影自洽性：C++ 侧正反变换应回到原像素
    back_errs = [max(abs(c["back_u"] - u), abs(c["back_v"] - v))
                 for (u, v), c in zip(points, cpp["points"])
                 if c["valid"] and "back_u" in c]
    check(back_errs and max(back_errs) < 1e-3,
          f"C++ 正反变换自洽（最大回环 {max(back_errs):.2e} px）" if back_errs else "无反投影数据")

    # 退化标定文件：C++ 必须拒绝载入，与 Python 侧行为一致
    print("\n  退化标定文件处理:")
    bad_cases = {
        "奇异矩阵": [[1, 2, 3], [2, 4, 6], [3, 6, 9]],
        "末行全零": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    }
    for tag, matrix in bad_cases.items():
        bad_path = os.path.join(tmpdir, f"bad_{abs(hash(tag))}.json")
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump({"homography": matrix, "z_plane": 30.0}, f)
        c = run_cpp(args.cpp_dump, bad_path, [(320, 240)])
        py = HandEyeCalibrator()
        py_loaded = py.load(bad_path)
        if c is None:
            failures.append(f"{tag} C++ 执行失败")
            continue
        check(not c["loaded"] and not py_loaded,
              f"{tag}: 双方均拒绝载入 (C++={c['loaded']}, Python={py_loaded})")
        os.remove(bad_path)

    os.remove(calib_path)
    os.rmdir(tmpdir)

    print("\n" + "=" * 74)
    if failures:
        print(f"  ❌ {len(failures)} 项未通过：")
        for f in failures:
            print(f"     - {f}")
    else:
        print("  ✅ 两套手眼变换一致（合成标定数据，非实机精度）")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
