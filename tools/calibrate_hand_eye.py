#!/usr/bin/env python3
"""
手眼标定作业工具（硬件到货后使用）

标定原理：本项目为 3-DOF 机械臂在固定工作平面抓取，像素平面到物理平面
是射影变换，用单应矩阵描述即可，无需标定相机内参。

作业步骤：
  1. 在工作台上放 4~9 个标记点（贴纸、螺栓孔、棋盘格角点均可）
  2. 运行本工具，相机拍照并保存
  3. 在图上读出每个标记点的像素坐标 (u, v)
  4. 手动示教机械臂末端逐个触碰标记点，记录基座坐标 (x, y)
  5. 把对应点填入 --points 或交互输入，求解并保存

用法：
  # 先拍一张标定图
  python3 tools/calibrate_hand_eye.py --capture

  # 交互输入对应点并求解
  python3 tools/calibrate_hand_eye.py --interactive

  # 直接从 JSON 文件读取对应点
  python3 tools/calibrate_hand_eye.py --points calib_points.json

  # 验算已有标定
  python3 tools/calibrate_hand_eye.py --verify 320 240
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.hand_eye import HandEyeCalibrator

CALIB_PATH = "configs/hand_eye_calib.json"


def capture_image(device: int, out: str) -> bool:
    try:
        import cv2
    except ImportError:
        print("[错误] 需要 opencv")
        return False
    if not os.path.exists(f"/dev/video{device}"):
        print(f"[错误] /dev/video{device} 不存在，请确认相机已接入")
        return False
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("[错误] 相机打开失败")
        return False
    for _ in range(10):        # 丢弃前几帧，等自动曝光稳定
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[错误] 取帧失败")
        return False
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, frame)
    h, w = frame.shape[:2]
    print(f"[完成] 标定图已保存 {out} ({w}x{h})")
    print("       请在图片中读出各标记点的像素坐标，然后运行 --interactive")
    return True


def input_points():
    """交互式录入对应点"""
    print("\n逐个录入标记点（至少 4 组，留空结束）")
    print("格式：像素u 像素v 机械臂x 机械臂y      例如：  210 168 199.2 2.3\n")
    px, rb = [], []
    while True:
        try:
            line = input(f"点 #{len(px) + 1} >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        parts = line.replace(",", " ").split()
        if len(parts) != 4:
            print("  格式错误，需要 4 个数字")
            continue
        try:
            u, v, x, y = (float(p) for p in parts)
        except ValueError:
            print("  含非法数字")
            continue
        px.append([u, v])
        rb.append([x, y])
        print(f"  已录入 {len(px)} 组")
    return px, rb


def main():
    ap = argparse.ArgumentParser(description="手眼标定作业工具")
    ap.add_argument("--capture", action="store_true", help="拍摄标定图")
    ap.add_argument("--device", type=int, default=0, help="相机设备号")
    ap.add_argument("--image", default="data/calibration/hand_eye_board.jpg")
    ap.add_argument("--interactive", action="store_true", help="交互录入对应点")
    ap.add_argument("--points", default="", help="从 JSON 读取对应点")
    ap.add_argument("--z-plane", type=float, default=30.0, help="工作平面高度 mm")
    ap.add_argument("--output", default=CALIB_PATH)
    ap.add_argument("--verify", nargs=2, type=float, metavar=("U", "V"),
                    help="用已有标定验算某像素点")
    args = ap.parse_args()

    if args.capture:
        capture_image(args.device, args.image)
        return

    if args.verify:
        c = HandEyeCalibrator(args.output)
        if not c.load():
            print(f"[错误] 未找到标定文件 {args.output}")
            sys.exit(1)
        u, v = args.verify
        r = c.pixel_to_robot(u, v)
        back = c.robot_to_pixel(r["x"], r["y"])
        print(f"\n像素 ({u}, {v})")
        print(f"  → 基座坐标 x={r['x']} mm, y={r['y']} mm, z={r['z']} mm")
        print(f"  → 反投影校验 {back}（应接近输入像素）")
        return

    if args.points:
        if not os.path.isfile(args.points):
            print(f"[错误] 对应点文件不存在: {args.points}")
            sys.exit(1)
        try:
            with open(args.points, "r", encoding="utf-8") as f:
                d = json.load(f)
            px, rb = d["pixel_points"], d["robot_points"]
        except json.JSONDecodeError as e:
            print(f"[错误] {args.points} 不是合法 JSON: {e}")
            print('       期望格式: {"pixel_points": [[u,v],...], '
                  '"robot_points": [[x,y],...]}')
            sys.exit(1)
        except KeyError as e:
            print(f"[错误] 缺少字段 {e}")
            print('       期望格式: {"pixel_points": [[u,v],...], '
                  '"robot_points": [[x,y],...]}')
            sys.exit(1)
    elif args.interactive:
        px, rb = input_points()
    else:
        ap.print_help()
        return

    if len(px) < 4:
        print(f"\n[错误] 至少需要 4 组对应点，当前 {len(px)} 组")
        sys.exit(1)

    c = HandEyeCalibrator(args.output)
    if not c.calibrate(px, rb, z_plane=args.z_plane):
        sys.exit(1)
    c.save()

    print("\n标定结果自检（把标定用的像素点回代）：")
    for (u, v), (gx, gy) in zip(px, rb):
        r = c.pixel_to_robot(u, v)
        e = ((r["x"] - gx) ** 2 + (r["y"] - gy) ** 2) ** 0.5
        flag = "✅" if e < 2.0 else ("⚠" if e < 5.0 else "❌")
        print(f"  {flag} 像素({u:>6.1f},{v:>6.1f}) → ({r['x']:>7.2f},{r['y']:>7.2f}) "
              f"| 实测({gx:>7.2f},{gy:>7.2f}) | 偏差 {e:.2f} mm")

    print(f"\n标定已保存至 {args.output}，pipeline 会自动加载。")


if __name__ == "__main__":
    main()
