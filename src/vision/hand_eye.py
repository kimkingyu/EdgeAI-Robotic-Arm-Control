"""
Eye-to-Hand 手眼标定：像素坐标 → 机械臂基座物理坐标

为什么用单应矩阵而不是完整 PnP：
  本项目是 3-DOF 机械臂在**固定工作平面**上抓取（工件都放在桌面同一高度）。
  这种平面场景下，像素平面到物理平面是一个射影变换，用单应矩阵 H 即可精确描述：

      [x_mm]       [u]
      [y_mm] ~ H · [v]
      [  1  ]      [1]

  相比完整 PnP 的优势：
    - 不需要标定相机内参与畸变系数（省掉棋盘格多姿态采集）
    - 只需 4 组以上「像素点 ↔ 实测物理坐标」对应点即可求解
    - 直接吸收了镜头畸变、安装倾角等系统误差，工程上更鲁棒

  标定作业流程（硬件到货后）：
    1. 在工作平面上放 4~9 个标记点（贴纸/角点均可）
    2. 相机拍一张，记下每个点的像素坐标 (u, v)
    3. 手动示教机械臂末端依次触碰各点，读取基座坐标 (x, y)
    4. 把对应点喂给 calibrate()，保存标定文件
"""
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class HandEyeCalibrator:
    """平面手眼标定求解器（像素 ↔ 基座物理坐标）"""

    def __init__(self, calib_path: str = "configs/hand_eye_calib.json"):
        self.calib_path = calib_path
        self.H: Optional["np.ndarray"] = None       # 像素 → 物理
        self.H_inv: Optional["np.ndarray"] = None   # 物理 → 像素（用于反投影校验）
        self.z_plane: float = 30.0                  # 工作平面高度 (mm)
        self.rms_error: Optional[float] = None
        self.n_points: int = 0

    # ── 标定求解 ────────────────────────────────────────────────────────

    def calibrate(self, pixel_pts: Sequence[Sequence[float]],
                  robot_pts: Sequence[Sequence[float]],
                  z_plane: float = 30.0) -> bool:
        """
        用对应点求解单应矩阵

        pixel_pts : [[u1,v1], [u2,v2], ...]  相机像素坐标，至少 4 组
        robot_pts : [[x1,y1], [x2,y2], ...]  对应的机械臂基座坐标 (mm)
        z_plane   : 该工作平面在基座系下的高度 (mm)
        """
        if not (HAS_NUMPY and HAS_CV2):
            print("[HandEye] 缺少 numpy / opencv，无法标定")
            return False

        src = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(robot_pts, dtype=np.float64).reshape(-1, 2)

        if len(src) != len(dst):
            print(f"[HandEye] 点数不匹配: 像素 {len(src)} vs 物理 {len(dst)}")
            return False
        if len(src) < 4:
            print(f"[HandEye] 至少需要 4 组对应点，当前 {len(src)} 组")
            return False

        # 点数 > 4 时用 RANSAC 抑制单点示教误差；恰好 4 点则直接精确求解
        if len(src) > 4:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else len(src)
            if inliers < 4:
                print(f"[HandEye] RANSAC 内点不足 ({inliers})，改用最小二乘")
                H, _ = cv2.findHomography(src, dst, 0)
        else:
            H, _ = cv2.findHomography(src, dst, 0)

        if H is None:
            print("[HandEye] 单应矩阵求解失败（点可能共线或退化）")
            return False

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.z_plane = float(z_plane)
        self.n_points = len(src)
        self.rms_error = self._compute_rms(src, dst)

        print(f"[HandEye] 标定完成 | {self.n_points} 组点 | "
              f"RMS 重投影误差 {self.rms_error:.3f} mm")
        if self.rms_error > 5.0:
            print("           ⚠ 误差偏大，建议检查示教精度或增加标记点")
        return True

    def _compute_rms(self, src: "np.ndarray", dst: "np.ndarray") -> float:
        """标定质量自检：把像素点正向映射后与实测物理坐标比对"""
        errs = []
        for (u, v), (gx, gy) in zip(src, dst):
            p = self._apply(self.H, u, v)
            errs.append(((p[0] - gx) ** 2 + (p[1] - gy) ** 2) ** 0.5)
        return float(np.sqrt(np.mean(np.square(errs))))

    @staticmethod
    def _apply(M: "np.ndarray", a: float, b: float) -> Tuple[float, float]:
        """射影变换：齐次坐标相乘后归一化"""
        v = M @ np.array([a, b, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(v[0] / v[2]), float(v[1] / v[2])

    # ── 坐标变换 ────────────────────────────────────────────────────────

    def pixel_to_robot(self, u: float, v: float,
                       z: Optional[float] = None) -> Optional[Dict[str, float]]:
        """像素坐标 → 机械臂基座坐标"""
        if self.H is None:
            return None
        x, y = self._apply(self.H, u, v)
        if x != x or y != y:      # NaN 检查
            return None
        return {"x": round(x, 2), "y": round(y, 2),
                "z": float(z if z is not None else self.z_plane)}

    def robot_to_pixel(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """基座坐标 → 像素坐标（用于把规划结果画回画面做可视化校验）"""
        if self.H_inv is None:
            return None
        u, v = self._apply(self.H_inv, x, y)
        return (None if u != u else (round(u, 1), round(v, 1)))

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    # ── 持久化 ──────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> bool:
        if self.H is None:
            print("[HandEye] 尚未标定，无内容可保存")
            return False
        p = path or self.calib_path
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "homography": self.H.tolist(),
                "z_plane": self.z_plane,
                "n_points": self.n_points,
                "rms_error_mm": self.rms_error,
            }, f, ensure_ascii=False, indent=2)
        print(f"[HandEye] 标定参数已保存至 {p}")
        return True

    def load(self, path: Optional[str] = None) -> bool:
        p = path or self.calib_path
        if not os.path.exists(p):
            return False
        if not HAS_NUMPY:
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.H = np.asarray(d["homography"], dtype=np.float64)
            self.H_inv = np.linalg.inv(self.H)
            self.z_plane = float(d.get("z_plane", 30.0))
            self.n_points = int(d.get("n_points", 0))
            self.rms_error = d.get("rms_error_mm")
            print(f"[HandEye] 已载入标定参数 {p} "
                  f"(RMS {self.rms_error} mm, {self.n_points} 点)")
            return True
        except Exception as e:
            print(f"[HandEye] 载入失败: {str(e)[:80]}")
            return False
