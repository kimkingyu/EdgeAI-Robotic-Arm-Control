import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any


class KinematicsBase(ABC):
    """机械臂运动学解算基类"""

    def __init__(self, dof: int = 6):
        self.dof = dof

    @abstractmethod
    def forward_kinematics(self, joint_angles: List[float]) -> Dict[str, Any]:
        """正运动学 (FK): 关节角度 (度或弧度) -> 末端位姿 (x, y, z, roll, pitch, yaw)"""
        pass

    @abstractmethod
    def inverse_kinematics(self, target_pose: Dict[str, float], current_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """逆运动学 (IK): 目标位姿 -> 关节角度列表。若无解返回 None"""
        pass


class SimpleArmKinematics(KinematicsBase):
    """通用连杆机械臂几何法/数值法逆解模板 (供初期快速调试与验证)"""

    def __init__(self, link_lengths: Optional[List[float]] = None):
        super().__init__(dof=len(link_lengths) if link_lengths else 4)
        # 各连杆长度 (单位: mm)，例如 [底座高, 大臂, 小臂, 末端夹爪]
        self.link_lengths = link_lengths or [100.0, 150.0, 150.0, 80.0]
        # 最近一次 IK 被拒绝的原因，求解成功时为 None
        self.last_reject_reason: Optional[str] = None

    def forward_kinematics(self, joint_angles: List[float]) -> Dict[str, float]:
        """简易平面投影正解示例"""
        # 默认 angles 为角度制
        rads = [math.radians(a) for a in joint_angles]
        base_angle = rads[0]
        # 俯仰角度叠加
        theta1 = rads[1]
        theta2 = rads[2]

        l1, l2 = self.link_lengths[1], self.link_lengths[2]
        r = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
        z = self.link_lengths[0] + l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
        x = r * math.cos(base_angle)
        y = r * math.sin(base_angle)

        return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}

    # 工作空间安全裕度：贴着理论极限求解会导致关节角接近奇异，
    # 实机上表现为抖动或堵转，故内缩一段作为硬边界。
    REACH_MARGIN_MM = 5.0
    # 肩关节到目标的最小距离，低于此值视为过近死区（小臂对折会撞底座）
    MIN_DIST_MM = 1e-3

    # 各关节的 IK 几何角有效范围（度）。
    # 这是**舵机物理行程**倒推出来的约束：数学上可解不代表舵机转得到。
    # 与 I2CArmController.DEFAULT_JOINT_CALIB 的 offset/direction 对应：
    #   舵机角 = offset + direction * ik_角  ∈ [0, 180]
    JOINT_LIMITS = [
        (-90.0,  90.0),    # 关节0 底座:  offset 90  → 舵机 [0,180]
        (-90.0,  90.0),    # 关节1 大臂:  offset 90  → 舵机 [0,180]
        (-180.0,  0.0),    # 关节2 小臂:  offset 180 → 舵机 [0,180]
        (-90.0,  90.0),    # 关节3 腕部:  offset 90  → 舵机 [0,180]
    ]

    def inverse_kinematics(self, target_pose: Dict[str, float], current_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """
        几何解析逆解（带工作空间边界检查）

        角度约定（与 forward_kinematics 严格一致）：
          base     : 绕 Z 轴航向角，atan2(y, x)
          shoulder : 大臂与水平面夹角，逆时针为正
          elbow    : 小臂相对大臂延长线的转角（**相对角**，非外角）
          FK 中末端位置 = l1*cos(θ1) + l2*cos(θ1+θ2)，故 elbow 必须是相对角
        """
        x = target_pose.get("x", 0.0)
        y = target_pose.get("y", 0.0)
        z = target_pose.get("z", 0.0)

        l1, l2 = self.link_lengths[1], self.link_lengths[2]

        # 1. 基座航向角
        base_deg = math.degrees(math.atan2(y, x))

        # 2. 投影到大小臂所在的竖直平面
        r = math.sqrt(x ** 2 + y ** 2)
        dz = z - self.link_lengths[0]
        dist = math.sqrt(r ** 2 + dz ** 2)

        # 3. 工作空间硬边界检查（含安全裕度，防止贴极限求解）
        reach_max = l1 + l2 - self.REACH_MARGIN_MM
        reach_min = max(abs(l1 - l2) + self.REACH_MARGIN_MM, self.MIN_DIST_MM)
        # 拒绝原因记录在 last_reject_reason 中而非直接打印：
        # IK 在实机上被高频调用，逐次打印会淹没日志；调用方需要时自行读取。
        if dist > reach_max:
            self.last_reject_reason = (
                f"超出最大臂展 (距离 {dist:.1f}mm > {reach_max:.1f}mm)")
            return None
        if dist < reach_min:
            self.last_reject_reason = (
                f"进入过近死区 (距离 {dist:.1f}mm < {reach_min:.1f}mm)，"
                f"继续求解会导致小臂对折撞击底座")
            return None
        self.last_reject_reason = None

        # 4. 余弦定理求肘部相对转角
        #    三角形三边为 l1、l2、dist，夹在 l1 与 l2 之间的内角为 beta
        cos_beta = (l1 ** 2 + l2 ** 2 - dist ** 2) / (2 * l1 * l2)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)
        # elbow 取相对角：小臂完全伸直(beta=180°)时为 0°，对折(beta=0)时为 180°
        # 取负值代表"肘朝上"构型（elbow-up），避免小臂向下扫过桌面
        elbow_deg = -(180.0 - math.degrees(beta))

        # 5. 肩部俯仰角 = 目标方位角 + 三角形内角修正
        alpha1 = math.atan2(dz, r)
        cos_alpha2 = (l1 ** 2 + dist ** 2 - l2 ** 2) / (2 * l1 * dist)
        cos_alpha2 = max(-1.0, min(1.0, cos_alpha2))
        alpha2 = math.acos(cos_alpha2)
        shoulder_deg = math.degrees(alpha1 + alpha2)

        # 6. 腕部保持末端水平：抵消前两轴的累计俯仰
        wrist_deg = -(shoulder_deg + elbow_deg)

        result = [round(base_deg, 1), round(shoulder_deg, 1),
                  round(elbow_deg, 1), round(wrist_deg, 1)]

        # 7. 关节物理限位检查
        #    数学上有解 ≠ 舵机转得到。此处提前拒绝，避免把越界角度
        #    交给控制器兜底 —— 那样上层无法区分"不可达"与"下发失败"。
        for idx, ang in enumerate(result):
            if idx >= len(self.JOINT_LIMITS):
                break
            lo, hi = self.JOINT_LIMITS[idx]
            if not (lo <= ang <= hi):
                self.last_reject_reason = (
                    f"关节{idx} 需转到 {ang:.1f}°，超出物理行程 "
                    f"[{lo:.0f}, {hi:.0f}]（目标虽在臂展内但姿态不可达）")
                return None

        return result
