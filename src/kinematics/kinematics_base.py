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

    def inverse_kinematics(self, target_pose: Dict[str, float], current_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """几何解析逆解 (带工作空间边界检查)"""
        x = target_pose.get("x", 0.0)
        y = target_pose.get("y", 0.0)
        z = target_pose.get("z", 0.0)

        # 1. 基座航向角
        base_deg = math.degrees(math.atan2(y, x))

        # 2. 平面逆解投影
        r = math.sqrt(x**2 + y**2)
        dz = z - self.link_lengths[0]
        dist = math.sqrt(r**2 + dz**2)

        l1, l2 = self.link_lengths[1], self.link_lengths[2]
        # 检查是否在工作半径内
        if dist > (l1 + l2) or dist < abs(l1 - l2):
            print(f"[IK] 警告: 目标坐标超出工作空间 (距离: {dist:.1f}mm, 最大: {l1+l2:.1f}mm)")
            return None

        # 余弦定理求关节夹角
        cos_beta = (l1**2 + l2**2 - dist**2) / (2 * l1 * l2)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)
        elbow_deg = 180.0 - math.degrees(beta)

        alpha1 = math.atan2(dz, r)
        cos_alpha2 = (l1**2 + dist**2 - l2**2) / (2 * l1 * dist)
        cos_alpha2 = max(-1.0, min(1.0, cos_alpha2))
        alpha2 = math.acos(cos_alpha2)
        shoulder_deg = math.degrees(alpha1 + alpha2)

        # 返回各轴角度 [底座, 大臂, 小臂, 手腕/夹爪保持水平]
        wrist_deg = -(shoulder_deg + elbow_deg)
        return [round(base_deg, 1), round(shoulder_deg, 1), round(elbow_deg, 1), round(wrist_deg, 1)]
