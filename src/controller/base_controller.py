from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseArmController(ABC):
    """机械臂底层驱动接口定义"""

    @abstractmethod
    def connect(self) -> bool:
        """建立通信连接"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接并释放资源"""
        pass

    @abstractmethod
    def move_joints(self, angles: List[float], speed: int = 50, wait: bool = True) -> bool:
        """多轴关节空间直接联动控制"""
        pass

    @abstractmethod
    def gripper_control(self, open_ratio: float) -> bool:
        """夹爪开合控制 (0.0 = 完全闭合, 1.0 = 完全张开)"""
        pass

    @abstractmethod
    def get_joint_angles(self) -> Optional[List[float]]:
        """读取当前各关节实际角度"""
        pass
