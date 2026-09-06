import time
import json
from typing import List, Optional

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

from .base_controller import BaseArmController


class SerialArmController(BaseArmController):
    """通用串口/总线舵机机械臂控制器 (支持真实串口与 Mock 模式)"""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout: float = 0.5, mock: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.mock = mock or not HAS_SERIAL
        self.ser: Optional[serial.Serial] = None
        self.current_angles = [0.0] * 6

    def connect(self) -> bool:
        if self.mock:
            print(f"[ArmController] (Mock) 模拟连接机械臂串口: {self.port} @ {self.baudrate}")
            return True

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(1.0) # 等待下位机复位完成
            print(f"[ArmController] 串口打开成功: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[ArmController] 打开串口失败: {e}，将自动切换为 Mock 模式运行")
            self.mock = True
            return True

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("[ArmController] 串口已关闭")

    def move_joints(self, angles: List[float], speed: int = 50, wait: bool = True) -> bool:
        """发送多轴角度指令"""
        self.current_angles = list(angles)
        payload = {
            "cmd": "move_joints",
            "angles": [round(a, 2) for a in angles],
            "speed": speed
        }
        cmd_str = json.dumps(payload) + "\n"

        if self.mock:
            print(f"[ArmController] (Mock 指令发送) -> {cmd_str.strip()}")
            if wait:
                time.sleep(0.2)
            return True

        try:
            self.ser.write(cmd_str.encode("utf-8"))
            if wait:
                time.sleep(1.0) # 简单延时等待运动就位
            return True
        except Exception as e:
            print(f"[ArmController] 发送指令异常: {e}")
            return False

    def gripper_control(self, open_ratio: float) -> bool:
        payload = {"cmd": "gripper", "ratio": max(0.0, min(1.0, open_ratio))}
        cmd_str = json.dumps(payload) + "\n"

        if self.mock:
            print(f"[ArmController] (Mock 夹爪) -> {cmd_str.strip()}")
            return True

        try:
            self.ser.write(cmd_str.encode("utf-8"))
            return True
        except Exception as e:
            print(f"[ArmController] 夹爪控制异常: {e}")
            return False

    def get_joint_angles(self) -> Optional[List[float]]:
        return list(self.current_angles)
