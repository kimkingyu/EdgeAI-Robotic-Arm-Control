import time
import math
from typing import List, Optional, Dict, Any

from .base_controller import BaseArmController

try:
    import smbus2
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False


class PCA9685:
    """PCA9685 16通道 12位 PWM I2C 驱动"""
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    LED0_ON_H = 0x07
    LED0_OFF_L = 0x08
    LED0_OFF_H = 0x09

    def __init__(self, bus_num: int = 7, address: int = 0x40):
        self.bus_num = bus_num
        self.address = address
        self.bus = None

    def open(self):
        if not HAS_SMBUS:
            return False
        try:
            self.bus = smbus2.SMBus(self.bus_num)
            # 复位
            self.bus.write_byte_data(self.address, self.MODE1, 0x00)
            self.set_pwm_freq(50) # 舵机标准 50Hz (20ms 周期)
            return True
        except Exception as e:
            print(f"[PCA9685] 打开 I2C 总线 {self.bus_num} 失败: {e}")
            return False

    def set_pwm_freq(self, freq_hz: float = 50.0):
        if not self.bus:
            return
        prescaleval = 25000000.0  # 25MHz 内部时钟
        prescaleval /= 4096.0     # 12-bit
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        prescale = int(math.floor(prescaleval + 0.5))

        oldmode = self.bus.read_byte_data(self.address, self.MODE1)
        newmode = (oldmode & 0x7F) | 0x10 # 进入 sleep 模式以设置 prescale
        self.bus.write_byte_data(self.address, self.MODE1, newmode)
        self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.address, self.MODE1, oldmode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.MODE1, oldmode | 0xa1) # 重启并开启 auto-increment

    def set_pwm(self, channel: int, on: int, off: int):
        if not self.bus:
            return
        base_reg = self.LED0_ON_L + 4 * channel
        self.bus.write_byte_data(self.address, base_reg, on & 0xFF)
        self.bus.write_byte_data(self.address, base_reg + 1, on >> 8)
        self.bus.write_byte_data(self.address, base_reg + 2, off & 0xFF)
        self.bus.write_byte_data(self.address, base_reg + 3, off >> 8)

    def set_servo_angle(self, channel: int, angle: float, min_us: int = 500, max_us: int = 2500):
        """将角度 (0~180) 映射为高电平时间 (us)，再映射为 12位计数值 (0~4095)"""
        angle = max(0.0, min(180.0, angle))
        pulse_us = min_us + (angle / 180.0) * (max_us - min_us)
        # 50Hz 下，每个 tick 是 20000us / 4096 = 4.8828us
        ticks = int(pulse_us * 4096.0 / 20000.0)
        self.set_pwm(channel, 0, ticks)

    def close(self):
        if self.bus:
            self.bus.close()


class I2CArmController(BaseArmController):
    """基于 I2C (PCA9685) 的多自由度舵机机械臂控制器"""

    def __init__(self, bus_num: int = 7, address: int = 0x40, channel_map: Optional[List[int]] = None, mock: bool = False):
        self.bus_num = bus_num
        self.address = address
        # 各关节对应的 PCA9685 通道号，默认 0~5 号通道依次控制关节 1~6，6 号为夹爪
        self.channel_map = channel_map or [0, 1, 2, 3, 4, 5]
        self.gripper_channel = 6
        self.mock = mock or not HAS_SMBUS
        self.pca = PCA9685(bus_num=self.bus_num, address=self.address)
        self.current_angles = [90.0] * len(self.channel_map)

    def connect(self) -> bool:
        if self.mock:
            print(f"[I2CArm] (Mock) 模拟连接 I2C-总线 /dev/i2c-{self.bus_num} 设备地址 0x{self.address:02X}")
            return True

        if not self.pca.open():
            print(f"[I2CArm] 无法打开 I2C 舵机板，自动转为 Mock 模式")
            self.mock = True
        else:
            print(f"[I2CArm] 成功连接 PCA9685 舵机板 (/dev/i2c-{self.bus_num}, 地址: 0x{self.address:02X})")
        return True

    def disconnect(self):
        if not self.mock:
            self.pca.close()
        print("[I2CArm] I2C 舵机驱动已关闭")

    def move_joints(self, angles: List[float], speed: int = 50, wait: bool = True) -> bool:
        """驱动各舵机旋转到指定角度 (度，标准 0~180)"""
        self.current_angles = list(angles)
        for i, ang in enumerate(angles):
            if i >= len(self.channel_map):
                break
            ch = self.channel_map[i]
            # 转换负角为 0~180 舵机角
            servo_ang = max(0.0, min(180.0, ang + 90.0))
            if self.mock:
                print(f"[I2CArm] (Mock) 关节 {i+1} -> 舵机通道 {ch}: {servo_ang:.1f}°")
            else:
                self.pca.set_servo_angle(ch, servo_ang)

        if wait:
            time.sleep(0.2)
        return True

    def gripper_control(self, open_ratio: float) -> bool:
        """夹爪开合 (0.0 闭合 -> 0°, 1.0 全开 -> 90°)"""
        ratio = max(0.0, min(1.0, open_ratio))
        ang = ratio * 90.0
        if self.mock:
            print(f"[I2CArm] (Mock 夹爪) 通道 {self.gripper_channel} -> 开合比: {ratio*100:.0f}%, 角度: {ang:.1f}°")
        else:
            self.pca.set_servo_angle(self.gripper_channel, ang)
        return True

    def get_joint_angles(self) -> Optional[List[float]]:
        return list(self.current_angles)
