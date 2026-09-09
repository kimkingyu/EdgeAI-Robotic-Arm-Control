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

    # 舵机安全限位：多数 SG90/MG996R 类舵机机械行程为 0~180°，
    # 超出会顶死堵转，电流可达额定数倍，几十秒即可烧毁线圈。
    SERVO_MIN_DEG = 0.0
    SERVO_MAX_DEG = 180.0

    def __init__(self, bus_num: int = 7, address: int = 0x40):
        self.bus_num = bus_num
        self.address = address
        self.bus = None
        # PWM 实际周期(us)。必须由 set_pwm_freq 按芯片真实输出频率回填，
        # 不能沿用 1/50Hz=20000us 的理论值 —— 因为设频时做了 0.9 过冲修正，
        # 实际周期约 22282us，用理论值换算 tick 会产生 11.4% 脉宽偏差（约 15°）。
        self.period_us = 20000.0
        self.last_clip_warning = None

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
        # 参考 Adafruit-PWM-Servo 规范: 乘以 0.9 修正 PCA9685 内部 25MHz RC 振荡器频率过冲 (Issue #11)
        corrected_freq = freq_hz * 0.9
        prescaleval = 25000000.0  # 25MHz 内部时钟
        prescaleval /= 4096.0     # 12-bit
        prescaleval /= float(corrected_freq)
        prescaleval -= 1.0
        prescale = int(math.floor(prescaleval + 0.5))
        prescale = max(3, min(255, prescale))   # 芯片寄存器有效范围

        # 按 prescale 反算芯片真实输出频率与周期，供 tick 换算使用。
        # 即便 bus 未打开（Mock 模式）也要更新，保证离线换算与实机一致。
        actual_freq = 25000000.0 / 4096.0 / (prescale + 1)
        self.period_us = 1_000_000.0 / actual_freq

        if not self.bus:
            return

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

    def set_servo_angle(self, channel: int, angle: float,
                        min_us: int = 500, max_us: int = 2500,
                        strict: bool = True) -> bool:
        """
        将角度映射为脉宽再换算为 12 位计数值下发

        strict=True 时，越界角度会被拒绝并告警，而不是静默夹紧 ——
        静默夹紧会掩盖上层逻辑错误，让机械臂突然扫到极限位置。
        """
        if angle < self.SERVO_MIN_DEG or angle > self.SERVO_MAX_DEG:
            msg = (f"通道 {channel} 角度 {angle:.1f}° 超出舵机安全范围 "
                   f"[{self.SERVO_MIN_DEG}, {self.SERVO_MAX_DEG}]")
            self.last_clip_warning = msg
            if strict:
                print(f"[PCA9685] 拒绝: {msg}")
                return False
            print(f"[PCA9685] 警告: {msg}，已夹紧执行")
            angle = max(self.SERVO_MIN_DEG, min(self.SERVO_MAX_DEG, angle))
        else:
            self.last_clip_warning = None

        pulse_us = min_us + (angle / 180.0) * (max_us - min_us)
        # 用芯片真实周期换算，而非 1/50Hz 的理论值。
        # 用四舍五入而非截断：截断会系统性偏短约半个 tick(约2.7us)。
        # 但不能用内置 round —— 它是银行家舍入(四舍六入五成双)，而 C++ 侧
        # std::round 是"五入且远离零"。实测 108° 恰好算得 312.5 ticks，
        # 两者分别给出 312 与 313，同一角度差 1 tick(约0.49°)。
        # 用 floor(x+0.5) 复现 std::round 语义（此处被除数恒为正）。
        ticks = int(math.floor(pulse_us * 4096.0 / self.period_us + 0.5))
        ticks = max(0, min(4095, ticks))
        self.set_pwm(channel, 0, ticks)
        return True

    def close(self):
        if self.bus:
            self.bus.close()


class I2CArmController(BaseArmController):
    """基于 I2C (PCA9685) 的多自由度舵机机械臂控制器"""

    # 各关节的「IK 角度 → 舵机角度」标定参数。
    # IK 输出的是几何角（如 elbow ∈ [-180,0]），舵机只能接受 0~180°，
    # 必须按关节分别设定偏置与方向，不能所有关节统一 +90 ——
    # 实测统一偏置会导致 elbow 等关节的合法角被截断到 0°，实机猛扫极限位。
    #   servo_angle = offset + direction * ik_angle
    DEFAULT_JOINT_CALIB = [
        {"offset": 90.0,  "direction": 1.0},   # 关节0 底座  IK[-90,90]  → 舵机[0,180]
        {"offset": 90.0,  "direction": 1.0},   # 关节1 大臂  IK[-90,90]  → 舵机[0,180]
        {"offset": 180.0, "direction": 1.0},   # 关节2 小臂  IK[-180,0]  → 舵机[0,180]
        {"offset": 90.0,  "direction": 1.0},   # 关节3 腕部  IK[-90,90]  → 舵机[0,180]
    ]

    def __init__(self, bus_num: int = 7, address: int = 0x40, channel_map: Optional[List[int]] = None,
                 mock: bool = False, joint_calib: Optional[List[Dict[str, float]]] = None):
        self.bus_num = bus_num
        self.address = address
        # 各关节对应的 PCA9685 通道号，默认 0~5 号通道依次控制关节 1~6，6 号为夹爪
        self.channel_map = channel_map or [0, 1, 2, 3, 4, 5]
        self.gripper_channel = 6
        self.mock = mock or not HAS_SMBUS
        self.pca = PCA9685(bus_num=self.bus_num, address=self.address)
        self.current_angles = [90.0] * len(self.channel_map)
        self.joint_calib = joint_calib or self.DEFAULT_JOINT_CALIB
        # 记录最近一次被拒绝的关节指令，供上层排查
        self.last_rejected: List[str] = []

    def ik_to_servo(self, joint_idx: int, ik_angle: float) -> float:
        """把 IK 几何角换算为该关节对应的舵机角"""
        if joint_idx < len(self.joint_calib):
            c = self.joint_calib[joint_idx]
            return c["offset"] + c["direction"] * ik_angle
        return ik_angle + 90.0

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
        """
        驱动各舵机到指定 IK 角度

        任一关节越界即整体拒绝下发：机械臂是刚性联动结构，
        只执行一部分关节会导致姿态错乱，比不动更危险。
        """
        self.last_rejected = []
        plan = []
        for i, ang in enumerate(angles):
            if i >= len(self.channel_map):
                break
            servo_ang = self.ik_to_servo(i, ang)
            if servo_ang < PCA9685.SERVO_MIN_DEG or servo_ang > PCA9685.SERVO_MAX_DEG:
                self.last_rejected.append(
                    f"关节{i} IK {ang:.1f}° → 舵机 {servo_ang:.1f}° 超出 "
                    f"[{PCA9685.SERVO_MIN_DEG:.0f},{PCA9685.SERVO_MAX_DEG:.0f}]")
                continue
            plan.append((i, self.channel_map[i], servo_ang))

        if self.last_rejected:
            print("[I2CArm] 拒绝下发（存在越界关节，避免姿态错乱与堵转）:")
            for r in self.last_rejected:
                print(f"         - {r}")
            return False

        self.current_angles = list(angles)
        for i, ch, servo_ang in plan:
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
