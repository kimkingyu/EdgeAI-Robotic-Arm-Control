#!/usr/bin/env python3
"""
PCA9685 接线后首次通电分步自检。

设计前提：第一次给陌生接线通电时，最危险的操作是一次性驱动多个舵机。
本脚本因此只做单通道、小幅度、可中止的分步验证，且默认不写任何寄存器。

分级：
  阶段0 探测   —— 只读寄存器，默认执行，不写、不动作。
  阶段1 设频   —— 写 MODE1/PRESCALE 并回读校验。需 --allow-write。
                  此阶段要求舵机本体未接，因为退出 SLEEP 后通道会输出电平。
  阶段2 单通道 —— 仅对一个通道输出有限脉宽。需 --allow-write --servos-connected
                  并显式给出 --channel，还需终端输入确认。

刻意不提供"一键全臂上电"：整臂联动属于标定完成后的动作，
不属于首次通电自检。

用法：
  python3 tools/bringup_servo.py --bus 7                      # 只探测
  python3 tools/bringup_servo.py --bus 7 --allow-write        # 探测+设频(舵机未接)
  python3 tools/bringup_servo.py --bus 7 --allow-write \\
      --servos-connected --channel 0                          # 单通道小幅动作
"""
import argparse
import errno
import sys
import time

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
OSCILLATOR_HZ = 25_000_000.0
# 首次通电只在中位附近小幅移动：即便方向标定错误或限位未知，
# 小幅度也留有人工断电的余地。
SAFE_CENTER_DEG = 90.0
SAFE_SPAN_DEG = 15.0


def parse_int(text: str) -> int:
    return int(text, 0)


def read_regs(bus, address: int) -> dict:
    try:
        return {
            "ok": True,
            "mode1": bus.read_byte_data(address, MODE1),
            "prescale": bus.read_byte_data(address, PRESCALE),
        }
    except OSError as exc:
        name = errno.errorcode.get(exc.errno, str(exc.errno))
        return {"ok": False, "error": f"{name}: {exc.strerror or exc}"}


def freq_of(prescale: int) -> float:
    return OSCILLATOR_HZ / 4096.0 / (prescale + 1)


def stage0_probe(bus, address: int) -> dict:
    print("\n【阶段0】只读探测（不写寄存器）")
    info = read_regs(bus, address)
    if not info["ok"]:
        print(f"  ✗ 0x{address:02X} 无响应（{info['error']}）")
        print("    舵机板未接线/未供电，或地址与总线不符。后续阶段全部跳过。")
        return info

    mode1, prescale = info["mode1"], info["prescale"]
    print(f"  ✓ 0x{address:02X} 有响应")
    print(f"      MODE1=0x{mode1:02X}  PRESCALE=0x{prescale:02X}({prescale})")
    print(f"      按25MHz标称时钟估算 {freq_of(prescale):.2f} Hz（非示波器实测）")
    if mode1 & 0x10:
        print("      SLEEP 置位：当前不输出 PWM")
    else:
        print("      ⚠ SLEEP 未置位：芯片可能正在输出 PWM，已接舵机会保持力矩")
    return info


def stage1_set_freq(bus, address: int, target_hz: float, correction: float) -> bool:
    """写 PRESCALE 并回读校验。必须在 SLEEP 下改 PRESCALE（数据手册要求）。"""
    print(f"\n【阶段1】设置 PWM 频率至 {target_hz} Hz 并回读校验")
    raw = OSCILLATOR_HZ / (4096.0 * target_hz * correction) - 1.0
    prescale = int(round(raw))
    if not (3 <= prescale <= 255):
        print(f"  ✗ 计算出的 PRESCALE={prescale} 超出手册有效范围 3~255，拒绝写入")
        return False

    try:
        oldmode = bus.read_byte_data(address, MODE1)
        bus.write_byte_data(address, MODE1, (oldmode & 0x7F) | 0x10)  # 进入 SLEEP
        bus.write_byte_data(address, PRESCALE, prescale)
        bus.write_byte_data(address, MODE1, oldmode & 0x6F)           # 清 SLEEP 与 RESTART
        time.sleep(0.005)
        bus.write_byte_data(address, MODE1, (oldmode & 0x6F) | 0xA0)  # RESTART + 自动递增
        readback = bus.read_byte_data(address, PRESCALE)
        mode1 = bus.read_byte_data(address, MODE1)
    except OSError as exc:
        print(f"  ✗ I2C 读写失败: {exc.strerror or exc}")
        return False

    print(f"  写入 PRESCALE=0x{prescale:02X}({prescale})，回读=0x{readback:02X}({readback})")
    if readback != prescale:
        print("  ✗ 回读不一致，通信或供电异常，停止后续阶段")
        return False
    print(f"  ✓ 回读一致；MODE1=0x{mode1:02X}")
    print(f"      估算输出 {freq_of(readback):.2f} Hz / 周期 {1e6 / freq_of(readback):.1f} us（非实测）")
    print("      ⚠ 真实脉宽仍需示波器校准，本数值仅来自寄存器与标称时钟")
    return True


def set_pulse(bus, address: int, channel: int, pulse_us: float, period_us: float) -> int:
    ticks = int(round(pulse_us * 4096.0 / period_us))
    if not (1 <= ticks <= 4095):
        raise ValueError(f"脉宽 {pulse_us}us 换算得 {ticks} ticks，超出有效范围")
    base = LED0_ON_L + 4 * channel
    bus.write_byte_data(address, base, 0)
    bus.write_byte_data(address, base + 1, 0)
    bus.write_byte_data(address, base + 2, ticks & 0xFF)
    bus.write_byte_data(address, base + 3, (ticks >> 8) & 0x0F)
    return ticks


def stage2_single_channel(bus, address: int, channel: int, prescale: int,
                          min_us: float, max_us: float, assume_yes: bool) -> bool:
    print(f"\n【阶段2】单通道 {channel} 小幅动作（{SAFE_CENTER_DEG}° ± {SAFE_SPAN_DEG}°）")
    print("  仅驱动这一个通道；其余通道不写。请把手远离机械结构，随时准备断电。")
    if not assume_yes:
        try:
            reply = input("  确认继续？输入 yes 执行，其他任意输入取消: ").strip().lower()
        except EOFError:
            print("  ✗ 非交互终端且未加 --yes，已取消（这是安全默认，不是故障）")
            return False
        if reply != "yes":
            print("  已取消，未写入任何通道寄存器")
            return False

    period_us = 1e6 / freq_of(prescale)
    angles = [SAFE_CENTER_DEG, SAFE_CENTER_DEG - SAFE_SPAN_DEG,
              SAFE_CENTER_DEG, SAFE_CENTER_DEG + SAFE_SPAN_DEG, SAFE_CENTER_DEG]
    try:
        for angle in angles:
            pulse = min_us + (angle / 180.0) * (max_us - min_us)
            ticks = set_pulse(bus, address, channel, pulse, period_us)
            print(f"    {angle:5.1f}° → {pulse:7.1f} us → {ticks:4d} ticks")
            time.sleep(0.6)
    except (OSError, ValueError) as exc:
        print(f"  ✗ 写入中止: {exc}")
        return False

    print("  ✓ 指令序列已下发完毕，舵机停在中位")
    print("    请人工观察：是否平滑到位、有无异响或发烫。脚本无法感知实际转角。")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA9685 首次通电分步自检")
    parser.add_argument("--bus", type=int, default=7, help="I2C 总线号，默认 7")
    parser.add_argument("--address", type=parse_int, default=0x40, help="从机地址，默认 0x40")
    parser.add_argument("--freq", type=float, default=50.0, help="目标 PWM 频率，默认 50Hz")
    parser.add_argument("--correction", type=float, default=0.9,
                        help="振荡器修正系数，默认 0.9（沿用项目历史值，非实测校准）")
    parser.add_argument("--min-us", type=float, default=500.0, help="0° 脉宽，默认 500us")
    parser.add_argument("--max-us", type=float, default=2500.0, help="180° 脉宽，默认 2500us")
    parser.add_argument("--allow-write", action="store_true",
                        help="允许进入阶段1（写 PRESCALE）。要求舵机本体未接")
    parser.add_argument("--servos-connected", action="store_true",
                        help="声明舵机已接好并确认限位，允许进入阶段2")
    parser.add_argument("--channel", type=int, help="阶段2 要驱动的单个通道 0~15")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认（自动化场景）")
    args = parser.parse_args()

    if args.correction <= 0 or args.freq <= 0:
        print("频率与修正系数必须为正数")
        return 2
    if not (0 < args.min_us < args.max_us):
        print("脉宽范围非法：需满足 0 < min-us < max-us")
        return 2
    if args.servos_connected and args.channel is None:
        print("阶段2 必须用 --channel 显式指定单个通道，拒绝默认驱动任何通道")
        return 2
    if args.channel is not None and not (0 <= args.channel <= 15):
        print("通道号必须在 0~15 之间")
        return 2

    try:
        import smbus2
    except ImportError:
        print("未安装 smbus2：这是环境缺失，不是硬件探测结论。请 pip3 install smbus2")
        return 2

    print("=" * 68)
    print("  PCA9685 首次通电分步自检")
    print(f"  总线 /dev/i2c-{args.bus} 地址 0x{args.address:02X}")
    print("=" * 68)

    try:
        bus = smbus2.SMBus(args.bus)
    except OSError as exc:
        print(f"无法打开 /dev/i2c-{args.bus}: {exc.strerror or exc}")
        return 1

    try:
        info = stage0_probe(bus, args.address)
        if not info["ok"]:
            return 1

        if not args.allow_write:
            print("\n未加 --allow-write，止步于只读探测（安全默认）。")
            return 0

        if args.servos_connected:
            print("\n注意：已声明舵机接好。阶段1 退出 SLEEP 时通道即输出电平。")
        else:
            print("\n阶段1 假定舵机本体未接。若已接线请补 --servos-connected 重新评估风险。")

        if not stage1_set_freq(bus, args.address, args.freq, args.correction):
            return 1

        if not args.servos_connected:
            print("\n未声明 --servos-connected，跳过阶段2（不驱动任何通道）。")
            return 0

        prescale = bus.read_byte_data(args.address, PRESCALE)
        ok = stage2_single_channel(bus, args.address, args.channel, prescale,
                                   args.min_us, args.max_us, args.yes)
        return 0 if ok else 1
    finally:
        bus.close()
        print("\nI2C 已关闭。注意：关闭句柄不会停止 PCA9685 的 PWM 输出，")
        print("舵机若已接线仍会保持当前位置的力矩。")


if __name__ == "__main__":
    sys.exit(main())
