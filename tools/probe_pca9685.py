#!/usr/bin/env python3
"""
PCA9685 只读在线探测：确认舵机板是否已接入、地址是否正确。

本脚本只做 SMBus 读操作，绝不写任何寄存器，因此不会改变 PWM 输出、
不会让已上电的舵机产生动作。写寄存器属于实机联调步骤，必须在确认
接线、供电与限位之后单独进行。

注意：读到寄存器只能证明 I2C 通信正常，不能证明舵机接线正确、
供电充足或机械限位安全，也不代表 PWM 波形已用示波器校准。

用法：
  python3 tools/probe_pca9685.py                # 扫描常见总线
  python3 tools/probe_pca9685.py --bus 7        # 指定总线
  python3 tools/probe_pca9685.py --bus 7 --address 0x40
"""
import argparse
import errno
import sys

MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
# PCA9685 上电默认值（NXP 数据手册）：MODE1=0x11(SLEEP|ALLCALL)、PRESCALE=0x1E(约200Hz)
DEFAULT_MODE1 = 0x11
DEFAULT_PRESCALE = 0x1E
OSCILLATOR_HZ = 25_000_000.0


def parse_int(text: str) -> int:
    return int(text, 0)


def describe_mode1(value: int) -> str:
    flags = []
    if value & 0x80:
        flags.append("RESTART")
    if value & 0x40:
        flags.append("EXTCLK")
    if value & 0x20:
        flags.append("AI(自动递增)")
    flags.append("SLEEP(低功耗,不输出PWM)" if value & 0x10 else "正常运行")
    if value & 0x08:
        flags.append("SUB1")
    if value & 0x04:
        flags.append("SUB2")
    if value & 0x02:
        flags.append("SUB3")
    if value & 0x01:
        flags.append("ALLCALL")
    return " | ".join(flags)


def probe_address(bus, address: int) -> dict:
    """只读读取关键寄存器。失败时返回错误原因而非抛出。"""
    try:
        mode1 = bus.read_byte_data(address, MODE1)
        mode2 = bus.read_byte_data(address, MODE2)
        prescale = bus.read_byte_data(address, PRESCALE)
    except OSError as exc:
        name = errno.errorcode.get(exc.errno, str(exc.errno))
        return {"ok": False, "error": f"{name}: {exc.strerror or exc}"}
    return {"ok": True, "mode1": mode1, "mode2": mode2, "prescale": prescale}


def report(bus_num: int, address: int, info: dict) -> None:
    tag = f"/dev/i2c-{bus_num} 地址 0x{address:02X}"
    if not info["ok"]:
        print(f"  ✗ {tag}: 无响应（{info['error']}）")
        return

    mode1, prescale = info["mode1"], info["prescale"]
    # 频率由读到的 prescale 反算，属于按数据手册公式的估算值，
    # 不是示波器实测；真实脉宽仍需仪器校准。
    freq = OSCILLATOR_HZ / 4096.0 / (prescale + 1)
    period_us = 1_000_000.0 / freq
    print(f"  ✓ {tag}: 有响应")
    print(f"      MODE1    = 0x{mode1:02X}  [{describe_mode1(mode1)}]")
    print(f"      MODE2    = 0x{info['mode2']:02X}")
    print(f"      PRESCALE = 0x{prescale:02X} ({prescale})")
    print(f"      按 25MHz 标称时钟估算: {freq:.2f} Hz / 周期 {period_us:.1f} us（非实测）")

    if mode1 == DEFAULT_MODE1 and prescale == DEFAULT_PRESCALE:
        print("      状态: 与上电默认值一致，芯片处于 SLEEP，未输出 PWM")
    elif mode1 & 0x10:
        print("      状态: SLEEP 位置位，当前不输出 PWM")
    else:
        print("      状态: 已退出 SLEEP，可能正在输出 PWM —— 舵机若已接线会保持力矩")

    if not (3 <= prescale <= 255):
        print("      ⚠ PRESCALE 超出手册有效范围 3~255，读数可疑")


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA9685 只读探测（不写寄存器）")
    parser.add_argument("--bus", type=int, action="append",
                        help="指定 I2C 总线号，可重复；默认扫描常见总线")
    parser.add_argument("--address", type=parse_int, action="append",
                        help="指定从机地址，可重复；默认扫描 0x40~0x47")
    args = parser.parse_args()

    try:
        import smbus2
    except ImportError:
        print("未安装 smbus2，无法进行在线探测。")
        print("这是环境缺失，不是探测结果；请先安装：pip3 install smbus2")
        return 2

    buses = args.bus or [7, 0, 2, 3, 6]
    addresses = args.address or list(range(0x40, 0x48))

    print("=" * 68)
    print("  PCA9685 只读探测（仅读寄存器，不写、不驱动舵机）")
    print("=" * 68)

    found = []
    for bus_num in buses:
        try:
            bus = smbus2.SMBus(bus_num)
        except OSError as exc:
            print(f"\n/dev/i2c-{bus_num}: 无法打开（{exc.strerror or exc}）")
            continue
        print(f"\n/dev/i2c-{bus_num}:")
        try:
            for address in addresses:
                info = probe_address(bus, address)
                if info["ok"]:
                    found.append((bus_num, address))
                    report(bus_num, address, info)
        finally:
            bus.close()
        if not any(b == bus_num for b, _ in found):
            print(f"  指定地址范围内无响应设备")

    print("\n" + "=" * 68)
    if found:
        for bus_num, address in found:
            print(f"  发现候选设备: /dev/i2c-{bus_num} 0x{address:02X}")
        print("  仅表示 I2C 通信正常；接线正确性、供电与机械限位仍需人工确认。")
    else:
        print("  未发现任何响应设备：舵机板可能未接线、未供电或接在其他总线。")
    print("=" * 68)
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
