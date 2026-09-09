#!/usr/bin/env python3
"""
bringup_servo 的离线验证：用假 SMBus 替换 smbus2，捕获寄存器写入序列。

为什么需要：无硬件时阶段1/阶段2 永远走不到，而它们恰恰是唯一会真正
驱动舵机的代码路径。若等接线后才第一次执行这段代码，任何低级错误
（写错寄存器、脉宽换算错误、确认逻辑失效）都会直接作用在实物上。

本测试不访问任何真实 I2C 设备。

用法：
  python3 tools/test_bringup_servo.py
"""
import io
import os
import sys
import types
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bringup_servo as bs

MODE1, PRESCALE = 0x00, 0xFE
# 上电默认：MODE1=0x11(SLEEP|ALLCALL), PRESCALE=0x1E
POWERON = {MODE1: 0x11, PRESCALE: 0x1E}


class FakeSMBus:
    """记录全部读写的假总线。present=False 时模拟无应答设备。"""

    def __init__(self, present=True, fail_on_write=None):
        self.regs = dict(POWERON)
        self.writes = []
        self.present = present
        self.fail_on_write = fail_on_write
        self.closed = False

    def read_byte_data(self, address, reg):
        if not self.present:
            raise OSError(6, "No such device or address")
        return self.regs.get(reg, 0)

    def write_byte_data(self, address, reg, value):
        if not self.present:
            raise OSError(6, "No such device or address")
        if self.fail_on_write is not None and len(self.writes) == self.fail_on_write:
            raise OSError(5, "Input/output error")
        self.writes.append((reg, value))
        self.regs[reg] = value

    def close(self):
        self.closed = True


failures = []

# 通道寄存器区间：LED0_ON_L(0x06) ~ LED15_OFF_H(0x45)。
# 不能简单用 reg >= LED0_ON_L 判断，那会把 PRESCALE(0xFE) 误算成通道寄存器。
CH_REG_LO = bs.LED0_ON_L
CH_REG_HI = bs.LED0_ON_L + 4 * 16 - 1


def channel_writes_of(bus):
    return [(r, v) for r, v in bus.writes if CH_REG_LO <= r <= CH_REG_HI]


def check(condition, label):
    if condition:
        print(f"  ✅ {label}")
    else:
        print(f"  ❌ {label}")
        failures.append(label)


def run(argv, bus, stdin_text=None):
    """以给定参数运行 main，返回 (退出码, 标准输出)。"""
    fake_module = types.SimpleNamespace(SMBus=lambda num: bus)
    out = io.StringIO()
    with mock.patch.dict(sys.modules, {"smbus2": fake_module}), \
         mock.patch.object(sys, "argv", ["bringup_servo.py"] + argv), \
         mock.patch.object(sys, "stdout", out):
        if stdin_text is not None:
            with mock.patch.object(sys, "stdin", io.StringIO(stdin_text)), \
                 mock.patch("builtins.input", lambda _="": stdin_text.strip()):
                code = bs.main()
        else:
            code = bs.main()
    return code, out.getvalue()


def main():
    print("=" * 70)
    print("  bringup_servo 离线验证（假总线，不访问真实 I2C）")
    print("=" * 70)

    # ── 阶段0：设备无应答时，即便允许写也绝不写 ──
    print("\n【用例1】设备无应答 + --allow-write 时零写入")
    bus = FakeSMBus(present=False)
    code, _ = run(["--bus", "7", "--allow-write"], bus)
    check(code == 1, "退出码为 1")
    check(bus.writes == [], "未写入任何寄存器")
    check(bus.closed, "已关闭总线句柄")

    # ── 只读默认：有设备也不写 ──
    print("\n【用例2】默认模式（无 --allow-write）对在线设备仍零写入")
    bus = FakeSMBus()
    code, text = run(["--bus", "7"], bus)
    check(code == 0, "退出码为 0")
    check(bus.writes == [], "未写入任何寄存器")
    check("SLEEP 置位" in text, "正确识别上电默认的 SLEEP 状态")

    # ── 阶段1：设频序列 ──
    print("\n【用例3】阶段1 设频的寄存器写入序列")
    bus = FakeSMBus()
    code, text = run(["--bus", "7", "--allow-write", "--freq", "50"], bus)
    check(code == 0, "退出码为 0")
    regs_written = [reg for reg, _ in bus.writes]
    check(regs_written == [MODE1, PRESCALE, MODE1, MODE1],
          f"写入顺序为 MODE1→PRESCALE→MODE1→MODE1（实际 {regs_written}）")
    sleep_value = bus.writes[0][1]
    check(sleep_value & 0x10 == 0x10, "改 PRESCALE 前先置 SLEEP（手册要求）")
    prescale_value = bus.writes[1][1]
    check(prescale_value == 135, f"50Hz×0.9 修正得 PRESCALE=135（实际 {prescale_value}）")
    check(bus.writes[2][1] & 0x10 == 0, "随后清除 SLEEP 位")
    check(bus.writes[3][1] & 0x20 == 0x20, "最终开启自动递增(AI)")
    check("回读一致" in text, "回读校验通过")
    check("非实测" in text, "输出注明频率为估算值而非示波器实测")

    # ── 阶段1 未声明接线时不进阶段2 ──
    check("跳过阶段2" in text, "未声明 --servos-connected 时不驱动任何通道")
    check(channel_writes_of(bus) == [], "未写入任何通道寄存器")

    # ── 回读不一致必须中止 ──
    print("\n【用例4】PRESCALE 回读不一致时中止")
    bus = FakeSMBus()
    original = bus.write_byte_data

    def tamper(address, reg, value):
        # 模拟写入未生效（供电不足/通信异常）
        original(address, reg, 0x1E if reg == PRESCALE else value)

    bus.write_byte_data = tamper
    code, text = run(["--bus", "7", "--allow-write"], bus)
    check(code == 1, "退出码为 1")
    check("回读不一致" in text, "报告回读不一致")

    # ── 阶段1 写入中途失败 ──
    print("\n【用例5】阶段1 I2C 写入失败时安全退出")
    bus = FakeSMBus(fail_on_write=1)
    code, text = run(["--bus", "7", "--allow-write"], bus)
    check(code == 1, "退出码为 1")
    check("I2C 读写失败" in text, "报告 I2C 失败而非冒充成功")

    # ── 阶段2：非交互终端且无 --yes 时取消 ──
    print("\n【用例6】阶段2 非交互且未加 --yes 时取消")
    bus = FakeSMBus()
    fake_module = types.SimpleNamespace(SMBus=lambda num: bus)
    out = io.StringIO()
    with mock.patch.dict(sys.modules, {"smbus2": fake_module}), \
         mock.patch.object(sys, "argv", ["x", "--bus", "7", "--allow-write",
                                         "--servos-connected", "--channel", "0"]), \
         mock.patch.object(sys, "stdout", out), \
         mock.patch("builtins.input", side_effect=EOFError):
        code = bs.main()
    text = out.getvalue()
    check(code == 1, "退出码为 1")
    check(channel_writes_of(bus) == [], "未写入任何通道寄存器")
    check("安全默认" in text, "说明这是安全默认而非故障")

    # ── 阶段2：输入非 yes 时取消 ──
    print("\n【用例7】阶段2 确认输入非 yes 时取消")
    bus = FakeSMBus()
    code, text = run(["--bus", "7", "--allow-write", "--servos-connected",
                      "--channel", "0"], bus, stdin_text="no\n")
    check(code == 1, "退出码为 1")
    check(channel_writes_of(bus) == [], "未写入任何通道寄存器")

    # ── 阶段2：确认后只驱动指定通道 ──
    print("\n【用例8】阶段2 确认后仅驱动指定通道且脉宽正确")
    bus = FakeSMBus()
    code, text = run(["--bus", "7", "--allow-write", "--servos-connected",
                      "--channel", "3", "--yes"], bus)
    check(code == 0, "退出码为 0")

    channel_writes = channel_writes_of(bus)
    base = bs.LED0_ON_L + 4 * 3
    touched = sorted({r for r, _ in channel_writes})
    check(touched == [base, base + 1, base + 2, base + 3],
          f"只触碰通道3的四个寄存器（实际 {touched}）")

    # 5 个角度 × 4 个寄存器
    check(len(channel_writes) == 20, f"共 20 次通道写入（实际 {len(channel_writes)}）")

    # 独立复算首个角度的 ticks：90° → 1500us，周期由 PRESCALE=135 反推
    period_us = 1e6 / (25_000_000.0 / 4096.0 / 136)
    expected = round(1500.0 * 4096.0 / period_us)
    got = channel_writes[2][1] | (channel_writes[3][1] << 8)
    check(got == expected, f"90° 换算得 {expected} ticks（实际 {got}）")
    check(all(v == 0 for r, v in channel_writes if r in (base, base + 1)),
          "ON 计数恒为 0")
    check(all((v >> 4) == 0 for r, v in channel_writes if r == base + 3),
          "OFF 高字节仅使用低 4 位")

    # 首尾都应停在中位
    first = channel_writes[2][1] | (channel_writes[3][1] << 8)
    last = channel_writes[-2][1] | (channel_writes[-1][1] << 8)
    check(first == last, "序列首尾均停在中位")

    # 幅度限制在中位 ±15°
    ticks_seq = [channel_writes[i][1] | (channel_writes[i + 1][1] << 8)
                 for i in range(2, len(channel_writes), 4)]
    span_us = (max(ticks_seq) - min(ticks_seq)) * period_us / 4096.0
    expected_span = 2 * bs.SAFE_SPAN_DEG / 180.0 * 2000.0
    check(abs(span_us - expected_span) < period_us / 4096.0 * 2,
          f"总摆幅约 {span_us:.1f}us，对应 ±{bs.SAFE_SPAN_DEG}°")

    print("\n" + "=" * 70)
    if failures:
        print(f"  ❌ {len(failures)} 项未通过：")
        for f in failures:
            print(f"     - {f}")
    else:
        print("  ✅ 全部通过（假总线验证，非实物舵机行为）")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
