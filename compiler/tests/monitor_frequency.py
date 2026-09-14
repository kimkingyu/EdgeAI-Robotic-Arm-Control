#!/usr/bin/env python3
"""在基准运行期间连续采样绑定核频率与温度。

边界快照只能证明起止状态，无法排除过程中的短暂降频。这个脚本以独立线程持续
采样，给出过程级证据。它不修改 governor，也不写任何 sysfs，只读。
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


class Sampler(threading.Thread):
    def __init__(self, policy, interval_ms):
        super().__init__(daemon=True)
        self.freq_path = Path("/sys/devices/system/cpu/cpufreq/%s/scaling_cur_freq" % policy)
        self.zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
        self.interval = interval_ms / 1000.0
        self.samples = []
        # 不能叫 _stop：threading.Thread 自己有个 _stop() 内部方法，覆盖掉会让
        # join() 在超时路径上崩。
        self._halt = threading.Event()
        if not self.freq_path.is_file():
            raise SystemExit("找不到 " + str(self.freq_path))

    def run(self):
        while not self._halt.is_set():
            entry = {"monotonic_ns": time.monotonic_ns()}
            try:
                entry["frequency_khz"] = int(self.freq_path.read_text().strip())
            except (OSError, ValueError) as error:
                entry["frequency_error"] = str(error)
            temps = []
            for zone in self.zones:
                try:
                    temps.append(int(zone.read_text().strip()))
                except (OSError, ValueError):
                    pass
            if temps:
                entry["thermal_max_millidegrees"] = max(temps)
            self.samples.append(entry)
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        if self.is_alive():
            raise RuntimeError("采样线程未能在 5 秒内退出，监控数据不可信")


def summarize(samples, available):
    freqs = [s["frequency_khz"] for s in samples if "frequency_khz" in s]
    temps = [s["thermal_max_millidegrees"] for s in samples if "thermal_max_millidegrees" in s]
    if not freqs:
        return {"error": "没有取到任何频率采样"}
    top = max(available) if available else max(freqs)
    below = [f for f in freqs if f < top]
    return {
        "sample_count": len(freqs),
        "frequency_khz_min": min(freqs),
        "frequency_khz_max": max(freqs),
        "frequency_khz_unique": sorted(set(freqs)),
        "top_available_khz": top,
        "samples_below_top": len(below),
        "share_below_top_percent": round(100.0 * len(below) / len(freqs), 3),
        "held_top_throughout": not below,
        "temperature_c_min": round(min(temps) / 1000, 2) if temps else None,
        "temperature_c_max": round(max(temps) / 1000, 2) if temps else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="policy4")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--keep-samples", action="store_true",
                        help="保留逐次采样原始值（体积较大）")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- 之后是被监控的命令")
    args = parser.parse_args(argv)
    command = [c for c in args.command if c != "--"]
    if not command:
        parser.error("需要在 -- 之后给出被监控的命令")
    if args.report.exists():
        parser.error("报告已存在，拒绝覆盖: " + str(args.report))

    available = []
    path = Path("/sys/devices/system/cpu/cpufreq/%s/scaling_available_frequencies" % args.policy)
    if path.is_file():
        available = [int(v) for v in path.read_text().split()]

    sampler = Sampler(args.policy, args.interval_ms)
    started = time.monotonic_ns()
    sampler.start()
    try:
        completed = subprocess.run(command)
    finally:
        # 被监控命令已经跑完，采集到的样本不能因为收尾出错而丢掉。
        try:
            sampler.stop()
        except RuntimeError as error:
            print("monitor_frequency: " + str(error), file=sys.stderr)
    elapsed = time.monotonic_ns() - started

    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "基准运行期间对绑定核的连续只读采样；不修改 governor，不写 sysfs",
        "policy": args.policy,
        "interval_ms": args.interval_ms,
        "command": command,
        "command_returncode": completed.returncode,
        "wall_clock_ns": elapsed,
        "available_frequencies_khz": available,
        "summary": summarize(sampler.samples, available),
    }
    if args.keep_samples:
        result["samples"] = sampler.samples
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    summary = result["summary"]
    print("\n监控结果: %d 次采样，频率 %s" % (
        summary.get("sample_count", 0), summary.get("frequency_khz_unique")))
    print("低于最高档的采样: %d 次 (%.3f%%)  温度 %s-%s C" % (
        summary.get("samples_below_top", 0), summary.get("share_below_top_percent", 0),
        summary.get("temperature_c_min"), summary.get("temperature_c_max")))
    print("全程保持最高档: %s" % summary.get("held_top_throughout"))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
