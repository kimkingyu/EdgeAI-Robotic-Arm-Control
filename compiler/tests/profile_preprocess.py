#!/usr/bin/env python3
"""用 PMU 硬件计数器定位 MLIR 预处理慢于 OpenCV 的主要来源。

回答的问题：4.38 倍差距主要来自访存（cache miss / 后端停顿）还是指令执行
（指令数、前端停顿）。只报硬件实测计数与由它直接导出的比率，不做外推。

每次测量都校验输出，算错或空转的内核不能显得更省。
"""
import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_reference import reference_resize_rgb
from preprocess_aot_checks import Kernel, require, sha

UNSUPPORTED = (1 << 64) - 1


def build_pmu(source, output):
    """就地编译 PMU 读取库；编译失败是错误，不静默跳过测量。"""
    command = ["gcc", "-O2", "-fPIC", "-shared", "-o", str(output), str(source)]
    done = subprocess.run(command, capture_output=True, text=True)
    require(done.returncode == 0, "PMU 库编译失败: " + done.stderr.strip()[:400])
    return command


# ARMv8 通用计数器只有 6 个。分组测量，每组不超过 4 个事件加留白，避免复用缩放。
EVENT_GROUPS = [(0, 4), (4, 8), (8, 12), (12, 16)]


class Pmu:
    def __init__(self, library):
        self.lib = C.CDLL(str(library))
        self.lib.pmu_open.restype = C.c_int
        self.lib.pmu_open_range.restype = C.c_int
        self.lib.pmu_open_range.argtypes = [C.c_int, C.c_int]
        self.lib.pmu_count.restype = C.c_int
        self.lib.pmu_name.restype = C.c_char_p
        self.lib.pmu_name.argtypes = [C.c_int]
        self.lib.pmu_supported.restype = C.c_int
        self.lib.pmu_supported.argtypes = [C.c_int]
        self.lib.pmu_read.restype = C.c_int
        self.lib.pmu_read.argtypes = [C.POINTER(C.c_uint64), C.POINTER(C.c_double)]
        self.count = self.lib.pmu_count()
        self.names = [self.lib.pmu_name(i).decode() for i in range(self.count)]
        self.buffer = (C.c_uint64 * self.count)()
        self.scaling = C.c_double(1.0)
        self.supported = [False] * self.count

    def open_group(self, begin, end):
        opened = self.lib.pmu_open_range(begin, end)
        require(opened > 0, "PMU 事件组 [%d,%d) 全部打开失败" % (begin, end))
        self.supported = [bool(self.lib.pmu_supported(i)) for i in range(self.count)]
        return opened

    def measure(self, function):
        self.lib.pmu_reset_and_enable()
        result = function()
        self.lib.pmu_disable()
        require(self.lib.pmu_read(self.buffer, C.byref(self.scaling)) == 0, "PMU 读取失败")
        values = {self.names[i]: (None if self.buffer[i] == UNSUPPORTED else int(self.buffer[i]))
                  for i in range(self.count)}
        return values, float(self.scaling.value), result

    def close(self):
        self.lib.pmu_close()


def derive(values, pixels):
    """只做除法，不引入任何模型假设。"""
    def ratio(a, b):
        x, y = values.get(a), values.get(b)
        return None if not x or not y else round(x / y, 4)

    cycles, instructions = values.get("cycles"), values.get("instructions")
    out = {
        "ipc": None if not cycles or not instructions else round(instructions / cycles, 4),
        "cache_miss_rate": ratio("cache_misses", "cache_references"),
        "l1d_read_miss_rate": ratio("l1d_read_misses", "l1d_read_accesses"),
        "ll_read_miss_rate": ratio("ll_read_misses", "ll_read_accesses"),
        "branch_miss_rate": ratio("branch_misses", "branch_instructions"),
        "backend_stall_share": ratio("stalled_cycles_backend", "cycles"),
        "frontend_stall_share": ratio("stalled_cycles_frontend", "cycles"),
        "armv8_branch_miss_rate": ratio("armv8_br_mis_pred", "armv8_br_immed_retired"),
        "armv8_l1i_refill_rate": ratio("armv8_l1i_cache_refill", "armv8_l1i_cache"),
    }
    if pixels:
        for key in ("cycles", "instructions", "cache_misses", "ll_read_misses",
                    "stalled_cycles_backend"):
            value = values.get(key)
            out["%s_per_output_pixel" % key] = None if not value else round(value / pixels, 4)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--variants", nargs="+",
                        default=["fused_destination", "tile_4x32_vector_packed_source"])
    parser.add_argument("--profile", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--input", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args(argv)

    require(platform.machine() == "aarch64", "必须在板端运行")
    require(not args.report.exists() and args.report.parent.is_dir(), "报告必须是新文件")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    source = Path(__file__).resolve().parent / "pmu_counters.c"
    library = args.report.parent / "libpmu_counters.so"
    if library.exists():
        library.unlink()
    command = build_pmu(source, library)
    pmu = Pmu(library)

    manifest = json.loads(args.manifest.read_text())
    require(manifest["status"] == "built", "构建清单不完整")
    height, width = args.profile
    pixels = height * width * 3

    files = sorted(p for p in args.images_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:4]
    require(len(files) == 4, "需要 4 张真实图片")
    images, provenance = [], []
    for path in files:
        encoded = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(decoded is not None, "无法解码 " + str(path))
        image = np.ascontiguousarray(cv2.resize(decoded, (args.input[1], args.input[0]),
                                               interpolation=cv2.INTER_LINEAR))
        images.append(image)
        provenance.append({"path": str(path),
                           "sha256": hashlib.sha256(encoded).hexdigest()})
    expected = [reference_resize_rgb(image, height, width) for image in images]

    runners = {}
    output = np.empty((height, width, 3), dtype=np.uint8)

    def opencv_run(image):
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=output)
        return output

    runners["opencv"] = (opencv_run, None)
    base = args.manifest.parent
    kernels = []
    for item in manifest["variants"]:
        if item.get("status") != "built" or tuple(item["profile"]) != tuple(args.profile):
            continue
        if item["variant"] not in args.variants:
            continue
        kernel = Kernel(base / item["library"], height, width, item["variant"])
        contexts = {}
        kernels.append(kernel)

        def make(kernel=kernel, contexts=contexts):
            def run(image):
                shape = image.shape[:2]
                if shape not in contexts:
                    contexts[shape] = kernel.context(*shape)
                status = kernel.run(contexts[shape], image.ctypes.data, image.nbytes,
                                    shape[1] * 3, output.ctypes.data, output.nbytes, width * 3)
                require(status == 0, "内核返回 " + str(status))
                return output
            return run
        runners[item["variant"]] = (make(), contexts)
    require(len(runners) > 1, "该 profile 下没有匹配的内核")

    result = {
        "schema_version": 1, "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "MLIR 预处理慢于 OpenCV 的主要来源是访存还是指令执行",
        "scope": "单核单线程 PMU 硬件计数；不含解码与输出分配",
        "method": [
            "用 perf_event_open 直接读 ARMv8 PMU，不依赖 perf 工具、不需要 root。",
            "exclude_kernel=1，只统计用户态；计数器复用时按运行时间缩放并上报缩放比。",
            "每次测量都校验完整输出，算错或空转的内核不能显得更省。",
            "各 variant 在同一轮内交错测量，分散热漂移。",
            "事件分 3 组依次打开（ARMv8 通用计数器只有 6 个），避免复用导致缩放失真。",
            "报告的都是硬件计数与其直接比率，不含任何外推模型。",
        ],
        "profile": list(args.profile), "input_shape": list(args.input),
        "output_values": pixels, "warmup": args.warmup, "repeats": args.repeats,
        "pmu_build_command": command,
        "pmu_events_supported": {name: ok for name, ok in zip(pmu.names, pmu.supported)},
        "images": provenance,
        "manifest_sha256": sha(args.manifest), "script_sha256": sha(Path(__file__)),
        "pmu_source_sha256": sha(source),
        "opencv": cv2.__version__, "python": sys.version, "platform": platform.platform(),
        "measurements": {},
    }

    try:
        order = list(runners)
        for name in order:
            run, _ = runners[name]
            for index in range(args.warmup):
                run(images[index % len(images)])

        samples = {name: {} for name in order}
        timings = {name: [] for name in order}
        scalings = {name: [] for name in order}
        supported_map = {}
        # 逐组打开事件，避免 6 个硬件槽位被 12 个事件复用而失真。
        for begin, end in EVENT_GROUPS:
            pmu.open_group(begin, end)
            group_names = [pmu.names[i] for i in range(begin, end)]
            for name in group_names:
                supported_map[name] = pmu.supported[pmu.names.index(name)]
            for name in order:
                run, _ = runners[name]
                for index in range(args.warmup):
                    run(images[index % len(images)])
            for index in range(args.repeats):
                image = images[index % len(images)]
                reference = expected[index % len(images)]
                for name in order:
                    run, _ = runners[name]
                    start = time.perf_counter_ns()
                    values, scaling, produced = pmu.measure(lambda: run(image))
                    elapsed = time.perf_counter_ns() - start
                    if not np.array_equal(produced, reference):
                        require(name == "opencv", "MLIR 输出与参考不一致: " + name)
                        delta = np.abs(produced.astype(np.int16) - reference.astype(np.int16))
                        require(int(delta.max()) <= 1, "OpenCV 超出冻结的 1 灰度级容差")
                    for key in group_names:
                        if values.get(key) is not None:
                            samples[name].setdefault(key, []).append(values[key])
                    timings[name].append(elapsed)
                    scalings[name].append(scaling)
            pmu.close()

        result["pmu_events_supported"] = supported_map
        result["pmu_event_groups"] = [[pmu.names[i] for i in range(b, e)] for b, e in EVENT_GROUPS]
        for name in order:
            aggregated = {key: int(statistics.median(values))
                          for key, values in samples[name].items()}
            for key in pmu.names:
                aggregated.setdefault(key, None)
            entry = {
                "wall_p50_us": round(statistics.median(timings[name]) / 1000, 1),
                "counters_median": aggregated,
                "derived": derive(aggregated, pixels),
                "pmu_scaling_min": round(min(scalings[name]), 4),
            }
            result["measurements"][name] = entry

        baseline = result["measurements"]["opencv"]
        result["comparison_vs_opencv"] = {}
        for name in order:
            if name == "opencv":
                continue
            entry = result["measurements"][name]
            compare = {}
            for key in ("cycles", "instructions", "cache_misses", "ll_read_misses",
                        "stalled_cycles_backend", "stalled_cycles_frontend"):
                mine = entry["counters_median"].get(key)
                theirs = baseline["counters_median"].get(key)
                compare[key + "_x"] = None if not mine or not theirs else round(mine / theirs, 3)
            compare["wall_x"] = round(entry["wall_p50_us"] / baseline["wall_p50_us"], 3)
            result["comparison_vs_opencv"][name] = compare
        result["status"] = "measured"
    except Exception as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        pmu.close()
        for kernel, (_, contexts) in ((k, runners[k.variant]) for k in kernels):
            for handle in contexts.values():
                kernel.destroy(handle)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")

    for name, entry in result["measurements"].items():
        derived = entry["derived"]
        print("%-32s wall %8.1f us  IPC %-7s 后端停顿 %-7s LL缺失率 %s"
              % (name, entry["wall_p50_us"], derived["ipc"],
                 derived["backend_stall_share"], derived["ll_read_miss_rate"]))
    print()
    for name, compare in result.get("comparison_vs_opencv", {}).items():
        print("%s 相对 OpenCV: 周期 %sx  指令 %sx  LL缺失 %sx  后端停顿 %sx"
              % (name, compare["cycles_x"], compare["instructions_x"],
                 compare["ll_read_misses_x"], compare["stalled_cycles_backend_x"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print("profile_preprocess: " + str(error), file=sys.stderr)
        raise SystemExit(1)
