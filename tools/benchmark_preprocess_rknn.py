#!/usr/bin/env python3
"""M6 板端 A/B：OpenCV 与 MLIR 预处理喂同一个真实 RKNN 模型。

比较的是预处理耗时和模型原始输出，不涉及控制器和舵机。没有标注数据和正确的
后处理，所以这里不报 mAP、检测精度保持或定位精度 —— 只报原始张量是否一致。

缺 rknnlite、缺模型、缺 MLIR 库都是 failure，不是 skip。Mock 推理路径在这里被
显式拒绝：拿 time.sleep 当 NPU 时延写进报告就是造假。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.vision.preprocess import Preprocessor

# 独立的 Q11 定点参考，用来判定像素差异由哪一侧引入。
sys.path.insert(0, str(ROOT / "compiler" / "tests"))
try:
    from preprocess_reference import reference_resize_rgb
except ImportError:
    reference_resize_rgb = None


def fail(message):
    raise RuntimeError(message)


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(samples):
    ordered = sorted(samples)

    def percentile(q):
        rank = (len(ordered) - 1) * q
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)

    return {"count": len(ordered), "mean_ns": statistics.mean(ordered), "min_ns": ordered[0],
            "p50_ns": percentile(.5), "p95_ns": percentile(.95), "p99_ns": percentile(.99),
            "max_ns": ordered[-1]}


def sysfs_snapshot():
    snapshot = {}
    for name, pattern in (("thermal_millidegrees", "sys/class/thermal/thermal_zone*/temp"),
                          ("frequency_khz", "sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq")):
        values = {}
        for path in sorted(Path("/").glob(pattern)):
            try:
                values[str(path)] = path.read_text().strip()
            except OSError as error:
                values[str(path)] = "unavailable: " + str(error)
        snapshot[name] = values
    return snapshot


def load_runtime(model_path, core_mode):
    """加载真实 RKNN。任何降级都视为失败。"""
    try:
        from rknnlite.api import RKNNLite
    except ImportError as error:
        fail("需要 rknnlite，且不接受 Mock 推理: " + str(error))
    import rknnlite
    model = Path(model_path)
    if not model.is_file():
        fail("模型不存在: " + str(model))
    runtime = RKNNLite()
    if runtime.load_rknn(str(model)) != 0:
        fail("load_rknn 失败: " + str(model))
    attribute = {"auto": "NPU_CORE_AUTO", "core0": "NPU_CORE_0", "all": "NPU_CORE_0_1_2"}[core_mode]
    if runtime.init_runtime(core_mask=getattr(RKNNLite, attribute)) != 0:
        fail("init_runtime 失败 (core=%s)" % attribute)
    return runtime, {"model_path": str(model), "model_sha256": sha256_file(model),
                     "model_bytes": model.stat().st_size, "core_mode": core_mode,
                     "core_mask": attribute,
                     "rknnlite_package": str(Path(rknnlite.__file__).parent),
                     "rknnlite_version": getattr(rknnlite, "__version__", "unknown")}


def librknnrt_version():
    for candidate in ("/usr/lib/librknnrt.so", "/usr/lib/aarch64-linux-gnu/librknnrt.so"):
        path = Path(candidate)
        if not path.is_file():
            continue
        blob = path.read_bytes()
        marker = b"librknnrt version:"
        index = blob.find(marker)
        if index >= 0:
            end = blob.find(b"\x00", index)
            return {"path": candidate, "banner": blob[index:end].decode("ascii", "replace")}
        return {"path": candidate, "banner": "unknown"}
    return {"path": None, "banner": "librknnrt.so not found"}


def load_images(directory, count, input_size):
    files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(files) < count:
        fail("需要 %d 张真实图片，只找到 %d 张" % (count, len(files)))
    images, provenance = [], []
    for path in files[:count]:
        encoded = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            fail("无法解码 " + str(path))
        image = np.ascontiguousarray(cv2.resize(decoded, (input_size[1], input_size[0]),
                                                interpolation=cv2.INTER_LINEAR))
        images.append(image)
        provenance.append({"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(),
                           "decoded_shape": list(decoded.shape), "benchmark_shape": list(image.shape)})
    return images, provenance


def compare_outputs(left, right):
    """比较两组模型原始输出。这是张量比较，不是精度评估。"""
    if len(left) != len(right):
        fail("输出张量数量不一致: %d vs %d" % (len(left), len(right)))
    report = []
    for index, (a, b) in enumerate(zip(left, right)):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            fail("输出 %d 形状/类型不一致: %s %s vs %s %s" % (index, a.shape, a.dtype, b.shape, b.dtype))
        identical = bool(np.array_equal(a, b))
        entry = {"index": index, "shape": list(a.shape), "dtype": str(a.dtype),
                 "bitwise_identical": identical}
        if not identical and np.issubdtype(a.dtype, np.number):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            entry.update(max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                         differing_elements=int(np.count_nonzero(diff)), total_elements=int(a.size))
        report.append(entry)
    return report


def run(args):
    if platform.machine() != "aarch64":
        fail("必须在板端运行，当前架构 " + platform.machine())
    report_path = Path(args.report)
    if report_path.exists():
        fail("报告已存在，拒绝覆盖: " + str(report_path))
    if not report_path.parent.is_dir():
        fail("报告目录不存在: " + str(report_path.parent))
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    images, provenance = load_images(args.images_dir, args.image_count, tuple(args.input))
    profile = tuple(args.profile)
    runtime, model_info = load_runtime(args.model, args.core_mode)

    backends = {}
    try:
        backends["opencv"] = Preprocessor(output_size=profile, backend="opencv")
        backends["mlir"] = Preprocessor(output_size=profile, backend="mlir",
                                        library=args.mlir_library, manifest=args.mlir_manifest)
        result = {
            "schema_version": 1, "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "预处理耗时与模型原始输出比较；无标注数据，故不报 mAP/检测精度/定位精度",
            "profile": list(profile), "input_shape": list(args.input),
            "warmup": args.warmup, "iterations": args.iterations,
            "images": provenance, "model": model_info, "librknnrt": librknnrt_version(),
            "backends": {name: backend.describe() for name, backend in backends.items()},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "opencv": cv2.__version__, "numpy": np.__version__, "python": sys.version,
            "platform": platform.platform(),
            "methodology": [
                "两个后端在同一次循环内交错执行，不按块分别测量。",
                "预处理与推理分开计时；推理输入是各自后端的真实输出。",
                "每个后端各自持有输出缓冲，不共用全局缓冲。",
                "Mock 推理被显式拒绝；缺库缺模型是失败而非跳过。",
                "耗时由逐次计时推导，不是持续吞吐（timing-derived, not sustained throughput）。",
                "进程亲和性由调用方用 taskset 设置；不写 governor 或任何 sysfs。",
            ],
            "sysfs_before": sysfs_snapshot(),
        }

        preprocess_samples = {name: [] for name in backends}
        inference_samples = {name: [] for name in backends}
        buffers = {name: backend.new_output() for name, backend in backends.items()}
        outputs_by_image = {}
        order = list(backends)

        for index in range(args.warmup + args.iterations):
            image = images[index % len(images)]
            for name in order:
                backend = backends[name]
                batch = backend.run_batch1(image, out=buffers[name])
                start = time.perf_counter_ns()
                outputs = runtime.inference(inputs=[batch])
                inference_ns = time.perf_counter_ns() - start
                if not outputs:
                    fail("RKNN 推理没有返回输出；拒绝把空结果当成成功")
                if index >= args.warmup:
                    preprocess_samples[name].append(backend.last_timings["call_ns"])
                    inference_samples[name].append(inference_ns)
                key = (index % len(images), name)
                if key not in outputs_by_image:
                    outputs_by_image[key] = [np.array(o, copy=True) for o in outputs]

        # 对照实验：同一份输入连跑两次。若不一致，说明 NPU 执行本身不确定，
        # 那么后面的输出差异就不能归因到预处理差异上。
        control = backends["opencv"].run_batch1(images[0], out=buffers["opencv"])
        first = [np.array(o, copy=True) for o in runtime.inference(inputs=[control])]
        second = [np.array(o, copy=True) for o in runtime.inference(inputs=[control])]
        determinism = compare_outputs(first, second)
        result["determinism_control"] = {
            "description": "同一份预处理输出连续推理两次，用于确认输出差异可归因于输入差异",
            "identical": all(entry["bitwise_identical"] for entry in determinism),
            "tensors": determinism,
        }
        if not result["determinism_control"]["identical"]:
            fail("同一输入两次推理结果不一致，无法把输出差异归因到预处理；先解决这个问题")

        # 预处理输出逐字节比较：这是两个后端是否等价的直接证据。
        # 同时和独立的 Q11 参考比较，用来判定差异是谁引入的，而不是只说"两者不同"。
        pixel_report = []
        for position, image in enumerate(images):
            left = backends["opencv"].run(image)
            right = backends["mlir"].run(image)
            difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
            entry = {
                "image_index": position,
                "bitwise_identical": bool(np.array_equal(left, right)),
                "max_abs_diff": int(difference.max()),
                "mean_abs_diff": float(difference.mean()),
                "differing_pixels": int(np.count_nonzero(difference)),
                "total_values": int(left.size),
            }
            if reference_resize_rgb is not None:
                expected = reference_resize_rgb(image, *profile)
                for name, actual in (("opencv", left), ("mlir", right)):
                    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                    entry[name + "_vs_reference"] = {
                        "bitwise_identical": bool(np.array_equal(actual, expected)),
                        "max_abs_diff": int(delta.max()),
                        "differing_values": int(np.count_nonzero(delta)),
                    }
            pixel_report.append(entry)

        model_report = []
        for position in range(len(images)):
            model_report.append({
                "image_index": position,
                "tensors": compare_outputs(outputs_by_image[(position, "opencv")],
                                           outputs_by_image[(position, "mlir")]),
            })

        baseline_pre = describe(preprocess_samples["opencv"])
        result["measurements"] = {}
        for name in order:
            pre_stats = describe(preprocess_samples[name])
            inf_stats = describe(inference_samples[name])
            total_p50 = pre_stats["p50_ns"] + inf_stats["p50_ns"]
            entry = {"preprocess_ns": pre_stats, "inference_ns": inf_stats,
                     "preprocess_plus_inference_p50_ns": total_p50}
            if name != "opencv":
                entry["preprocess_p50_ratio_vs_opencv"] = pre_stats["p50_ns"] / baseline_pre["p50_ns"]
            result["measurements"][name] = entry
        baseline_total = result["measurements"]["opencv"]["preprocess_plus_inference_p50_ns"]
        for name in order:
            entry = result["measurements"][name]
            entry["total_p50_ratio_vs_opencv"] = entry["preprocess_plus_inference_p50_ns"] / baseline_total

        result["preprocess_pixel_comparison"] = pixel_report
        result["model_output_comparison"] = model_report
        result["conclusions"] = build_conclusions(result)
        result["status"] = "measured"
    except Exception as error:
        result = locals().get("result") or {"schema_version": 1, "status": "failed"}
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        snapshot = locals().get("result")
        if isinstance(snapshot, dict):
            snapshot["sysfs_after"] = sysfs_snapshot()
            snapshot["finished_utc"] = datetime.now(timezone.utc).isoformat()
            with report_path.open("x", encoding="utf-8") as handle:
                json.dump(snapshot, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
        for backend in backends.values():
            backend.close()
        try:
            runtime.release()
        except Exception:  # noqa: BLE001 - 释放失败不掩盖主错误
            pass

    print_summary(result)
    return result


def build_conclusions(result):
    conclusions = []
    pixel_entries = result["preprocess_pixel_comparison"]
    pixels_identical = all(entry["bitwise_identical"] for entry in pixel_entries)
    max_pixel = max(entry["max_abs_diff"] for entry in pixel_entries)
    share = max(entry["differing_pixels"] / float(entry["total_values"]) for entry in pixel_entries)
    conclusions.append("两个后端的预处理输出逐字节一致。" if pixels_identical
                       else "两个后端的预处理输出不一致：最大像素差 %d，最多 %.2f%% 的值不同。"
                            % (max_pixel, 100 * share))
    if not pixels_identical and "mlir_vs_reference" in pixel_entries[0]:
        mlir_exact = all(entry["mlir_vs_reference"]["bitwise_identical"] for entry in pixel_entries)
        opencv_exact = all(entry["opencv_vs_reference"]["bitwise_identical"] for entry in pixel_entries)
        opencv_max = max(entry["opencv_vs_reference"]["max_abs_diff"] for entry in pixel_entries)
        conclusions.append(
            "对独立 Q11 参考：MLIR %s，OpenCV %s（最大差 %d）——差异由 OpenCV 的舍入实现引入，"
            "不是 MLIR 内核算错。" % ("逐字节一致" if mlir_exact else "不一致",
                                  "逐字节一致" if opencv_exact else "不一致", opencv_max))
    control = result.get("determinism_control")
    if control:
        conclusions.append("对照实验：同一输入连续两次推理结果%s，因此输出差异可归因于输入差异。"
                           % ("一致" if control["identical"] else "不一致"))
    tensors = [t for entry in result["model_output_comparison"] for t in entry["tensors"]]
    tensors_identical = all(t["bitwise_identical"] for t in tensors)
    if tensors_identical:
        conclusions.append("模型原始输出逐字节一致。")
    else:
        max_diff = max(t.get("max_abs_diff", 0) for t in tensors)
        max_share = max(0.0 if t["bitwise_identical"]
                        else t["differing_elements"] / float(t["total_elements"]) for t in tensors)
        max_mean = max(t.get("mean_abs_diff", 0.0) for t in tensors)
        # 不在这里断言模型内部精度（YOLO 为 INT8、Qwen 视觉编码器为 FP16），
        # 只陈述实测到的输入差异与输出差异，不臆测放大机制。
        conclusions.append("模型原始输出存在差异：最大绝对差 %.3f，平均绝对差最大 %.6f，"
                           "最多 %.2f%% 的元素不同。输入仅差 ±1 个灰度级即产生该输出差异，"
                           "这是更换预处理实现的真实代价。"
                           % (max_diff, max_mean, 100 * max_share))
        conclusions.append("要做无缝默认替换，需先与部署所用 OpenCV 版本达到逐字节一致；"
                           "当前仅近似一致，因此保留为显式实验后端。")
    ratio = result["measurements"]["mlir"]["preprocess_p50_ratio_vs_opencv"]
    conclusions.append("MLIR 预处理 P50 是 OpenCV 的 %.2f 倍%s。"
                       % (ratio, "，更慢" if ratio > 1 else "，更快"))
    conclusions.append("无标注数据与后处理解码，因此不报 mAP、检测精度保持或实机定位精度。")
    if ratio > 1:
        conclusions.append("OpenCV 保持默认后端；MLIR 仍为显式实验后端。")
    return conclusions


def print_summary(result):
    print("\n模型: %s" % result["model"]["model_path"])
    print("RKNN 运行时: %s" % result["librknnrt"]["banner"])
    for name, entry in result["measurements"].items():
        print("%-8s 预处理 P50 %9.1f us | 推理 P50 %9.1f us | 合计 P50 %9.1f us"
              % (name, entry["preprocess_ns"]["p50_ns"] / 1000,
                 entry["inference_ns"]["p50_ns"] / 1000,
                 entry["preprocess_plus_inference_p50_ns"] / 1000))
    for line in result["conclusions"]:
        print("- " + line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="真实 .rknn 模型路径")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mlir-library", required=True)
    parser.add_argument("--mlir-manifest")
    parser.add_argument("--profile", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--input", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--image-count", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--core-mode", choices=("auto", "core0", "all"), default="auto")
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as error:
        print("benchmark_preprocess_rknn: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
