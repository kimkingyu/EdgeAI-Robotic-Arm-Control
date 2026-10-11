#!/usr/bin/env python3
"""Real CPU ResNet18 classification and an offline, self-contained HTML report.

No camera, RKNN mock, VLM, or motor controller is instantiated by this tool.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DemoError(RuntimeError):
    """Invalid demo input or a failed correctness check."""


STAGES = ("preprocess_ms", "inference_ms", "postprocess_ms", "pipeline_ms")


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def summarize(values):
    """Linear percentiles over actual samples; never manufacture missing data."""
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(v) or v < 0 for v in ordered):
        raise DemoError("Timing samples must be nonempty, finite and nonnegative")

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"count": len(ordered), "p50_ms": percentile(0.5),
            "p95_ms": percentile(0.95), "mean_ms": sum(ordered) / len(ordered),
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def summarize_stages(samples):
    return {stage: summarize([item[stage] for item in samples]) for stage in STAGES}


def timed_prediction(model, image, categories, k=5, *, clock=time.perf_counter_ns):
    """Decoded RGB PIL image -> predictions, excluding I/O and model setup.

Inference-mode context entry/exit is outside timing. Input validation and output
validation are part of preprocessing and postprocessing respectively.
"""
    import torch
    from src.vision.resnet_classifier import prepare_image, topk_predictions

    with torch.inference_mode():
        start = clock()
        tensor = prepare_image(image)
        prepared = clock()
        logits = model(tensor)
        inferred = clock()
        predictions = topk_predictions(logits, categories, k)
        finished = clock()
    timing = dict(zip(STAGES, ((prepared - start) / 1e6,
                              (inferred - prepared) / 1e6,
                              (finished - inferred) / 1e6,
                              (finished - start) / 1e6)))
    return logits, predictions, timing


def benchmark(models, image, categories, *, warmup=10, iterations=50, rounds=3, k=5):
    """Alternate variants in every iteration, reversing order between rounds."""
    import torch
    from src.vision.resnet_classifier import prepare_image

    if not models or warmup < 0 or iterations < 1 or rounds < 1:
        raise DemoError("Benchmark requires models, warmup >= 0 and positive iterations/rounds")
    tensor = prepare_image(image)
    wall_start = time.perf_counter_ns()
    with torch.inference_mode():
        for _ in range(warmup):
            for model in models.values():
                model(tensor)
    timed_wall_start = time.perf_counter_ns()
    collected = {name: [] for name in models}
    round_results = []
    for index in range(rounds):
        order = list(models)
        if index % 2:
            order.reverse()
        samples = {name: [] for name in models}
        round_start = time.perf_counter_ns()
        for _ in range(iterations):
            for name in order:
                _, _, timing = timed_prediction(models[name], image, categories, k)
                samples[name].append(timing)
                collected[name].append(timing)
        round_results.append({
            "round": index + 1, "variant_order_per_iteration": order,
            "wall_ms": (time.perf_counter_ns() - round_start) / 1e6,
            "variants": {name: {"summary": summarize_stages(values), "samples": values}
                         for name, values in samples.items()},
        })
    finish = time.perf_counter_ns()
    return {
        "kind": "single_image_cpu_microbenchmark", "warmup_per_variant": warmup,
        "iterations_per_round": iterations, "round_count": rounds,
        "sample_count_per_variant": iterations * rounds,
        "timed_wall_ms": (finish - timed_wall_start) / 1e6,
        "wall_including_warmup_ms": (finish - wall_start) / 1e6,
        "aggregate": {name: summarize_stages(values) for name, values in collected.items()},
        "rounds": round_results,
        "limitations": ["Same decoded image repeated; not camera throughput or dataset accuracy.",
                        "Rounds reuse the process and model; iterations are not independent trials.",
                        "Wall time includes Python/report-statistics overhead; stage timings do not."],
    }


def check_output_paths(directory, overwrite=False):
    directory = Path(directory)
    paths = (directory / "results.json", directory / "demo.html")
    if directory.exists() and not directory.is_dir():
        raise DemoError("Output path is not a directory: %s" % directory)
    if not overwrite and any(path.exists() for path in paths):
        raise DemoError("Report already exists; choose another --output-dir or explicitly --overwrite")
    return paths


def render_report(report, image):
    """No JavaScript, remote resources, or unescaped user/model strings."""
    thumbnail = image.copy()
    thumbnail.thumbnail((720, 640))
    stream = io.BytesIO()
    thumbnail.save(stream, format="PNG")
    image_uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    blocks = []
    for name, variant in report["variants"].items():
        rows = []
        for rank, prediction in enumerate(variant["top5"], 1):
            probability = float(prediction["probability"])
            rows.append('<div class="prediction"><div><span>%d. %s</span><b>%.2f%%</b></div>'
                        '<div class="track"><i style="width:%.4f%%"></i></div></div>' %
                        (rank, escape(str(prediction["label"])), probability * 100, probability * 100))
        blocks.append('<section class="card"><h2>%s</h2>%s</section>' %
                      (escape(name), "".join(rows)))
    measured = report.get("benchmark")
    if measured:
        timing_rows = []
        for name, values in measured["aggregate"].items():
            timing_rows.append("<tr><td>%s</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>" %
                               (escape(name), values["preprocess_ms"]["p50_ms"],
                                values["inference_ms"]["p50_ms"], values["inference_ms"]["p95_ms"],
                                values["pipeline_ms"]["p50_ms"]))
        timing = ('<h2>CPU 单图重复微基准 / ms</h2><p>预热 %d 次，每版本 %d 个样本，%d 轮交错计时。</p>'
                  '<div class="scroll"><table><tr><th>版本</th><th>预处理 P50</th><th>推理 P50</th>'
                  '<th>推理 P95</th><th>分类流水线 P50</th></tr>%s</table></div>') % (
                      measured["warmup_per_variant"], measured["sample_count_per_variant"],
                      measured["round_count"], "".join(timing_rows))
        if "baseline" in measured["aggregate"] and "fused" in measured["aggregate"]:
            original = measured["aggregate"]["baseline"]["inference_ms"]["p50_ms"]
            optimized = measured["aggregate"]["fused"]["inference_ms"]["p50_ms"]
            if optimized > 0:
                timing += '<p>推理 P50 比值（原始 / 融合）：<b>%.3fx</b>；小于 1 表示融合版本更慢。</p>' % (original / optimized)
    else:
        timing = "<h2>未运行基准</h2><p>JSON 中的单次时间包含冷启动影响，不作为性能结论。加 --benchmark 可重复测量。</p>"
    comparison = report.get("comparison")
    check = "未选择 --compare，仅运行原始模型。"
    if comparison:
        check = ("融合一致性：%s；最大 logits 绝对误差 %.8g；Top-1 一致：%s；融合 %d 对 Conv/BN。" %
                 ("通过" if comparison["passed"] else "未通过", comparison["max_abs_error"],
                  comparison["top1_equal"], report["variants"]["fused"]["fused_pairs"]))
    metadata = report["model"]
    details = escape(json.dumps({key: value for key, value in metadata.items() if key != "categories"},
                               ensure_ascii=False, indent=2))
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ResNet18 CPU Demo</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101826;color:#e7eef8;font:16px/1.65 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:32px 22px}h1{font-size:34px;margin:0 0 8px}h2{font-size:20px;margin:0 0 16px}
p{color:#bac8dd}.tag{color:#75dfc7;font-weight:700;letter-spacing:2px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.card{padding:24px;background:#19273b;border:1px solid #2d405a;border-radius:14px;margin-bottom:20px}
img{width:100%%;height:auto;border-radius:9px;display:block}.prediction{margin:15px 0}.prediction>div:first-child{display:flex;justify-content:space-between;gap:15px}
.track{height:7px;background:#2d405a;border-radius:8px;margin-top:7px;overflow:hidden}.track i{height:100%%;display:block;background:#75dfc7}
table{width:100%%;border-collapse:collapse;text-align:left}td,th{padding:12px;border-bottom:1px solid #344963;white-space:nowrap}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#c7d5e8}.scroll{overflow:auto}.note{border-left:3px solid #e7ba69;padding-left:15px}
@media(max-width:760px){.grid{grid-template-columns:1fr}main{padding:20px 14px}h1{font-size:28px}}
</style></head><body><main><div class="tag">EDGE AI / REAL CPU INFERENCE</div>
<h1>ResNet18 图片分类与融合对照</h1><p>静态离线报告 · %s · 不连接机械臂或 NPU</p>
<div class="grid"><section class="card"><h2>输入图片预览</h2><img src="%s" alt="输入图片预览"><p>%s</p></section>
<div>%s</div></div><section class="card"><h2>输出一致性</h2><p>%s</p></section>
<section class="card">%s</section><section class="card note"><h2>结果边界</h2>
<p>这里只做 ImageNet 分类，不产生目标框或抓取坐标。百分比为模型 softmax 分数，不是已校准的正确率。
流水线计时从已解码 RGB 图片到分类结果，不含磁盘 I/O、权重下载、模型加载或报告写入。
少量图片的一致性不等于数据集精度；本机 CPU 数据不代表 RK3588 NPU 性能。</p></section>
<details class="card"><summary>模型、预处理与环境记录</summary><pre>%s</pre></details>
</main></body></html>''' % (escape(report["created_at"]), image_uri, escape(report["image"]["path"]),
                         "".join(blocks), escape(check), timing, details)


def write_reports(directory, report, image, *, overwrite=False):
    paths = check_output_paths(directory, overwrite)
    html = render_report(report, image)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    Path(directory).mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with paths[0].open(mode, encoding="utf-8") as handle:
        handle.write(payload + "\n")
    with paths[1].open(mode, encoding="utf-8") as handle:
        handle.write(html)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="真实图片路径")
    parser.add_argument("--weights", type=Path, help="本地 ResNet18 state_dict；默认项目权重路径")
    parser.add_argument("--download-weights", action="store_true", help="允许下载缺失的官方权重至项目")
    parser.add_argument("--compare", action="store_true", help="对照原模型与 Conv-BN 融合版本")
    parser.add_argument("--benchmark", action="store_true", help="同一图片重复测量，非数据集准确率")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--warmup", type=nonnegative_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/resnet-demo")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        check_output_paths(args.output_dir, args.overwrite)
        if not args.image.is_file():
            raise DemoError("Image file not found: %s" % args.image)
        from src.vision.resnet_classifier import (
            load_classifier, open_rgb_image, fuse_resnet18, compare_logits,
        )
        start = time.perf_counter_ns()
        image = open_rgb_image(args.image)
        decode_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        model, metadata = load_classifier(args.weights, download_weights=args.download_weights,
                                          threads=args.threads)
        setup_ms = (time.perf_counter_ns() - start) / 1e6
        models = {"baseline": model}
        fused_pairs = 0
        fusion_ms = None
        if args.compare:
            start = time.perf_counter_ns()
            models["fused"], fused_pairs = fuse_resnet18(model)
            fusion_ms = (time.perf_counter_ns() - start) / 1e6
        variants, logits = {}, {}
        for name, instance in models.items():
            logits[name], predictions, timing = timed_prediction(instance, image, metadata["categories"])
            variants[name] = {"top5": predictions, "single_observation_ms": timing,
                              "fused_pairs": fused_pairs if name == "fused" else 0}
        comparison = compare_logits(logits["baseline"], logits["fused"]) if args.compare else None
        if comparison and not comparison["passed"]:
            raise DemoError("Conv-BN comparison failed: %s" % json.dumps(comparison))
        measured = benchmark(models, image, metadata["categories"], warmup=args.warmup,
                             iterations=args.iterations, rounds=args.rounds) if args.benchmark else None
        report = {
            "schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
            "status": "measured", "device": "cpu", "model": metadata,
            "image": {"path": str(args.image.resolve()), "width": image.width, "height": image.height,
                      "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest()},
            "startup": {"image_decode_wall_ms": decode_ms, "model_setup_wall_ms": setup_ms,
                        "model_setup_includes_download_if_requested": True, "fusion_setup_ms": fusion_ms},
            "variants": variants, "comparison": comparison, "benchmark": measured,
            "claims": {"dataset_accuracy_measured": False, "npu_measured": False,
                       "robot_controlled": False, "image_top1_ground_truth_verified": False},
        }
        json_path, html_path = write_reports(args.output_dir, report, image, overwrite=args.overwrite)
        for name, variant in variants.items():
            top = variant["top5"][0]
            print("%s: %s (%.2f%%)" % (name, top["label"], top["probability"] * 100))
        if measured:
            for name, summary in measured["aggregate"].items():
                print("%s CPU inference P50=%.3f ms, P95=%.3f ms; pipeline P50=%.3f ms" %
                      (name, summary["inference_ms"]["p50_ms"], summary["inference_ms"]["p95_ms"],
                       summary["pipeline_ms"]["p50_ms"]))
        print("JSON: %s\nHTML: %s" % (json_path, html_path))
        return 0
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print("[ResNet error] %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
