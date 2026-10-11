#!/usr/bin/env python3
"""Offline unit tests. Synthetic images/weights here are NOT benchmark evidence."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from src.vision.resnet_classifier import (
    ResNetError, load_classifier, open_rgb_image, prepare_image,
    fuse_resnet18, topk_predictions, compare_logits,
)
from tools.run_resnet_demo import (
    DemoError, summarize, timed_prediction, benchmark, render_report,
    write_reports, check_output_paths,
)
from tools.export_resnet_onnx import export_and_verify


def report_fixture():
    """In-memory fixture only; unit-test predictions are never shipped as results."""
    return {
        "created_at": "unit-test", "model": {"categories": ["test"], "source": "<script>test</script>"},
        "image": {"path": '<img src=x onerror="alert(1)">'},
        "variants": {"baseline": {"top5": [{"index": 0, "label": "<script>label</script>",
                                              "probability": 0.8}], "fused_pairs": 0}},
        "benchmark": None, "comparison": None,
    }


class PreprocessingTests(unittest.TestCase):
    def test_contract_matches_weights_transform(self):
        image = Image.new("RGB", (320, 180), (64, 128, 192))
        actual = prepare_image(image)
        expected = ResNet18_Weights.IMAGENET1K_V1.transforms()(image).unsqueeze(0)
        self.assertEqual(tuple(actual.shape), (1, 3, 224, 224))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, "cpu")
        self.assertTrue(torch.equal(actual, expected))

    def test_rgb_channel_order_and_normalization(self):
        tensor = prepare_image(Image.new("RGB", (280, 300), (255, 0, 0)))
        expected = torch.tensor([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
        self.assertTrue(torch.allclose(tensor[0, :, 100, 100], expected, atol=1e-6))

    def test_open_converts_grayscale_and_releases_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.png"
            Image.new("L", (20, 10), 127).save(path)
            image = open_rgb_image(path)
            path.unlink()
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), (127, 127, 127))

    def test_missing_and_corrupt_images_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.png"
            with self.assertRaises(ResNetError):
                open_rgb_image(missing)
            corrupt = Path(directory) / "bad.png"
            corrupt.write_bytes(b"not an image")
            with self.assertRaises(ResNetError):
                open_rgb_image(corrupt)


class PredictionTests(unittest.TestCase):
    def test_softmax_is_stable_and_topk_ordered(self):
        result = topk_predictions(torch.tensor([[1000.0, 1002.0, 1001.0]]), ["a", "b", "c"], 3)
        self.assertEqual([item["index"] for item in result], [1, 2, 0])
        self.assertEqual([item["label"] for item in result], ["b", "c", "a"])
        self.assertAlmostEqual(sum(item["probability"] for item in result), 1.0, places=6)
        json.dumps(result, allow_nan=False)

    def test_rejects_malformed_outputs(self):
        for logits in (torch.zeros(3), torch.zeros(2, 3), torch.zeros(1, 2),
                       torch.tensor([[float("nan"), 1.0, 2.0]]),
                       torch.tensor([[float("inf"), 1.0, 2.0]]),
                       torch.tensor([[1, 2, 3]])):
            with self.subTest(shape=logits.shape, dtype=logits.dtype):
                with self.assertRaises(ResNetError):
                    topk_predictions(logits, ["a", "b", "c"], 2)

    def test_invalid_k(self):
        for k in (0, -1, 4):
            with self.assertRaises(ResNetError):
                topk_predictions(torch.zeros(1, 3), ["a", "b", "c"], k)

    def test_comparison_records_but_does_not_hide_difference(self):
        reference = torch.tensor([[3.0, 2.0, 1.0]])
        same = compare_logits(reference, reference.clone())
        self.assertTrue(same["passed"])
        self.assertEqual(same["max_abs_error"], 0)
        changed = compare_logits(reference, torch.tensor([[0.0, 2.0, 4.0]]))
        self.assertFalse(changed["passed"])
        self.assertFalse(changed["top1_equal"])
        self.assertGreater(changed["max_abs_error"], 0)

    def test_comparison_rejects_nonfinite_or_mismatched_inputs(self):
        reference = torch.ones(1, 3)
        for candidate in (torch.ones(1, 4), torch.full((1, 3), float("nan"))):
            with self.assertRaises(ResNetError):
                compare_logits(reference, candidate)


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(21)
        cls.baseline = resnet18(weights=None).eval()
        cls.fused, cls.pairs = fuse_resnet18(cls.baseline)

    def test_covers_stem_blocks_and_residual_downsample(self):
        self.assertEqual(self.pairs, 20)
        self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm2d) for m in self.baseline.modules()), 20)
        self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm2d) for m in self.fused.modules()), 0)
        self.assertIsInstance(self.fused.layer2[0].downsample[1], torch.nn.Identity)
        self.assertFalse(self.fused.training)
        self.assertEqual(next(self.fused.parameters()).dtype, torch.float32)

    def test_does_not_mutate_baseline_storage(self):
        self.assertIsInstance(self.baseline.bn1, torch.nn.BatchNorm2d)
        self.assertNotEqual(self.baseline.conv1.weight.data_ptr(), self.fused.conv1.weight.data_ptr())
        self.assertIsNone(self.baseline.conv1.bias)

    def test_fused_random_weight_logits_are_equivalent(self):
        # Test structure/correctness only. Real pretrained evidence is obtained separately.
        with torch.inference_mode():
            for seed in (31, 32):
                torch.manual_seed(seed)
                tensor = torch.randn(1, 3, 224, 224)
                comparison = compare_logits(self.baseline(tensor), self.fused(tensor))
                self.assertTrue(comparison["passed"], comparison)

    def test_training_model_rejected(self):
        model = copy.deepcopy(self.baseline).train()
        with self.assertRaises(ResNetError):
            fuse_resnet18(model)

    def test_different_resnet_depth_is_not_silently_partially_fused(self):
        model = copy.deepcopy(self.baseline)
        model.layer1 = torch.nn.Sequential(model.layer1[0])
        with self.assertRaises(ResNetError):
            fuse_resnet18(model)


class WeightsTests(unittest.TestCase):
    def test_missing_weights_do_not_trigger_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.pth"
            with self.assertRaises(ResNetError):
                load_classifier(path, download_weights=False)
            self.assertFalse(path.exists())

    def test_invalid_threads(self):
        for count in (0, -1):
            with self.assertRaises(ResNetError):
                load_classifier(threads=count)

    def test_malformed_checkpoint_is_not_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pth"
            torch.save({"not_a_resnet_parameter": torch.zeros(1)}, path)
            with self.assertRaises(ResNetError):
                load_classifier(path)

    def test_safe_load_never_falls_back_to_unrestricted_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsupported.pth"
            path.write_bytes(b"safe-load API test")
            with patch("torch.load", side_effect=TypeError("weights_only not supported")) as loader:
                with self.assertRaises(ResNetError):
                    load_classifier(path)
                self.assertEqual(loader.call_count, 1)
                self.assertTrue(loader.call_args.kwargs["weights_only"])


class TimingTests(unittest.TestCase):
    def test_linear_percentiles(self):
        summary = summarize([1, 2, 3, 4])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)

    def test_bad_samples_are_not_filled_or_dropped(self):
        for values in ([], [float("nan")], [float("inf")], [-1]):
            with self.assertRaises(DemoError):
                summarize(values)

    def test_stage_timer_excludes_image_io_and_model_loading(self):
        model = lambda _: torch.tensor([[1.0, 2.0, 3.0]])
        ticks = iter([0, 1_000_000, 3_000_000, 6_000_000])
        with patch("src.vision.resnet_classifier.prepare_image", return_value=torch.zeros(1, 3, 224, 224)), \
             patch("src.vision.resnet_classifier.open_rgb_image", side_effect=AssertionError("I/O in timer")), \
             patch("src.vision.resnet_classifier.load_classifier", side_effect=AssertionError("load in timer")):
            _, _, times = timed_prediction(model, object(), ["a", "b", "c"], k=3,
                                           clock=lambda: next(ticks))
        self.assertEqual(times, {"preprocess_ms": 1.0, "inference_ms": 2.0,
                                "postprocess_ms": 3.0, "pipeline_ms": 6.0})

    def test_benchmark_counts_warmup_and_alternates_variants(self):
        calls, order = {"baseline": 0, "fused": 0}, []

        def model(name):
            def call(_):
                calls[name] += 1
                return torch.zeros(1, 3)
            return call

        models = {name: model(name) for name in calls}
        names = {id(value): name for name, value in models.items()}

        def prediction(instance, *args):
            order.append(names[id(instance)])
            return None, [], {"preprocess_ms": 1, "inference_ms": 2,
                              "postprocess_ms": 1, "pipeline_ms": 4}

        with patch("src.vision.resnet_classifier.prepare_image", return_value=torch.zeros(1, 3, 224, 224)), \
             patch("tools.run_resnet_demo.timed_prediction", side_effect=prediction):
            result = benchmark(models, object(), ["a", "b", "c"], warmup=2, iterations=3, rounds=2, k=3)
        self.assertEqual(calls, {"baseline": 2, "fused": 2})
        self.assertEqual(order[:6], ["baseline", "fused"] * 3)
        self.assertEqual(order[6:], ["fused", "baseline"] * 3)
        self.assertEqual(result["sample_count_per_variant"], 6)
        self.assertEqual(result["aggregate"]["baseline"]["inference_ms"]["count"], 6)


class ReportTests(unittest.TestCase):
    def test_html_escapes_labels_paths_and_metadata(self):
        html = render_report(report_fixture(), Image.new("RGB", (8, 8)))
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;label&lt;/script&gt;", html)
        self.assertIn("&lt;img", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("width:100%;", html)
        self.assertNotIn("<script src=", html)

    def test_report_files_are_parseable_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            report = report_fixture()
            image = Image.new("RGB", (8, 8))
            paths = write_reports(directory, report, image)
            self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8")), report)
            self.assertTrue(paths[1].read_text(encoding="utf-8").startswith("<!doctype html>"))
            with self.assertRaises(DemoError):
                write_reports(directory, report, image)
            write_reports(directory, report, image, overwrite=True)

    def test_existing_html_does_not_create_json(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "demo.html").write_text("user report", encoding="utf-8")
            with self.assertRaises(DemoError):
                check_output_paths(directory)
            self.assertFalse((Path(directory) / "results.json").exists())

    def test_nonfinite_report_is_not_written(self):
        with tempfile.TemporaryDirectory() as directory:
            report = report_fixture()
            report["invalid"] = float("nan")
            with self.assertRaises(ValueError):
                write_reports(directory, report, Image.new("RGB", (8, 8)))
            self.assertFalse((Path(directory) / "results.json").exists())


class CliAndExportTests(unittest.TestCase):
    def test_help_works_without_site_packages(self):
        for tool in ("run_resnet_demo.py", "export_resnet_onnx.py"):
            result = subprocess.run([sys.executable, "-S", str(ROOT / "tools" / tool), "--help"],
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(b"--image", result.stdout)

    def test_no_dependency_fallback_to_mock(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "real_input.png"
            Image.new("RGB", (8, 8)).save(image)
            result = subprocess.run([sys.executable, "-S", str(ROOT / "tools/run_resnet_demo.py"),
                                     "--image", str(image), "--output-dir", str(Path(directory) / "out")],
                                    capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"ResNet error", result.stderr)
            self.assertNotIn(b"Traceback", result.stderr)
            self.assertFalse((Path(directory) / "out/results.json").exists())

    def test_export_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "resnet.onnx"
            target.write_bytes(b"user artifact")
            with self.assertRaises(ValueError):
                export_and_verify(Path(directory) / "missing.jpg", target)
            self.assertEqual(target.read_bytes(), b"user artifact")

    def test_export_requires_onnx_suffix_and_real_image(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_and_verify("missing.jpg", Path(directory) / "model.rknn")
            with self.assertRaises(ValueError):
                export_and_verify("missing.jpg", Path(directory) / "model.onnx")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
