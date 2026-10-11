#!/usr/bin/env python3
"""多模型优化基准与分类流水线离线单元测试。

测试说明：
- 严格离线运行，不联网下载权重，不访问硬件设备。
- 测试中若使用随机生成权重，仅用于单测结构融合与数值算子等价性校验，绝非正式精度评估证据。
- 保证 --help 零依赖可用，验证 overwrite 保护与 baseline 隔离性。
"""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import torchvision.models as models

from src.vision.classification import (
    MODEL_IDS,
    ClassificationError,
    get_model_spec,
    configure_cpu_threads,
    load_classifier,
    open_rgb_image,
    prepare_image,
    compute_sha256,
    topk_predictions,
    compare_logits,
    environment_metadata,
)
from src.vision.classification_variants import (
    VARIANT_IDS,
    TORCH_VARIANT_IDS,
    ORT_VARIANT_IDS,
    fuse_classifier,
    build_torch_variant,
    create_ort_session,
    graph_statistics,
)
from tools.export_classifier_onnx import export_and_verify


def _create_dummy_onnx(path: Path) -> None:
    """在指定路径创建一个小型的合法 ONNX 模型供测试使用。"""
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()
            self.fc = torch.nn.Linear(8 * 224 * 224, 10)

        def forward(self, x):
            return self.fc(self.relu(self.conv(x)).flatten(1))

    m = TinyModel().eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        m,
        (dummy,),
        str(path),
        opset_version=17,
        dynamo=False,
        external_data=False,
        input_names=["images"],
        output_names=["logits"],
    )


class ModelRegistrationTests(unittest.TestCase):
    def test_model_ids_whitelist(self):
        self.assertEqual(
            MODEL_IDS,
            ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"),
        )

    def test_get_model_spec_contract(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                spec = get_model_spec(model_id)
                self.assertEqual(spec["model_id"], model_id)
                self.assertEqual(spec["architecture"], model_id)
                self.assertIn("IMAGENET1K_V1", spec["weights_id"])
                self.assertTrue(spec["weights_url"].startswith("https://download.pytorch.org/models/"))
                self.assertTrue(spec["checkpoint_filename"].endswith(".pth"))
                self.assertGreater(len(spec["hash_prefix"]), 6)
                self.assertEqual(len(spec["categories"]), 1000)
                self.assertEqual(spec["preprocess"]["input_shape"], [1, 3, 224, 224])
                self.assertEqual(spec["preprocess"]["layout"], "NCHW")
                self.assertEqual(spec["preprocess"]["color_space"], "RGB")

    def test_unknown_model_raises(self):
        for bad_id in ("unknown_net", "resnet50", "", None):
            with self.assertRaises(ClassificationError):
                get_model_spec(bad_id)  # type: ignore

    def test_environment_metadata_does_not_import_heavy_backends(self):
        # 独立干净子进程测试：确认调用 environment_metadata 不污染加载 torch / ORT 等后端
        script = (
            "import sys\n"
            "from src.vision.classification import environment_metadata\n"
            "meta = environment_metadata()\n"
            "assert meta['torch'] is not None, 'torch version should be detected'\n"
            "assert 'torch' not in sys.modules, f'torch polluted: {sys.modules.get(\"torch\")}'\n"
            "assert 'onnxruntime' not in sys.modules, f'ort polluted: {sys.modules.get(\"onnxruntime\")}'\n"
            "assert 'onnx' not in sys.modules, 'onnx polluted'\n"
            "print('METADATA_CLEAN_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        self.assertIn("METADATA_CLEAN_OK", result.stdout)


class PreprocessingTests(unittest.TestCase):
    def test_all_models_produce_nchw_fp32_cpu_tensors(self):
        image = Image.new("RGB", (300, 200), (100, 150, 200))
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                tensor = prepare_image(image, model_id=model_id)
                self.assertEqual(list(tensor.shape), [1, 3, 224, 224])
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device.type, "cpu")

    def test_prepare_image_rejects_non_pil(self):
        for bad_input in (np.zeros((224, 224, 3)), "path.jpg", 123):
            with self.assertRaises(ClassificationError):
                prepare_image(bad_input)

    def test_open_rgb_image_correctness_and_file_release(self):
        with tempfile.TemporaryDirectory() as directory:
            gray_path = Path(directory) / "test_gray.png"
            Image.new("L", (32, 32), 128).save(gray_path)
            img = open_rgb_image(gray_path)
            gray_path.unlink()  # 必须能成功删除，证明文件句柄已被立即释放
            self.assertEqual(img.mode, "RGB")
            self.assertEqual(img.getpixel((0, 0)), (128, 128, 128))

    def test_open_rgb_image_missing_and_corrupt_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jpg"
            with self.assertRaises(ClassificationError):
                open_rgb_image(missing)

            corrupt = Path(directory) / "bad.jpg"
            corrupt.write_bytes(b"not an image data")
            with self.assertRaises(ClassificationError):
                open_rgb_image(corrupt)


class WeightLoadingSafetyTests(unittest.TestCase):
    def test_missing_weights_refuse_silent_download(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nonexistent.pth"
            for model_id in MODEL_IDS:
                with self.subTest(model_id=model_id):
                    with self.assertRaises(ClassificationError):
                        load_classifier(model_id, missing, download_weights=False)
                    self.assertFalse(missing.exists())

    def test_configure_cpu_threads_validation_and_idempotency(self):
        for bad in (0, -1, "1"):
            with self.assertRaises(ClassificationError):
                configure_cpu_threads(bad)  # type: ignore

        # 正常设置线程
        configure_cpu_threads(threads=1, inter_op_threads=1)
        # 重复调用保持幂等
        configure_cpu_threads(threads=2, inter_op_threads=1)

        # 尝试在运行后修改 inter_op 线程数，必须显式抛出异常
        with self.assertRaises(ClassificationError):
            configure_cpu_threads(threads=1, inter_op_threads=4)

    def test_safe_load_enforces_weights_only(self):
        with tempfile.TemporaryDirectory() as directory:
            dummy_pth = Path(directory) / "weights.pth"
            dummy_pth.write_bytes(b"dummy")
            with patch("torch.load", side_effect=TypeError("weights_only error")) as mocked_load:
                with self.assertRaises(ClassificationError):
                    load_classifier("resnet18", dummy_pth)
                self.assertEqual(mocked_load.call_count, 1)
                self.assertTrue(mocked_load.call_args.kwargs.get("weights_only"))

    def test_strict_state_dict_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            dummy_pth = Path(directory) / "corrupt_state.pth"
            torch.save({"unrelated_key": torch.zeros(10)}, dummy_pth)
            with self.assertRaises(ClassificationError):
                load_classifier("resnet18", dummy_pth)


class FusionAndVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_fuse_classifier_all_three_models(self):
        # 针对 3 个模型，离线验证 Conv-BN 算子融合与无残留 BN
        test_specs = [
            ("resnet18", models.resnet18, 20),
            ("mobilenet_v3_small", models.mobilenet_v3_small, 34),
            ("shufflenet_v2_x1_0", models.shufflenet_v2_x1_0, 56),
        ]
        for model_id, builder, expected_pairs in test_specs:
            with self.subTest(model_id=model_id):
                baseline = builder(weights=None).eval()
                initial_bns = sum(isinstance(m, torch.nn.BatchNorm2d) for m in baseline.modules())
                self.assertEqual(initial_bns, expected_pairs)

                fused, meta = fuse_classifier(baseline, model_id)
                self.assertEqual(meta["fused_pair_count"], expected_pairs)
                self.assertEqual(meta["variant_id"], "F1")
                self.assertEqual(len(meta["unfused_bn_paths"]), 0)

                remaining_bns = sum(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
                self.assertEqual(remaining_bns, 0)

                # 随机权重下的前向 logits 数值等价性验证（仅单测离线校验）
                dummy_input = torch.randn(1, 3, 224, 224)
                with torch.inference_mode():
                    out_base = baseline(dummy_input)
                    out_fused = fused(dummy_input)
                comparison = compare_logits(out_base, out_fused, rtol=1e-4, atol=1e-4)
                self.assertTrue(comparison["passed"], f"{model_id} 融合前后数值不一致: {comparison}")

    def test_fusion_does_not_mutate_baseline(self):
        baseline = models.resnet18(weights=None).eval()
        orig_ptr = baseline.conv1.weight.data_ptr()
        self.assertIsInstance(baseline.bn1, torch.nn.BatchNorm2d)

        fused, _ = fuse_classifier(baseline, "resnet18")
        # baseline 的权重指针和 BatchNorm 属性绝对未被修改
        self.assertEqual(baseline.conv1.weight.data_ptr(), orig_ptr)
        self.assertIsInstance(baseline.bn1, torch.nn.BatchNorm2d)
        self.assertNotEqual(fused.conv1.weight.data_ptr(), orig_ptr)

    def test_build_torch_variant_all_variants(self):
        baseline = models.mobilenet_v3_small(weights=None).eval()

        for variant_id in TORCH_VARIANT_IDS:
            with self.subTest(variant_id=variant_id):
                variant_model, meta = build_torch_variant(baseline, "mobilenet_v3_small", variant_id)
                self.assertEqual(meta["variant_id"], variant_id)
                self.assertIn(meta["memory_format"], ("contiguous", "channels_last"))
                if variant_id in ("F1", "FL1"):
                    self.assertTrue(meta["fused"])
                    self.assertEqual(meta["fused_pair_count"], 34)
                else:
                    self.assertFalse(meta["fused"])
                    self.assertEqual(meta["fused_pair_count"], 0)

                if variant_id in ("L1", "FL1"):
                    self.assertEqual(meta["memory_format"], "channels_last")
                else:
                    self.assertEqual(meta["memory_format"], "contiguous")

        # 确保 baseline 对象的内存格式未被污染为 channels_last
        first_param = next(baseline.parameters())
        self.assertTrue(first_param.is_contiguous())

    def test_fusion_training_mode_rejected(self):
        train_model = models.resnet18(weights=None).train()
        with self.assertRaises(ClassificationError):
            fuse_classifier(train_model, "resnet18")


class OrtSessionAndGraphStatsTests(unittest.TestCase):
    def test_ort_session_creation_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            onnx_path = Path(directory) / "tiny.onnx"
            _create_dummy_onnx(onnx_path)

            # 图统计
            stats = graph_statistics(onnx_path)
            self.assertGreater(stats["node_count"], 0)
            self.assertGreater(stats["size_bytes"], 0)
            self.assertEqual(len(stats["sha256"]), 64)
            self.assertIn("images", stats["inputs"])
            self.assertIn("logits", stats["outputs"])
            self.assertEqual(stats["opset_imports"].get("ai.onnx"), 17)

            # ORT 会话创建 (O0 与 O1)
            for v_id in ORT_VARIANT_IDS:
                with self.subTest(variant_id=v_id):
                    sess = create_ort_session(onnx_path, threads=1, variant_id=v_id)
                    meta = getattr(sess, "_benchmark_metadata")
                    self.assertEqual(meta["variant_id"], v_id)
                    self.assertEqual(meta["threads"], 1)
                    self.assertEqual(meta["inter_op_threads"], 1)
                    self.assertEqual(meta["execution_mode"], "ORT_SEQUENTIAL")
                    self.assertEqual(meta["allow_spinning"], "0")
                    self.assertEqual(meta["providers"], ["CPUExecutionProvider"])


class PredictionAndComparisonTests(unittest.TestCase):
    def test_topk_predictions_torch_and_numpy_parity(self):
        cats = [f"cat_{i}" for i in range(10)]
        logits_np = np.array([[0.1, 2.5, 0.3, 1.2, 0.05, 3.1, -0.4, 0.0, 0.2, 0.8]], dtype=np.float32)
        logits_torch = torch.from_numpy(logits_np)

        res_np = topk_predictions(logits_np, cats, k=5)
        res_torch = topk_predictions(logits_torch, cats, k=5)

        self.assertEqual(len(res_np), 5)
        self.assertEqual(len(res_torch), 5)
        for item_np, item_torch in zip(res_np, res_torch):
            self.assertEqual(item_np["index"], item_torch["index"])
            self.assertEqual(item_np["label"], item_torch["label"])
            self.assertAlmostEqual(item_np["probability"], item_torch["probability"], places=5)

        # 概率降序
        probs = [item["probability"] for item in res_np]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertLessEqual(sum(probs), 1.0)

        # 当 k 等于总类别数时，概率和严格为 1
        res_all = topk_predictions(logits_np, cats, k=len(cats))
        probs_all = [item["probability"] for item in res_all]
        self.assertAlmostEqual(sum(probs_all), 1.0, places=5)

    def test_topk_predictions_rejects_malformed_inputs(self):
        cats = ["c0", "c1", "c2"]
        # NaN / Inf / 维度错误
        for bad in (
            np.array([[np.nan, 1.0, 2.0]], dtype=np.float32),
            np.array([[np.inf, 1.0, 2.0]], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([[1.0, 2.0]], dtype=np.float32),
            "invalid",
        ):
            with self.assertRaises(ClassificationError):
                topk_predictions(bad, cats, k=2)

        # 非法 k
        for bad_k in (0, -1, 4):
            with self.assertRaises(ClassificationError):
                topk_predictions(np.zeros((1, 3), dtype=np.float32), cats, k=bad_k)

    def test_compare_logits_behavior(self):
        ref = np.array([[5.0, 2.0, 1.0]], dtype=np.float32)
        cand_same = np.array([[5.0, 2.0, 1.0]], dtype=np.float32)
        res_same = compare_logits(ref, cand_same)
        self.assertTrue(res_same["passed"])
        self.assertEqual(res_same["max_abs_error"], 0.0)
        self.assertTrue(res_same["top1_equal"])
        self.assertTrue(res_same["top5_equal"])
        self.assertFalse(res_same["rank_difference"])

        cand_diff = np.array([[1.0, 2.0, 5.0]], dtype=np.float32)
        res_diff = compare_logits(ref, cand_diff)
        self.assertFalse(res_diff["passed"])
        self.assertFalse(res_diff["top1_equal"])
        self.assertTrue(res_diff["rank_difference"])


class CliAndExportTests(unittest.TestCase):
    def test_help_works_without_site_packages(self):
        tool = ROOT / "tools" / "export_classifier_onnx.py"
        result = subprocess.run(
            [sys.executable, "-S", str(tool), "--help"],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"--model", result.stdout)
        self.assertIn(b"--image", result.stdout)
        self.assertIn(b"--output", result.stdout)

    def test_export_refuses_existing_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test.onnx"
            target.write_bytes(b"existing content")
            with self.assertRaises(ValueError) as ctx:
                export_and_verify("resnet18", Path(directory) / "none.jpg", target, overwrite=False)
            self.assertIn("Output exists", str(ctx.exception))
            self.assertEqual(target.read_bytes(), b"existing content")

    def test_export_requires_onnx_suffix_and_real_image(self):
        with tempfile.TemporaryDirectory() as directory:
            # 错误后缀
            with self.assertRaises(ValueError) as ctx:
                export_and_verify("resnet18", Path(directory) / "none.jpg", Path(directory) / "test.bin")
            self.assertIn("--output must end in .onnx", str(ctx.exception))

            # 缺失图片
            with self.assertRaises(ValueError) as ctx:
                export_and_verify("resnet18", Path(directory) / "missing.jpg", Path(directory) / "test.onnx")
            self.assertIn("Image file not found", str(ctx.exception))

    def test_full_export_and_verify_pipeline_with_local_weights(self):
        # 使用现有的 ResNet18 权重与 demo 测试图片执行完整的真实导出与验证
        local_weights = ROOT / "models" / "weights" / "resnet18-f37072fd.pth"
        demo_image = ROOT / "data" / "resnet-demo" / "dog.jpg"

        if not local_weights.is_file() or not demo_image.is_file():
            self.skipTest("本地权重或 demo 测试图片不存在，跳过完整导出测试")

        with tempfile.TemporaryDirectory() as directory:
            out_onnx = Path(directory) / "resnet18_test.onnx"
            manifest = export_and_verify(
                model_id="resnet18",
                image_paths=[demo_image],
                output_path=out_onnx,
                weights_path=local_weights,
                threads=1,
                overwrite=False,
            )
            self.assertTrue(out_onnx.is_file())
            out_json = out_onnx.with_suffix(".json")
            self.assertTrue(out_json.is_file())

            self.assertEqual(manifest["model_id"], "resnet18")
            self.assertEqual(manifest["onnx"]["opset"], 17)
            self.assertEqual(manifest["onnx"]["input_shape"], [1, 3, 224, 224])
            self.assertEqual(manifest["onnx"]["output_shape"], [1, 1000])
            self.assertEqual(manifest["validation"]["verified_images_count"], 1)
            self.assertTrue(manifest["validation"]["results"][0]["comparison"]["passed"])

            # 验证默认不可重复覆盖
            with self.assertRaises(ValueError):
                export_and_verify(
                    model_id="resnet18",
                    image_paths=[demo_image],
                    output_path=out_onnx,
                    weights_path=local_weights,
                    overwrite=False,
                )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
