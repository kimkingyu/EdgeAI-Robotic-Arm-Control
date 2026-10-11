#!/usr/bin/env python3
"""离线单测：所有图片/随机小权重/tiny ONNX 仅是临时合成夹具。

没有真实权重、公开数据、真实 holdout、硬件或性能基准。
缩小门槛只使用本文件内的显式 mock；公共 API/CLI 门槛不变。
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import classification_quantization as quant
from src.vision import classification_benchmark as benchmark

HAS_DEPS = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "PIL", "onnx", "onnxruntime", "torch", "torchvision"))
TEST_COUNTS = {"calibration": 4, "development": 4, "holdout": 5}


def dump(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def synthetic_gates():
    """明确的测试 mock；正式 CLI 不提供降低数据门槛的开关。"""
    ready = {"formal": {"ready": True, "reason": "UNITTEST mocked count gates only",
                        "development_count": 4, "performance_count": 4},
             "quantization": {"ready": True, "reason": "UNITTEST mocked count gates only",
                              "calibration_count": 4, "development_labeled_count": 4,
                              "holdout_labeled_count": 5}}
    with patch.object(quant, "_MINIMUM_COUNTS", TEST_COUNTS.copy()), patch.object(
            benchmark, "dataset_readiness", return_value=ready):
        yield


def tiny_onnx(path: Path, *, input_shape=(1, 3, 224, 224), opset=17,
              nonfinite=False, has_target=True, input_dtype=None) -> None:
    """只有 4000 个合成随机参数；不是三个模型的真实权重。"""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    input_dtype = input_dtype or TensorProto.FLOAT
    initializers = []
    if has_target:
        generator = np.random.default_rng(20261001)
        weight = generator.normal(0, 0.001, size=(3, 1000)).astype(np.float32)
        bias = np.linspace(-0.01, 0.01, 1000, dtype=np.float32)
        bias[999] = np.nan if nonfinite else 1.0
        initializers = [numpy_helper.from_array(weight, "synthetic_weight"),
                        numpy_helper.from_array(bias, "synthetic_bias")]
        nodes = [helper.make_node("GlobalAveragePool", ["images"], ["pool"], name="tiny_pool"),
                 helper.make_node("Flatten", ["pool"], ["flat"], name="tiny_flatten", axis=1),
                 helper.make_node("Gemm", ["flat", "synthetic_weight", "synthetic_bias"],
                                  ["logits"], name="tiny_gemm")]
    else:
        initializers = [numpy_helper.from_array(np.array([1], dtype=np.int64), "axis"),
                        numpy_helper.from_array(np.array([1, 1000], dtype=np.int64), "out_shape")]
        nodes = [helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
                 helper.make_node("Unsqueeze", ["mean", "axis"], ["column"]),
                 helper.make_node("Expand", ["column", "out_shape"], ["logits"])]
    graph = helper.make_graph(nodes, "SYNTHETIC UNITTEST ONLY",
                              [helper.make_tensor_value_info("images", input_dtype, list(input_shape))],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1000])],
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                              producer_name="unit-test-not-real-model")
    model.ir_version = 10
    onnx.save(model, str(path))


def synthetic_images(directory: Path) -> tuple[Path, dict]:
    import numpy as np
    from PIL import Image
    images = []
    for split, count in TEST_COUNTS.items():
        for index in range(count):
            number = len(images)
            pixels = ((np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
                       + number * 17) % 256).astype(np.uint8)
            path = directory / f"synthetic-{number:03d}.png"
            Image.fromarray(pixels).save(path)
            images.append({"id": f"synthetic-{number:03d}", "path": path.name,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "width": 32, "height": 24, "split": split,
                           "label": None if split == "calibration" else 999,
                           "source_id": f"generated-test-source-{number}",
                           "source": "generated unittest pixels; NOT dataset/model evidence",
                           "license": "test-owned generated pixels"})
    manifest = {"schema_version": 1, "root": ".", "images": images,
                "label_mapping": {"description": "Synthetic 999 only; test fixture, not inferred labels"}}
    path = directory / "synthetic-manifest.json"
    dump(path, manifest)
    return path, manifest


def accuracy_fixture(top1_correct, top5_correct=None) -> dict:
    top5_correct = top5_correct if top5_correct is not None else top1_correct
    rows = []
    for index, (one, five) in enumerate(zip(top1_correct, top5_correct)):
        top5 = [999, 400, 401, 402, 403] if one else (
            [400, 999, 401, 402, 403] if five else [400, 401, 402, 403, 404])
        rows.append({"id": f"unit-test-{index:04d}", "label": 999, "top1": top5[0],
                     "top5": top5, "top1_correct": bool(one), "top5_correct": bool(five),
                     "sha256": hashlib.sha256(f"synthetic-only-{index}".encode()).hexdigest()})
    return {"schema_version": 1, "model_id": "resnet18", "manifest_sha256": "a" * 64,
            "artifact_sha256": "b" * 64, "split": "development", "image_count": len(rows),
            "per_image": rows, "top1_accuracy": sum(top1_correct) / len(rows),
            "top5_accuracy": sum(top5_correct) / len(rows)}


class StandardLibraryTests(unittest.TestCase):
    def test_import_and_help_without_site_packages_or_torch(self):
        code = ("import runpy,sys; sys.path.insert(0,sys.argv[1]); "
                "from src.vision import classification_quantization; "
                "assert not any(n in sys.modules for n in "
                "('torch','torchvision','numpy','onnx','onnxruntime','PIL')); "
                "sys.argv=[sys.argv[2],'--help']; runpy.run_path(sys.argv[0],run_name='__main__')")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", code,
                                 str(ROOT), str(ROOT / "tools/quantize_classifier_onnx.py")],
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--expected-frozen-sha256", result.stdout)

    def test_public_variant_ids_and_stage_a_matrix_unchanged(self):
        self.assertEqual(quant.QUANT_VARIANTS, ("QB", "Q1", "Q2", "Q3"))
        self.assertEqual(quant._MINIMUM_COUNTS, {"calibration": 256, "development": 128, "holdout": 1000})
        self.assertEqual(len(benchmark.expand_experiments(benchmark.load_config())), 30)
        self.assertEqual(benchmark.VARIANT_IDS, ("B0", "F1", "L1", "FL1", "O0", "O1"))

    def test_holdout_rejected_before_loading_data_or_heavy_dependencies(self):
        with patch.object(quant, "_load_dataset") as load:
            with self.assertRaisesRegex(quant.QuantizationError, "holdout guard"):
                quant.evaluate_classifier("resnet18", "unused", "unused", split="holdout")
            load.assert_not_called()

    def test_model_split_seed_and_threads_rejected(self):
        for arguments in (("unknown", "unused", "unused"),):
            with self.assertRaises(quant.QuantizationError):
                quant.evaluate_classifier(*arguments)
        for bad in (0, True, -1, "1"):
            with self.assertRaises(quant.QuantizationError):
                quant.evaluate_classifier("resnet18", "unused", "unused", threads=bad)
        for split in ("calibration", "smoke", "test"):
            with self.assertRaises(quant.QuantizationError):
                quant.evaluate_classifier("resnet18", "unused", "unused", split=split)
        with self.assertRaises(quant.QuantizationError):
            quant.prepare_quantized_family("resnet18", "unused", "unused", "unused", seed=True)

    def test_qdq_conversion_is_not_integer_compute(self):
        events = [{"cat": "Node", "args": {"op_name": op, "provider": "CPUExecutionProvider"}}
                  for op in ("QuantizeLinear", "DequantizeLinear", "Conv", "MysteryInt8LikeName")]
        # 一个无 provider 的 QGemm fence 不是整数 kernel 证据。
        events.append({"cat": "Node", "args": {"op_name": "QGemm"}})
        result = quant._profile_events(events)
        self.assertEqual(result["int8_operator_counts"], {})
        self.assertEqual(result["int8_operator_events"], [])
        self.assertEqual(result["float_operator_counts"], {"Conv": 1})
        self.assertIn("未观察", result["precision_note"])

    def test_known_integer_event_requires_cpu_provider(self):
        events = [{"cat": "Node", "name": "tiny_integer_kernel", "dur": 4,
                   "args": {"op_name": "QLinearConv", "provider": "CPUExecutionProvider"}},
                  {"cat": "Node", "args": {"op_name": "QGemm", "provider": "AzureExecutionProvider"}}]
        result = quant._profile_events(events)
        self.assertEqual(result["int8_operator_counts"], {"QLinearConv": 1})
        self.assertEqual(len(result["int8_operator_events"]), 1)
        self.assertIn("不代表全 INT8", result["precision_note"])


@unittest.skipUnless(HAS_DEPS, "缺本地 ONNX/ORT/PIL/torch 测试依赖；不自动安装")
class TinyOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="classifier-quant-UNITTEST-")
        cls.directory = Path(cls.temporary.name)
        cls.raw = cls.directory / "synthetic-raw.onnx"
        tiny_onnx(cls.raw)
        cls.manifest_path, cls.manifest = synthetic_images(cls.directory)
        cls.output = cls.directory / "synthetic-family"
        with synthetic_gates():
            cls.family = quant.prepare_quantized_family("resnet18", cls.raw, cls.manifest_path, cls.output)
        if cls.family["status"] != "prepared":
            raise AssertionError(json.dumps(cls.family["variants"], ensure_ascii=False, indent=2))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def new_path(self, filename):
        return self.directory / f"{self._testMethodName}-{filename}"

    def modified_manifest(self, edit):
        value = copy.deepcopy(self.manifest)
        edit(value)
        path = self.new_path("manifest.json")
        dump(path, value)
        return path

    def test_family_same_qb_parent_and_actual_s8_coverage(self):
        family = self.family
        self.assertEqual(family["schema_version"], 1)
        self.assertEqual(family["input_model"]["sha256"], quant._hash_file(self.raw))
        self.assertEqual(family["manifest_sha256"], quant._hash_file(self.manifest_path))
        self.assertTrue((self.output / "family.json").is_file())
        qb_sha = family["variants"]["QB"]["onnx_sha256"]
        for variant, result in family["variants"].items():
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["onnx_sha256"], quant._hash_file(result["path"]))
            self.assertEqual(result["size_bytes"], Path(result["path"]).stat().st_size)
            self.assertTrue(result["validation"]["passed"])
            if variant != "QB":
                self.assertEqual(result["config"]["parent_onnx_sha256"], qb_sha)
                coverage = result["graph_statistics"]["quantization_coverage"]
                self.assertEqual(coverage["qdq_target_node_count"], 1)
                self.assertGreater(coverage["qdq_node_counts"]["DequantizeLinear"], 0)
                self.assertIn("不是全 INT8", coverage["precision_note"])
        self.assertEqual(self.family["variants"]["QB"]["config"]["quant_pre_process"]["skip_optimization"], True)
        for row in self.family["variants"]["QB"]["validation"]["results"]:
            self.assertTrue(row["comparison"]["passed"])
            self.assertEqual(row["comparison"]["atol"], 1e-4)

    def test_unicode_source_and_output_use_ascii_native_io(self):
        import shutil
        directory = self.new_path("中文源图与输出")
        directory.mkdir()
        source = directory / "原始模型.onnx"
        shutil.copyfile(self.raw, source)
        output = directory / "量化产物"
        api = quant._quantization_api()
        native_calls, replacements = [], []
        preprocess = api["shape"].quant_pre_process
        quantize = api["quant"].quantize_static
        replace = quant.os.replace

        def checked_preprocess(*args, **kwargs):
            native_calls.extend([str(kwargs["input_model"]), str(kwargs["output_model_path"])])
            self.assertTrue(all(path.isascii() for path in native_calls))
            return preprocess(*args, **kwargs)

        def checked_quantize(*args, **kwargs):
            native_calls.append(str(kwargs["model_input"]))
            self.assertTrue(native_calls[-1].isascii())
            return quantize(*args, **kwargs)

        def checked_replace(source_path, destination):
            if Path(destination).parent == output:
                replacements.append((Path(source_path), Path(destination)))
                self.assertEqual(Path(source_path).parent.parent, output.parent)
            return replace(source_path, destination)

        with synthetic_gates(), patch.object(quant, "_quantization_api", return_value=api), patch.object(
                api["shape"], "quant_pre_process", side_effect=checked_preprocess), patch.object(
                api["quant"], "quantize_static", side_effect=checked_quantize), patch.object(
                quant.os, "replace", side_effect=checked_replace):
            family = quant.prepare_quantized_family("resnet18", source, self.manifest_path, output)
        self.assertEqual(family["status"], "prepared", {key: value.get("reason") for key, value in family["variants"].items()})
        self.assertEqual(family["input_model"]["sha256"], quant._hash_file(source))
        self.assertEqual(replacements[-1][1].name, "family.json")
        self.assertEqual(len(native_calls), 5)
        for variant in quant.QUANT_VARIANTS:
            record = family["variants"][variant]
            self.assertEqual(record["status"], "prepared")
            self.assertEqual(record["onnx_sha256"], quant._hash_file(output / f"{variant}.onnx"))
        self.assertIn("atomic os.replace", family["io_policy"]["publication"])

    def test_corrupted_native_copy_is_rejected(self):
        output = self.new_path("bad-native-copy")

        def corrupt(source, destination):
            Path(destination).write_bytes(b"UNITTEST intentionally corrupted copy")

        with synthetic_gates(), patch.object(quant.shutil, "copyfile", side_effect=corrupt):
            with self.assertRaisesRegex(quant.QuantizationError, "临时副本 SHA256"):
                quant.prepare_quantized_family("resnet18", self.raw, self.manifest_path, output)
        self.assertFalse((output / "family.json").exists())
        self.assertEqual(quant._hash_file(self.raw), self.family["input_model"]["sha256"])

    def test_corrupted_publication_copy_is_rejected_before_any_replace(self):
        output = self.new_path("bad-publish-copy")
        copyfile = quant.shutil.copyfile
        replace = quant.os.replace

        def checked_replace(source, destination):
            if Path(destination).parent == output:
                raise AssertionError("must not publish a corrupt copy")
            return replace(source, destination)

        def corrupt_publication(source, destination):
            copyfile(source, destination)
            if Path(destination).parent.name.startswith(".resnet18-publish-"):
                Path(destination).write_bytes(b"UNITTEST intentionally corrupted publication")

        with synthetic_gates(), patch.object(quant.shutil, "copyfile", side_effect=corrupt_publication), patch.object(
                quant.os, "replace", side_effect=checked_replace):
            with self.assertRaisesRegex(quant.QuantizationError, "发布副本 SHA256"):
                quant.prepare_quantized_family("resnet18", self.raw, self.manifest_path, output)
        self.assertFalse((output / "family.json").exists())
        self.assertFalse((output / "QB.onnx").exists())

    def test_only_approved_factors_change(self):
        first = copy.deepcopy(self.family["variants"]["Q1"]["config"])
        second = copy.deepcopy(self.family["variants"]["Q2"]["config"])
        self.assertFalse(first.pop("per_channel"))
        self.assertTrue(second.pop("per_channel"))
        self.assertEqual(first, second)
        third = copy.deepcopy(self.family["variants"]["Q3"]["config"])
        second = copy.deepcopy(self.family["variants"]["Q2"]["config"])
        self.assertEqual(second.pop("calibrate_method"), "MinMax")
        self.assertEqual(third.pop("calibrate_method"), "Percentile")
        self.assertEqual(third["extra_options"].pop("CalibPercentile"), 99.99)
        self.assertEqual(third["calibration_collection"].pop("percentile"), 99.99)
        self.assertEqual(third["calibration_collection"].pop("histogram_num_bins"), 2048)
        second.pop("calibration_cache")
        third.pop("calibration_cache")
        self.assertEqual(second, third)
        self.assertEqual(first["activation_type"], "QInt8")
        self.assertEqual(first["weight_type"], "QInt8")
        self.assertEqual(first["extra_options"], {"ActivationSymmetric": False,
                                               "WeightSymmetric": True, "CalibTensorRangeSymmetric": False})
        self.assertEqual(first["op_types_to_quantize"], ["Conv", "Gemm", "MatMul"])
        self.assertFalse(first["reduce_range"])

    def test_calibration_ids_exclude_development_holdout_and_cache_binding(self):
        calibration = self.family["calibration"]
        expected = {row["id"] for row in self.manifest["images"] if row["split"] == "calibration"}
        self.assertEqual(set(calibration["image_ids"]), expected)
        self.assertFalse(calibration["source_policy"]["holdout_pixels_read"])
        self.assertEqual(calibration["count"], 4)
        for variant in ("Q1", "Q2", "Q3"):
            cache = self.family["variants"][variant]["config"]["calibration_cache"]
            self.assertEqual(cache["qb_sha256"], self.family["variants"]["QB"]["onnx_sha256"])
            self.assertEqual(cache["calibration_image_ids"], calibration["image_ids"])
            self.assertEqual(cache["calibration_image_list_sha256"], calibration["sha256"])
            self.assertEqual(cache["collect_count"], 4)
            self.assertEqual(cache["streaming_batch_size"], 1)
            self.assertEqual(cache["cache_sha256"], quant._hash_file(self.output / cache["cache_filename"]))
        first = self.family["variants"]["Q1"]["config"]["calibration_cache"]
        second = self.family["variants"]["Q2"]["config"]["calibration_cache"]
        self.assertEqual(first["cache_sha256"], second["cache_sha256"])

    def test_reader_rewind_reset_reproducible_and_lazy(self):
        import numpy as np
        from onnxruntime.quantization import CalibrationDataReader as OrtReader
        manifest = quant._load_dataset(self.manifest_path)
        images = quant._ordered_images(manifest, "calibration", 20261001)
        self.assertEqual([row["id"] for row in images], self.family["calibration"]["image_ids"])
        with patch.object(quant, "_prepare_array", wraps=quant._prepare_array) as prepare:
            reader = quant.CalibrationDataReader("resnet18", images, "images")
            self.assertIsInstance(reader, OrtReader)
            self.assertEqual(prepare.call_count, 0)
            first = reader.get_next()["images"].copy()
            self.assertEqual(prepare.call_count, 1)
            remaining = list(reader)
            self.assertEqual(len(remaining), 3)
            self.assertIsNone(reader.get_next())
            reader.rewind()
            np.testing.assert_array_equal(reader.get_next()["images"], first)
            reader.reset()
            np.testing.assert_array_equal(reader.get_next()["images"], first)
            self.assertTrue(all(call.args[0]["split"] == "calibration" for call in prepare.call_args_list))
        with self.assertRaises(quant.QuantizationError):
            quant.CalibrationDataReader("resnet18", quant._ordered_images(manifest, "holdout"), "images")
        reader.set_range(1, 3)
        self.assertEqual(len(reader), 2)
        self.assertEqual(len(list(reader)), 2)
        reader.rewind()
        self.assertEqual(len(list(reader)), 2)

    def test_cpu_sequential_one_thread_spinning_zero(self):
        import onnxruntime as ort
        session = quant._create_session(self.raw)
        options = session.get_session_options()
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(options.graph_optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
            self.assertEqual(options.get_session_config_entry(key), "0")

    def test_default_refuses_overwrite_and_original_unchanged(self):
        original_sha = quant._hash_file(self.raw)
        with synthetic_gates(), self.assertRaisesRegex(quant.QuantizationError, "覆盖"):
            quant.prepare_quantized_family("resnet18", self.raw, self.manifest_path, self.output)
        self.assertEqual(quant._hash_file(self.raw), original_sha)
        with self.assertRaisesRegex(quant.QuantizationError, "旧 ONNX"):
            quant.prepare_quantized_family("resnet18", self.output / "QB.onnx", self.manifest_path,
                                            self.output, overwrite=True)

    def test_real_gates_do_not_accept_tiny_fixture(self):
        with self.assertRaisesRegex(quant.QuantizationError, "blocked_data"):
            quant.prepare_quantized_family("resnet18", self.raw, self.manifest_path,
                                            self.new_path("blocked"))
        for split in ("development", "holdout"):
            kwargs = {"split": split}
            if split == "holdout":
                kwargs.update({"frozen_path": "unused", "expected_frozen_sha256": "a" * 64})
            with self.assertRaisesRegex(quant.QuantizationError, "blocked_data"):
                quant.evaluate_classifier("resnet18", self.raw, self.manifest_path, **kwargs)
        self.assertFalse(self.new_path("blocked").exists())

    def test_gate_all_three_splits_no_hidden_user_readiness_bypass(self):
        with synthetic_gates():
            manifest = quant._load_dataset(self.manifest_path)
            for split in TEST_COUNTS:
                missing = copy.deepcopy(manifest)
                missing["images"] = [entry for entry in missing["images"] if entry["split"] != split]
                with self.assertRaisesRegex(quant.QuantizationError, split):
                    quant._require_stage_b_data(missing)

    def test_manifest_provenance_bad_labels_leakage_and_hash_rejected(self):
        edits = [lambda value: value["images"][0].update({"license": ""}),
                 lambda value: value["images"][4].update({"label": 1000}),
                 lambda value: value["images"][4].update({"label": True}),
                 lambda value: value["images"][4].update({"sha256": value["images"][0]["sha256"]}),
                 lambda value: value["images"][4].update({"source_id": value["images"][0]["source_id"]}),
                 lambda value: value["images"][0].update({"width": 0})]
        for edit in edits:
            with self.subTest(edit=edit), self.assertRaises(quant.QuantizationError):
                quant._load_dataset(self.modified_manifest(edit))
        path = self.modified_manifest(lambda value: value["images"][4].update({"sha256": "f" * 64}))
        with synthetic_gates(), self.assertRaisesRegex(quant.QuantizationError, "SHA256"):
            quant.evaluate_classifier("resnet18", self.raw, path)

    def test_shape_opset_dtype_and_nonfinite_refused(self):
        import numpy as np
        import onnx
        for name, kwargs in (("shape", {"input_shape": (1, 3, 112, 224)}),
                             ("opset", {"opset": 16}),
                             ("dtype", {"input_dtype": onnx.TensorProto.DOUBLE})):
            path = self.new_path(f"{name}.onnx")
            tiny_onnx(path, **kwargs)
            with self.subTest(name=name), self.assertRaises(quant.QuantizationError):
                quant._check_graph(path)
        path = self.new_path("nan.onnx")
        tiny_onnx(path, nonfinite=True)
        session = quant._create_session(path)
        with self.assertRaisesRegex(quant.QuantizationError, "有限"):
            quant._run(session, np.zeros((1, 3, 224, 224), dtype=np.float32))
        for array in (np.zeros((1, 3, 224, 224), dtype=np.float64),
                      np.full((1, 3, 224, 224), np.inf, dtype=np.float32),
                      np.zeros((1, 3, 2, 2), dtype=np.float32)):
            with self.assertRaises(quant.QuantizationError):
                quant.profile_quantized_execution(self.raw, array, self.new_path("bad-profile"))

    def test_development_true_labels_full_1000_class_ranking(self):
        with synthetic_gates():
            result = quant.evaluate_classifier("resnet18", self.raw, self.manifest_path)
        self.assertEqual(result["image_count"], 4)
        self.assertEqual(result["top1_accuracy"], 1.0)
        self.assertEqual(result["top5_accuracy"], 1.0)
        self.assertEqual(result["manifest_sha256"], quant._hash_file(self.manifest_path))
        self.assertTrue(all(row["top1"] == 999 and 999 in row["top5"] for row in result["per_image"]))
        # 修改真值为 0 后不能拿预测 999 替换标签。
        path = self.modified_manifest(lambda value: [row.update({"label": 0}) for row in value["images"]
                                                    if row["split"] == "development"])
        with synthetic_gates():
            wrong_label = quant.evaluate_classifier("resnet18", self.raw, path)
        self.assertEqual(wrong_label["top1_accuracy"], 0.0)
        self.assertEqual(wrong_label["top5_accuracy"], 0.0)
        self.assertTrue(all(row["label"] == 0 for row in wrong_label["per_image"]))

    def test_all_calibration_and_development_hashes_checked_before_prepare(self):
        # 第4张开发图不一定在QB的固定数值检查前列；全部字节仍必须可信。
        path = self.modified_manifest(lambda value: value["images"][7].update({"sha256": "f" * 64}))
        destination = self.new_path("corrupt-data-family")
        with synthetic_gates(), self.assertRaisesRegex(quant.QuantizationError, "SHA256"):
            quant.prepare_quantized_family("resnet18", self.raw, path, destination)
        self.assertFalse(destination.exists())

    def test_zero_target_graph_is_explicitly_unsupported_not_float_fallback(self):
        path = self.new_path("no-target.onnx")
        tiny_onnx(path, has_target=False)
        destination = self.new_path("no-target-family")
        with synthetic_gates():
            family = quant.prepare_quantized_family("resnet18", path, self.manifest_path, destination)
        self.assertEqual(family["variants"]["QB"]["status"], "prepared")
        for variant in ("Q1", "Q2", "Q3"):
            self.assertEqual(family["variants"][variant]["status"], "unsupported")
            self.assertIn("目标算子", family["variants"][variant]["reason"])
            self.assertFalse((destination / f"{variant}.onnx").exists())

    def test_real_v1_transform_reused_for_all_three_model_ids(self):
        manifest = quant._load_dataset(self.manifest_path)
        entry = quant._ordered_images(manifest, "calibration")[0]
        with patch.object(quant._core(), "prepare_image", wraps=quant._core().prepare_image) as prepare:
            for model_id in quant.MODEL_IDS:
                result = quant._prepare_array(entry, model_id)
                self.assertEqual(result.shape, (1, 3, 224, 224))
                self.assertEqual(prepare.call_args.kwargs["model_id"], model_id)

    def test_actual_weight_scale_granularity_not_just_config_flags(self):
        import onnx
        scale_shapes = {}
        for variant in ("Q1", "Q2", "Q3"):
            model = onnx.load(str(self.output / f"{variant}.onnx"))
            scales = [tensor for tensor in model.graph.initializer
                      if tensor.name == "synthetic_weight_scale"]
            self.assertEqual(len(scales), 1)
            scale_shapes[variant] = list(scales[0].dims)
            self.assertFalse(self.family["variants"][variant]["validation"]["fp32_equivalence_required"])
        self.assertEqual(scale_shapes, {"Q1": [], "Q2": [1000], "Q3": [1000]})

    def test_input_manifest_cannot_be_overwritten_even_with_flag(self):
        directory = self.new_path("input-protection")
        directory.mkdir()
        manifest_path = directory / "family.json"
        dump(manifest_path, self.manifest)
        with self.assertRaisesRegex(quant.QuantizationError, "输入数据 manifest"):
            quant.prepare_quantized_family("resnet18", self.raw, manifest_path, directory, overwrite=True)

    def test_no_label_is_not_accuracy(self):
        path = self.modified_manifest(lambda value: value["images"][4].update({"label": None}))
        with synthetic_gates(), self.assertRaisesRegex(quant.QuantizationError, "真实"):
            quant.evaluate_classifier("resnet18", self.raw, path)

    def freeze_fixture(self):
        return {"schema_version": 1, "status": "frozen",
                "manifest_sha256": quant._hash_file(self.manifest_path),
                "candidates": [{"model_id": "resnet18", "artifact_sha256": quant._hash_file(self.raw),
                                "threads": 1, "baseline_artifact_sha256": quant._hash_file(self.output / "QB.onnx"),
                                "baseline_threads": 2}]}

    def test_frozen_sha_schema_status_manifest_model_artifact_threads_guard(self):
        for field, edit, threads in (
                ("sha", lambda value: None, 1),
                ("schema", lambda value: value.update({"schema_version": True}), 1),
                ("status", lambda value: value.update({"status": "proposed"}), 1),
                ("manifest", lambda value: value.update({"manifest_sha256": "f" * 64}), 1),
                ("model", lambda value: value["candidates"][0].update({"model_id": "mobilenet_v3_small"}), 1),
                ("artifact", lambda value: value["candidates"][0].update({"artifact_sha256": "f" * 64}), 1),
                ("threads", lambda value: None, 2)):
            value = self.freeze_fixture()
            edit(value)
            path = self.new_path(f"frozen-{field}.json")
            digest = dump(path, value)
            if field == "sha":
                digest = "e" * 64
            with synthetic_gates(), patch.object(quant, "_prepare_array") as prepare:
                with self.subTest(field=field), self.assertRaisesRegex(quant.QuantizationError, "holdout guard"):
                    quant.evaluate_classifier("resnet18", self.raw, self.manifest_path, split="holdout",
                                               threads=threads, frozen_path=path, expected_frozen_sha256=digest)
                prepare.assert_not_called()

    def test_frozen_candidate_and_baseline_threads_are_separate(self):
        path = self.new_path("frozen.json")
        digest = dump(path, self.freeze_fixture())
        with synthetic_gates():
            candidate = quant.evaluate_classifier("resnet18", self.raw, self.manifest_path,
                                                    split="holdout", frozen_path=path,
                                                    expected_frozen_sha256=digest)
            baseline = quant.evaluate_classifier("resnet18", self.output / "QB.onnx", self.manifest_path,
                                                   split="holdout", threads=2, frozen_path=path,
                                                   expected_frozen_sha256=digest)
            with self.assertRaisesRegex(quant.QuantizationError, "holdout guard"):
                quant.evaluate_classifier("resnet18", self.output / "QB.onnx", self.manifest_path,
                                           split="holdout", threads=1, frozen_path=path,
                                           expected_frozen_sha256=digest)
        self.assertEqual(candidate["frozen"]["roles"], ["candidate"])
        self.assertEqual(baseline["frozen"]["roles"], ["baseline"])
        self.assertEqual(candidate["image_count"], 5)
        self.assertEqual(baseline["session"]["threads"], 2)

    def test_prepare_does_not_open_holdout_pixels(self):
        directory = self.new_path("no-holdout")
        opened = []
        original = quant._core().open_rgb_image
        def only_nonholdout(path):
            opened.append(str(path))
            return original(path)
        api = quant._quantization_api()
        with synthetic_gates(), patch.object(quant, "_quantization_api", return_value=api), patch.object(
                quant._core(), "open_rgb_image", side_effect=only_nonholdout), patch.object(
                api["quant"], "quantize_static", side_effect=NotImplementedError("UNITTEST intentional unsupported S8S8")):
            family = quant.prepare_quantized_family("resnet18", self.raw, self.manifest_path, directory)
        holdout_paths = {str(self.directory / row["path"]) for row in self.manifest["images"] if row["split"] == "holdout"}
        self.assertTrue(opened)
        self.assertTrue(set(opened).isdisjoint(holdout_paths))
        self.assertEqual(family["status"], "completed_with_failures")
        for variant in ("Q1", "Q2", "Q3"):
            self.assertEqual(family["variants"][variant]["status"], "unsupported")
            self.assertIn("UNITTEST intentional", family["variants"][variant]["reason"])
            self.assertIsNone(family["variants"][variant]["path"])
            self.assertFalse((directory / f"{variant}.onnx").exists())
        self.assertTrue((directory / "family.json").is_file())

    def test_profile_raw_has_no_integer_and_q_trace_preserved(self):
        import numpy as np
        array = np.zeros((1, 3, 224, 224), dtype=np.float32)
        raw = quant.profile_quantized_execution(self.raw, array, self.new_path("raw-profile"))
        self.assertEqual(raw["int8_operator_counts"], {})
        self.assertIn("未观察", raw["precision_note"])
        self.assertEqual(raw["providers"], ["CPUExecutionProvider"])
        for path in (self.raw, self.output / "Q2.onnx"):
            result = quant.profile_quantized_execution(path, array, self.new_path("profile"))
            self.assertEqual(result["trace_sha256"], quant._hash_file(result["trace_path"]))
            self.assertEqual(result["inputs"], 5)
            self.assertEqual(result["warmup"], 1)
            events = json.loads(Path(result["trace_path"]).read_text(encoding="utf-8"))
            self.assertEqual(result["operator_counts"], quant._profile_events(events)["operator_counts"])
            self.assertEqual(result["int8_operator_counts"], quant._profile_events(events)["int8_operator_counts"])

    def test_api_options_checked_from_installed_source(self):
        api = quant._quantization_api()
        self.assertIn("CalibPercentile", api["evidence"]["verified_options"])
        self.assertEqual(api["evidence"]["percentile_option_mapping"], "CalibPercentile -> percentile")
        self.assertIn("calibration_cache_path", api["evidence"]["quantize_static_signature"])


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "配对 bootstrap 需要本地 NumPy")
class AccuracyComparisonTests(unittest.TestCase):
    def test_paired_point_drop_ci_units_and_determinism(self):
        reference = accuracy_fixture([True] * 20)
        candidate = accuracy_fixture([True] * 10 + [False] * 10, [True] * 15 + [False] * 5)
        candidate["artifact_sha256"] = "c" * 64
        result = quant.compare_accuracies(reference, candidate, seed=20261001, bootstrap_samples=300)
        self.assertEqual(result["top1_drop_pp"], 50.0)
        self.assertEqual(result["top5_drop_pp"], 25.0)
        self.assertTrue(result["paired"])
        self.assertEqual(result["resampling_unit"], "paired image")
        self.assertEqual(result["budget_status"], "failed")
        self.assertEqual(result, quant.compare_accuracies(reference, candidate, bootstrap_samples=300))
        self.assertLessEqual(result["top1_ci95_pp"][0], 50.0)
        self.assertGreaterEqual(result["top1_ci95_pp"][1], 50.0)

    def test_identical_passes_and_interval_crossing_budget_inconclusive(self):
        reference = accuracy_fixture([True] * 128)
        same = quant.compare_accuracies(reference, copy.deepcopy(reference), bootstrap_samples=400)
        self.assertEqual(same["top1_ci95_pp"], [0.0, 0.0])
        self.assertTrue(same["budget_passed"])
        candidate = accuracy_fixture([True] * 127 + [False])
        uncertain = quant.compare_accuracies(reference, candidate, bootstrap_samples=2000)
        self.assertEqual(uncertain["budget_status"], "inconclusive")
        self.assertIsNone(uncertain["budget_passed"])
        self.assertLessEqual(uncertain["top1_ci95_pp"][0], 1.0)
        self.assertGreater(uncertain["top1_ci95_pp"][1], 1.0)

    def test_unpaired_manifest_model_split_order_sha_and_ground_truth_rejected(self):
        reference = accuracy_fixture([True, False, True, False])
        changes = [lambda value: value.update({"manifest_sha256": "c" * 64}),
                   lambda value: value.update({"model_id": "mobilenet_v3_small"}),
                   lambda value: value.update({"split": "holdout"}),
                   lambda value: value["per_image"].reverse(),
                   lambda value: value["per_image"][0].update({"sha256": "d" * 64}),
                   lambda value: value["per_image"][1].update({"label": 998})]
        for edit in changes:
            candidate = copy.deepcopy(reference)
            edit(candidate)
            with self.subTest(edit=edit), self.assertRaises(quant.QuantizationError):
                quant.compare_accuracies(reference, candidate, bootstrap_samples=10)

    def test_missing_labels_fabricated_accuracy_or_correctness_are_not_passes(self):
        reference = accuracy_fixture([True, False, True, False])
        changes = [lambda value: value["per_image"][0].pop("label"),
                   lambda value: value["per_image"][0].update({"label": None}),
                   lambda value: value["per_image"][0].update({"top1_correct": False}),
                   lambda value: value.update({"top1_accuracy": 0.95}),
                   lambda value: value.update({"image_count": 500}),
                   lambda value: value["per_image"][0].update({"top5": [999] * 5})]
        for edit in changes:
            candidate = copy.deepcopy(reference)
            edit(candidate)
            with self.subTest(edit=edit), self.assertRaises(quant.QuantizationError):
                quant.compare_accuracies(reference, candidate, bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
