#!/usr/bin/env python3
"""M5 后端测试：内存契约、像素、Python ABI 与后端选择。

真实 MLIR 内核需要 aarch64 上的产物库。没有库时这些用例 **失败**，不 skip ——
把缺失依赖记成通过，等于把没验证的东西说成验证过了。用 --require-mlir 控制。
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from src.vision.preprocess import (
    BackendUnavailable, MlirBackend, OpenCVBackend, Preprocessor, UnsupportedInput,
    buffer_span, check_source, from_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compiler" / "tests"))
from preprocess_reference import reference_resize_rgb

MLIR_LIBRARY = os.environ.get("EDGEAI_MLIR_LIBRARY")
MLIR_MANIFEST = os.environ.get("EDGEAI_MLIR_MANIFEST")
REQUIRE_MLIR = os.environ.get("EDGEAI_REQUIRE_MLIR") == "1"


def bgr_fixture(height, width, seed=7):
    """合成图案只作正确性 fixture，不是性能或模型证据。"""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


class ContractTests(unittest.TestCase):
    """输入契约必须在进入生成代码前拒绝，不能写一半再报错。"""

    def test_accepts_contiguous_and_positive_stride_roi(self):
        image = bgr_fixture(12, 16)
        self.assertEqual(check_source(image)[:3], (12, 16, 48))
        roi = image[2:9, :10]
        height, width, stride, span = check_source(roi)
        self.assertEqual((height, width, stride), (7, 10, 48))
        self.assertGreaterEqual(span, (height - 1) * stride + width * 3)

    def test_rejects_unsupported_inputs(self):
        image = bgr_fixture(8, 8)
        cases = {
            "float": image.astype(np.float32),
            "gray": image[:, :, 0].copy(),
            "rgba": np.zeros((8, 8, 4), dtype=np.uint8),
            "empty": np.zeros((0, 8, 3), dtype=np.uint8),
            "negative_row_stride": image[::-1],
            "negative_column_stride": image[:, ::-1],
            "sparse_columns": image[:, ::2],
            "channel_sliced": image[:, :, ::-1],
            "not_an_array": [[0, 0, 0]],
            "too_tall": np.lib.stride_tricks.as_strided(
                image, shape=(9000, 8, 3), strides=image.strides, writeable=False),
        }
        for name, value in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(UnsupportedInput):
                    check_source(value)

    def test_rejects_span_beyond_owner(self):
        """虚报的形状必须被拒绝，而不是越界读。"""
        image = bgr_fixture(4, 4)
        lying = np.lib.stride_tricks.as_strided(
            image, shape=(8, 4, 3), strides=image.strides, writeable=False)
        with self.assertRaises(UnsupportedInput):
            check_source(lying)

    def test_buffer_span_follows_base_chain(self):
        image = bgr_fixture(10, 10)
        view = image[3:][:, 1:]
        self.assertEqual(buffer_span(view), image.nbytes - (3 * 30 + 3))


class BackendSelectionTests(unittest.TestCase):
    def test_default_is_opencv_without_mlir_dependency(self):
        with Preprocessor(output_size=(64, 64)) as pre:
            self.assertEqual(pre.backend_name, "opencv")
            self.assertIsInstance(pre.backend, OpenCVBackend)
            self.assertEqual(pre.run(bgr_fixture(30, 40)).shape, (64, 64, 3))

    def test_config_drives_backend_and_defaults_to_opencv(self):
        self.assertEqual(from_config({}, (32, 32)).backend_name, "opencv")
        self.assertEqual(from_config(None, (32, 32)).backend_name, "opencv")
        with self.assertRaises(BackendUnavailable):
            from_config({"preprocess": {"backend": "mlir"}}, (32, 32))

    def test_config_selects_library_per_profile(self):
        """AOT 内核尺寸编译期固定，不能拿错 profile 的库硬凑。"""
        config = {"preprocess": {"backend": "mlir", "libraries": {"640x640": "/a/640.so"}}}
        with self.assertRaises(BackendUnavailable) as raised:
            from_config(config, (448, 448))
        self.assertIn("448x448", str(raised.exception))
        with self.assertRaises(BackendUnavailable) as raised:
            from_config(config, (640, 640))
        self.assertIn("640.so", str(raised.exception))
        # opencv 后端不受 libraries 影响。
        self.assertEqual(from_config(config | {"preprocess": {"backend": "opencv"}},
                                     (448, 448)).backend_name, "opencv")

    def test_config_accepts_per_profile_manifest(self):
        config = {"preprocess": {"backend": "mlir", "manifest": "/shared.json", "libraries": {
            "448x448": {"library": "/a/448.so", "manifest": "/a/448.json"}}}}
        with self.assertRaises(BackendUnavailable) as raised:
            from_config(config, (448, 448))
        self.assertIn("448.so", str(raised.exception))

    def test_unknown_backend_and_bad_output_size_rejected(self):
        for kwargs in ({"backend": "auto"}, {"backend": "rga"}, {"backend": ""}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Preprocessor(output_size=(32, 32), **kwargs)
        for size in ((0, 32), (32, 0), (9000, 32), (-1, 4)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                Preprocessor(output_size=size)

    def test_explicit_mlir_failures_are_loud(self):
        """缺库/错路径必须报错，不能静默退回 OpenCV 后还报 MLIR 耗时。"""
        with self.assertRaises(BackendUnavailable):
            Preprocessor(output_size=(32, 32), backend="mlir")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(BackendUnavailable):
                Preprocessor(output_size=(32, 32), backend="mlir", library=root / "absent.so")
            with self.assertRaises(BackendUnavailable):
                Preprocessor(output_size=(32, 32), backend="mlir", library=root)
            bogus = root / "libbogus.so"
            bogus.write_bytes(b"not an ELF shared object")
            with self.assertRaises(BackendUnavailable):
                Preprocessor(output_size=(32, 32), backend="mlir", library=bogus)


class OutputOwnershipTests(unittest.TestCase):
    def test_each_call_can_own_its_output(self):
        with Preprocessor(output_size=(16, 16)) as pre:
            first = pre.run(bgr_fixture(20, 20, seed=1))
            second = pre.run(bgr_fixture(20, 20, seed=2))
            self.assertNotEqual(first.ctypes.data, second.ctypes.data)
            self.assertFalse(np.array_equal(first, second))
            self.assertTrue(first.flags["C_CONTIGUOUS"] and first.flags["OWNDATA"])

    def test_batch_view_is_not_a_copy(self):
        with Preprocessor(output_size=(16, 16)) as pre:
            out = pre.new_output()
            batch = pre.run_batch1(bgr_fixture(20, 20), out=out)
            self.assertEqual(batch.shape, (1, 16, 16, 3))
            self.assertFalse(batch.flags["OWNDATA"])
            self.assertEqual(batch.__array_interface__["data"][0], out.ctypes.data)
            out[0, 0, 0] = 123
            self.assertEqual(batch[0, 0, 0, 0], 123)

    def test_caller_supplied_output_is_validated(self):
        with Preprocessor(output_size=(16, 16)) as pre:
            good = pre.new_output()
            self.assertIs(pre.run(bgr_fixture(20, 20), out=good), good)
            bad = {
                "wrong_shape": np.empty((15, 16, 3), dtype=np.uint8),
                "wrong_dtype": np.empty((16, 16, 3), dtype=np.uint16),
                "non_contiguous": np.empty((16, 32, 3), dtype=np.uint8)[:, ::2],
                "not_array": bytearray(16 * 16 * 3),
            }
            for name, value in bad.items():
                with self.subTest(case=name), self.assertRaises(UnsupportedInput):
                    pre.run(bgr_fixture(20, 20), out=value)
            readonly = pre.new_output()
            readonly.flags.writeable = False
            with self.assertRaises(UnsupportedInput):
                pre.run(bgr_fixture(20, 20), out=readonly)

    def test_timings_split_allocation_from_call(self):
        with Preprocessor(output_size=(16, 16)) as pre:
            pre.run(bgr_fixture(20, 20))
            allocated = dict(pre.last_timings)
            pre.run(bgr_fixture(20, 20), out=pre.new_output())
            reused = dict(pre.last_timings)
        self.assertGreater(allocated["allocate_ns"], 0)
        self.assertEqual(reused["allocate_ns"], 0)
        for timings in (allocated, reused):
            self.assertGreater(timings["call_ns"], 0)
            self.assertEqual(timings["total_ns"], timings["allocate_ns"] + timings["call_ns"])


class MlirBackendTests(unittest.TestCase):
    """真实产物库测试。没有库就失败，不 skip。"""

    library = None
    manifest = None
    profile = None

    @classmethod
    def setUpClass(cls):
        cls.library, cls.manifest = MLIR_LIBRARY, MLIR_MANIFEST
        if not cls.library and not REQUIRE_MLIR:
            raise unittest.SkipTest(
                "未提供 EDGEAI_MLIR_LIBRARY；板端验收必须设置 EDGEAI_REQUIRE_MLIR=1")
        # profile 从 manifest 推导，不写死 640，否则换 profile 时测的是测试自己的假设。
        cls.profile = (640, 640)
        if cls.library and cls.manifest:
            manifest_path = Path(cls.manifest)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            resolved = Path(cls.library).resolve()
            for entry in data.get("variants", []):
                if entry.get("status") == "built" and \
                        (manifest_path.parent / entry["library"]).resolve() == resolved:
                    cls.profile = tuple(entry["profile"])
                    break
            else:
                raise AssertionError("manifest 中没有该库的 built 记录: " + str(cls.library))

    def build(self, output_size=None, manifest=True):
        if not self.library:
            self.fail("EDGEAI_REQUIRE_MLIR=1 但未提供 EDGEAI_MLIR_LIBRARY")
        return Preprocessor(output_size=output_size or self.profile, backend="mlir",
                            library=self.library, manifest=self.manifest if manifest else None)

    def test_matches_reference_bytewise(self):
        with self.build() as pre:
            for shape in ((480, 640), (720, 1280), (479, 641)):
                with self.subTest(shape=shape):
                    source = bgr_fixture(*shape)
                    expected = reference_resize_rgb(source, *pre.output_size)
                    self.assertTrue(np.array_equal(pre.run(source), expected))

    def test_positive_stride_roi_matches_reference(self):
        with self.build() as pre:
            full = bgr_fixture(500, 700)
            roi = full[10:490, 20:660]
            self.assertGreater(roi.strides[0], roi.shape[1] * 3)
            expected = reference_resize_rgb(np.ascontiguousarray(roi), *pre.output_size)
            self.assertTrue(np.array_equal(pre.run(roi), expected))

    def test_repeated_calls_do_not_bleed_between_frames(self):
        """单实例连续调用：每帧结果必须只取决于该帧输入。"""
        with self.build() as pre:
            sources = [bgr_fixture(480, 640, seed=s) for s in (1, 2, 3)]
            expected = [reference_resize_rgb(s, *pre.output_size) for s in sources]
            for round_index in range(3):
                for index, source in enumerate(sources):
                    self.assertTrue(np.array_equal(pre.run(source), expected[index]),
                                    "第 %d 轮第 %d 帧串帧" % (round_index, index))

    def test_workers_hold_independent_outputs(self):
        """每个 worker 各自持有 Preprocessor 与输出缓冲，并发不得互相污染。"""
        sources = [bgr_fixture(480, 640, seed=100 + i) for i in range(4)]
        expected = [reference_resize_rgb(s, *self.profile) for s in sources]
        results, errors = [None] * len(sources), []

        def work(index):
            try:
                with self.build() as pre:
                    out = pre.new_output()
                    for _ in range(3):
                        pre.run(sources[index], out=out)
                    results[index] = out.copy()
            except Exception as error:  # noqa: BLE001 - 交回主线程判定
                errors.append(error)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(sources))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        for index, output in enumerate(results):
            self.assertTrue(np.array_equal(output, expected[index]), "worker %d 输出被污染" % index)

    def test_input_size_change_rebuilds_context_and_times_it(self):
        with self.build() as pre:
            pre.run(bgr_fixture(480, 640))
            self.assertIsNotNone(pre.last_timings["context_create_ns"])
            pre.run(bgr_fixture(480, 640))
            self.assertIsNone(pre.last_timings["context_create_ns"])
            pre.run(bgr_fixture(360, 540))
            self.assertIsNotNone(pre.last_timings["context_create_ns"])

    def test_unsupported_inputs_rejected_by_the_mlir_backend(self):
        with self.build() as pre:
            for value in (bgr_fixture(8, 8).astype(np.float32), bgr_fixture(8, 8)[:, ::-1],
                          np.zeros((8, 8, 4), dtype=np.uint8)):
                with self.assertRaises(UnsupportedInput):
                    pre.run(value)

    def test_manifest_and_profile_mismatch_rejected(self):
        if not self.manifest:
            self.fail("板端验收需要 EDGEAI_MLIR_MANIFEST")
        wrong = (448, 448) if self.profile != (448, 448) else (640, 640)
        with self.assertRaises(BackendUnavailable):
            self.build(output_size=wrong)
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "manifest.json"
            data = json.loads(Path(self.manifest).read_text())
            data["status"] = "running"
            tampered.write_text(json.dumps(data))
            with self.assertRaises(BackendUnavailable):
                Preprocessor(output_size=(640, 640), backend="mlir",
                             library=self.library, manifest=tampered)

    def test_reports_real_library_identity(self):
        with self.build() as pre:
            info = pre.describe()
            self.assertEqual(info["backend"], "mlir")
            self.assertEqual(info["semantic_version"], "linear_half_pixel_q11_v1")
            self.assertEqual(info["machine"], platform.machine())
            self.assertEqual(len(info["library_sha256"]), 64)
            self.assertEqual(info["variant"], info["manifest"]["variant"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="libedgeai_preprocess.so 路径")
    parser.add_argument("--manifest", help="构建 manifest.json 路径")
    parser.add_argument("--require-mlir", action="store_true", help="缺库时失败而非跳过")
    args, remaining = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    global MLIR_LIBRARY, MLIR_MANIFEST, REQUIRE_MLIR
    MLIR_LIBRARY = args.library or MLIR_LIBRARY
    MLIR_MANIFEST = args.manifest or MLIR_MANIFEST
    REQUIRE_MLIR = args.require_mlir or REQUIRE_MLIR
    return unittest.main(argv=[sys.argv[0]] + remaining, exit=False).result.wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
