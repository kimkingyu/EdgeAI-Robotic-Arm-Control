#!/usr/bin/env python3
"""M0 unittest checks; synthetic image fixtures are NOT performance evidence."""
import contextlib
import hashlib
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if __package__:
    from . import benchmark_preprocess as bench
else:
    import benchmark_preprocess as bench

HAS_IMAGES = bench.cv2 is not None and bench.np is not None


class BaselineTests(unittest.TestCase):
    def test_cli_defaults(self):
        args = bench.parse_args(["--images-dir", "images", "--report", "result.json"])
        self.assertEqual((args.image_count, args.sizes, args.mode), (16, [640, 448], "both"))
        self.assertEqual((args.warmup, args.iterations, args.batches, args.threads), (30, 1000, 5, 1))
        self.assertEqual(bench.input_dimensions("1920X1080"), (1920, 1080))
        self.assertEqual(bench.nonnegative_int("0"), 0)

    def test_cli_invalid_values(self):
        base = ["--images-dir", "images", "--report", "result.json"]
        invalid = [(flag, value) for flag in ("--image-count", "--sizes", "--iterations", "--batches", "--threads")
                   for value in ("0", "-1", "1.5")]
        invalid += [("--warmup", "-1"), ("--mode", "mlir")]
        invalid += [("--input-size", value) for value in ("2", "1x2x3", "0x4", "4x-1", "ax4")]
        for arguments in [[], ["--images-dir", "images"], ["--report", "r.json"]] + [base + list(p) for p in invalid]:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    bench.parse_args(arguments)
                self.assertNotEqual(raised.exception.code, 0)

    def test_selection_missing_empty_insufficient_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (root / "missing", root):
                with self.assertRaises(ValueError):
                    bench.select_images(path, 1)
            for name in ("z.PNG", "a.jpg", "m.jpeg", "ignored.txt"):
                (root / name).write_bytes(b"selection-only fixture")
            (root / "subdir.png").mkdir()
            self.assertEqual([p.name for p in bench.select_images(root, 2)], ["a.jpg", "m.jpeg"])
            with self.assertRaises(ValueError):
                bench.select_images(root, 4)
            with self.assertRaises(ValueError):
                bench.select_images(root, 0)

    def test_dependency_failure_has_nonzero_exit_and_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "absent.json"
            with mock.patch.object(bench, "np", None), mock.patch.object(bench, "cv2", None):
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    code = bench.main(["--images-dir", directory, "--report", str(report)])
            self.assertEqual(code, 1)
            self.assertIn("required", error.getvalue())
            self.assertFalse(report.exists())

    def test_statistics_and_percentile_interpolation(self):
        result = bench.describe_ns([40, 10, 30, 20])
        self.assertEqual((result["sample_count"], result["mean_ns"], result["p50_ns"]), (4, 25, 25))
        self.assertAlmostEqual(result["p95_ns"], 38.5)
        self.assertAlmostEqual(result["p99_ns"], 39.7)
        self.assertEqual(bench.describe_ns([7])["p99_ns"], 7)
        with self.assertRaises(ValueError):
            bench.describe_ns([])

    def test_fps_excludes_validation_and_batch_mean_summary(self):
        def raw(total):
            return {**dict.fromkeys(bench.STAGES, total), "validation_ns": 10**12}
        one = bench.summarize_batch([raw(1000), raw(3000)])
        self.assertEqual(one["sum_total_ns"], 4000)
        self.assertEqual(one["serial_pipeline_fps"], 500000)
        batches = [bench.summarize_batch([raw(value)]) for value in (1000, 3000)]
        summary = bench.summarize_batches(batches)
        self.assertEqual(summary["batch_means"]["total_ns"]["mean_ns"], 2000)
        self.assertEqual(summary["batch_means"]["total_ns"]["min_ns"], 1000)
        self.assertEqual(summary["batch_means"]["total_ns"]["max_ns"], 3000)
        self.assertEqual(summary["serial_pipeline_fps"], 500000)
        with self.assertRaises(ValueError):
            bench.summarize_batch([raw(0)])

    def test_stage_timer_boundaries_without_image_dependencies(self):
        fake_cv, fake_np = mock.Mock(), mock.Mock()
        events = []
        def clock():
            events.append("clock")
            return len(events) * 100
        fake_cv.resize.side_effect = lambda *a, **k: events.append("resize") or "resized"
        fake_cv.cvtColor.side_effect = lambda *a, **k: events.append("color") or "rgb"
        fake_np.expand_dims.side_effect = lambda *a, **k: events.append("view") or "output"
        with mock.patch.object(bench, "cv2", fake_cv), mock.patch.object(bench, "np", fake_np):
            with mock.patch.object(bench.time, "perf_counter_ns", clock):
                with mock.patch.object(bench, "array_sha256") as checksum:
                    products, timing = bench.timed_preprocess("input", 6)
        self.assertEqual(events, ["clock", "resize", "clock", "color", "clock", "view", "clock"])
        self.assertEqual(products, ("resized", "rgb", "output"))
        self.assertEqual(timing, dict(zip(bench.STAGES, (200, 200, 200, 600))))
        fake_cv.resize.assert_called_once_with("input", (6, 6), interpolation=fake_cv.INTER_LINEAR)
        fake_cv.cvtColor.assert_called_once_with("resized", fake_cv.COLOR_BGR2RGB)
        fake_np.expand_dims.assert_called_once_with("rgb", axis=0)
        checksum.assert_not_called()

    def test_warmup_excluded_and_modes_schedule_inputs(self):
        for mode in ("hot", "stream"):
            calls = []
            def fake_preprocess(image, size):
                calls.append(image)
                total = 10**12 if len(calls) <= 2 else 1000
                return object(), {"resize_ns": total - 2, "cvtColor_ns": 1,
                                  "batch_view_ns": 1, "total_ns": total}
            with self.subTest(mode=mode), mock.patch.object(bench, "timed_preprocess", fake_preprocess):
                with mock.patch.object(bench, "validate_output") as verify:
                    with mock.patch.object(bench.time, "perf_counter_ns", side_effect=itertools.count()):
                        batch = bench.run_batch(["a", "b", "c"], 4, mode, 2, 4, [None] * 3)
            self.assertEqual(calls, ["a"] * 6 if mode == "hot" else ["a", "b", "a", "b", "c", "a"])
            self.assertEqual(verify.call_count, 6)
            self.assertEqual((batch["sample_count"], len(batch["raw_samples"])), (4, 4))
            self.assertEqual(batch["sum_total_ns"], 4000)
            self.assertEqual(batch["validation_overhead"]["mean_ns"], 1)

    def test_readonly_sysfs_and_unavailable_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = bench.sysfs_snapshot(root)
            self.assertTrue(all(g["status"] == "unavailable" for g in snapshot["groups"].values()))
            temp = root / "class/thermal/thermal_zone0/temp"
            frequency = root / "devices/system/cpu/cpufreq/policy0/scaling_cur_freq"
            for path, value in ((temp, "42000\n"), (frequency, "1800000\n")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value, encoding="utf-8")
            snapshot = bench.sysfs_snapshot(root)
            self.assertEqual(snapshot["groups"]["thermal_temp"]["readings"][0]["value"], "42000")
            self.assertEqual(snapshot["groups"]["current_frequency"]["readings"][0]["unit"], "kHz")
            self.assertEqual(temp.read_text(encoding="utf-8"), "42000\n")
            with mock.patch.object(Path, "read_text", side_effect=OSError("denied")):
                failed = bench.sysfs_snapshot(root)
            self.assertIn("error", failed["groups"]["thermal_temp"]["readings"][0])


@unittest.skipUnless(HAS_IMAGES, "NumPy/OpenCV unavailable: image/pipeline tests NOT executed")
class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = bench.np.arange(7 * 11 * 3, dtype=bench.np.uint8).reshape(7, 11, 3)
        self.path = self.root / "真实路径.png"
        ok, encoded = bench.cv2.imencode(".png", self.image)
        self.assertTrue(ok)
        self.path.write_bytes(encoded.tobytes())

    def test_load_dimensions_derived_input_and_original_hash(self):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        for dimensions in (None, (13, 4), (11, 7)):
            images, records = bench.load_inputs([self.path], dimensions)
            expected_shape = (7, 11, 3) if dimensions is None else (dimensions[1], dimensions[0], 3)
            self.assertEqual(images[0].shape, expected_shape)
            self.assertEqual(records[0]["original_shape"], [7, 11, 3])
            self.assertEqual(records[0]["input_shape"], list(expected_shape))
            self.assertEqual(records[0]["file_sha256"], digest)
            self.assertEqual(records[0]["derived_input"], dimensions is not None)
            self.assertFalse(images[0].flags.writeable)

    def test_missing_corrupt_and_empty_files_fail(self):
        corrupt, empty = self.root / "broken.jpg", self.root / "empty.png"
        corrupt.write_bytes(b"not an image")
        empty.write_bytes(b"")
        for path in (self.root / "missing.jpg", corrupt, empty):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, path.name):
                bench.load_inputs([path])
        with contextlib.redirect_stderr(io.StringIO()):
            code = bench.main(["--images-dir", str(self.root), "--image-count", "1",
                               "--report", str(self.root / "failed.json")])
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "failed.json").exists())

    def test_pixels_shapes_contiguity_and_independent_products(self):
        for dimensions in (None, (13, 4)):
            images, _ = bench.load_inputs([self.path], dimensions)
            for size in (1, 5, 11):
                image = images[0]
                products, timing = bench.timed_preprocess(image, size)
                bench.validate_output(image, products, size, bench.reference_checksums(image, size))
                expected = bench.cv2.resize(image, (size, size), interpolation=bench.cv2.INTER_LINEAR)[:, :, ::-1]
                bench.np.testing.assert_array_equal(products[2][0], expected)
                self.assertTrue(bench.np.shares_memory(products[1], products[2]))
                self.assertEqual(timing["total_ns"], sum(timing[key] for key in bench.STAGES[:-1]))
                saved = products[2].copy()
                later, _ = bench.timed_preprocess(image, size)
                self.assertFalse(bench.np.shares_memory(products[2], later[2]))
                later[2].fill(0)
                bench.np.testing.assert_array_equal(products[2], saved)

    def test_validation_rejects_layout_dtype_copy_and_wrong_pixels(self):
        products, _ = bench.timed_preprocess(self.image, 5)
        resized, rgb, output = products
        wrong_rgb = rgb.copy()
        wrong_rgb[0, 0, 0] ^= 1
        invalid = [(resized.astype(bench.np.float32), rgb, output), (resized[::-1], rgb, output),
                   (resized, rgb, output[0]), (resized, rgb, output.copy()),
                   (resized, wrong_rgb, bench.np.expand_dims(wrong_rgb, 0))]
        for result in invalid:
            with self.assertRaises(ValueError):
                bench.validate_output(self.image, result, 5, bench.reference_checksums(self.image, 5))

    def test_small_report_contains_raw_data_and_provenance(self):
        path = self.root / "fixture-report.json"
        argv = ["--images-dir", str(self.root), "--image-count", "1", "--sizes", "2", "3",
                "--input-size", "9x6", "--warmup", "2", "--iterations", "3", "--batches", "2",
                "--report", str(path)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench.main(argv), 0)
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["backend"], "opencv")
        self.assertEqual(len(report["cases"]), 4)
        self.assertEqual(report["inputs"][0]["input_shape"], [6, 9, 3])
        self.assertEqual(report["environment"]["script_sha256"],
                         hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest())
        self.assertTrue(report["environment"]["opencv_build_information"])
        for case in report["cases"]:
            self.assertEqual(len(case["batches"]), 2)
            for batch in case["batches"]:
                self.assertEqual((batch["sample_count"], batch["warmup_samples_excluded"]), (3, 2))
                self.assertEqual(len(batch["raw_samples"]), 3)
                self.assertEqual(batch["sum_total_ns"], sum(r["total_ns"] for r in batch["raw_samples"]))
        original = path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bench.main(argv), 1)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    if not HAS_IMAGES:
        print("DEPENDENCY-LIMITED: image/pipeline tests skipped; NOT a full validation pass.")
    unittest.main(verbosity=2)
