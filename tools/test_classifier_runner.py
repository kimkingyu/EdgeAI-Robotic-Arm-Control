#!/usr/bin/env python3
"""Offline runner tests: fake clocks/artifacts are never benchmark evidence."""
import base64
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
from tools import run_classifier_benchmark as runner


class RunnerTests(unittest.TestCase):
    def test_help_has_no_optional_dependencies(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "tools/run_classifier_benchmark.py"), "--help"],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mode", result.stdout)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_vision_package_is_lazy(self):
        code = ("import sys,src.vision; assert 'numpy' not in sys.modules; "
                "assert 'cv2' not in sys.modules; assert 'src.vision.camera' not in sys.modules; "
                "assert 'src.vision.rknn_detector' not in sys.modules; "
                "assert src.vision.__all__==['USBCamera','RKNNObjectDetector']")
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pipeline_measures_adaptation_and_actual_wall(self):
        calls = []
        prepare = lambda value: calls.append("prepare") or value
        adapt = lambda value: calls.append("adapt") or value
        infer = lambda value: calls.append("infer") or value
        post = lambda value: calls.append("post") or value
        ticks = iter(range(0, 7_000_000, 1_000_000))
        result, timing = runner.timed_pipeline("input", prepare, adapt, infer, post, clock=lambda: next(ticks))
        self.assertEqual(result, "input")
        self.assertEqual(calls, ["prepare", "adapt", "infer", "post"])
        self.assertEqual(timing["input_adapt_ms"], 1)
        self.assertEqual(timing["inference_ms"], 1)
        self.assertEqual(timing["stages_total_ms"], 4)
        self.assertEqual(timing["pipeline_wall_ms"], 6)
        self.assertEqual(timing["wrapper_overhead_ms"], 2)

    def test_failed_stage_does_not_return_fake_samples(self):
        def fail(_):
            raise ValueError("backend failure")
        with self.assertRaisesRegex(ValueError, "backend failure"):
            runner.timed_pipeline(1, lambda x: x, lambda x: x, fail, lambda x: x)

    def test_atomic_json_refuses_overwrite_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "record.json"
            runner.atomic_json(output, {"real": True})
            with self.assertRaises(ValueError):
                runner.atomic_json(output, {"real": False})
            self.assertTrue(json.loads(output.read_text())["real"])
            other = Path(directory) / "nonfinite.json"
            with self.assertRaises(ValueError):
                runner.atomic_json(other, {"value": float("nan")})
            self.assertFalse(other.exists())
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_inline_image_never_invents_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "pixel.png"
            image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jX1cAAAAASUVORK5CYII="))
            result = runner.inline_manifest(image)
            self.assertIsNone(result["images"][0]["label"])
            self.assertEqual(result["images"][0]["split"], "smoke")
            self.assertEqual(result["images"][0]["sha256"], runner.sha256(image))
            self.assertEqual(len(result["manifest_sha256"]), 64)

    def test_worker_env_threads_are_set_before_import(self):
        environment = runner._worker_environment(4)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(environment[key], "4")
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")

    def test_child_interpreter_preserves_pid_and_venv(self):
        executable, environment = runner.child_python(1)
        code = "import os,sys,json; print(json.dumps({'pid':os.getpid(),'prefix':sys.prefix}))"
        process = subprocess.Popen([executable, "-c", code], env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stderr)
        child = json.loads(stdout)
        self.assertEqual(child["pid"], process.pid)
        self.assertEqual(Path(child["prefix"]).resolve(), Path(sys.prefix).resolve())

    def test_native_process_creation_identity_matches_parent_and_worker(self):
        executable, environment = runner.child_python(1)
        code = ("import json,time; from tools.run_classifier_benchmark import worker_process_identity; "
                "print(json.dumps(worker_process_identity()),flush=True); time.sleep(0.2)")
        process = subprocess.Popen([executable, "-c", code], env=environment, cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        parent = runner.worker_process_identity(process.pid)
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stderr)
        child = json.loads(stdout)
        self.assertEqual(parent, child)
        self.assertGreater(child["worker_creation_ticks"], 0)
        self.assertTrue(child["worker_identity"].startswith(str(process.pid) + ":"))

    def test_image_order_is_paired_across_variants(self):
        prepared = {"model_metadata": {}, "weights_path": "unused", "weights_sha256": "hash",
                    "reference_path": "unused", "reference_sha256": "hash"}
        run = {"prepared_models": {"resnet18": prepared}, "config": {"seed": 91, "rtol": 1e-4, "atol": 1e-4},
               "run_id": "test", "config_sha256": "c", "manifest_sha256": "m", "code_fingerprint": "s"}
        images = [{"id": str(i)} for i in range(4)]
        experiment = {"id": "resnet18.B0.t1", "model_id": "resnet18", "variant_id": "B0", "threads": 1}
        other = {**experiment, "id": "resnet18.FL1.t1", "variant_id": "FL1"}
        first = runner.make_job(run, Path("unused"), experiment, 0, "inference", images,
                                {"warmup": 3, "iterations": 10})
        second = runner.make_job(run, Path("unused"), other, 0, "inference", images,
                                 {"warmup": 3, "iterations": 10})
        self.assertEqual(first["image_sequence"], second["image_sequence"])
        self.assertEqual(len(first["image_sequence"]), 10)
        self.assertEqual(set(first["image_sequence"]), {0, 1, 2, 3})

    def test_sampled_rss_is_real_and_positive(self):
        if sys.platform not in ("win32", "linux"):
            self.skipTest("current RSS backend not available")
        reader = runner.RssReader()
        self.assertGreater(reader.read(), 0)
        with runner.RssSampler(reader, interval=0.001) as sampler:
            allocation = bytearray(2 * 1024 * 1024)
            allocation[0] = 1
        self.assertIsNone(sampler.error)
        self.assertGreaterEqual(len(sampler.values), 2)
        self.assertTrue(all(value > 0 for value in sampler.values))

    def test_worker_job_rejects_artifact_changes_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for subdirectory in ("jobs", "results", "prepared"):
                (root / subdirectory).mkdir()
            weights = root / "weights.pth"
            reference = root / "prepared" / "reference.npz"
            weights.write_bytes(b"test-only-not-a-model")
            reference.write_bytes(b"test-only-not-an-array")
            job = {"schema_version": 1, "run_dir": str(root), "kind": "inference",
                   "output_path": str(root / "results" / "result.json"),
                   "experiment": {"model_id": "resnet18", "variant_id": "B0", "threads": 1},
                   "iterations": 3, "warmup": 0, "block": 0, "rtol": 1e-4, "atol": 1e-4,
                   "weights_path": str(weights), "weights_sha256": runner.sha256(weights),
                   "reference_path": str(reference), "reference_sha256": runner.sha256(reference),
                   "images": [], "code_fingerprint": "test-fingerprint"}
            job_path = root / "jobs" / "job.json"
            with patch.object(runner, "source_fingerprint", return_value=("test-fingerprint", {})):
                self.assertEqual(runner._validate_job(job_path, job), root / "results" / "result.json")
                with self.assertRaises(ValueError):
                    runner._validate_job(job_path, {**job, "output_path": str(root / "escape.json")})
                weights.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    runner._validate_job(job_path, job)

    def test_resume_archives_failed_results_outside_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "jobs").mkdir()
            job = {"output_path": str(root / "results" / "old.json"), "job_name": "old",
                   "config_sha256": "c", "manifest_sha256": "m", "code_fingerprint": "s"}
            previous = {"job_sha256": runner.canonical_hash(job), "config_sha256": "c",
                        "manifest_sha256": "m", "code_fingerprint": "s", "status": "failed_runtime"}
            runner.atomic_json(job["output_path"], previous)
            runner.atomic_json(root / "jobs" / "old.json", job)
            self.assertIsNone(runner.resume_result(job, root))
            self.assertFalse(list((root / "results").glob("*.json")))
            self.assertEqual(len(list((root / "archived-results").glob("*.json"))), 2)

    def test_resume_does_not_trust_a_changed_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            job = {"output_path": str(root / "results" / "old.json"), "job_name": "old",
                   "config_sha256": "c", "manifest_sha256": "m", "code_fingerprint": "s"}
            previous = {"job_sha256": "wrong", "status": "passed"}
            runner.atomic_json(job["output_path"], previous)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                runner.resume_result(job, root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
