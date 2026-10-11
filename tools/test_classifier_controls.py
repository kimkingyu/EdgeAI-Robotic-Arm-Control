#!/usr/bin/env python3
"""Formal control tests use fake clocks/results only, never real benchmarks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.vision.classification_controls import drift_factor, run_formal_groups, ControlError
from tools.run_classifier_benchmark import atomic_json, canonical_hash


class FakeDriver:
    def __init__(self, root, after_values=(1.02, 1.02), interrupt=False):
        self.root = root
        self.after_values = after_values
        self.interrupt = interrupt
        self.calls = []
        for name in ("jobs", "results", "controls"):
            (root / name).mkdir()

    def make(self, run, root, experiment, block, kind, images, budget):
        name = f"{experiment['id']}.{kind}.b{block}"
        return {"experiment": experiment, "kind": kind, "block": block, "job_name": name,
                "output_path": str(root / ("controls" if kind == "control" else "results") / (name + ".json")),
                "warmup": budget["warmup"], "iterations": budget["iterations"]}

    def call(self, job, root, timeout):
        self.calls.append(job)
        if self.interrupt and job["kind"] == "inference":
            self.interrupt = False
            raise KeyboardInterrupt()
        value = (self.after_values[job["attempt"]] if job.get("control_role") == "after" else 1.0)
        result = {"status": "passed", "kind": job["kind"], "experiment": job["experiment"],
                  "block": job["block"], "worker_pid": 1000 + len(self.calls),
                  "job_sha256": canonical_hash(job),
                  "samples": [{"image_id": "fake", "inference_ms": value}] * job["iterations"],
                  "sample_count": job["iterations"]}
        atomic_json(Path(job["output_path"]), result)
        atomic_json(root / "jobs" / (job["job_name"] + ".json"), job)
        return result

    def restore(self, job, root):
        path = Path(job["output_path"])
        return json.loads(path.read_text()) if path.exists() else None

    def execute(self, *, resuming=False):
        run = {"config_sha256": "c", "manifest_sha256": "m", "code_fingerprint": "s",
               "config": {"seed": 11, "worker_timeout_s": 20}}
        experiments = [{"id": "resnet18.B0.t1", "model_id": "resnet18", "variant_id": "B0", "threads": 1},
                       {"id": "resnet18.O1.t1", "model_id": "resnet18", "variant_id": "O1", "threads": 1}]
        with contextlib.redirect_stdout(io.StringIO()):
            return run_formal_groups(run, self.root, experiments, [{"id": "fake"}],
                                     {"warmup": 0, "iterations": 4, "repeats": 1},
                                     make_job=self.make, run_job=self.call, resume_result=self.restore,
                                     write_json=atomic_json, resuming=resuming)


class ControlsTests(unittest.TestCase):
    def test_gate_uses_symmetric_medians(self):
        before = {"status": "passed", "samples": [{"inference_ms": 10}, {"inference_ms": 20}]}
        after = {"status": "passed", "samples": [{"inference_ms": 12}, {"inference_ms": 18}]}
        self.assertEqual(drift_factor(before, after), 1)
        faster = {"status": "passed", "samples": [{"inference_ms": 5}, {"inference_ms": 10}]}
        self.assertEqual(drift_factor(before, faster), 2)
        self.assertEqual(drift_factor(faster, before), 2)

    def test_failed_empty_controls_cannot_accept(self):
        valid = {"status": "passed", "samples": [{"inference_ms": 1}]}
        self.assertIsNone(drift_factor(valid, {"status": "failed_runtime", "samples": []}))
        self.assertIsNone(drift_factor(valid, {"status": "passed", "samples": []}))

    def test_stable_group_has_two_separate_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver(Path(directory))
            results = driver.execute()
            self.assertEqual(len(results), 2)
            self.assertTrue(all(row["environment_accepted"] for row in results))
            self.assertEqual(len([j for j in driver.calls if j["kind"] == "control"]), 2)
            self.assertEqual(len(list((driver.root / "results").glob("*.json"))), 2)
            self.assertEqual(len(list((driver.root / "controls").glob("*.json"))), 3)

    def test_one_retry_preserves_rejected_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver(Path(directory), after_values=(1.30, 1.03))
            results = driver.execute()
            self.assertTrue(all(row["attempt"] == 1 for row in results))
            self.assertTrue(all(row["environment_accepted"] for row in results))
            self.assertEqual(len(list((driver.root / "results").glob("*.json"))), 2)
            self.assertEqual(len(list((driver.root / "rejected-results").rglob("*.job.json"))), 2)
            self.assertEqual(len(driver.calls), 8)

    def test_permanent_drift_is_failure_not_ranked_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver(Path(directory), after_values=(1.30, 1.25))
            results = driver.execute()
            self.assertTrue(all(row["status"] == "failed_runtime" for row in results))
            self.assertTrue(all(row["samples"] == [] for row in results))
            self.assertTrue(all(row["environment_accepted"] is False for row in results))
            self.assertEqual(len(driver.calls), 8)
            self.assertEqual(len(list((driver.root / "rejected-results").rglob("*.job.json"))), 4)

    def test_completed_group_resume_does_not_run_models(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver(Path(directory))
            first = driver.execute()
            count = len(driver.calls)
            second = driver.execute(resuming=True)
            self.assertEqual(first, second)
            self.assertEqual(len(driver.calls), count)

    def test_interrupted_group_never_mixes_new_controls_with_old_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver(Path(directory), interrupt=True)
            with self.assertRaises(KeyboardInterrupt):
                driver.execute()
            results = driver.execute(resuming=True)
            self.assertTrue(all(row["status"] == "passed" for row in results))
            self.assertTrue(list((driver.root / "rejected-results").rglob("*.job.json")))

    def test_changed_control_ledger_fingerprint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver(Path(directory))
            driver.execute()
            ledger = driver.root / "controls" / "ledger.json"
            data = json.loads(ledger.read_text())
            data["code_fingerprint"] = "changed"
            atomic_json(ledger, data, overwrite=True)
            with self.assertRaises(ControlError):
                driver.execute(resuming=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
