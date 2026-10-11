#!/usr/bin/env python3
"""Offline Stage B contract tests. All artifacts/children are explicit test mocks.

Never execute real quantization, model inference, holdout evaluation or hardware.
Mock end-to-end runs suppress report publication; fixtures live in private tempdirs.
"""
from __future__ import annotations

import copy
import csv
import hashlib
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
from src.vision import classification_stage_b as stage
from tools import run_classifier_stage_b as cli

H = "a" * 64
C = "b" * 64


def fake_identity(pid=None, ticks=None):
    pid = 1 if pid is None else pid
    ticks = pid * 100000 if ticks is None else ticks
    return {"worker_pid": pid, "worker_identity": f"{pid}:{ticks}", "worker_creation_ticks": ticks,
            "worker_identity_source": "unittest native-identity stub; temporary fixtures only"}


def result_base(job, pid):
    return stage._result_base(job, pid=pid, identity=fake_identity(pid))


def accuracy(split="development", drop=0.0, ci=None):
    return {"split": split, "model_id": stage.MODELS[0], "paired": True,
            "top1": {"drop_pp": drop, "ci95_pp": ci or [drop, drop]},
            "top5": {"drop_pp": drop, "ci95_pp": ci or [drop, drop]},
            "budget_passed": True, "budget_status": "passed"}


def candidate(variant="Q2", threads=1, *, model=stage.MODELS[0], speedup=1.3):
    return {"id": f"{model}.{variant}.t{threads}", "model_id": model, "variant_id": variant,
            "threads": threads, "status": "passed", "validation_passed": True,
            "artifact_path": "unused", "artifact_sha256": C,
            "baseline_artifact_path": "unused-baseline", "baseline_artifact_sha256": H,
            "baseline_threads": 1, "baseline_variant_id": "QB" if variant.startswith("Q") and variant != "QB" else "O1",
            "phase": "screening", "development": {"split": "development", "model_id": model, "artifact_sha256": C},
            "development_comparison": accuracy(),
            "runtime_profile": {"int8_operator_events": [{"op_name": "QLinearConv"}],
                                "int8_operator_counts": {"QLinearConv": 1}, "providers": ["CPUExecutionProvider"]},
            "performance": {"status": "available", "gate_passed": True, "speedup": speedup,
                            "candidate_stats": {"p50": 10.0 if variant == "O1" and threads == 1 else 10.0 / speedup},
                            "ci95": {"low": 1.1, "high": 1.5}, "p95_ratio": .9}}


def process_blocks(value=10, *, start_pid=100, round_id="test:screening", iterations=200):
    results = []
    for block in range(5):
        sequence = stage.fixed_sequence(91, block, iterations)
        samples = [{"image_id": f"image-{index}", "image_sha256": H,
                    "inference_ms": value, "input_sha256": C} for index in sequence]
        results.append({"block": block, **fake_identity(start_pid + block), "model_id": stage.MODELS[0],
                        "round_id": round_id, "iterations": iterations, "warmup": 20,
                        "image_sequence": sequence, "status": "passed", "environment_accepted": True, "samples": samples})
    return results


class Fixture:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "images").mkdir()
        config = stage.read_json(ROOT / "configs/classifier_dataset.json")
        self.classes = config["classes"]
        self.images = []
        for split, count in (("calibration", 256), ("development", 128), ("holdout", 1000)):
            for index in range(count):
                mapping = self.classes[index % 20]
                identity = f"{split}-{index:04d}"
                data = f"TEST MOCK IMAGE {identity}".encode()
                path = self.dataset / "images" / f"{identity}.jpg"
                path.write_bytes(data)
                official = "test" if split == "holdout" else "trainval"
                self.images.append({"id": identity, "path": f"images/{identity}.jpg",
                                    "sha256": hashlib.sha256(data).hexdigest(), "source_id": identity,
                                    "split": split, "breed": mapping["prefix"], "canonical_name": mapping["canonical_name"],
                                    "label": None if split == "calibration" else mapping["imagenet_index"],
                                    "source": config["dataset"], "source_url": config["homepage"], "license": config["license"],
                                    "width": 224, "height": 224, "source_original_split": official,
                                    "ground_truth_source": f"annotations/{official}.txt"})
        manifest = {"schema_version": 1, "root": ".", "images": self.images,
                    "source_metadata": "source.json", "dataset": config["dataset"]}
        self.manifest_path = self.dataset / "manifest.json"
        stage.atomic_json(self.manifest_path, manifest)
        self.source = {"schema_version": 1, "license": config["license"], "weights_version": "IMAGENET1K_V1",
                       "subset": {"class_count": 20, "species": "dog", "mapping": self.classes,
                                  "counts": {"calibration": 256, "development": 128, "holdout": 1000}}}
        stage.atomic_json(self.dataset / "source.json", self.source)
        self.completion()
        self.manifest = stage.load_stage_b_manifest(self.manifest_path)
        self.performance = [row for row in self.manifest["images"] if row["split"] == "development"][:64]
        self.prepared = {}
        categories = [f"category-{index}" for index in range(1000)]
        for mapping in self.classes:
            categories[mapping["imagenet_index"]] = mapping["canonical_name"]
        for model in stage.MODELS:
            paths = {}
            for name, suffix in (("weights", ".pth"), ("onnx", ".onnx"), ("reference", ".npz")):
                path = self.root / f"{model}-{name}{suffix}"
                path.write_bytes(f"TEST MOCK {model} {name}".encode())
                paths[f"{name}_path"], paths[f"{name}_sha256"] = str(path), stage.sha256(path)
            metadata = {"model_id": model, "weights_source": "official_imagenet1k_v1", "matches_official_hash": True,
                        "category_mapping_assumed": False, "input_shape": [1, 3, 224, 224], "dtype": "float32",
                        "device": "cpu", "categories": categories, "preprocess": stage.PREPROCESS,
                        "weights_sha256": paths["weights_sha256"]}
            export = {"schema_version": 1, "status": "verified", "model_id": model, "model": metadata,
                      "onnx": {"sha256": paths["onnx_sha256"], "input_shape": [1, 3, 224, 224], "output_shape": [1, 1000],
                               "normalization_in_graph": False, "external_data": False, "graph_statistics": {},
                               "opset": 17, "input_name": "images", "output_name": "logits",
                               "input_dtype": "float32", "output_dtype": "float32"},
                      "validation": {"verified_images_count": 1, "provider": "CPUExecutionProvider",
                                     "results": [{"image_sha256": H, "comparison": {"passed": True, "allclose": True}}]}}
            stage.atomic_json(Path(paths["onnx_path"]).with_suffix(".json"), export)
            self.prepared[model] = {**paths, "model_metadata": metadata, "export_metadata": export}
        from src.vision.classification_benchmark import load_config, expand_experiments
        config = load_config()
        self.experiments = expand_experiments(config)
        self.fp32_dir = self.root / "fp32"
        self.fp32_dir.mkdir()
        (self.fp32_dir / "jobs").mkdir()
        (self.fp32_dir / "results").mkdir()
        self.fp32_run = {"schema_version": 1, "mode": "formal", "status": "completed", "run_id": "test-fp32",
                         "manifest_sha256": self.manifest["manifest_sha256"], "config": config,
                         "images": self.performance, "experiments": self.experiments, "prepared_models": self.prepared,
                         "config_sha256": stage.canonical_hash(config), "code_fingerprint": H}
        stage.atomic_json(self.fp32_dir / "run.json", self.fp32_run)
        rows = [{**experiment, "status": "passed", "scheduled": True,
                 "inference": {"stats": {"inference_ms": {"p50": 10 + experiment["threads"]}}}}
                for experiment in self.experiments]
        self.summary = {"status": "completed", "experiments": rows, "records": [], "integrity_errors": []}
        controls = self.fp32_dir / "controls"
        controls.mkdir()
        identity = {key: self.fp32_run[key] for key in ("config_sha256", "manifest_sha256", "code_fingerprint")}
        ledger = {"schema_version": 1, **identity, "threshold": 1.10, "max_attempts": 2, "groups": {}}
        next_pid = 10000
        for model in stage.MODELS:
            for block in range(5):
                group_key = f"{model}.b{block}"
                group = {"attempt": 0, "status": "accepted", "drift_factor": 1.0, "threshold": 1.10}
                sequence = stage.fixed_sequence(config["seed"], block, 200)
                def source_record(experiment, kind, name, folder, role=None):
                    nonlocal next_pid
                    next_pid += 1
                    job = {**identity, "run_id": self.fp32_run["run_id"], "experiment": experiment,
                           "block": block, "kind": kind, "attempt": 0, "warmup": 20, "iterations": 200,
                           "image_sequence": sequence, "images": self.performance, **self.prepared[model]}
                    if role is not None:
                        job["control_role"] = role
                    result = {**identity, "run_id": job["run_id"], "experiment": experiment, "block": block,
                              "kind": kind, "status": "passed", **fake_identity(next_pid), "job_sha256": stage.canonical_hash(job),
                              "samples": [{"image_id": self.performance[index]["id"], "inference_ms": 10.0} for index in sequence],
                              "validation": {"passed": True, "allclose": True, "image_count": 64,
                                             "rtol": 1e-4, "atol": 1e-4,
                                             "per_image": [{"id": image["id"], "sha256": image["sha256"],
                                                            "allclose": True, "passed": True} for image in self.performance]}}
                    if kind == "inference":
                        result.update(environment_accepted=True, drift_control_group=group_key, attempt=0, drift_factor=1.0)
                    path = folder / f"{name}.json"
                    stage.atomic_json(path, result)
                    stage.atomic_json(self.fp32_dir / "jobs" / f"{name}.json", job)
                    return result, path
                baseline = next(row for row in self.experiments if row["model_id"] == model and row["variant_id"] == "B0" and row["threads"] == 1)
                for role in ("before", "after"):
                    result, path = source_record(baseline, "control", f"{model}.control.{role}.b{block}", controls, role)
                    group[f"{role}_path"], group[f"{role}_pid"] = str(path), result["worker_pid"]
                    group[f"{role}_identity"] = result["worker_identity"]
                ledger["groups"][group_key] = group
                for experiment in [row for row in self.experiments if row["model_id"] == model]:
                    result, path = source_record(experiment, "inference", f"{experiment['id']}.inference.b{block}", self.fp32_dir / "results")
                    self.summary["records"].append({"data": result, "source_file": f"results/{path.name}",
                                                   "record_sha256": stage.sha256(path)})
        stage.atomic_json(controls / "ledger.json", ledger)
        stage.atomic_json(self.fp32_dir / "summary.json", self.summary)
        stage.atomic_json(self.fp32_dir / "complete.json", {"run_id": self.fp32_run["run_id"],
                          "status": self.fp32_run["status"], "code_fingerprint": self.fp32_run["code_fingerprint"],
                          "summary_sha256": stage.sha256(self.fp32_dir / "summary.json")})
        self.audit = {"run_dir": str(self.fp32_dir), "run": self.fp32_run, "summary": self.summary,
                      "run_sha256": stage.sha256(self.fp32_dir / "run.json"), "snapshot_sha256": H,
                      "visible_failures": [], "dependencies": {}, "allow_failures": False}

    def completion(self):
        stage.atomic_json(self.dataset / "prepared.complete.json", {
            "schema_version": 1, "status": "complete", "manifest_sha256": stage.sha256(self.manifest_path),
            "source_sha256": stage.sha256(self.dataset / "source.json"),
            "counts": {"calibration": 256, "development": 128, "holdout": 1000}}, overwrite=True)

    def stage_run(self, directory):
        directory = Path(directory).resolve()
        directory.mkdir()
        for folder in ("jobs", "results", "receipts", "logs", "families", "profiles"):
            (directory / folder).mkdir()
        config = stage._config(20261001, 900, 900, 900, False)
        fingerprint, sources = stage.code_fingerprint()
        run = {"schema_version": 1, "stage": "B", "mode": "formal", "run_id": "test-stage-b",
               "run_dir": str(directory), "status": "running", "phase": "reference", "images": self.performance,
               "config": config, "config_sha256": stage.canonical_hash(config), "code_fingerprint": fingerprint,
               "source_hashes": sources, "manifest_path": str(self.manifest_path),
               "manifest_sha256": self.manifest["manifest_sha256"], "source_path": self.manifest["source_path"],
               "source_sha256": self.manifest["source_sha256"], "dataset_completion_sha256": self.manifest["dataset_completion_sha256"],
               "environment": stage.environment_info(), "fp32_snapshot_sha256": H, "fp32_prepared": self.prepared,
               "artifacts": {}, "failures": [], "family_files": {}}
        stage.atomic_json(directory / "run.json", run)
        return run

    def family(self, model, directory):
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        calibration = sorted((row for row in self.manifest["images"] if row["split"] == "calibration"), key=lambda row: row["id"])
        import random
        random.Random(20261001).shuffle(calibration)
        variants = {}
        for variant in stage.FAMILY_VARIANTS:
            path = directory / f"{variant}.onnx"
            if not path.exists():
                path.write_bytes(f"TEST MOCK {model}.{variant}".encode())
            variants[variant] = {"status": "prepared", "path": str(path), "onnx_sha256": stage.sha256(path),
                                 "size_bytes": path.stat().st_size, "validation": {"passed": True}, "graph_statistics": {},
                                 "config": {"parent_onnx_sha256": self.prepared[model]["onnx_sha256"] if variant == "QB" else variants["QB"]["onnx_sha256"]}}
        family = {"schema_version": 1, "model_id": model, "manifest_sha256": self.manifest["manifest_sha256"],
                  "input_model": {"path": self.prepared[model]["onnx_path"], "sha256": self.prepared[model]["onnx_sha256"]},
                  "calibration": {"count": 256, "image_ids": [row["id"] for row in calibration],
                                  "images": [{key: row[key] for key in ("id", "sha256", "source_id")} for row in calibration]},
                  "variants": variants, "status": "prepared"}
        if not (directory / "family.json").exists():
            stage.atomic_json(directory / "family.json", family)
        return family

    def measurement(self, job, pid=321):
        result = result_base(job, pid=pid)
        result["status"] = "passed"
        result["validation"] = {"passed": True, "numerical_equivalence_passed": True}
        result["payload"] = {"artifact_sha256": job["artifact_sha256"], "providers": ["CPUExecutionProvider"],
                             "intra_op_threads": job["threads"], "inter_op_threads": 1,
                             "execution_mode": "ORT_SEQUENTIAL", "graph_optimization": "ORT_ENABLE_ALL", "allow_spinning": 0}
        result["samples"] = []
        for index, image_index in enumerate(job["image_sequence"]):
            image = job["images"][image_index]
            result["samples"].append({"sequence_index": index, "input_index": image_index,
                                       "image_id": image["id"], "image_sha256": image["sha256"], "input_sha256": C,
                                       "inference_ms": 10.0})
        stage._write_samples(job["output_path"], result)
        return result


class PureTests(unittest.TestCase):
    def test_lightweight_import_and_help(self):
        code = "import sys; import src.vision.classification_stage_b; assert not set(('torch','numpy','onnxruntime','PIL')).intersection(sys.modules)"
        result = subprocess.run([sys.executable, "-B", "-S", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "tools/run_classifier_stage_b.py"), "--help"],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--fp32-run", result.stdout)
        self.assertIn("--holdout-timeout-s", result.stdout)
        self.assertNotIn("[--worker]", result.stdout)
        self.assertNotIn("--job JOB", result.stdout)

    def test_no_stage_a_white_list_extension(self):
        from src.vision.classification_benchmark import load_config, expand_experiments
        configurations = expand_experiments(load_config())
        self.assertEqual(len(configurations), 30)
        self.assertFalse(set(stage.FAMILY_VARIANTS).intersection(row["variant_id"] for row in configurations))

    def test_fixed_sequences_are_paired_and_rounds_distinct(self):
        self.assertEqual(stage.fixed_sequence(1, 0, 200), stage.fixed_sequence(1, 0, 200))
        self.assertNotEqual(stage.fixed_sequence(1, 0, 200), stage.fixed_sequence(20001, 0, 200))
        self.assertEqual(set(stage.fixed_sequence(1, 0, 200)), set(range(64)))

    def test_timing_statistics_failure_has_no_zero(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")], [True]):
            with self.subTest(values=values), self.assertRaises(stage.StageBError):
                stage.describe(values)
        self.assertEqual(stage.describe([1, 2, 3])["p50"], 2)

    def test_paired_block_speedup_and_ci(self):
        result = stage.paired_performance(process_blocks(10), process_blocks(5, start_pid=200))
        self.assertAlmostEqual(result["speedup"], 2)
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["ci95"], {"low": 2.0, "high": 2.0})
        self.assertEqual(result["resampling_unit"], "paired process block")

    def test_recycled_pid_is_accepted_only_with_distinct_os_process_instances(self):
        left, right = process_blocks(10), process_blocks(5, start_pid=200)
        right[0].update(fake_identity(left[0]["worker_pid"], left[0]["worker_creation_ticks"] + 10))
        self.assertTrue(stage.paired_performance(left, right)["gate_passed"])
        right[0].update(fake_identity(left[0]["worker_pid"], left[0]["worker_creation_ticks"]))
        self.assertFalse(stage.paired_performance(left, right)["gate_passed"])
        right = process_blocks(5, start_pid=200)
        right[0].pop("worker_identity")
        self.assertFalse(stage.paired_performance(left, right)["gate_passed"])

    def test_cross_model_round_sequence_pid_and_failure_excluded(self):
        baseline = process_blocks()
        for field, value in (("model_id", stage.MODELS[1]), ("round_id", "old-round"),
                             ("image_sequence", []), ("worker_pid", baseline[0]["worker_pid"]), ("status", "failed_runtime")):
            right = process_blocks(5, start_pid=200)
            right[0][field] = value
            with self.subTest(field=field):
                result = stage.paired_performance(baseline, right)
                self.assertFalse(result["gate_passed"])
                self.assertIsNone(result["speedup"])
        self.assertFalse(stage.paired_performance(baseline[:4], process_blocks())['gate_passed'])

    def test_degradation_and_tail_failure_do_not_fakepass(self):
        result = stage.paired_performance(process_blocks(10), process_blocks(12, start_pid=200))
        self.assertFalse(result["gate_passed"])
        self.assertGreater(result["p95_ratio"], 1.05)
        baseline, right = process_blocks(10), process_blocks(5, start_pid=200)
        for block in right:
            for sample in block["samples"][:30]:
                sample["inference_ms"] = 100
        result = stage.paired_performance(baseline, right)
        self.assertGreater(result["speedup"], 1.05)
        self.assertFalse(result["gate_passed"])

    def test_accuracy_budget_boundary_and_uncertainty(self):
        self.assertTrue(stage.accuracy_gate(accuracy(drop=1, ci=[0, 1]))["passed"])
        self.assertEqual(stage.accuracy_gate(accuracy(drop=.5, ci=[-.5, 1.1]))["status"], "inconclusive")
        self.assertEqual(stage.accuracy_gate(accuracy(drop=2, ci=[1.5, 3]))["status"], "failed")
        self.assertFalse(stage.accuracy_gate(accuracy("holdout"))["passed"])
        self.assertFalse(stage.accuracy_gate({"paired": True, "split": "development", "budget_passed": True})["passed"])

    def test_selection_accepts_confirmed_integer_events_list(self):
        chosen = stage.select_candidate(stage.MODELS[0], [candidate("O1"), candidate()])
        self.assertEqual(chosen["candidate"]["variant_id"], "Q2")
        self.assertTrue(chosen["int8_screening_passed"])
        self.assertEqual(chosen["selection_basis"], "development_only")

    def test_selection_is_insensitive_to_holdout_fields(self):
        rows = [candidate("O1"), candidate()]
        first = stage.select_candidate(stage.MODELS[0], rows)
        for row in rows:
            row["holdout"] = {"top1_accuracy": 0 if row["variant_id"] == "Q2" else 1}
        self.assertEqual(first["candidate"]["id"], stage.select_candidate(stage.MODELS[0], rows)["candidate"]["id"])
        rows[1]["development"]["split"] = "holdout"
        self.assertEqual(stage.select_candidate(stage.MODELS[0], rows)["candidate"]["variant_id"], "O1")

    def test_negative_int8_falls_back_to_legal_fp32(self):
        for update in ({"status": "unsupported"}, {"validation_passed": False},
                       {"development_comparison": accuracy(drop=2)},
                       {"development_comparison": accuracy(drop=.5, ci=[0, 2])},
                       {"runtime_profile": {"int8_operator_events": [], "int8_operator_counts": {}, "providers": ["CPUExecutionProvider"]}},
                       {"performance": {"status": "available", "speedup": 3, "ci95": {"low": .9}, "p95_ratio": .5, "gate_passed": True}}):
            quant = candidate()
            quant.update(update)
            fp32 = candidate("O1", speedup=1.0)
            fp32["performance"]["gate_passed"] = False
            with self.subTest(update=update):
                selection = stage.select_candidate(stage.MODELS[0], [fp32, quant])
                self.assertEqual(selection["status"], "retained_fp32")
                self.assertFalse(selection["int8_screening_passed"])
                self.assertEqual(selection["candidate"]["variant_id"], "O1")

    def test_no_legal_fp32_means_no_frozen_candidate(self):
        row = candidate("O1")
        row["status"] = "failed_correctness"
        self.assertIsNone(stage.select_candidate(stage.MODELS[0], [row])["candidate"])

    def test_wrong_baseline_or_artifact_never_enters_selection(self):
        row = candidate()
        row["baseline_variant_id"] = "O1"
        with self.assertRaisesRegex(stage.StageBError, "QB"):
            stage.select_candidate(stage.MODELS[0], [row])
        row = candidate()
        row["development"]["artifact_sha256"] = H
        with self.assertRaisesRegex(stage.StageBError, "binding"):
            stage.select_candidate(stage.MODELS[0], [row])

    def test_candidate_ranking_uses_latency_not_different_baseline_ratios(self):
        fp32 = candidate("O1", 2, speedup=1.2)
        quant = candidate("Q2", speedup=2.0)
        # A giant QB denominator must not make a slower integer model win.
        fp32["performance"]["candidate_stats"]["p50"] = 8
        quant["performance"]["candidate_stats"]["p50"] = 20
        baseline = candidate("O1", 1, speedup=1.0)
        selected = stage.select_candidate(stage.MODELS[0], [baseline, fp32, quant])
        self.assertEqual(selected["candidate"]["id"], fp32["id"])
        self.assertFalse(selected["int8_screening_passed"])
        self.assertEqual(selected["qualified_int8_count"], 1)

    def test_selection_rejects_confirmation_round_and_bool_threads(self):
        row = candidate()
        row["phase"] = "confirmation"
        self.assertIsNone(stage.select_candidate(stage.MODELS[0], [row])["candidate"])
        row = candidate()
        row["threads"] = True
        with self.assertRaises(stage.StageBError):
            stage.select_candidate(stage.MODELS[0], [row])

    def test_finite_cli_timeout_and_no_hidden_mock(self):
        self.assertEqual(cli._positive("900"), 900)
        for text in ("0", "-1", "inf", "86401"):
            with self.assertRaises(Exception):
                cli._positive(text)
        self.assertEqual(cli.main(["--manifest", "missing", "--fp32-run", "missing", "--output-dir", "unused"]), 2)

    def test_int8_evidence_status_all_failed_runtime(self):
        q1 = candidate("Q1", speedup=1.3)
        q1["status"] = "failed_runtime"
        q1["validation_passed"] = False
        q2 = candidate("Q2", speedup=1.3)
        q2["status"] = "failed_runtime"
        q2["validation_passed"] = False
        q3 = candidate("Q3", speedup=1.3)
        q3["status"] = "failed_runtime"
        q3["validation_passed"] = False
        fp32_base = candidate("O1", 1, speedup=1.0)
        fp32_fast = candidate("O1", 4, speedup=1.4)
        fp32_fast["performance"]["candidate_stats"]["p50"] = 7.0

        selected = stage.select_candidate(stage.MODELS[0], [fp32_base, fp32_fast, q1, q2, q3])
        self.assertEqual(selected["int8_evidence_status"], "failed_runtime")
        self.assertFalse(selected["int8_screening_passed"])
        self.assertEqual(selected["candidate"]["id"], fp32_fast["id"])
        self.assertNotIn("No INT8 adoption benefit", selected["reason"])
        self.assertIn("failed runtime preparation/execution without evaluation", selected["reason"])

    def test_int8_evidence_status_all_unsupported(self):
        q1 = candidate("Q1")
        q1["status"] = "unsupported"
        q1["validation_passed"] = False
        q2 = candidate("Q2")
        q2["status"] = "unsupported"
        q2["validation_passed"] = False
        q3 = candidate("Q3")
        q3["status"] = "unsupported"
        q3["validation_passed"] = False
        fp32_base = candidate("O1", 1, speedup=1.0)

        selected = stage.select_candidate(stage.MODELS[0], [fp32_base, q1, q2, q3])
        self.assertEqual(selected["int8_evidence_status"], "unsupported")
        self.assertFalse(selected["int8_screening_passed"])
        self.assertEqual(selected["candidate"]["id"], fp32_base["id"])
        self.assertNotIn("No INT8 adoption benefit", selected["reason"])
        self.assertIn("unsupported", selected["reason"])

    def test_int8_evidence_status_truly_evaluated_negative(self):
        q1 = candidate("Q1", speedup=1.3)
        q1["development_comparison"] = accuracy(drop=2.0)  # dropped too much accuracy
        q2 = candidate("Q2", speedup=1.3)
        q2["runtime_profile"] = {"int8_operator_events": [], "int8_operator_counts": {}, "providers": ["CPUExecutionProvider"]}  # no int8 ops
        q3 = candidate("Q3", speedup=1.02)
        q3["performance"] = {"status": "available", "speedup": 1.02, "ci95": {"low": 0.99}, "p95_ratio": 1.0, "gate_passed": False}  # speedup < 1.05
        fp32_base = candidate("O1", 1, speedup=1.0)

        selected = stage.select_candidate(stage.MODELS[0], [fp32_base, q1, q2, q3])
        self.assertEqual(selected["int8_evidence_status"], "evaluated_no_benefit")
        self.assertFalse(selected["int8_screening_passed"])
        self.assertEqual(selected["candidate"]["id"], fp32_base["id"])
        self.assertIn("No INT8 adoption benefit", selected["reason"])

    def test_int8_evidence_status_partial_failure(self):
        q1 = candidate("Q1", speedup=1.3)
        q1["status"] = "failed_runtime"
        q1["validation_passed"] = False
        q2 = candidate("Q2")
        q2["status"] = "unsupported"
        q2["validation_passed"] = False
        q3 = candidate("Q3", speedup=1.3)
        q3["development_comparison"] = accuracy(drop=2.5)  # evaluated but failed accuracy gate
        fp32_base = candidate("O1", 1, speedup=1.0)

        selected = stage.select_candidate(stage.MODELS[0], [fp32_base, q1, q2, q3])
        self.assertEqual(selected["int8_evidence_status"], "partial_failure")
        self.assertFalse(selected["int8_screening_passed"])
        self.assertEqual(selected["candidate"]["id"], fp32_base["id"])
        self.assertNotIn("No INT8 adoption benefit", selected["reason"])
        self.assertIn("incomplete", selected["reason"])

    def test_int8_evidence_status_qualified_int8(self):
        # Case A: qualified INT8 is faster than FP32
        q2 = candidate("Q2", speedup=1.5)
        q2["performance"]["candidate_stats"]["p50"] = 6.0
        fp32_base = candidate("O1", 1, speedup=1.0)
        fp32_base["performance"]["candidate_stats"]["p50"] = 10.0

        selected = stage.select_candidate(stage.MODELS[0], [fp32_base, q2])
        self.assertEqual(selected["int8_evidence_status"], "qualified_available")
        self.assertTrue(selected["int8_screening_passed"])
        self.assertEqual(selected["candidate"]["id"], q2["id"])
        self.assertIn("INT8 gates passed", selected["reason"])

        # Case B: qualified INT8 exists, but FP32 candidate is even faster
        fp32_fast = candidate("O1", 4, speedup=2.0)
        fp32_fast["performance"]["candidate_stats"]["p50"] = 4.0
        selected_fp32 = stage.select_candidate(stage.MODELS[0], [fp32_base, fp32_fast, q2])
        self.assertEqual(selected_fp32["int8_evidence_status"], "qualified_available")
        self.assertFalse(selected_fp32["int8_screening_passed"])
        self.assertEqual(selected_fp32["candidate"]["id"], fp32_fast["id"])
        self.assertIn("lower nominal P50 than qualified INT8", selected_fp32["reason"])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fixture = Fixture(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_data_counts_null_calibration_and_full1000_scope(self):
        manifest = self.fixture.manifest
        self.assertEqual(manifest["dataset_scope"]["class_count"], 20)
        self.assertEqual(manifest["dataset_scope"]["output_class_count"], 1000)
        self.assertFalse(manifest["dataset_scope"]["is_full_imagenet_validation"])
        self.assertFalse(manifest["dataset_scope"]["pretraining_unseen_proven"])
        self.assertTrue(all(row["label"] is None for row in manifest["images"] if row["split"] == "calibration"))
        labels = {row["label"] for row in manifest["images"] if row["split"] == "development"}
        self.assertEqual(labels, {row["imagenet_index"] for row in self.fixture.classes})
        self.assertGreater(min(labels), 20)

    def test_manifest_audit_does_not_read_holdout_bytes(self):
        actual = stage.sha256
        def guard(path):
            self.assertFalse(Path(path).name.startswith("holdout-"), "Holdout byte opened before freeze")
            return actual(path)
        with patch.object(stage, "sha256", side_effect=guard):
            stage.load_stage_b_manifest(self.fixture.manifest_path)

    def test_pinned_mirror_truth_revision_lfs_and_full1000_mapping_are_strict(self):
        fixture = self.fixture
        mirror = {"dataset_id": "unittest/pets", "revision": "d" * 40, "card_license": fixture.source["license"],
                  "label_names": {str(index): row["prefix"] for index, row in enumerate(fixture.classes)},
                  "files": {split: {"mirror_relative_path": f"{split}.parquet", "row_count": 2000, "split": split,
                                     "sha256_expected": H, "sha256_observed": H,
                                     "sha256_matches_lfs_reference": True, "unchanged_during_preparation": True}
                            for split in ("train", "test")}}
        source = copy.deepcopy(fixture.source)
        source["mirror"] = mirror
        source["subset"]["original_split_roles"] = {"train": ["calibration", "development"], "test": ["holdout"]}
        manifest = copy.deepcopy(fixture.manifest)
        for index, entry in enumerate(manifest["images"]):
            split = "test" if entry["split"] == "holdout" else "train"
            breed_index = next(i for i, row in enumerate(fixture.classes) if row["prefix"] == entry["breed"])
            truth = f"mirror:unittest/pets@{'d' * 40}:{split}.parquet#row={index}"
            entry.update(source_original_split=split, ground_truth_source=truth,
                original_annotation={"mirror_row_index": index, "label": breed_index, "label_cat_dog": 1,
                    "label_name": entry["breed"], "image_id": entry["id"], "source_original_split": split,
                    "ground_truth_source": truth})
        actual_read = stage.read_json
        def read(path):
            return source if Path(path) == fixture.dataset / "source.json" else actual_read(path)
        with patch("src.vision.classification_benchmark.load_manifest", return_value=manifest), \
             patch.object(stage, "read_json", side_effect=read):
            result = stage.load_stage_b_manifest(fixture.manifest_path, check_files=False)
            self.assertEqual(result["dataset_scope"]["output_class_count"], 1000)
            mirror["files"]["test"]["sha256_observed"] = C
            with self.assertRaisesRegex(stage.StageBError, "LFS"):
                stage.load_stage_b_manifest(fixture.manifest_path, check_files=False)
            mirror["files"]["test"]["sha256_observed"] = H
            manifest["images"][0]["original_annotation"]["label_name"] = "model-predicted-breed"
            with self.assertRaisesRegex(stage.StageBError, "truth"):
                stage.load_stage_b_manifest(fixture.manifest_path, check_files=False)

    def test_manifest_bad_label_scope_or_source_hash_is_rejected(self):
        original = stage.read_json(self.fixture.manifest_path)
        for index, label in ((0, 151), (256, 0), (256, None)):
            changed = copy.deepcopy(original)
            changed["images"][index]["label"] = label
            with patch("src.vision.classification_benchmark.load_manifest", return_value={
                    **changed, "resolved_root": str(self.fixture.dataset), "manifest_sha256": H}):
                with self.assertRaises(stage.StageBError):
                    stage.load_stage_b_manifest(self.fixture.manifest_path, check_files=False)

    def test_fp32_smoke_incomplete_budget_and_manifest_are_rejected(self):
        for update in ({"mode": "smoke"}, {"status": "running"}, {"manifest_sha256": H},
                       {"config": {"formal": {"warmup": 3, "iterations": 10, "repeats": 2}}}):
            changed = {**self.fixture.fp32_run, **update}
            original_read = stage.read_json
            def read(path):
                return changed if Path(path) == self.fixture.fp32_dir / "run.json" else original_read(path)
            with self.subTest(update=update), patch.object(stage, "read_json", side_effect=read):
                with self.assertRaises(stage.StageBError):
                    stage.audit_fp32_run(self.fixture.fp32_dir, self.fixture.manifest, allow_failures=True)

    def test_fp32_good_fixture_and_explicit_failures(self):
        with patch("src.vision.classification_benchmark.aggregate_results", return_value=self.fixture.summary):
            audit = stage.audit_fp32_run(self.fixture.fp32_dir, self.fixture.manifest)
        self.assertEqual(set(audit["run"]["prepared_models"]), set(stage.MODELS))
        self.assertGreater(len(audit["dependencies"]), 9)
        summary = copy.deepcopy(self.fixture.summary)
        summary["experiments"][0]["status"] = "failed_correctness"
        with patch("src.vision.classification_benchmark.aggregate_results", return_value=summary):
            with self.assertRaisesRegex(stage.StageBError, "must pass"):
                stage.audit_fp32_run(self.fixture.fp32_dir, self.fixture.manifest)
            audit = stage.audit_fp32_run(self.fixture.fp32_dir, self.fixture.manifest, allow_failures=True)
        self.assertEqual(len(audit["visible_failures"]), 1)

    def test_cached_one_image_sidecar_requires_all_current_64_image_onnx_checks(self):
        fixture = self.fixture
        self.assertTrue(all(item["export_metadata"]["validation"]["verified_images_count"] == 1
                            for item in fixture.prepared.values()))
        with patch("src.vision.classification_benchmark.aggregate_results", return_value=fixture.summary):
            stage.audit_fp32_run(fixture.fp32_dir, fixture.manifest)
        for problem in ("one_image", "missing_item", "failed_item", "swapped_item", "missing_block"):
            summary = copy.deepcopy(fixture.summary)
            record = next(row for row in summary["records"] if row["data"]["experiment"]["variant_id"] == "O0"
                          and row["data"]["experiment"]["model_id"] == stage.MODELS[0])
            validation = record["data"]["validation"]
            if problem == "one_image":
                validation["image_count"] = 1
                validation["per_image"] = validation["per_image"][:1]
            elif problem == "missing_item":
                validation["per_image"].pop()
            elif problem == "failed_item":
                validation["per_image"][1]["allclose"] = False
            elif problem == "swapped_item":
                validation["per_image"][0], validation["per_image"][1] = validation["per_image"][1], validation["per_image"][0]
            else:
                summary["records"].remove(record)
            with self.subTest(problem=problem), patch("src.vision.classification_benchmark.aggregate_results", return_value=summary):
                with self.assertRaises(stage.StageBError):
                    stage.audit_fp32_run(fixture.fp32_dir, fixture.manifest, allow_failures=True)

    def test_current_onnx_proof_is_bound_to_job_image_order_and_physical_reference(self):
        fixture = self.fixture
        for problem in ("images", "reference_sha", "reference_path"):
            summary = copy.deepcopy(fixture.summary)
            record = next(row for row in summary["records"] if row["data"]["experiment"]["variant_id"] == "O1")
            job_path = fixture.fp32_dir / "jobs" / Path(record["source_file"]).name
            changed_job = copy.deepcopy(stage.read_json(job_path))
            if problem == "images":
                changed_job["images"][0], changed_job["images"][1] = changed_job["images"][1], changed_job["images"][0]
            elif problem == "reference_sha":
                changed_job["reference_sha256"] = H
            else:
                changed_job["reference_path"] = str(fixture.root / "unbound-reference.npz")
            record["data"]["job_sha256"] = stage.canonical_hash(changed_job)
            actual_read = stage.read_json
            def read(path):
                return changed_job if Path(path) == job_path else actual_read(path)
            with self.subTest(problem=problem), patch.object(stage, "read_json", side_effect=read), \
                 patch("src.vision.classification_benchmark.aggregate_results", return_value=summary):
                with self.assertRaises(stage.StageBError):
                    stage.audit_fp32_run(fixture.fp32_dir, fixture.manifest)

    def test_fp32_allow_failure_never_ignores_corruption_or_missing(self):
        for update in ({"integrity_errors": [{"issues": "bad"}]}, {"status": "incomplete"}):
            summary = {**self.fixture.summary, **update}
            with patch("src.vision.classification_benchmark.aggregate_results", return_value=summary):
                with self.assertRaises(stage.StageBError):
                    stage.audit_fp32_run(self.fixture.fp32_dir, self.fixture.manifest, allow_failures=True)

    def test_m4_ledger_missing_identity_unstable_group_or_unaccepted_measurement_rejected(self):
        fixture = self.fixture
        ledger_path = fixture.fp32_dir / "controls" / "ledger.json"
        original = stage.read_json(ledger_path)
        for kind in ("identity", "missing_group", "unstable"):
            changed = copy.deepcopy(original)
            if kind == "identity":
                changed["manifest_sha256"] = H
            elif kind == "missing_group":
                changed["groups"].pop(next(iter(changed["groups"])))
            else:
                changed["groups"][next(iter(changed["groups"]))]["status"] = "unstable"
            actual_read = stage.read_json
            def read(path):
                return changed if Path(path) == ledger_path else actual_read(path)
            with self.subTest(kind=kind), patch.object(stage, "read_json", side_effect=read):
                with self.assertRaises(stage.StageBError):
                    stage.audit_fp32_controls(fixture.fp32_dir, fixture.fp32_run, fixture.summary["records"])
        records = copy.deepcopy(fixture.summary["records"])
        records[0]["data"]["environment_accepted"] = False
        with self.assertRaisesRegex(stage.StageBError, "environment_accepted"):
            stage.audit_fp32_controls(fixture.fp32_dir, fixture.fp32_run, records)
        with self.assertRaisesRegex(stage.StageBError, "complete"):
            stage.audit_fp32_controls(fixture.fp32_dir, fixture.fp32_run, fixture.summary["records"][:-1])

    def test_freeze_requires_all_three_and_real_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            selections, families = [], {}
            for model in stage.MODELS:
                family = self.fixture.family(model, directory / model)
                families[model] = directory / model / "family.json"
                row = candidate(model=model)
                row.update(artifact_path=family["variants"]["Q2"]["path"], artifact_sha256=family["variants"]["Q2"]["onnx_sha256"],
                           baseline_artifact_path=family["variants"]["QB"]["path"], baseline_artifact_sha256=family["variants"]["QB"]["onnx_sha256"])
                selections.append({"candidate": row})
            with self.assertRaises(stage.StageBError):
                stage.freeze_candidates(selections[:2], self.fixture.manifest["manifest_sha256"], H, families, directory / "partial.json")
            value, digest = stage.freeze_candidates(selections, self.fixture.manifest["manifest_sha256"], H, families, directory / "frozen.json")
            self.assertEqual(value["status"], "frozen")
            self.assertEqual(stage.sha256(directory / "frozen.json"), digest)
            stage.validate_frozen(value, manifest_sha256=self.fixture.manifest["manifest_sha256"], code=H, family_files=families)
            chosen = value["candidates"][0]
            guard_job = {"kind": "evaluate", "model_id": chosen["model_id"], "role": "candidate",
                         "variant_id": chosen["variant_id"], "threads": chosen["threads"],
                         "artifact_sha256": chosen["artifact_sha256"], "manifest_sha256": value["manifest_sha256"],
                         "frozen_path": str(directory / "frozen.json"), "frozen_sha256": digest}
            stage._guard_frozen_job(guard_job, {"family_files": families}, H)
            with self.assertRaisesRegex(stage.StageBError, "role"):
                stage._guard_frozen_job({**guard_job, "artifact_sha256": chosen["baseline_artifact_sha256"]},
                                       {"family_files": families}, H)
            changed = copy.deepcopy(value)
            changed["candidates"][0]["threads"] = True
            with self.assertRaises(stage.StageBError):
                stage.validate_frozen(changed, manifest_sha256=self.fixture.manifest["manifest_sha256"], check_files=False)
            changed = copy.deepcopy(value)
            changed["candidates"][0]["selection_basis"] = "holdout"
            with self.assertRaises(stage.StageBError):
                stage.validate_frozen(changed, manifest_sha256=self.fixture.manifest["manifest_sha256"], check_files=False)
            Path(value["candidates"][0]["artifact_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(stage.StageBError, "SHA"):
                stage.validate_frozen(value, manifest_sha256=self.fixture.manifest["manifest_sha256"])

    def test_holdout_never_launches_without_unified_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run = self.fixture.stage_run(root)
            job = stage.make_job(run, root, stage.MODELS[0], "O1", 1, kind="evaluate", phase="holdout",
                                 artifact=stage._raw_artifact(run, stage.MODELS[0]), split="holdout")
            dispatcher = stage.Dispatcher(root, run, self.fixture.manifest)
            with patch.object(stage.subprocess, "Popen") as spawn:
                with self.assertRaises(stage.StageBError):
                    dispatcher.execute(job, 900)
                spawn.assert_not_called()

    def test_job_sha_pid_sequence_and_raw_jsonl_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run = self.fixture.stage_run(root)
            job = stage.make_job(run, root, stage.MODELS[0], "O1", 1, kind="measure", phase="screening",
                                 artifact=stage._raw_artifact(run, stage.MODELS[0]), budget=stage.SCREENING)
            result = self.fixture.measurement(job)
            stage.validate_result(job, result, expected_pid=321, input_hashes={row["id"]: C for row in run["images"]})
            for field, value in (("worker_pid", 99), ("job_sha256", H), ("input_sequence_sha256", H)):
                changed = {**result, field: value}
                with self.subTest(field=field), self.assertRaises(stage.StageBError):
                    stage.validate_result(job, changed, expected_pid=321)
            changed = copy.deepcopy(result)
            changed["samples"][0]["image_sha256"] = H
            with self.assertRaisesRegex(stage.StageBError, "SHA"):
                stage.validate_result(job, changed, expected_pid=321)
            with self.assertRaisesRegex(stage.StageBError, "normalization"):
                stage.validate_result(job, result, expected_pid=321, input_hashes={row["id"]: H for row in run["images"]})
            Path(result["samples_path"]).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(stage.StageBError, "SHA"):
                stage.validate_result(job, result, expected_pid=321)

    def test_resume_wrong_fingerprint_and_artifact_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run = self.fixture.stage_run(root)
            arguments = {"config": run["config"], "manifest": self.fixture.manifest, "audit": self.fixture.audit,
                         "fingerprint": run["code_fingerprint"], "environment": run["environment"]}
            empty_log = root / "logs" / "silent-worker.log"
            empty_log.write_bytes(b"")
            run["artifacts"][str(empty_log)] = stage.sha256(empty_log)
            stage.validate_resume(run, **arguments)
            for key in ("code_fingerprint", "manifest_sha256", "source_sha256", "fp32_snapshot_sha256"):
                with self.subTest(key=key), self.assertRaisesRegex(stage.StageBError, "fingerprint"):
                    stage.validate_resume({**run, key: C}, **arguments)
            artifact = root / "immutable.json"
            artifact.write_bytes(b"old")
            run["artifacts"][str(artifact)] = stage.sha256(artifact)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(stage.StageBError, "SHA"):
                stage.validate_resume(run, **arguments)

    def test_failed_result_never_counts_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run = self.fixture.stage_run(root)
            job = stage.make_job(run, root, stage.MODELS[0], "O1", 1, kind="measure", phase="screening",
                                 artifact=stage._raw_artifact(run, stage.MODELS[0]), budget=stage.SCREENING)
            result = result_base(job, pid=1)
            result.update(status="failed_runtime", error="unsupported kernel", samples=[])
            stage.validate_result(job, result, expected_pid=1)
            self.assertIsNone(result["payload"])
            self.assertFalse(stage.paired_performance([result], [result])["gate_passed"])

    def test_evaluator_labels_and_full1000_ranking_are_rechecked(self):
        manifest = self.fixture.manifest
        images = sorted((row for row in manifest["images"] if row["split"] == "development"), key=lambda row: row["id"])
        rows = []
        for image in images:
            label = image["label"]
            rows.append({"id": image["id"], "label": label, "sha256": image["sha256"], "top1": label,
                         "top5": [label, 0, 1, 2, 999], "top1_correct": True, "top5_correct": True})
        job = {"split": "development", "manifest_sha256": manifest["manifest_sha256"], "artifact_sha256": H,
               "model_id": stage.MODELS[0], "threads": 2}
        value = {**job, "image_count": 128, "per_image": rows, "top1_accuracy": 1, "top5_accuracy": 1,
                 "session": {"threads": 2, "providers": ["CPUExecutionProvider"]}}
        stage._evaluation_check(value, job, manifest)
        changed = copy.deepcopy(value)
        changed["per_image"][0]["label"] = 0
        with self.assertRaisesRegex(stage.StageBError, "label"):
            stage._evaluation_check(changed, job, manifest)
        changed = copy.deepcopy(value)
        changed["per_image"][0]["top5"][4] = 1000
        with self.assertRaises(stage.StageBError):
            stage._evaluation_check(changed, job, manifest)
        changed = {**value, "top1_accuracy": .99}
        with self.assertRaisesRegex(stage.StageBError, "summary"):
            stage._evaluation_check(changed, job, manifest)

    def test_family_parent_graph_calibration_order_and_validation_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = stage.MODELS[0]
            family = self.fixture.family(model, directory / model)
            job = {"model_id": model, "manifest_sha256": self.fixture.manifest["manifest_sha256"],
                   "raw_onnx_sha256": self.fixture.prepared[model]["onnx_sha256"], "seed": 20261001,
                   "family_output_dir": str(directory / model)}
            stage._family_check(family, job, self.fixture.manifest)
            for change in ("parent", "calibration", "validation"):
                altered = copy.deepcopy(family)
                if change == "parent":
                    altered["variants"]["Q2"]["config"]["parent_onnx_sha256"] = H
                elif change == "calibration":
                    altered["calibration"]["image_ids"] = list(reversed(altered["calibration"]["image_ids"]))
                else:
                    altered["variants"]["QB"]["validation"]["passed"] = False
                with self.subTest(change=change), self.assertRaises(stage.StageBError):
                    stage._family_check(altered, job, self.fixture.manifest)

    def test_job_guard_refuses_artifact_change_or_escaping_output_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run = self.fixture.stage_run(root)
            job = stage.make_job(run, root, stage.MODELS[0], "O1", 1, kind="reference", phase="reference")
            path = root / "jobs" / f"{job['job_name']}.json"
            stage.atomic_json(path, job)
            stage.validate_job(path, job)
            with self.assertRaisesRegex(stage.StageBError, "output"):
                stage.validate_job(path, {**job, "output_path": str(root / "escape.json")})
            with self.assertRaisesRegex(stage.StageBError, "SHA"):
                stage.validate_job(path, {**job, "reference_sha256": H})
            with self.assertRaisesRegex(stage.StageBError, "Stage B"):
                stage.validate_job(path, {**job, "schema_version": True})


class ObservationTests(unittest.TestCase):
    def tiny_job(self):
        return {"warmup": 20, "iterations": 200,
                "image_sequence": stage.fixed_sequence(1, 0, 200, count=4),
                "images": [{"id": str(index), "sha256": H} for index in range(4)]}

    def test_pipeline_injected_clock_measures_actual_wall_not_sum_of_medians(self):
        import itertools
        job = self.tiny_job()
        calls = []
        ticks = itertools.count(0, 1000000)
        def operation(name):
            return lambda value: calls.append(name) or value
        samples = stage.collect_pipeline_samples(job, list(range(4)), operation("prepare"), operation("adapt"),
                    operation("infer"), operation("full1000_top5"), [C] * 4, clock=lambda: next(ticks))
        self.assertEqual(len(samples), 200)
        self.assertEqual(len(calls), (20 + 200) * 4)
        self.assertEqual(calls[:4], ["prepare", "adapt", "infer", "full1000_top5"])
        self.assertEqual(samples[0]["pipeline_wall_ms"], 6)
        self.assertEqual(samples[0]["stages_total_ms"], 4)
        self.assertEqual(samples[0]["wrapper_overhead_ms"], 2)
        self.assertEqual(samples[0]["input_sha256"], C)
        self.assertEqual(samples[0]["image_id"], str(job["image_sequence"][0]))

    def test_pipeline_failed_inference_propagates_without_zero_samples(self):
        def fail(value):
            raise RuntimeError("test-only failed inference")
        with self.assertRaisesRegex(RuntimeError, "test-only"):
            stage.collect_pipeline_samples(self.tiny_job(), list(range(4)), lambda x: x, lambda x: x,
                                           fail, lambda x: x, [C] * 4)

    def test_memory_workload_and_native_sampler_values_are_not_latency(self):
        class Reader:
            def read(self):
                return 500
        class Sampler:
            values = [550, 900, 600]
            interval = .02
            error = None
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return None
        calls = []
        samples, loaded, values = stage.collect_memory_samples(self.tiny_job(), list(range(4)),
                                        lambda value: calls.append(value), [C] * 4, Reader(), Sampler())
        self.assertEqual(len(calls), 220)
        self.assertEqual(len(samples), 200)
        self.assertEqual(loaded, 500)
        self.assertEqual(values, [550, 900, 600])
        self.assertFalse(any(key.endswith("_ms") for sample in samples for key in sample))
        sampler = Sampler()
        sampler.error = "test-only native counter error"
        with self.assertRaisesRegex(stage.StageBError, "counter error"):
            stage.collect_memory_samples(self.tiny_job(), list(range(4)), lambda x: x, [C] * 4, Reader(), sampler)

    def test_rss_jsonl_sha_and_csv_observation_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "memory.json"
            trace = stage._write_rss_log(output, [10, 15, 11])
            self.assertEqual(stage.sha256(trace["path"]), trace["sha256"])
            rows = [json.loads(line) for line in Path(trace["path"]).read_text().splitlines()]
            self.assertEqual(rows[-1], {"sample_index": 2, "rss_bytes": 11})
            summary = {"rows": [], "observations": {stage.MODELS[0]: {"candidate": {
                "pipeline": {"status": "passed", "payload": {"pipeline_stats": {"pipeline_wall_ms": {"p50": 6}}}, "frozen_sha256": H},
                "memory": {"status": "passed", "payload": {"memory": {"rss_peak_bytes": 15}}, "frozen_sha256": H}}}}}
            table = list(csv.DictReader(io.StringIO(stage._report_csv(summary))))
            self.assertEqual(len(table), 2)
            self.assertEqual(table[0]["phase"], "observations")
            self.assertEqual(table[0]["p50_ms"], "")
            self.assertIn("pipeline_wall_ms", table[0]["pipeline_stages_and_wall"])
            self.assertIn("rss_peak_bytes", table[1]["independent_rss"])


class DriftRoundTests(unittest.TestCase):
    def run_case(self, always_unstable=False, missing_qb=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = Fixture(temporary.name)
        directory = Path(temporary.name) / "stage-b"
        run = fixture.stage_run(directory)
        model = stage.MODELS[0]
        family = fixture.family(model, directory / "families" / model)
        run["families"] = {model: family}
        if missing_qb:
            family["variants"]["QB"]["status"] = "unsupported"
        configs = [{"id": f"{model}.{variant}.t1", "model_id": model, "variant_id": variant, "threads": 1,
                    "artifact": family["variants"][variant]} for variant in ("QB", "Q2")]
        dispatcher = stage.Dispatcher(directory, run, fixture.manifest)
        calls = []
        def execute(job, timeout):
            calls.append(job)
            result = result_base(job, pid=1000 + len(calls))
            value = 10 if job["variant_id"] == "QB" else 5
            if job["kind"] == "control" and job["control_role"] == "after" and (always_unstable or job["attempt"] == 0):
                value = 12
            result.update(status="passed", validation={"passed": True}, payload={},
                samples=[{"image_id": job["images"][index]["id"], "image_sha256": job["images"][index]["sha256"],
                          "input_sha256": C, "inference_ms": value} for index in job["image_sequence"]])
            stage.atomic_json(job["output_path"], result)
            return result
        with patch.object(dispatcher, "execute", side_effect=execute):
            measurements = stage._measure_round(dispatcher, configs, "screening", stage.SCREENING)
        return directory, run, calls, measurements

    def test_one_whole_group_retry_preserves_rejects_and_only_accepts_new_attempt(self):
        directory, run, calls, measurements = self.run_case()
        self.assertEqual(len(calls), 5 * 2 * 4)
        self.assertTrue(all(record["attempt"] == 1 and record["environment_accepted"] is True
                            for records in measurements.values() for record in records))
        self.assertTrue((directory / "rejected-results").is_dir())
        self.assertTrue(all(len(ledger["attempts"]) == 2 and ledger["status"] == "accepted"
                            for ledger in run["drift_controls"].values()))
        results = list(measurements.values())
        self.assertTrue(stage.paired_performance(results[0], results[1])["gate_passed"])

    def test_second_instability_is_terminal_not_extra_successful_blocks(self):
        directory, run, calls, measurements = self.run_case(always_unstable=True)
        self.assertEqual(len(calls), 40)
        self.assertTrue(all(record["status"] == "failed_runtime" and record["samples"] == []
                            for records in measurements.values() for record in records))
        self.assertTrue(all(len(ledger["attempts"]) == 2 and ledger["status"] == "unstable"
                            for ledger in run["drift_controls"].values()))
        self.assertFalse(stage.paired_performance(*list(measurements.values()))["gate_passed"])

    def test_missing_qb_is_disclosed_without_fabricated_environment_acceptance(self):
        directory, run, calls, measurements = self.run_case(missing_qb=True)
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(record["environment_accepted"] is None for records in measurements.values() for record in records))
        self.assertTrue(all(ledger["status"] == "uncontrolled" for ledger in run["drift_controls"].values()))
        self.assertFalse(stage.paired_performance(*list(measurements.values()))["gate_passed"])


class ReportTests(unittest.TestCase):
    def summary(self):
        row = candidate()
        row.update(model_id="=HYPERLINK(\"bad\")", phase="screening", status="unsupported", errors="<script>alert(1)</script>",
                   development={"split": "development", "top1_accuracy": .5, "top5_accuracy": .8}, size_bytes=123)
        return {"schema_version": 1, "stage": "B", "run_id": "<script>&\"", "status": "completed_with_failures",
                "rows": [row], "final": {}, "dataset": {"scope": "20 breeds/full1000", "license": "<img src=x onerror=alert(1)>"},
                "fp32_stage_a": {"experiments": []}, "limitations": ["<bad>"], "failures": ["unsupported"]}

    def test_html_static_escape_csv_formula_and_no_original_images(self):
        summary = self.summary()
        rendered = stage._render_html(summary)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img ", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("Content-Security-Policy", rendered)
        self.assertIn("default-src 'none'", rendered)
        rows = list(csv.reader(io.StringIO(stage._report_csv(summary))))
        self.assertTrue(rows[1][1].startswith("'="))
        for value in ("=1", " +1", "\t@sum()", "\r-1", "\ufeff=1"):
            self.assertTrue(stage._csv_cell(value).startswith("'"))

    def test_report_refuses_mock_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.summary()
            files = stage.write_report(root, summary)
            before = {name: Path(path).read_bytes() for name, path in files.items()}
            with self.assertRaises(stage.StageBError):
                stage.write_report(root, summary)
            for name, path in files.items():
                self.assertEqual(Path(path).read_bytes(), before[name])
            summary["rows"][0]["evidence_kind"] = "test_mock"
            with self.assertRaisesRegex(stage.StageBError, "mock"):
                stage.write_report(root, summary, overwrite=True)

    def test_atomic_json_nonfinite_and_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            stage.atomic_json(path, {"old": True})
            with patch.object(stage.os, "replace", side_effect=OSError("publish failed")):
                with self.assertRaises(OSError):
                    stage.atomic_json(path, {"new": True}, overwrite=True)
            self.assertEqual(stage.read_json(path), {"old": True})
            self.assertFalse(list(Path(temporary).glob("*.tmp")))
            for value in (float("nan"), float("inf")):
                with self.assertRaises(stage.StageBError):
                    stage.atomic_json(path, {"bad": value}, overwrite=True)

    def test_report_bundle_rolls_back_and_marker_sha_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.summary()
            files = stage.write_report(root, summary)
            before = {name: Path(path).read_bytes() for name, path in files.items()}
            actual_replace = stage.os.replace
            calls = []
            def replace(source, target):
                calls.append(str(target))
                if len(calls) == 2:
                    raise OSError("test failure publishing html")
                return actual_replace(source, target)
            with patch.object(stage.os, "replace", side_effect=replace):
                with self.assertRaises(OSError):
                    stage.write_report(root, {**summary, "status": "changed"}, overwrite=True)
            self.assertEqual({name: Path(path).read_bytes() for name, path in files.items()}, before)
            marker = stage.read_json(root / "report.complete.json")
            for name, digest in marker["files"].items():
                self.assertEqual(stage.sha256(root / name), digest)
            self.assertFalse(list(root.glob(".report-stage-b-*")))


class DispatcherTests(unittest.TestCase):
    def test_dispatch_child_python_pid_and_failure_resume_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            directory = Path(temporary) / "run"
            run = fixture.stage_run(directory)
            job = stage.make_job(run, directory, stage.MODELS[0], "O1", 1, kind="reference", phase="reference")
            class FakeProcess:
                pid = 1357
                returncode = 2
                def wait(self, timeout):
                    result = result_base(job, pid=self.pid)
                    result.update(status="failed_runtime", error="explicit test-only child failure")
                    stage.atomic_json(job["output_path"], result)
                    return 2
            with patch("tools.run_classifier_benchmark.child_python", return_value=("TEST_PYTHON", {"TEST_ONLY": "1"})) as child, \
                 patch("tools.run_classifier_benchmark.worker_process_identity", side_effect=fake_identity), \
                 patch.object(stage.subprocess, "Popen", return_value=FakeProcess()) as spawn:
                dispatcher = stage.Dispatcher(directory, run, fixture.manifest)
                result = dispatcher.execute(job, 900)
                self.assertEqual(result["status"], "failed_runtime")
                self.assertEqual(len(run["failures"]), 1)
                child.assert_called_once_with(1)
                self.assertEqual(spawn.call_args.args[0][0], "TEST_PYTHON")
                self.assertIn("--worker", spawn.call_args.args[0])
                self.assertIn("--job", spawn.call_args.args[0])
                result = dispatcher.execute(job, 900)
                self.assertEqual(result["status"], "failed_runtime")
                self.assertEqual(len(run["failures"]), 1)
                self.assertEqual(spawn.call_count, 1)

    def test_fake_positive_child_exit_code_pid_binding_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            directory = Path(temporary) / "run"
            run = fixture.stage_run(directory)
            job = stage.make_job(run, directory, stage.MODELS[0], "O1", 1, kind="reference", phase="reference")
            class FakeProcess:
                pid = 1357
                returncode = 2
                def wait(self, timeout):
                    result = result_base(job, pid=999)
                    result.update(status="failed_runtime", error="wrong PID fixture")
                    stage.atomic_json(job["output_path"], result)
                    return 2
            with patch("tools.run_classifier_benchmark.child_python", return_value=("TEST_PYTHON", {})), \
                 patch("tools.run_classifier_benchmark.worker_process_identity", side_effect=fake_identity), \
                 patch.object(stage.subprocess, "Popen", return_value=FakeProcess()):
                dispatcher = stage.Dispatcher(directory, run, fixture.manifest)
                with self.assertRaisesRegex(stage.StageBError, "PID"):
                    dispatcher.execute(job, 900)

    def test_timeout_kills_child_and_keeps_real_failure_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            directory = Path(temporary) / "run"
            run = fixture.stage_run(directory)
            job = stage.make_job(run, directory, stage.MODELS[0], "O1", 1, kind="reference", phase="reference")
            class FakeProcess:
                pid = 2468
                returncode = -9
                waits = 0
                killed = False
                def wait(self, timeout):
                    self.waits += 1
                    if self.waits == 1:
                        raise subprocess.TimeoutExpired("test mock worker", timeout)
                    return self.returncode
                def kill(self):
                    self.killed = True
            process = FakeProcess()
            with patch("tools.run_classifier_benchmark.child_python", return_value=("TEST_PYTHON", {})), \
                 patch("tools.run_classifier_benchmark.worker_process_identity", side_effect=fake_identity), \
                 patch.object(stage.subprocess, "Popen", return_value=process) as spawn:
                dispatcher = stage.Dispatcher(directory, run, fixture.manifest)
                result = dispatcher.execute(job, 7)
                self.assertTrue(process.killed)
                self.assertEqual(result["status"], "failed_runtime")
                self.assertIn("7s", result["error"])
                self.assertEqual(result["samples"], [])
                self.assertIsNone(result["payload"])
                dispatcher.execute(job, 7)
                self.assertEqual(spawn.call_count, 1)

    def test_success_payload_with_failed_process_exit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            directory = Path(temporary) / "run"
            run = fixture.stage_run(directory)
            job = stage.make_job(run, directory, stage.MODELS[0], "O1", 1, kind="reference", phase="reference")
            class FakeProcess:
                pid = 2468
                returncode = 2
                def wait(self, timeout):
                    result = result_base(job, pid=self.pid)
                    result.update(status="passed", payload={"passed": True, "image_count": 64,
                                  "reference_sha256": job["reference_sha256"], "weights_sha256": job["weights_sha256"],
                                  "preprocess": stage.PREPROCESS, "per_image": [
                                      {"id": row["id"], "sha256": row["sha256"], "input_sha256": C}
                                      for row in job["images"]]})
                    stage.atomic_json(job["output_path"], result)
                    return self.returncode
            with patch("tools.run_classifier_benchmark.child_python", return_value=("TEST_PYTHON", {})), \
                 patch("tools.run_classifier_benchmark.worker_process_identity", side_effect=fake_identity), \
                 patch.object(stage.subprocess, "Popen", return_value=FakeProcess()):
                dispatcher = stage.Dispatcher(directory, run, fixture.manifest)
                with self.assertRaisesRegex(stage.StageBError, "declaring success"):
                    dispatcher.execute(job, 7)


class WorkflowTests(unittest.TestCase):
    def test_strict_mock_workflow_freezes_before_confirm_holdout_and_does_not_reselect(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            if getattr(self, "force_combination", False):
                for row in fixture.summary["experiments"]:
                    if row["variant_id"] == "O1":
                        row["inference"]["stats"]["inference_ms"]["p50"] = 1 if row["threads"] == 2 else 20
            calls, publications = [], []
            pid = [1000]
            def execute(dispatcher, job, timeout):
                calls.append(copy.deepcopy(job))
                pid[0] += 1
                result = result_base(job, pid=pid[0])
                result["status"] = "passed"
                kind = job["kind"]
                model = job["model_id"]
                if kind == "reference":
                    result["payload"] = {"passed": True, "per_image": [{"id": row["id"], "sha256": row["sha256"], "input_sha256": C} for row in job["images"]]}
                elif kind == "prepare":
                    result["payload"] = fixture.family(model, job["family_output_dir"])
                elif kind in ("measure", "control"):
                    value = (4 if job["threads"] == 2 else 5) if job["variant_id"] in ("Q1", "Q2", "Q3") else 10
                    if job["phase"] == "confirmation":
                        self.assertEqual(job["frozen_sha256"], stage.sha256(job["frozen_path"]))
                        self.assertEqual(len(stage.read_json(job["frozen_path"])["candidates"]), 3)
                    result["samples"] = [{"image_id": job["images"][index]["id"], "image_sha256": job["images"][index]["sha256"],
                                           "input_sha256": C, "inference_ms": value} for index in job["image_sequence"]]
                    result["validation"] = {"passed": True}
                    result["payload"] = {}
                elif kind in ("pipeline", "memory"):
                    self.assertTrue(Path(job["frozen_path"]).exists())
                    self.assertEqual(job["iterations"], 200)
                    self.assertEqual(job["warmup"], 20)
                    result["payload"] = {"evidence_kind": "test_mock", "pipeline_stats": {} if kind == "pipeline" else None,
                                         "memory": {"sampling_interval_ms": 20, "timing_included": False} if kind == "memory" else None}
                elif kind == "evaluate":
                    if job["split"] == "holdout":
                        self.assertTrue(Path(job["frozen_path"]).exists())
                        frozen = stage.read_json(job["frozen_path"])
                        self.assertEqual(len(frozen["candidates"]), 3)
                        self.assertEqual(job["frozen_sha256"], stage.sha256(job["frozen_path"]))
                    result["payload"] = {"split": job["split"], "model_id": model, "artifact_sha256": job["artifact_sha256"],
                                         "top1_accuracy": .8, "top5_accuracy": .9, "evidence_kind": "test_mock"}
                elif kind == "profile":
                    result["payload"] = {"providers": ["CPUExecutionProvider"], "int8_operator_events": [{"op_name": "QLinearConv"}],
                                         "int8_operator_counts": {"QLinearConv": 1}, "float_operator_counts": {"Relu": 1}}
                elif kind == "compare":
                    result["payload"] = accuracy(job["split"], drop=2 if job["split"] == "holdout" else 0)
                    result["payload"]["model_id"] = model
                stage.atomic_json(job["output_path"], result)
                return result
            def report(directory, summary, **kwargs):
                publications.append(summary)
                self.assertFalse((Path(directory) / "report.html").exists())
                with self.assertRaisesRegex(stage.StageBError, "mock"):
                    stage._reject_mock(summary)
                return {}
            with patch.object(stage, "audit_fp32_run", return_value=fixture.audit), \
                 patch.object(stage.Dispatcher, "execute", new=execute), patch.object(stage, "write_report", side_effect=report):
                summary = stage.run_stage_b(fixture.manifest_path, fixture.fp32_dir, Path(temporary) / "stage-b")
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(publications), 1)
            self.assertTrue(all(row["status"] == "not_adopted_accuracy_budget" for row in summary["final"].values()))
            self.assertTrue(all(not row["selection_changed_after_holdout"] for row in summary["final"].values()))
            self.assertTrue(all(row["selection_basis"] == "development_only" for row in summary["selections"]))
            frozen = summary["frozen"]["value"]
            self.assertTrue(all(row["variant_id"] in ("Q1", "Q2", "Q3") for row in frozen["candidates"]))
            first_holdout = next(index for index, job in enumerate(calls) if job.get("split") == "holdout")
            self.assertFalse(any(job.get("split") == "development" for job in calls[first_holdout:]))
            self.assertEqual(len([job for job in calls if job["phase"] == "confirmation" and job["kind"] == "measure"]), 30)
            self.assertEqual(len([job for job in calls if job["phase"] == "confirmation" and job["kind"] == "control"]), 30)
            self.assertTrue(all(job["iterations"] == 500 and job["warmup"] == 20 for job in calls if job["phase"] == "confirmation" and job["kind"] == "measure"))
            self.assertEqual(len([job for job in calls if job["phase"] == "observations"]), 12)
            self.assertEqual(set(summary["observations"]), set(stage.MODELS))
            self.assertTrue(all(job["iterations"] == 200 and job["warmup"] == 20 for job in calls if job["phase"] == "screening"))
            self.assertTrue(all(job["threads"] == 1 for job in calls if job["phase"] == "prepare"))
            self.assertEqual(len([job for job in calls if job["kind"] == "prepare"]), 3)
            self.assertEqual(len([job for job in calls if job["kind"] == "evaluate" and job["split"] == "holdout"]), 6)
            if getattr(self, "force_combination", False):
                combos = [row for row in summary["rows"] if row["phase"] == "combination"]
                self.assertEqual(len(combos), 3)
                self.assertEqual(len({row["model_id"] for row in combos}), 3)
                self.assertTrue(all(row["threads"] == 2 and "not a pure INT8" in row["attribution"] for row in combos))
                self.assertTrue(all(row["threads"] == 2 for row in frozen["candidates"]))
                self.assertEqual(len([job for job in calls if job["phase"] == "combination" and job["kind"] == "measure"]), 30)

    def test_one_thread_combination_per_model_is_development_validated_before_freeze(self):
        self.force_combination = True
        self.test_strict_mock_workflow_freezes_before_confirm_holdout_and_does_not_reselect()

    def test_failed_reference_stops_before_calibration_confirmation_or_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            calls = []
            def execute(dispatcher, job, timeout):
                calls.append(job)
                result = result_base(job, pid=1234 + len(calls))
                result.update(status="failed_correctness", error="test-only corrupted normalization reference")
                stage.atomic_json(job["output_path"], result)
                return result
            with patch.object(stage, "audit_fp32_run", return_value=fixture.audit), \
                 patch.object(stage.Dispatcher, "execute", new=execute), patch.object(stage, "write_report") as publish:
                summary = stage.run_stage_b(fixture.manifest_path, fixture.fp32_dir, Path(temporary) / "stage-b")
            self.assertEqual(summary["status"], "blocked_correctness")
            self.assertEqual(summary["phase"], "holdout_not_executed")
            self.assertTrue(all(job["kind"] == "reference" for job in calls))
            self.assertEqual(len(calls), 3)
            self.assertIsNone(summary["frozen"])
            self.assertFalse((Path(temporary) / "stage-b" / "frozen.json").exists())
            self.assertEqual(summary["final"], {})
            publish.assert_called_once()

    def test_workflow_int8_all_failed_runtime_verdict_and_completed_with_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            calls, publications = [], []
            pid = [2000]

            def execute(dispatcher, job, timeout):
                calls.append(copy.deepcopy(job))
                pid[0] += 1
                result = result_base(job, pid=pid[0])
                result["status"] = "passed"
                kind = job["kind"]
                model = job["model_id"]
                if kind == "reference":
                    result["payload"] = {"passed": True, "per_image": [{"id": row["id"], "sha256": row["sha256"], "input_sha256": C} for row in job["images"]]}
                elif kind == "prepare":
                    fam = fixture.family(model, job["family_output_dir"])
                    fam["status"] = "completed_with_failures"
                    for q in ("Q1", "Q2", "Q3"):
                        fam["variants"][q] = {
                            "status": "failed_runtime",
                            "reason": "FileNotFoundError: QB-inferred.onnx",
                            "path": None,
                            "onnx_sha256": None,
                            "size_bytes": None,
                            "validation": {"passed": False},
                        }
                    stage.atomic_json(Path(job["family_output_dir"]) / "family.json", fam, overwrite=True)
                    result["payload"] = fam
                elif kind in ("measure", "control"):
                    result["samples"] = [{"image_id": job["images"][index]["id"], "image_sha256": job["images"][index]["sha256"],
                                           "input_sha256": C, "inference_ms": 6.0 if job["threads"] == 4 else 10.0} for index in job["image_sequence"]]
                    result["validation"] = {"passed": True}
                    result["payload"] = {}
                elif kind in ("pipeline", "memory"):
                    result["payload"] = {"evidence_kind": "test_mock", "pipeline_stats": {} if kind == "pipeline" else None,
                                         "memory": {"sampling_interval_ms": 20, "timing_included": False} if kind == "memory" else None}
                elif kind == "evaluate":
                    result["payload"] = {"split": job["split"], "model_id": model, "artifact_sha256": job["artifact_sha256"],
                                         "top1_accuracy": .8, "top5_accuracy": .9, "evidence_kind": "test_mock"}
                elif kind == "profile":
                    result["payload"] = {"providers": ["CPUExecutionProvider"], "int8_operator_events": [],
                                         "int8_operator_counts": {}, "float_operator_counts": {"Relu": 1}}
                elif kind == "compare":
                    result["payload"] = accuracy(job["split"], drop=0.0)
                    result["payload"]["model_id"] = model
                stage.atomic_json(job["output_path"], result)
                return result

            def report(directory, summary, **kwargs):
                publications.append(summary)
                return {}

            with patch.object(stage, "audit_fp32_run", return_value=fixture.audit), \
                 patch.object(stage.Dispatcher, "execute", new=execute), \
                 patch.object(stage, "write_report", side_effect=report):
                summary = stage.run_stage_b(fixture.manifest_path, fixture.fp32_dir, Path(temporary) / "stage-b")

            self.assertEqual(summary["status"], "completed_with_failures")
            self.assertEqual(len(publications), 1)
            for model in stage.MODELS:
                final = summary["final"][model]
                self.assertEqual(final["status"], "fp32_confirmed_int8_not_evaluated")
                self.assertEqual(final["int8_evidence_status"], "failed_runtime")
                self.assertTrue(final["adopted"])
                self.assertFalse(final["int8_adopted"])
                self.assertIn("INT8 candidates not evaluated", final["reason"])
                self.assertNotIn("no INT8 adoption benefit", final["reason"])
                selection = next(s for s in summary["selections"] if s["model_id"] == model)
                self.assertEqual(selection["int8_evidence_status"], "failed_runtime")

            # Verify CLI prints completed_with_failures and fp32_confirmed_int8_not_evaluated and returns 2
            with patch.object(stage, "run_stage_b", return_value=summary):
                with patch("sys.stdout", new=io.StringIO()) as fake_out:
                    code = cli.main(["--manifest", "unused", "--fp32-run", "unused", "--output-dir", "unused"])
                    self.assertEqual(code, 2)
                    output = fake_out.getvalue()
                    self.assertIn("Stage B: completed_with_failures", output)
                    self.assertIn("fp32_confirmed_int8_not_evaluated", output)

    def test_workflow_int8_truly_evaluated_negative_adopt_fp32_verdict(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            calls, publications = [], []
            pid = [3000]

            def execute(dispatcher, job, timeout):
                calls.append(copy.deepcopy(job))
                pid[0] += 1
                result = result_base(job, pid=pid[0])
                result["status"] = "passed"
                kind = job["kind"]
                model = job["model_id"]
                if kind == "reference":
                    result["payload"] = {"passed": True, "per_image": [{"id": row["id"], "sha256": row["sha256"], "input_sha256": C} for row in job["images"]]}
                elif kind == "prepare":
                    result["payload"] = fixture.family(model, job["family_output_dir"])
                elif kind in ("measure", "control"):
                    result["samples"] = [{"image_id": job["images"][index]["id"], "image_sha256": job["images"][index]["sha256"],
                                           "input_sha256": C, "inference_ms": 6.0 if job["threads"] == 4 else 10.0} for index in job["image_sequence"]]
                    result["validation"] = {"passed": True}
                    result["payload"] = {}
                elif kind in ("pipeline", "memory"):
                    result["payload"] = {"evidence_kind": "test_mock", "pipeline_stats": {} if kind == "pipeline" else None,
                                         "memory": {"sampling_interval_ms": 20, "timing_included": False} if kind == "memory" else None}
                elif kind == "evaluate":
                    result["payload"] = {"split": job["split"], "model_id": model, "artifact_sha256": job["artifact_sha256"],
                                         "top1_accuracy": .8, "top5_accuracy": .9, "evidence_kind": "test_mock"}
                elif kind == "profile":
                    result["payload"] = {"providers": ["CPUExecutionProvider"], "int8_operator_events": [{"op_name": "QLinearConv"}],
                                         "int8_operator_counts": {"QLinearConv": 1}, "float_operator_counts": {"Relu": 1}}
                elif kind == "compare":
                    # Reject all INT8 in development due to accuracy drop; accept FP32 candidate
                    is_int8 = job["variant_id"] in ("Q1", "Q2", "Q3")
                    drop = 2.0 if is_int8 else 0.0
                    result["payload"] = accuracy(job["split"], drop=drop)
                    result["payload"]["model_id"] = model
                stage.atomic_json(job["output_path"], result)
                return result

            def report(directory, summary, **kwargs):
                publications.append(summary)
                return {}

            with patch.object(stage, "audit_fp32_run", return_value=fixture.audit), \
                 patch.object(stage.Dispatcher, "execute", new=execute), \
                 patch.object(stage, "write_report", side_effect=report):
                summary = stage.run_stage_b(fixture.manifest_path, fixture.fp32_dir, Path(temporary) / "stage-b")

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(publications), 1)
            for model in stage.MODELS:
                final = summary["final"][model]
                self.assertEqual(final["status"], "adopt_fp32_only_no_int8_benefit")
                self.assertEqual(final["int8_evidence_status"], "evaluated_no_benefit")
                self.assertTrue(final["adopted"])
                self.assertFalse(final["int8_adopted"])
                self.assertIn("no INT8 adoption benefit", final["reason"])
                selection = next(s for s in summary["selections"] if s["model_id"] == model)
                self.assertEqual(selection["int8_evidence_status"], "evaluated_no_benefit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
