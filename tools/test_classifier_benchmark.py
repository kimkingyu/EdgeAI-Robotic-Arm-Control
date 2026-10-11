#!/usr/bin/env python3
"""Offline protocol tests; all fabricated bytes/results stay in temporary test dirs."""
from __future__ import annotations

import copy
import csv
import hashlib
from html.parser import HTMLParser
import io
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

from src.vision.classification_benchmark import (
    BenchmarkError, MODEL_IDS, aggregate_results, dataset_readiness,
    expand_experiments, load_config, load_manifest, select_images,
    stage_stats, summarize, write_report,
)


def dump(path: Path, value: dict) -> bytes:
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return encoded


def image_entry(index: int, split: str = "development", label=None) -> dict:
    return {"id": f"image-{index:05d}", "path": f"image-{index}.bin",
            "sha256": hashlib.sha256(f"test bytes {index}".encode()).hexdigest(),
            "split": split, "label": label}


def data_fixture(development=0, calibration=0, holdout=0, *, labeled=False, smoke=0) -> dict:
    images = []
    for split, count in (("smoke", smoke), ("development", development),
                         ("calibration", calibration), ("holdout", holdout)):
        for _ in range(count):
            index = len(images)
            images.append(image_entry(index, split, index % 1000 if labeled and split in ("development", "holdout") else None))
    return {"schema_version": 1, "root": ".", "images": images}


class RunFixture:
    """Fake worker records are explicit unittest fixtures, never demonstration evidence."""
    def __init__(self, directory: Path, *, mode="smoke", iterations=3, repeats=None,
                 status="completed", all_experiments=False):
        self.directory = directory
        self.results = directory / "results"
        self.results.mkdir()
        self.config = load_config()
        self.config[mode]["iterations"] = iterations
        if repeats is not None:
            self.config[mode]["repeats"] = repeats
        experiments = expand_experiments(self.config)
        self.known = {experiment["id"]: experiment for experiment in experiments}
        chosen = experiments if all_experiments else [self.known[f"resnet18.{variant}.t1"] for variant in ("B0", "O0", "O1")]
        self.run = {"schema_version": 1, "run_id": "unit-test-not-benchmark-evidence", "mode": mode,
                    "config": self.config, "config_sha256": hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
                    "manifest_sha256": "b" * 64, "code_fingerprint": "c" * 64,
                    "environment": {"platform": "synthetic-unittest"},
                    "dataset_readiness": dataset_readiness(data_fixture(development=128) if mode == "formal" else data_fixture(smoke=1)),
                    "experiments": chosen, "status": status, "prepared_models": {}, "limitations": []}
        self.next_pid = 1000
        self.write_run()

    def write_run(self):
        dump(self.directory / "run.json", self.run)

    def record(self, name="resnet18.B0.t1", *, block=0, kind="inference", status="passed", latency=10.0):
        self.next_pid += 1
        samples = [{"image_id": "test-image", "inference_ms": latency}
                   for _ in range(self.config[self.run["mode"]]["iterations"])]
        if kind == "pipeline":
            for sample in samples:
                sample.update({"preprocess_ms": 2.0, "input_adapt_ms": 0.0,
                               "postprocess_ms": 1.0, "pipeline_wall_ms": latency + 4.0})
        if kind == "memory":
            samples = []
        value = {"schema_version": 1, "status": status,
                 "experiment": copy.deepcopy(self.known[name]), "block": block,
                 "worker_pid": self.next_pid if status == "passed" else None, "kind": kind,
                 **{key: self.run[key] for key in ("config_sha256", "manifest_sha256", "code_fingerprint")},
                 "samples": samples if status == "passed" else [],
                 "validation": {"passed": True, "allclose": True, "max_abs_error": 0.00001,
                                "mean_abs_error": 0.000001, "top1_equal": True, "top5_equal": True},
                 "model_metadata": {"weights_sha256": "d" * 64, "parameter_count": 100},
                 "variant_metadata": {"layout": "NCHW"}, "load_ms": 15.0,
                 "first_inference_ms": 25.0,
                 "predictions": [{"image_id": "test-image", "top5": [{"index": 0, "label": "test only", "probability": 0.5}]}]}
        if kind == "memory":
            value["memory"] = {"rss_baseline_bytes": 1000000, "rss_loaded_bytes": 2000000,
                               "rss_peak_bytes": 2500000, "sampling_interval_ms": 5.0,
                               "limitations": ["synthetic unit test; RSS includes runtime"]}
        if status != "passed":
            value["error"] = "deliberate unittest failure"
        return value

    def save(self, value: dict, filename=None):
        filename = filename or f"{value['experiment']['id']}.{value['kind']}.b{value['block']}.json"
        dump(self.results / filename, value)

    def passing_blocks(self, name, latency=10.0, kind="inference"):
        for block in range(self.config[self.run["mode"]]["repeats"]):
            self.save(self.record(name, block=block, latency=latency, kind=kind))

    def row(self, summary, name="resnet18.B0.t1"):
        return next(row for row in summary["experiments"] if row["id"] == name)


class ConfigTests(unittest.TestCase):
    def test_pinned_defaults_and_no_calibration_for_smoke(self):
        value = load_config()
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["models"], list(MODEL_IDS))
        self.assertEqual(value["variants"], ["B0", "F1", "L1", "FL1", "O0", "O1"])
        self.assertEqual(value["thread_sweep"], [1, 2, 4])
        self.assertEqual(value["smoke"], {"warmup": 3, "iterations": 10, "repeats": 2})
        self.assertEqual(value["formal"], {"warmup": 20, "iterations": 200, "repeats": 5})
        self.assertEqual(value["seed"], 20261001)
        self.assertEqual(value["rtol"], 1e-4)
        self.assertEqual(value["atol"], 1e-4)
        self.assertEqual(value["worker_timeout_s"], 300)
        self.assertEqual(value["dataset_requirements"], {"development": 128, "performance": 64, "calibration": 256, "holdout": 1000})
        self.assertTrue(dataset_readiness(data_fixture(smoke=1))["smoke"]["ready"])

    def test_matrix_is_30_unique_not_repeated_t1(self):
        value = load_config()
        full = expand_experiments(value)
        self.assertEqual(len(full), 30)
        self.assertEqual(len({entry["id"] for entry in full}), 30)
        self.assertEqual(len(expand_experiments(value, include_thread_sweep=False)), 18)
        for model in MODEL_IDS:
            entries = [entry for entry in full if entry["model_id"] == model]
            self.assertEqual(len(entries), 10)
            self.assertEqual(sum(entry["threads"] == 1 for entry in entries), 6)
            for entry in entries:
                self.assertEqual(entry["id"], f"{model}.{entry['variant_id']}.t{entry['threads']}")
                if entry["threads"] > 1:
                    self.assertIn(entry["variant_id"], ("B0", "O1"))

    def test_invalid_config_types_ids_and_budgets(self):
        changes = [("schema_version", True), ("schema_version", 2), ("models", ["resnet18"]),
                   ("models", ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]),
                   ("variants", ["B0", "F1", "L1", "FL1", "O0", "Q1"]),
                   ("variants", ["B0", "F1", "L1", "FL1", "O0", "O0"]),
                   ("thread_sweep", [True, 2, 4]), ("thread_sweep", [1, 2, 8]),
                   ("thread_sweep_variants", ["B0", "F1"]), ("batch_size", True),
                   ("inter_op_threads", 2), ("seed", True), ("seed", -1),
                   ("worker_timeout_s", 0), ("worker_timeout_s", "300"),
                   ("rtol", 0.001), ("atol", -1), ("weights_version", "DEFAULT"), ("device", "cuda")]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for key, invalid in changes:
                with self.subTest(key=key, value=invalid):
                    value = load_config()
                    value[key] = invalid
                    dump(path, value)
                    with self.assertRaises(BenchmarkError):
                        load_config(path)
            for key, invalid in (("warmup", True), ("warmup", -1), ("iterations", 0),
                                 ("iterations", 1.5), ("repeats", False), ("repeats", 1001)):
                with self.subTest(budget=key, value=invalid):
                    value = load_config()
                    value["smoke"][key] = invalid
                    dump(path, value)
                    with self.assertRaises(BenchmarkError):
                        load_config(path)
            value = load_config()
            value["dataset_requirements"]["holdout"] = 999
            dump(path, value)
            with self.assertRaises(BenchmarkError):
                load_config(path)
            value = load_config()
            value["ort"]["providers"] = ["AzureExecutionProvider", "CPUExecutionProvider"]
            dump(path, value)
            with self.assertRaises(BenchmarkError):
                load_config(path)

    def test_nonfinite_and_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for value in ("NaN", "Infinity", "-Infinity", "1e999"):
                text = json.dumps(load_config()).replace('"worker_timeout_s": 300', '"worker_timeout_s": ' + value)
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(BenchmarkError):
                    load_config(path)
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(BenchmarkError):
                load_config(path)


class ManifestTests(unittest.TestCase):
    def create_manifest(self, temporary):
        directory = Path(temporary)
        root = directory / "local-data"
        root.mkdir()
        value = data_fixture(smoke=2)
        value["root"] = "local-data"
        value["images"][0].update({"source_id": "same-original", "source": "licensed unittest", "license": "test only", "source_url": "https://example.invalid/test"})
        value["images"][1]["source_id"] = "same-original"
        for index, entry in enumerate(value["images"]):
            (root / entry["path"]).write_bytes(f"test bytes {index}".encode())
        path = directory / "manifest.json"
        raw = dump(path, value)
        return path, value, raw

    def test_resolved_relative_root_hash_and_preserved_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, value, raw = self.create_manifest(temporary)
            actual = load_manifest(path)
            self.assertEqual(actual["root"], "local-data")
            self.assertEqual(actual["manifest_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(actual["images"][0]["source"], value["images"][0]["source"])
            self.assertEqual(actual["images"][0]["label"], None)
            self.assertEqual(Path(actual["images"][0]["resolved_path"]), (path.parent / "local-data" / "image-0.bin").resolve())
            self.assertEqual(len(actual["images"]), 2)  # Same-split derived images are allowed.

    def test_byte_mismatch_empty_missing_and_check_files_false(self):
        for replacement in (b"changed", b"", None):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as temporary:
                path, value, _ = self.create_manifest(temporary)
                image = path.parent / "local-data" / value["images"][0]["path"]
                if replacement is None:
                    image.unlink()
                else:
                    image.write_bytes(replacement)
                with self.assertRaises(BenchmarkError):
                    load_manifest(path)
                self.assertEqual(len(load_manifest(path, check_files=False)["images"]), 2)

    def test_duplicate_ids_hashes_and_cross_split_source_leakage(self):
        for field, change in (("id", "image-00000"), ("sha256", "duplicate"), ("split", "holdout")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                path, value, _ = self.create_manifest(temporary)
                value["images"][1][field] = value["images"][0]["sha256"] if change == "duplicate" else change
                dump(path, value)
                with self.assertRaises(BenchmarkError):
                    load_manifest(path, check_files=False)

    def test_labels_path_traversal_and_unknown_splits(self):
        changes = [("label", True), ("label", -1), ("label", 1000), ("label", 1.5), ("label", "0"),
                   ("path", "../outside.bin"), ("path", "..\\outside.bin"), ("path", "/absolute.bin"),
                   ("path", "C:\\absolute.bin"), ("path", "image-0.bin:stream"), ("path", "."),
                   ("split", "validation"), ("split", ["smoke"]), ("sha256", "not-a-sha"), ("source_id", "")]
        with tempfile.TemporaryDirectory() as temporary:
            path, original, _ = self.create_manifest(temporary)
            for field, invalid in changes:
                with self.subTest(field=field, value=invalid):
                    value = copy.deepcopy(original)
                    value["images"][0][field] = invalid
                    dump(path, value)
                    with self.assertRaises(BenchmarkError):
                        load_manifest(path, check_files=False)

    def test_empty_manifest_and_no_fake_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            dump(path, data_fixture())
            manifest = load_manifest(path)
            self.assertFalse(dataset_readiness(manifest)["smoke"]["ready"])
            with self.assertRaises(BenchmarkError):
                select_images(manifest)


class ReadinessTests(unittest.TestCase):
    def test_one_image_smoke_without_calibration_or_accuracy(self):
        manifest = data_fixture(smoke=1)
        readiness = dataset_readiness(manifest)
        self.assertTrue(readiness["smoke"]["ready"])
        self.assertFalse(readiness["formal"]["ready"])
        self.assertFalse(readiness["quantization"]["ready"])
        self.assertEqual(readiness["quantization"]["development_labeled_count"], 0)
        self.assertEqual(len(select_images(manifest)), 1)
        with self.assertRaises(BenchmarkError):
            select_images(manifest, mode="formal")

    def test_formal_dev128_and_performance64_fixed_deterministic(self):
        for count in (64, 127):
            manifest = data_fixture(development=count)
            self.assertFalse(dataset_readiness(manifest)["formal"]["ready"])
            with self.assertRaises(BenchmarkError):
                select_images(manifest, mode="formal")
        manifest = data_fixture(development=128, calibration=256, holdout=1000)
        readiness = dataset_readiness(manifest)
        self.assertTrue(readiness["formal"]["ready"])
        self.assertEqual(readiness["formal"]["performance_count"], 64)
        self.assertFalse(readiness["quantization"]["ready"])
        first = [entry["id"] for entry in select_images(manifest, mode="formal")]
        changed = copy.deepcopy(manifest)
        changed["images"].reverse()
        self.assertEqual(first, [entry["id"] for entry in select_images(changed, mode="formal")])
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, [entry["id"] for entry in select_images(manifest, mode="formal", seed=20261002)])
        self.assertTrue(all(entry["split"] == "development" for entry in select_images(manifest, mode="formal")))
        with self.assertRaises(BenchmarkError):
            select_images(manifest, seed=True)
        with self.assertRaises(BenchmarkError):
            select_images(manifest, mode="holdout")

    def test_quantization_requires_independent_calibration_and_real_labels(self):
        manifest = data_fixture(development=128, calibration=256, holdout=1000, labeled=True)
        readiness = dataset_readiness(manifest)
        self.assertTrue(readiness["quantization"]["ready"])
        self.assertEqual(readiness["quantization"]["calibration_count"], 256)
        self.assertEqual(readiness["quantization"]["development_labeled_count"], 128)
        self.assertEqual(readiness["quantization"]["holdout_labeled_count"], 1000)
        for split in ("development", "holdout"):
            changed = copy.deepcopy(manifest)
            next(entry for entry in changed["images"] if entry["split"] == split)["label"] = None
            self.assertFalse(dataset_readiness(changed)["quantization"]["ready"])
        changed = copy.deepcopy(manifest)
        changed["images"] = [entry for entry in changed["images"] if entry["id"] != "image-00128"]
        self.assertFalse(dataset_readiness(changed)["quantization"]["ready"])


class StatisticsTests(unittest.TestCase):
    def test_linear_quantiles_population_std_and_positive_values(self):
        value = summarize([1, 2, 3, 4])
        self.assertEqual(value["count"], 4)
        self.assertEqual(value["p50_ms"], 2.5)
        self.assertAlmostEqual(value["p95_ms"], 3.85)
        self.assertEqual(value["mean_ms"], 2.5)
        self.assertAlmostEqual(value["std_ms"], (1.25)**0.5)
        self.assertEqual(value["min_ms"], 1)
        self.assertEqual(value["max_ms"], 4)
        self.assertEqual(summarize([3])["std_ms"], 0)
        self.assertEqual(summarize([1e308, 1e308])["mean_ms"], 1e308)
        for values in ([], [0], [-1], [float("nan")], [float("inf")], [float("-inf")], [True], ["1"]):
            with self.subTest(values=values), self.assertRaises(BenchmarkError):
                summarize(values)

    def test_stage_wall_is_measured_not_added_medians(self):
        samples = [
            {"image_id": "a", "preprocess_ms": 1, "input_adapt_ms": 0, "inference_ms": 5, "postprocess_ms": 1, "pipeline_wall_ms": 8},
            {"image_id": "a", "preprocess_ms": 5, "input_adapt_ms": 0, "inference_ms": 1, "postprocess_ms": 1, "pipeline_wall_ms": 9},
        ]
        value = stage_stats(samples)
        self.assertEqual(value["pipeline_wall_ms"]["p50_ms"], 8.5)
        self.assertEqual(value["stage_sum_ms"]["p50_ms"], 7)
        self.assertEqual(value["input_adapt_ms"]["p50_ms"], 0)
        self.assertEqual(value["wrapper_overhead_ms"]["p50_ms"], 1.5)
        samples[0]["pipeline_wall_ms"] = 1
        with self.assertRaises(BenchmarkError):
            stage_stats(samples)


class AggregationTests(unittest.TestCase):
    def test_recycled_pid_with_distinct_creation_markers_is_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RunFixture(Path(directory), repeats=1)
            for name, ticks in (("resnet18.B0.t1", 100), ("resnet18.O0.t1", 200)):
                record = fixture.record(name)
                record.update(worker_pid=4321, worker_creation_ticks=ticks,
                              worker_identity=f"4321:{ticks}", worker_identity_source="synthetic-unittest-start")
                fixture.save(record)
            summary = aggregate_results(directory)
            self.assertFalse(summary["integrity_errors"])
            self.assertTrue(all(r["trusted"] for r in summary["records"]))


    def test_same_process_creation_marker_cannot_be_reused_across_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RunFixture(Path(directory), repeats=1)
            for name in ("resnet18.B0.t1", "resnet18.O0.t1"):
                record = fixture.record(name)
                record.update(worker_pid=4321, worker_creation_ticks=100,
                              worker_identity="4321:100", worker_identity_source="synthetic-unittest-start")
                fixture.save(record)
            summary = aggregate_results(directory)
            self.assertEqual(len(summary["integrity_errors"]), 2)
            self.assertTrue(all(not r["trusted"] for r in summary["records"]))


    def test_process_marker_must_bind_pid_and_creation_not_numeric_prefix(self):
        for identity in ("4321:1000", "9999:100", ["4321:100"]):
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as directory:
                fixture = RunFixture(Path(directory), repeats=1)
                record = fixture.record()
                record.update(worker_pid=4321, worker_creation_ticks=100,
                              worker_identity=identity, worker_identity_source="synthetic-unittest-start")
                fixture.save(record)
                summary = aggregate_results(directory)
                self.assertTrue(summary["integrity_errors"])

    def test_complete_matrix_30_rows_and_smoke_never_recommended(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary), all_experiments=True)
            for experiment in fixture.run["experiments"]:
                fixture.passing_blocks(experiment["id"])
            summary = aggregate_results(temporary)
            self.assertEqual(summary["configuration_count"], 30)
            self.assertEqual(len(summary["experiments"]), 30)
            self.assertEqual(summary["status_counts"], {"passed": 30})
            self.assertEqual(summary["measured_unique_image_count"], 1)
            self.assertTrue(any("single_image" in text for text in summary["limitations"]))
            self.assertEqual(summary["stage_b"]["status"], "not_executed")
            self.assertEqual(len(summary["records"]), 60)
            for row in summary["experiments"]:
                self.assertFalse(row["recommendation"]["eligible"])
                self.assertFalse(row["recommendation"]["screening_gate_passed"])
                self.assertIsNone(row["comparisons"]["overall"]["ci95"])

    def test_paired_ci_uses_blocks_not_hundreds_of_iterations(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary), iterations=200, repeats=2)
            fixture.passing_blocks("resnet18.B0.t1", latency=20)
            fixture.passing_blocks("resnet18.O1.t1", latency=10)
            summary = aggregate_results(temporary)
            row = fixture.row(summary, "resnet18.O1.t1")
            comparison = row["comparisons"]["overall"]
            self.assertEqual(row["inference"]["sample_count"], 400)
            self.assertEqual(comparison["paired_block_count"], 2)
            self.assertAlmostEqual(comparison["speedup"], 2.0)
            self.assertIsNone(comparison["ci95"])
            self.assertIn("independent", comparison["ci_reason"])

    def test_five_blocks_ci_and_o1_increment_versus_o0(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary), mode="formal")
            for block in range(5):
                fixture.save(fixture.record("resnet18.B0.t1", block=block, latency=10 * (block + 1)))
                fixture.save(fixture.record("resnet18.O0.t1", block=block, latency=8 * (block + 1)))
                fixture.save(fixture.record("resnet18.O1.t1", block=block, latency=5 * (block + 1)))
            summary = aggregate_results(temporary)
            row = fixture.row(summary, "resnet18.O1.t1")
            overall = row["comparisons"]["overall"]
            increment = row["comparisons"]["ort_incremental"]
            self.assertAlmostEqual(overall["speedup"], 2.0)
            self.assertEqual(overall["ci95"]["resampling_unit"], "repeat_block")
            self.assertEqual(overall["ci95"]["block_count"], 5)
            self.assertAlmostEqual(overall["ci95"]["low"], 2.0)
            self.assertAlmostEqual(overall["ci95"]["high"], 2.0)
            self.assertEqual(increment["baseline_id"], "resnet18.O0.t1")
            self.assertAlmostEqual(increment["speedup"], 1.6)
            self.assertEqual(row["comparisons"]["factor"]["baseline_id"], "resnet18.O0.t1")
            self.assertEqual(row["recommendation"]["status"], "confirmation_required")
            self.assertFalse(row["recommendation"]["eligible"])
            second = aggregate_results(temporary)
            self.assertEqual(overall["ci95"], fixture.row(second, "resnet18.O1.t1")["comparisons"]["overall"]["ci95"])

    def test_thread_factor_and_combo_controls_do_not_mix_threads(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary), all_experiments=True)
            for name, latency in (("resnet18.B0.t1", 20), ("resnet18.B0.t2", 10),
                                  ("resnet18.O0.t1", 16), ("resnet18.O1.t1", 10),
                                  ("resnet18.O1.t4", 5), ("resnet18.FL1.t1", 8),
                                  ("resnet18.F1.t1", 12), ("resnet18.L1.t1", 16)):
                fixture.passing_blocks(name, latency=latency)
            summary = aggregate_results(temporary)
            row = fixture.row(summary, "resnet18.O1.t4")
            self.assertAlmostEqual(row["comparisons"]["overall"]["speedup"], 4)
            self.assertAlmostEqual(row["comparisons"]["threads"]["speedup"], 2)
            self.assertEqual(row["comparisons"]["factor"]["baseline_id"], "resnet18.O1.t1")
            self.assertIsNone(row["comparisons"]["ort_incremental"]["speedup"])
            row = fixture.row(summary, "resnet18.FL1.t1")
            self.assertAlmostEqual(row["comparisons"]["fusion_given_layout"]["speedup"], 2)
            self.assertAlmostEqual(row["comparisons"]["layout_given_fusion"]["speedup"], 1.5)

    def test_failed_unsupported_blocked_and_interrupted_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary), status="interrupted", all_experiments=True)
            for index, status in enumerate(("failed_runtime", "failed_correctness", "unsupported", "blocked_data", "interrupted")):
                name = fixture.run["experiments"][index]["id"]
                fixture.save(fixture.record(name, status=status))
            fixture.save(fixture.record("resnet18.O1.t1"))
            summary = aggregate_results(temporary)
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(len(summary["records"]), 6)
            for index, status in enumerate(("failed_runtime", "failed_correctness", "unsupported", "blocked_data", "interrupted")):
                row = fixture.row(summary, fixture.run["experiments"][index]["id"])
                self.assertEqual(row["status"], status)
                self.assertEqual(row["inference"]["sample_count"], 0)
                self.assertEqual(row["inference"]["stats"], {})
                self.assertIsNone(row["comparisons"]["overall"]["speedup"])
            self.assertEqual(fixture.row(summary, "resnet18.O1.t1")["status"], "interrupted")

    def test_corrupt_file_remains_visible_and_untrusted(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary))
            (fixture.results / "broken.json").write_bytes(b"{not valid JSON")
            fixture.save(fixture.record())
            summary = aggregate_results(temporary)
            self.assertEqual(summary["status"], "untrusted")
            self.assertEqual(len(summary["records"]), 2)
            self.assertEqual(summary["integrity_errors"][0]["source_file"], "results/broken.json")
            self.assertIsNone(next(record for record in summary["records"] if record["source_file"] == "results/broken.json")["data"])

    def test_hash_version_sample_count_pid_status_and_validation_strict(self):
        mutations = [("config_sha256", "a" * 64), ("manifest_sha256", "a" * 64),
                     ("code_fingerprint", "a" * 64), ("schema_version", True),
                     ("worker_pid", True), ("worker_pid", 0), ("block", True),
                     ("status", "completed"), ("samples", []),
                     ("validation", {"passed": False}), ("validation", {"passed": "true"})]
        for key, value in mutations:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temporary:
                fixture = RunFixture(Path(temporary))
                record = fixture.record()
                record[key] = value
                fixture.save(record)
                summary = aggregate_results(temporary)
                row = fixture.row(summary)
                self.assertEqual(row["status"], "untrusted")
                self.assertEqual(row["inference"]["sample_count"], 0)
                self.assertEqual(len(summary["integrity_errors"]), 1)

    def test_duplicate_records_and_reused_pids_both_excluded(self):
        for duplicate_pid in (False, True):
            with self.subTest(duplicate_pid=duplicate_pid), tempfile.TemporaryDirectory() as temporary:
                fixture = RunFixture(Path(temporary))
                first = fixture.record(block=0)
                second = fixture.record(block=1 if duplicate_pid else 0)
                if duplicate_pid:
                    second["worker_pid"] = first["worker_pid"]
                fixture.save(first, "one.json")
                fixture.save(second, "two.json")
                summary = aggregate_results(temporary)
                self.assertEqual(len(summary["integrity_errors"]), 2)
                self.assertEqual(fixture.row(summary)["inference"]["sample_count"], 0)

    def test_mismatched_image_sequences_never_paired(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary))
            fixture.passing_blocks("resnet18.B0.t1")
            for block in range(2):
                record = fixture.record("resnet18.O1.t1", block=block)
                record["samples"][0]["image_id"] = "different-image"
                fixture.save(record)
            comparison = fixture.row(aggregate_results(temporary), "resnet18.O1.t1")["comparisons"]["overall"]
            self.assertEqual(comparison["paired_block_count"], 0)
            self.assertIsNone(comparison["speedup"])
            self.assertIn("mismatch", comparison["reason"])

    def test_pipeline_and_memory_are_independent_and_not_inference_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary))
            fixture.passing_blocks("resnet18.B0.t1", latency=10)
            fixture.passing_blocks("resnet18.B0.t1", latency=10, kind="pipeline")
            fixture.save(fixture.record(kind="memory"))
            row = fixture.row(aggregate_results(temporary))
            self.assertEqual(row["inference"]["sample_count"], 6)
            self.assertEqual(row["pipeline"]["sample_count"], 6)
            self.assertEqual(row["pipeline"]["stats"]["pipeline_wall_ms"]["p50_ms"], 14)
            self.assertEqual(row["memory"]["sample_count"], 0)
            self.assertEqual(row["memory"]["rss_peak_bytes"]["median"], 2500000)
            self.assertTrue(row["memory"]["independent_process"])
            self.assertEqual(row["inference"]["load_ms"]["p50_ms"], 15)

    def test_nonfinite_and_bad_stage_or_memory_not_successful(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary))
            value = fixture.record()
            text = json.dumps(value).replace('"inference_ms": 10.0', '"inference_ms": NaN')
            (fixture.results / "nan.json").write_text(text, encoding="utf-8")
            pipeline = fixture.record(kind="pipeline")
            pipeline["samples"][0]["pipeline_wall_ms"] = 1
            fixture.save(pipeline)
            memory = fixture.record(kind="memory")
            memory["memory"]["rss_peak_bytes"] = 1
            fixture.save(memory)
            summary = aggregate_results(temporary)
            self.assertEqual(len(summary["integrity_errors"]), 3)
            self.assertEqual(fixture.row(summary)["memory"]["block_count"], 0)

    def test_missing_run_invalid_run_and_forged_data_readiness_raise(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(BenchmarkError):
                aggregate_results(temporary)
            fixture = RunFixture(Path(temporary))
            fixture.run["schema_version"] = True
            fixture.write_run()
            with self.assertRaises(BenchmarkError):
                aggregate_results(temporary)
            fixture.run["schema_version"] = 1
            fixture.run["dataset_readiness"]["formal"]["ready"] = True
            fixture.write_run()
            with self.assertRaises(BenchmarkError):
                aggregate_results(temporary)

    def test_code_fingerprint_object_matches_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(Path(temporary))
            fixture.run["code_fingerprint"] = {"files": {"test.py": "d" * 64}, "sha256": "c" * 64}
            fixture.write_run()
            fixture.save(fixture.record())
            self.assertFalse(aggregate_results(temporary)["integrity_errors"])


class ElementAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class ReportTests(unittest.TestCase):
    def fixture_summary(self, directory):
        fixture = RunFixture(directory, status="interrupted")
        fixture.passing_blocks("resnet18.B0.t1")
        value = fixture.record("resnet18.O0.t1", status="unsupported")
        value["error"] = '<script>alert("error")</script><img src="https://example.invalid/x" onerror="run()">'
        value["traceback"] = "https://example.invalid/no-network < & >"
        fixture.save(value)
        value = fixture.record("resnet18.O1.t1", status="interrupted")
        fixture.save(value)
        fixture.run["environment"]["untrusted"] = '<iframe src="https://example.invalid"></iframe>'
        fixture.run["limitations"] = ['<img src="//cdn.invalid/x">']
        fixture.write_run()
        return aggregate_results(directory)

    def test_json_csv_html_escaping_offline_resources_and_failed_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary = self.fixture_summary(directory)
            paths = write_report(directory, summary)
            self.assertEqual(set(paths), {"summary_json", "summary_csv", "report_html"})
            self.assertEqual(json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))["status"], "interrupted")
            rows = list(csv.DictReader(io.StringIO(Path(paths["summary_csv"]).read_text(encoding="utf-8"))))
            self.assertEqual(len(rows), 30)
            failed = next(row for row in rows if row["id"] == "resnet18.O0.t1")
            self.assertEqual(failed["status"], "unsupported")
            self.assertEqual(failed["p50_ms"], "")
            self.assertEqual(failed["overall_speedup"], "")
            content = Path(paths["report_html"]).read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;", content)
            self.assertIn("&quot;error&quot;", content)
            self.assertIn("&lt;img", content)
            self.assertIn("unsupported", content)
            self.assertIn("interrupted", content)
            self.assertIn("Top-5", content)
            self.assertIn("独立RSS", content)
            self.assertIn("single_image", content)
            audit = ElementAudit()
            audit.feed(content)
            for tag, attrs in audit.tags:
                self.assertNotIn(tag, {"script", "iframe", "link", "img", "object", "embed"})
                self.assertNotIn("src", attrs)
                self.assertNotIn("href", attrs)
                self.assertFalse(any(key.startswith("on") for key in attrs))
            self.assertTrue(any(tag == "svg" for tag, _ in audit.tags))
            self.assertFalse(list(directory.glob(".*.tmp")))

    def test_refuse_default_overwrite_then_explicit_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary = self.fixture_summary(directory)
            paths = write_report(directory, summary)
            original = {key: Path(path).read_bytes() for key, path in paths.items()}
            with self.assertRaises(BenchmarkError):
                write_report(directory, summary)
            self.assertEqual(original, {key: Path(path).read_bytes() for key, path in paths.items()})
            changed = copy.deepcopy(summary)
            changed["run_id"] = "explicit-replacement"
            write_report(directory, changed, overwrite=True)
            self.assertEqual(json.loads((directory / "summary.json").read_text(encoding="utf-8"))["run_id"], "explicit-replacement")

    def test_existing_single_target_prevents_any_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary = self.fixture_summary(directory)
            (directory / "report.html").write_text("user existing file", encoding="utf-8")
            with self.assertRaises(BenchmarkError):
                write_report(directory, summary)
            self.assertFalse((directory / "summary.json").exists())
            self.assertFalse((directory / "summary.csv").exists())
            self.assertEqual((directory / "report.html").read_text(encoding="utf-8"), "user existing file")

    def test_atomic_publish_failure_rolls_back_new_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary = self.fixture_summary(directory)
            from src.vision import classification_benchmark as module
            real_link = module.os.link
            count = 0

            def failing_link(source, destination):
                nonlocal count
                count += 1
                if count == 2:
                    raise OSError("synthetic publish failure")
                return real_link(source, destination)

            with patch.object(module.os, "link", side_effect=failing_link):
                with self.assertRaises(BenchmarkError):
                    write_report(directory, summary)
            self.assertFalse(any((directory / filename).exists() for filename in ("summary.json", "summary.csv", "report.html")))
            self.assertFalse(list(directory.glob(".*.tmp")))

    def test_reject_nan_summary_without_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary = self.fixture_summary(directory)
            summary["environment"]["bad"] = float("nan")
            with self.assertRaises(BenchmarkError):
                write_report(directory, summary)
            self.assertFalse((directory / "summary.json").exists())


class ImportAndCliTests(unittest.TestCase):
    def blocker(self):
        return """import importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'onnx', 'onnxruntime', 'numpy', 'cv2', 'PIL', 'psutil'}:
            raise AssertionError('Heavy import forbidden: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
"""

    def test_summarizer_help_works_without_heavy_packages(self):
        script = ROOT / "tools" / "summarize_classifier_benchmark.py"
        program = self.blocker() + f"import runpy\nsys.argv = [{str(script)!r}, '--help']\nrunpy.run_path({str(script)!r}, run_name='__main__')\n"
        completed = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--run-dir", completed.stdout)
        self.assertIn("--overwrite", completed.stdout)

    def test_protocol_import_is_standard_library_only(self):
        program = self.blocker() + "from src.vision.classification_benchmark import load_config, expand_experiments\nassert len(expand_experiments(load_config())) == 30\n"
        completed = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_reads_existing_results_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = RunFixture(directory, status="interrupted")
            fixture.save(fixture.record(status="failed_runtime"))
            script = ROOT / "tools" / "summarize_classifier_benchmark.py"
            program = self.blocker() + f"import runpy\nsys.argv = [{str(script)!r}, '--run-dir', {temporary!r}]\nrunpy.run_path({str(script)!r}, run_name='__main__')\n"
            completed = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                       capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((directory / "summary.json").is_file())
            self.assertIn("30 configurations", completed.stdout)
            duplicate = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                       capture_output=True, text=True, timeout=20)
            self.assertEqual(duplicate.returncode, 2)
            self.assertIn("Refusing to overwrite", duplicate.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
