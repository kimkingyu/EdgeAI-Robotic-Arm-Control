#!/usr/bin/env python3
"""Offline unittest fixtures only: small synthetic JPEG/tars, never real datasets.

Successful small prepare_dataset calls explicitly mock benchmark readiness;
separate tests prove that the production 256/128/1000 gates still reject them.
The frozen production MD5 references are never replaced on disk.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import prepare_classifier_dataset as prepare
from src.vision import classification_benchmark as benchmark

SMALL = {"calibration_count": 20, "development_count": 20, "holdout_count": 20}
REAL_READINESS = benchmark.dataset_readiness


def jpeg(identity: str, *, orientation: int | None = None) -> bytes:
    pixels = hashlib.shake_256(identity.encode()).digest(12 * 10 * 3)
    output = io.BytesIO()
    with Image.frombytes("RGB", (12, 10), pixels) as image:
        options = {"quality": 95, "subsampling": 0}
        if orientation is not None:
            exif = Image.Exif()
            exif[274] = orientation
            options["exif"] = exif
        image.save(output, format="JPEG", **options)
    return output.getvalue()


def comment_jpeg(payload: bytes) -> bytes:
    text = b"TEST-ONLY comment; identical decoded RGB"
    return payload[:2] + b"\xff\xfe" + struct.pack(">H", len(text) + 2) + text + payload[2:]


def digest(path: Path, algorithm="sha256") -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def synthetic_readiness(manifest: dict) -> dict:
    value = REAL_READINESS(manifest)
    for gate in value.values():
        gate["ready"] = True
        gate["reason"] = "TEST-ONLY synthetic readiness mock; not production evidence"
    return value


class SmallArchives:
    """Synthetic official-shaped records; all artifacts stay in a unittest tmpdir."""

    def __init__(self, directory: Path, *, train_count=3, test_count=3):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.images = {}
        self.lines = {"trainval": [], "test": []}
        self.extras = {"images": [], "annotations": []}
        self.reverse = False
        for prefix, _, _ in prepare.FROZEN_MAPPING:
            class_id, species, breed_id = prepare.OFFICIAL_ANNOTATIONS[prefix]
            for split, numbers in (("trainval", range(1, train_count + 1)),
                                   ("test", range(101, 101 + test_count))):
                for number in numbers:
                    image_id = f"{prefix}_{number}"
                    self.lines[split].append(f"{image_id} {class_id} {species} {breed_id}")
                    self.images[f"images/{image_id}.jpg"] = jpeg(image_id)
        self.images_path = directory / "images.tar.gz"
        self.annotations_path = directory / "annotations.tar.gz"

    def ordered_ids(self, prefix: str, split="trainval", seed=prepare.DEFAULT_SEED) -> list:
        identities = [line.split()[0] for line in self.lines[split]
                      if line.split()[0].rsplit("_", 1)[0] == prefix]
        return sorted(identities, key=lambda image_id: prepare._rank(seed, f"candidates:{split}", image_id))

    def write(self) -> dict:
        for role, path in (("images", self.images_path), ("annotations", self.annotations_path)):
            if role == "images":
                members = list(sorted(self.images.items()))
            else:
                members = [(f"annotations/{split}.txt", ("# Synthetic unittest records only\n" +
                            "\n".join(lines) + "\n").encode("ascii"))
                           for split, lines in sorted(self.lines.items())]
                members += [("annotations/trimaps/ignored.png", b"not an extracted PNG"),
                            ("annotations/xmls/ignored.xml", b"<synthetic-test/>")]
            if self.reverse:
                members.reverse()
            with tarfile.open(path, "w:gz") as archive:
                directory = tarfile.TarInfo(role)
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
                for name, payload in members:
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
                for member, payload in self.extras[role]:
                    archive.addfile(member, io.BytesIO(payload) if payload is not None else None)
        config = copy.deepcopy(prepare.load_dataset_config())
        config["task_usage"] += " SYNTHETIC UNITTEST FIXTURE ONLY; not real benchmark data."
        for role, path in (("images", self.images_path), ("annotations", self.annotations_path)):
            config["archive_checksums"][role]["md5"] = digest(path, "md5")
            config["archive_checksums"][role]["md5_reference"] = (
                "TEST-ONLY synthetic tar checksum; not the production torchvision _RESOURCES reference.")
        return config


class FrozenConfigurationTests(unittest.TestCase):
    def test_exact_approved_mapping_and_attribution(self):
        config = prepare.load_dataset_config()
        expected = {
            "basset_hound": ("basset", 161), "beagle": ("beagle", 162),
            "boxer": ("boxer", 242), "chihuahua": ("Chihuahua", 151),
            "english_setter": ("English setter", 212),
            "german_shorthaired": ("German short-haired pointer", 210),
            "great_pyrenees": ("Great Pyrenees", 257), "japanese_chin": ("Japanese spaniel", 152),
            "keeshond": ("keeshond", 261), "leonberger": ("Leonberg", 255),
            "miniature_pinscher": ("miniature pinscher", 237), "newfoundland": ("Newfoundland", 256),
            "pomeranian": ("Pomeranian", 259), "pug": ("pug", 254),
            "saint_bernard": ("Saint Bernard", 247), "samoyed": ("Samoyed", 258),
            "scottish_terrier": ("Scotch terrier", 199),
            "staffordshire_bull_terrier": ("Staffordshire bullterrier", 179),
            "wheaten_terrier": ("soft-coated wheaten terrier", 202),
            "yorkshire_terrier": ("Yorkshire terrier", 187),
        }
        self.assertEqual({item["prefix"]: (item["canonical_name"], item["imagenet_index"])
                          for item in config["classes"]}, expected)
        self.assertEqual(config["license"], "CC BY-SA 4.0")
        self.assertEqual(len(config["authors"]), 4)
        self.assertIn("original owners", config["license_notes"])
        self.assertIn("Local research", config["task_usage"])
        self.assertEqual(config["counts"], {"calibration": 256, "development": 128, "holdout": 1000})
        for item in config["classes"]:
            self.assertEqual(item["species"], 2)
            self.assertTrue(item["alias"].strip())
        for role in ("images", "annotations"):
            self.assertEqual(config["archive_checksums"][role]["md5"], prepare.ARCHIVE_MD5[role])
            self.assertIn("_RESOURCES", config["archive_checksums"][role]["md5_reference"])

    def test_config_cannot_expand_mapping_or_change_indices_species_checksums(self):
        original = prepare.load_dataset_config()
        variants = []
        for key, value in (("prefix", "unknown_dog"), ("canonical_name", "bloodhound"),
                           ("imagenet_index", 163), ("imagenet_index", True),
                           ("species", 1), ("breed_id", 4)):
            config = copy.deepcopy(original)
            config["classes"][0][key] = value
            variants.append(config)
        extra = copy.deepcopy(original)
        extra["classes"].append(copy.deepcopy(extra["classes"][0]))
        variants.append(extra)
        altered_hash = copy.deepcopy(original)
        altered_hash["archive_checksums"]["images"]["md5"] = "0" * 32
        variants.append(altered_hash)
        with tempfile.TemporaryDirectory(prefix="classifier-config-unit-") as temporary:
            path = Path(temporary) / "config.json"
            for config in variants:
                with self.subTest(config=config["classes"][0]):
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with patch.object(prepare, "CONFIG_PATH", path):
                        with self.assertRaises(prepare.DatasetPreparationError):
                            prepare.load_dataset_config()

    def test_optional_categories_are_checked_offline(self):
        categories = ["unused synthetic category"] * 1000
        for _, name, index in prepare.FROZEN_MAPPING:
            categories[index] = name
        prepare.validate_category_names(categories)
        categories[161] = "basset hound"
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "mismatch"):
            prepare.validate_category_names(categories)
        with self.assertRaises(prepare.DatasetPreparationError):
            prepare.validate_category_names(categories[:999])

    def test_budget_quotas_are_balanced_and_holdout_exactly_fifty(self):
        prefixes = [item[0] for item in prepare.FROZEN_MAPPING]
        for split, count in prepare.DEFAULT_COUNTS.items():
            quotas = prepare._class_quotas(prefixes, count, prepare.DEFAULT_SEED, split)
            self.assertEqual(sum(quotas.values()), count)
            self.assertLessEqual(max(quotas.values()) - min(quotas.values()), 1)
            if split == "holdout":
                self.assertEqual(set(quotas.values()), {50})

    def test_annotation_validation_is_strict_including_unknown_prefix(self):
        for line in ("unknown_dog_1 4 2 3", "basset_hound_1 5 2 3", "basset_hound_1 4 1 3",
                     "basset_hound_1 4 2 4", "basset_hound_1 4 2", "basset_hound_1 4 2 3 extra",
                     "../basset_hound_1 4 2 3", "basset_hound_0 4 2 3", "basset_hound_1 x 2 3"):
            with self.subTest(line=line):
                with self.assertRaises(prepare.DatasetPreparationError):
                    prepare._annotation_records((line + "\n").encode(), "trainval")
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "Duplicate annotation"):
            prepare._annotation_records(b"basset_hound_1 4 2 3\nbasset_hound_1 4 2 3\n", "trainval")

    def test_invalid_argument_types_do_not_touch_archives(self):
        for changes in ({"seed": True}, {"seed": -1}, {"seed": 2**32},
                        {"calibration_count": 0}, {"development_count": True},
                        {"holdout_count": 1001}, {"overwrite": "yes"}):
            with self.subTest(changes=changes):
                with self.assertRaises(prepare.DatasetPreparationError):
                    prepare.prepare_dataset("not-read.tar", "not-read.tar", "not-created", **changes)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="classifier-dataset-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.fixture = SmallArchives(self.directory / "raw")
        self.output = self.directory / "prepared"

    def run_prepare(self, *, output=None, fake_readiness=True, config=None, **kwargs):
        config = self.fixture.write() if config is None else config
        with ExitStack() as stack:
            stack.enter_context(patch.object(prepare, "load_dataset_config", return_value=config))
            if fake_readiness:
                stack.enter_context(patch.object(benchmark, "dataset_readiness", side_effect=synthetic_readiness))
            return prepare.prepare_dataset(self.fixture.images_path, self.fixture.annotations_path,
                                           self.output if output is None else output, **(SMALL | kwargs))

    def read_manifest(self, output=None):
        return json.loads(((output or self.output) / "manifest.json").read_text(encoding="utf-8"))

    def read_source(self, output=None):
        return json.loads(((output or self.output) / "source.json").read_text(encoding="utf-8"))

    def assert_unpublished(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.glob(".prepared.prepare-*")), [])
        self.assertEqual(list(self.directory.glob("prepared/**/prepared.complete.json")), [])

    def test_manifest_source_license_hashes_and_original_bytes(self):
        result = self.run_prepare()
        manifest, source = self.read_manifest(), self.read_source()
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["root"], ".")
        self.assertEqual(dict(Counter(entry["split"] for entry in manifest["images"])),
                         {"calibration": 20, "development": 20, "holdout": 20})
        self.assertEqual(source["license"], "CC BY-SA 4.0")
        self.assertEqual(source["authors"], prepare.load_dataset_config()["authors"])
        self.assertIn("Original image owners", source["copyright_owner"])
        self.assertFalse(source["acquisition"]["tool_downloaded"])
        self.assertFalse(source["acquisition"]["tool_uploaded"])
        self.assertIn("not an official", source["archives"]["images"]["sha256_origin"])
        self.assertEqual(source["archives"]["images"]["sha256_observed"], digest(self.fixture.images_path))
        self.assertEqual(source["archives"]["annotations"]["md5_observed"], digest(self.fixture.annotations_path, "md5"))
        self.assertEqual(source["subset"]["class_count"], 20)
        self.assertTrue(any("near duplicates" in text for text in source["limitations"]))
        self.assertTrue(any("pretrained" in text for text in source["limitations"]))
        mapping = {item["prefix"]: item for item in prepare.load_dataset_config()["classes"]}
        for entry in manifest["images"]:
            path = self.output / entry["path"]
            self.assertEqual(path.read_bytes(), self.fixture.images[entry["path"]])
            self.assertEqual(digest(path), entry["sha256"])
            self.assertEqual(entry["pixel_sha256"], prepare._decoded_pixels(path.read_bytes())["pixel_sha256"])
            self.assertEqual(entry["original_annotation"]["SPECIES"], 2)
            self.assertEqual(entry["label"], None if entry["split"] == "calibration" else mapping[entry["breed"]]["imagenet_index"])
            self.assertEqual(entry["source_original_split"], "test" if entry["split"] == "holdout" else "trainval")
        for key in ("sha256", "pixel_sha256", "source_id"):
            self.assertEqual(len({entry[key] for entry in manifest["images"]}), 60)
        marker = json.loads((self.output / "prepared.complete.json").read_text())
        self.assertEqual(marker["manifest_sha256"], digest(self.output / "manifest.json"))
        self.assertEqual(marker["source_sha256"], digest(self.output / "source.json"))
        self.assertEqual(result["manifest_sha256"], marker["manifest_sha256"])
        self.assertFalse((self.output / ".candidate-cache").exists())
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"images", "manifest.json", "source.json", "prepared.complete.json"})
        # The production reader works, and its real gates remain false for this fixture.
        real = REAL_READINESS(benchmark.load_manifest(self.output / "manifest.json"))
        self.assertFalse(real["formal"]["ready"])
        self.assertFalse(real["quantization"]["ready"])

    def test_identical_seed_produces_identical_manifest(self):
        first = self.run_prepare()
        second = self.run_prepare(output=self.directory / "second")
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(self.read_manifest(), self.read_manifest(self.directory / "second"))

    def test_tar_member_and_annotation_order_cannot_change_selection(self):
        self.run_prepare()
        first = [(entry["id"], entry["split"], entry["label"]) for entry in self.read_manifest()["images"]]
        self.fixture.reverse = True
        for lines in self.fixture.lines.values():
            lines.reverse()
        self.run_prepare(output=self.directory / "second")
        second = [(entry["id"], entry["split"], entry["label"]) for entry in self.read_manifest(self.directory / "second")["images"]]
        self.assertEqual(first, second)

    def test_different_seed_changes_selection(self):
        self.run_prepare(seed=17)
        self.run_prepare(output=self.directory / "second", seed=18)
        self.assertNotEqual([entry["id"] for entry in self.read_manifest()["images"]],
                            [entry["id"] for entry in self.read_manifest(self.directory / "second")["images"]])

    def test_bad_jpeg_replaced_by_next_seeded_same_class_candidate(self):
        first, second, _ = self.fixture.ordered_ids("beagle")
        self.fixture.images[f"images/{first}.jpg"] = b"not JPEG test bytes"
        self.run_prepare()
        rejects = self.read_source()["rejections"]
        self.assertTrue(any(item["image_id"] == first and item["reason"] == "invalid_jpeg" for item in rejects))
        entry = next(item for item in self.read_manifest()["images"] if item["split"] == "calibration" and item["breed"] == "beagle")
        self.assertEqual(entry["id"], second)

    def test_missing_jpeg_replaced_without_lowering_quota(self):
        first = self.fixture.ordered_ids("beagle")[0]
        del self.fixture.images[f"images/{first}.jpg"]
        self.run_prepare()
        self.assertTrue(any(item["reason"] == "missing_jpeg" for item in self.read_source()["rejections"]))
        self.assertEqual(len(self.read_manifest()["images"]), 60)

    def test_sha_duplicate_between_calibration_and_development_is_replaced(self):
        first, second, _ = self.fixture.ordered_ids("beagle")
        self.fixture.images[f"images/{second}.jpg"] = self.fixture.images[f"images/{first}.jpg"]
        self.run_prepare()
        reject = next(item for item in self.read_source()["rejections"] if item["image_id"] == second)
        self.assertEqual(reject["reason"], "duplicate_sha256")
        self.assertEqual(reject["duplicate_of_split"], "calibration")
        self.assertFalse(reject["cross_official_split"])

    def test_different_jpeg_bytes_with_identical_rgb_are_replaced(self):
        first, second, _ = self.fixture.ordered_ids("beagle")
        original = self.fixture.images[f"images/{first}.jpg"]
        different = comment_jpeg(original)
        self.assertNotEqual(hashlib.sha256(original).hexdigest(), hashlib.sha256(different).hexdigest())
        self.assertEqual(prepare._decoded_pixels(original)["pixel_sha256"], prepare._decoded_pixels(different)["pixel_sha256"])
        self.fixture.images[f"images/{second}.jpg"] = different
        self.run_prepare()
        reject = next(item for item in self.read_source()["rejections"] if item["image_id"] == second)
        self.assertEqual(reject["reason"], "duplicate_rgb_pixels")

    def test_cross_official_split_sha_duplicate_retains_diagnostic_and_no_leakage(self):
        training = self.fixture.ordered_ids("beagle")[0]
        testing = self.fixture.ordered_ids("beagle", "test")[0]
        self.fixture.images[f"images/{testing}.jpg"] = self.fixture.images[f"images/{training}.jpg"]
        self.run_prepare()
        reject = next(item for item in self.read_source()["rejections"] if item["image_id"] == testing)
        self.assertEqual(reject["reason"], "duplicate_sha256")
        self.assertTrue(reject["cross_official_split"])
        self.assertEqual(reject["duplicate_of_original_split"], "trainval")
        self.assertNotIn(testing, [entry["id"] for entry in self.read_manifest()["images"]])
        benchmark.load_manifest(self.output / "manifest.json")

    def test_cross_official_split_pixel_duplicate_is_replaced(self):
        training = self.fixture.ordered_ids("beagle")[0]
        testing = self.fixture.ordered_ids("beagle", "test")[0]
        self.fixture.images[f"images/{testing}.jpg"] = comment_jpeg(self.fixture.images[f"images/{training}.jpg"])
        self.run_prepare()
        reject = next(item for item in self.read_source()["rejections"] if item["image_id"] == testing)
        self.assertEqual(reject["reason"], "duplicate_rgb_pixels")
        self.assertTrue(reject["cross_official_split"])

    def test_exif_is_used_for_pixel_hash_and_not_removed_from_source_bytes(self):
        first = self.fixture.ordered_ids("beagle")[0]
        self.fixture.images[f"images/{first}.jpg"] = jpeg(first, orientation=6)
        self.run_prepare()
        entry = next(item for item in self.read_manifest()["images"] if item["id"] == first)
        self.assertEqual((entry["original_width"], entry["original_height"]), (12, 10))
        self.assertEqual((entry["width"], entry["height"]), (10, 12))
        self.assertEqual((self.output / entry["path"]).read_bytes(), self.fixture.images[entry["path"]])

    def test_png_disguised_as_jpeg_is_rejected(self):
        first = self.fixture.ordered_ids("beagle")[0]
        output = io.BytesIO()
        Image.new("RGB", (8, 8)).save(output, format="PNG")
        self.fixture.images[f"images/{first}.jpg"] = output.getvalue()
        self.run_prepare()
        self.assertTrue(any(item["image_id"] == first and item["reason"] == "invalid_jpeg"
                            for item in self.read_source()["rejections"]))

    def test_decompression_bomb_candidate_is_rejected_and_replaced(self):
        first = self.fixture.ordered_ids("beagle")[0]
        bad_payload = self.fixture.images[f"images/{first}.jpg"]
        real_open = Image.open

        def bomb_one(stream, *args, **kwargs):
            if isinstance(stream, io.BytesIO) and stream.getvalue() == bad_payload:
                raise Image.DecompressionBombError("synthetic bomb candidate")
            return real_open(stream, *args, **kwargs)

        with patch.object(Image, "open", side_effect=bomb_one):
            self.run_prepare()
        reject = next(item for item in self.read_source()["rejections"] if item["image_id"] == first)
        self.assertEqual(reject["reason"], "invalid_jpeg")
        self.assertIn("decompression bomb", reject["detail"])

    def test_decoded_pixel_budget_replaces_oversized_candidate(self):
        first = self.fixture.ordered_ids("beagle")[0]
        output = io.BytesIO()
        with Image.new("RGB", (16, 16), (30, 80, 120)) as image:
            image.save(output, format="JPEG")
        self.fixture.images[f"images/{first}.jpg"] = output.getvalue()
        with patch.object(prepare, "MAX_PIXELS", 200):
            self.run_prepare()
        reject = next(item for item in self.read_source()["rejections"] if item["image_id"] == first)
        self.assertEqual(reject["reason"], "invalid_jpeg")
        self.assertIn("pixel budget", reject["detail"])

    def test_final_source_and_pixel_identity_checks_are_stricter_than_reader(self):
        real_select = prepare._select_images
        for identity_key in ("source_id", "pixel_sha256"):
            with self.subTest(identity_key=identity_key):
                def corrupt_identity(*args, **kwargs):
                    images, rejects, quotas = real_select(*args, **kwargs)
                    images[1][identity_key] = images[0][identity_key]
                    return images, rejects, quotas

                with patch.object(prepare, "_select_images", side_effect=corrupt_identity):
                    with self.assertRaisesRegex(prepare.DatasetPreparationError, f"duplicate {identity_key}"):
                        self.run_prepare()
                self.assert_unpublished()

    def test_insufficient_class_fails_without_false_complete(self):
        self.fixture = SmallArchives(self.directory / "too-small", train_count=1, test_count=1)
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "quotas will not be reduced"):
            self.run_prepare()
        self.assert_unpublished()

    def test_rejected_cross_split_duplicates_survive_failure_as_diagnostics(self):
        training = self.fixture.ordered_ids("beagle")[0]
        for testing in self.fixture.ordered_ids("beagle", "test"):
            self.fixture.images[f"images/{testing}.jpg"] = self.fixture.images[f"images/{training}.jpg"]
        with self.assertRaises(prepare.DatasetPreparationError) as context:
            self.run_prepare()
        self.assertTrue(any(item.get("cross_official_split") for item in context.exception.diagnostics))
        self.assert_unpublished()

    def test_official_source_id_overlap_is_a_fatal_annotation_error(self):
        self.fixture.lines["test"].append(self.fixture.lines["trainval"][0])
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "source_id overlap") as context:
            self.run_prepare()
        self.assertEqual(context.exception.diagnostics[0]["reason"], "official_split_source_id_overlap")
        self.assert_unpublished()

    def test_malformed_annotation_fails_before_publication(self):
        self.fixture.lines["trainval"][0] = "beagle_1 5 1 4"
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "mismatch"):
            self.run_prepare()
        self.assert_unpublished()

    def test_known_excluded_dog_and_cat_do_not_expand_frozen_subset(self):
        for prefix in ("shiba_inu", "Abyssinian"):
            class_id, species, breed_id = prepare.OFFICIAL_ANNOTATIONS[prefix]
            for split, number in (("trainval", 1), ("test", 101)):
                image_id = f"{prefix}_{number}"
                self.fixture.lines[split].append(f"{image_id} {class_id} {species} {breed_id}")
                self.fixture.images[f"images/{image_id}.jpg"] = b"Ignored non-selected bad JPEG"
        self.run_prepare()
        breeds = {item["breed"] for item in self.read_manifest()["images"]}
        self.assertEqual(breeds, {item[0] for item in prepare.FROZEN_MAPPING})
        self.assertNotIn("shiba_inu", breeds)
        self.assertNotIn("Abyssinian", breeds)

    def test_real_md5_mismatch_is_rejected_and_not_reported_as_official_sha(self):
        self.fixture.write()
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "MD5 mismatch.*not an official SHA256"):
            prepare.prepare_dataset(self.fixture.images_path, self.fixture.annotations_path, self.output, **SMALL)
        self.assert_unpublished()

    def test_annotations_archive_md5_mismatch_is_rejected(self):
        config = self.fixture.write()
        config["archive_checksums"]["annotations"]["md5"] = "0" * 32
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "annotations archive MD5 mismatch"):
            self.run_prepare(config=config)
        self.assert_unpublished()

    def test_actual_data_gates_are_not_lowered_for_small_function_counts(self):
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "data gates are not met"):
            self.run_prepare(fake_readiness=False)
        self.assert_unpublished()

    def test_refuse_existing_output_even_if_empty(self):
        self.output.mkdir()
        sentinel = self.output / "user-original.txt"
        sentinel.write_text("do not delete", encoding="utf-8")
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "already exists"):
            self.run_prepare()
        self.assertEqual(sentinel.read_text(), "do not delete")
        self.assertFalse((self.output / "prepared.complete.json").exists())
        self.assertEqual(list(self.directory.glob(".prepared.prepare-*")), [])

    def test_explicit_overwrite_preserves_old_directory_as_backup(self):
        self.output.mkdir()
        (self.output / "user-original.txt").write_text("keep original", encoding="utf-8")
        result = self.run_prepare(overwrite=True)
        backup = Path(result["backup_dir"])
        self.assertEqual(backup.parent, self.output.parent)
        self.assertEqual((backup / "user-original.txt").read_text(), "keep original")
        self.assertTrue((self.output / "prepared.complete.json").is_file())

    def test_failed_overwrite_keeps_original_data_untouched(self):
        self.output.mkdir()
        sentinel = self.output / "user-original.txt"
        sentinel.write_text("keep on failure", encoding="utf-8")
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "data gates are not met"):
            self.run_prepare(overwrite=True, fake_readiness=False)
        self.assertEqual(sentinel.read_text(), "keep on failure")
        self.assertEqual(list(self.directory.glob(".prepared.backup-*")), [])
        self.assertFalse((self.output / "prepared.complete.json").exists())

    def test_publication_failure_rolls_back_old_output(self):
        self.output.mkdir()
        sentinel = self.output / "user-original.txt"
        sentinel.write_text("rollback original", encoding="utf-8")
        real_rename = prepare.os.rename
        calls = []

        def fail_second(source, destination):
            calls.append((source, destination))
            if len(calls) == 2:
                raise OSError("synthetic publication failure")
            return real_rename(source, destination)

        with patch.object(prepare.os, "rename", side_effect=fail_second):
            with self.assertRaisesRegex(prepare.DatasetPreparationError, "synthetic publication failure"):
                self.run_prepare(overwrite=True)
        self.assertEqual(sentinel.read_text(), "rollback original")
        self.assertFalse((self.output / "prepared.complete.json").exists())
        self.assertEqual(list(self.directory.glob(".prepared.prepare-*")), [])
        self.assertEqual(list(self.directory.glob(".prepared.backup-*")), [])

    def test_output_must_not_contain_raw_archives(self):
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "must not contain"):
            self.run_prepare(output=self.fixture.directory, overwrite=True)
        self.assertTrue(self.fixture.images_path.exists())
        self.assertTrue(self.fixture.annotations_path.exists())

    def test_no_extractall_and_complete_marker_is_written_last(self):
        real_write = prepare._write_json
        writes = []

        def track(path, value):
            if path.name == "prepared.complete.json":
                self.assertTrue((path.parent / "manifest.json").is_file())
                self.assertTrue((path.parent / "source.json").is_file())
            writes.append(path.name)
            return real_write(path, value)

        with patch.object(tarfile.TarFile, "extractall", side_effect=AssertionError("extractall is forbidden")), \
                patch.object(prepare, "_write_json", side_effect=track):
            self.run_prepare()
        self.assertEqual(writes, ["manifest.json", "source.json", "prepared.complete.json"])

    def test_manifest_byte_validation_is_called_before_marker(self):
        real_write = prepare._write_json

        def corrupt_after_source(path, value):
            result = real_write(path, value)
            if path.name == "source.json":
                first = next((path.parent / "images").iterdir())
                first.write_bytes(first.read_bytes() + b"synthetic tamper")
            return result

        with patch.object(prepare, "_write_json", side_effect=corrupt_after_source):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.run_prepare()
        self.assert_unpublished()

    def test_raw_archive_rehash_detects_changes_during_preparation(self):
        original_spool = prepare._spool_candidates

        def change_after_spool(*args, **kwargs):
            original_spool(*args, **kwargs)
            with self.fixture.images_path.open("ab") as handle:
                handle.write(b"synthetic raw archive mutation")

        with patch.object(prepare, "_spool_candidates", side_effect=change_after_spool):
            with self.assertRaisesRegex(prepare.DatasetPreparationError, "archive changed"):
                self.run_prepare()
        self.assert_unpublished()

    def test_duplicate_tar_headers_fail_without_extracting(self):
        name, payload = next(iter(self.fixture.images.items()))
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        self.fixture.extras["images"].append((member, payload))
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "Duplicate tar member"):
            self.run_prepare()
        self.assert_unpublished()

    def test_unsafe_tar_paths_and_links_are_rejected(self):
        variants = [
            ("../escape.jpg", tarfile.REGTYPE), ("/images/absolute.jpg", tarfile.REGTYPE),
            ("C:/images/absolute.jpg", tarfile.REGTYPE), ("images/../escape.jpg", tarfile.REGTYPE),
            ("images\\escape.jpg", tarfile.REGTYPE), ("images/nested/escape.jpg", tarfile.REGTYPE),
            ("images/link.jpg", tarfile.SYMTYPE), ("images/hard.jpg", tarfile.LNKTYPE),
            ("images/device", tarfile.CHRTYPE), ("images/fifo", tarfile.FIFOTYPE),
        ]
        for name, kind in variants:
            with self.subTest(name=name, kind=kind):
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "../../outside"
                payload = b"x" if kind == tarfile.REGTYPE else None
                member.size = 1 if payload else 0
                self.fixture.extras["images"] = [(member, payload)]
                with self.assertRaises(prepare.DatasetPreparationError):
                    self.run_prepare()
                self.assert_unpublished()
        self.assertFalse((self.directory / "escape.jpg").exists())

    def test_truncated_tar_fails_without_complete(self):
        config = self.fixture.write()
        raw = self.fixture.images_path.read_bytes()
        self.fixture.images_path.write_bytes(raw[:len(raw) // 2])
        config["archive_checksums"]["images"]["md5"] = digest(self.fixture.images_path, "md5")
        with self.assertRaises(prepare.DatasetPreparationError):
            self.run_prepare(config=config)
        self.assert_unpublished()


class SafetyAndHelpTests(unittest.TestCase):
    def test_header_size_and_member_budgets_are_finite(self):
        for role, name, size in (("images", "images/beagle_1.jpg", prepare.MAX_JPEG_BYTES + 1),
                                 ("annotations", "annotations/trainval.txt", prepare.MAX_TEXT_BYTES + 1)):
            with self.subTest(role=role):
                member = tarfile.TarInfo(name)
                member.size = size
                with self.assertRaisesRegex(prepare.DatasetPreparationError, "size budget"):
                    prepare._header_index([member], role)
        members = [tarfile.TarInfo("images"), tarfile.TarInfo("images/ignored")]
        members[0].type = tarfile.DIRTYPE
        with patch.object(prepare, "MAX_MEMBERS", 1):
            with self.assertRaisesRegex(prepare.DatasetPreparationError, "too many headers"):
                prepare._header_index(members, "images")
        members[1].size = prepare.MAX_EXPANDED_BYTES + 1
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "expanded byte budget"):
            prepare._header_index(members, "images")
        sparse = tarfile.TarInfo("images/beagle_1.jpg")
        sparse.sparse = [(0, 1)]
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "sparse"):
            prepare._safe_member_name(sparse, "images")

    def test_missing_required_annotation_text_is_rejected(self):
        member = tarfile.TarInfo("annotations/trainval.txt")
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "missing annotations"):
            prepare._header_index([member], "annotations")

    def test_rgb_dimensions_are_part_of_pixel_identity(self):
        first = jpeg("dimensions")
        metadata = prepare._decoded_pixels(first)
        with Image.open(io.BytesIO(first)) as image:
            rgb = image.convert("RGB")
            expected = hashlib.sha256(b"RGB\0" + struct.pack(">II", 12, 10) + rgb.tobytes()).hexdigest()
            rgb.close()
        self.assertEqual(metadata["pixel_sha256"], expected)

    def test_help_has_no_torch_torchvision_or_pillow_dependency(self):
        script = ROOT / "tools" / "prepare_classifier_dataset.py"
        blocked_runner = (
            "import builtins,runpy,sys; original=builtins.__import__; "
            "blocked={'torch','torchvision','PIL','onnxruntime','numpy','pyarrow'}; "
            "exec('def guarded(name,*args,**kwargs):\\n"
            " if name.split(\".\")[0] in blocked: raise AssertionError(\"heavy import: \"+name)\\n"
            " return original(name,*args,**kwargs)'); "
            "builtins.__import__=guarded; "
            f"sys.argv=[{str(script)!r},'--help']; runpy.run_path({str(script)!r},run_name='__main__')"
        )
        result = subprocess.run([sys.executable, "-B", "-I", "-S", "-c", blocked_runner],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--images-archive", result.stdout)
        self.assertIn("--annotations-archive", result.stdout)
        self.assertIn("--overwrite", result.stdout)
        self.assertNotIn("--holdout-count", result.stdout)

    def test_help_and_missing_input_return_no_dataset(self):
        script = ROOT / "tools" / "prepare_classifier_dataset.py"
        with tempfile.TemporaryDirectory(prefix="classifier-cli-unit-") as temporary:
            output = Path(temporary) / "no-output"
            result = subprocess.run([sys.executable, "-B", "-I", "-S", str(script),
                                     "--images-archive", str(Path(temporary) / "missing-images.tar.gz"),
                                     "--annotations-archive", str(Path(temporary) / "missing-annotations.tar.gz"),
                                     "--output-dir", str(output)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(result.stderr)["status"], "failed")
            self.assertFalse(output.exists())


class SyntheticMirror:
    """Small fake Parquet streams; checksums/readiness are explicitly test mocks."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir()
        self.rows = {"train": [], "test": []}
        self.batch_sizes = []
        self.closed = []
        self.columns = ["image", "label", "image_id", "label_cat_dog"]
        self.reported_rows = {}
        self.config = copy.deepcopy(prepare.load_dataset_config())
        names = {str(index): prefix.lower() for index, (prefix, _, _) in enumerate(prepare._OXFORD_CLASS_ORDER)}
        self.metadata = {"schema_version": 1, "dataset_id": prepare.MIRROR_DATASET_ID,
                         "revision": prepare.MIRROR_REVISION, "license": "cc-by-sa-4.0",
                         "license_basis": {"dataset_card": "TEST-ONLY fixture card, not real source evidence"},
                         "authors": self.config["authors"], "publisher_url": self.config["homepage"],
                         "label_names": names, "files": {}}
        self.config["task_usage"] += " SYNTHETIC UNITTEST MIRROR ONLY."
        self.config["mirror"]["sha256_reference"] = "TEST-ONLY synthetic file digest, not an actual LFS object"
        for prefix, _, _ in prepare.FROZEN_MAPPING:
            label = next(int(key) for key, name in names.items() if name == prefix)
            for split, numbers in (("train", (1, 2, 3)), ("test", (101, 102, 103))):
                for number in numbers:
                    image_id = f"{prefix}_{number}"
                    self.rows[split].append({"image": {"bytes": jpeg(image_id), "path": f"{image_id}.jpg"},
                                             "image_id": image_id, "label": label, "label_cat_dog": 1})
        self.write()

    def write(self, *, actual_parquet=False):
        for role in ("train", "test"):
            path = self.directory / f"{role}.parquet"
            if actual_parquet:
                import pyarrow as arrow
                import pyarrow.parquet as parquet
                schema = arrow.schema([
                    ("image", arrow.struct([("bytes", arrow.binary()), ("path", arrow.string())])),
                    ("label", arrow.int64()), ("image_id", arrow.string()), ("label_cat_dog", arrow.int64())])
                parquet.write_table(arrow.Table.from_pylist(self.rows[role], schema=schema), path, row_group_size=17)
            else:
                path.write_bytes(f"SYNTHETIC UNITTEST ONLY; fake {role} Parquet container".encode())
            sha, size = digest(path), path.stat().st_size
            self.config["mirror"]["files"][role] = {"sha256": sha, "size_bytes": size,
                                                     "row_count": len(self.rows[role])}
            self.metadata["files"][role] = {"path": path.name, "sha256": sha, "size_bytes": size, "split": role}
        self.write_metadata()

    def write_metadata(self):
        (self.directory / "mirror-source.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def ordered(self, prefix, role="train", seed=prepare.DEFAULT_SEED):
        domain = "trainval" if role == "train" else "test"
        return sorted([row for row in self.rows[role] if row["image_id"].rsplit("_", 1)[0] == prefix],
                      key=lambda row: prepare._rank(seed, f"candidates:{domain}", row["image_id"]))

    def reader(self, path):
        role = path.stem
        fixture = self

        class Reader:
            schema_arrow = SimpleNamespace(names=fixture.columns)
            metadata = SimpleNamespace(num_rows=fixture.reported_rows.get(role, len(fixture.rows[role])))

            def iter_batches(self, *, batch_size, columns, use_threads):
                fixture.batch_sizes.append(batch_size)
                if columns != ["image", "label", "image_id", "label_cat_dog"] or use_threads:
                    raise AssertionError("Unexpected columns or threaded mirror reading")
                for start in range(0, len(fixture.rows[role]), batch_size):
                    rows = copy.deepcopy(fixture.rows[role][start:start + batch_size])
                    yield SimpleNamespace(num_rows=len(rows), to_pylist=lambda rows=rows: rows)

            def close(self):
                fixture.closed.append(role)

        return Reader()


class MirrorPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="classifier-mirror-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.fixture = SyntheticMirror(self.directory / "mirror")
        self.output = self.directory / "prepared"

    def run_prepare(self, *, output=None, fake_readiness=True, fake_parquet=True, **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch.object(prepare, "load_dataset_config", return_value=self.fixture.config))
            if fake_parquet:
                stack.enter_context(patch.object(prepare, "_parquet_reader", side_effect=self.fixture.reader))
            if fake_readiness:
                stack.enter_context(patch.object(benchmark, "dataset_readiness", side_effect=synthetic_readiness))
            return prepare.prepare_dataset_from_mirror(self.fixture.directory, output or self.output, **(SMALL | kwargs))

    def assert_unpublished(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.glob(".prepared.prepare-*")), [])

    def source(self):
        return json.loads((self.output / "source.json").read_text(encoding="utf-8"))

    def manifest(self, output=None):
        return json.loads(((output or self.output) / "manifest.json").read_text(encoding="utf-8"))

    def test_pinned_mirror_manifest_provenance_and_bounded_stream(self):
        result = self.run_prepare()
        source = self.source()
        self.assertEqual(result["mirror_revision"], prepare.MIRROR_REVISION)
        self.assertEqual(source["mirror"]["dataset_id"], prepare.MIRROR_DATASET_ID)
        self.assertEqual(source["license"], prepare.SOURCE_LICENSE)
        self.assertIn("Original image owners", source["copyright_owner"])
        self.assertEqual(self.fixture.batch_sizes, [32, 32])
        self.assertEqual(self.fixture.closed, ["train", "test"])
        for role in ("images", "annotations"):
            self.assertFalse(source["archives"][role]["md5_verified"])
            self.assertIsNone(source["archives"][role]["md5_observed"])
            self.assertIsNone(source["archives"][role]["sha256_observed"])
        for role in ("train", "test"):
            self.assertTrue(source["mirror"]["files"][role]["sha256_matches_lfs_reference"])
            self.assertEqual(source["mirror"]["files"][role]["sha256_observed"], digest(self.fixture.directory / f"{role}.parquet"))
        expected = {row["image_id"]: row for rows in self.fixture.rows.values() for row in rows}
        for entry in self.manifest()["images"]:
            self.assertEqual(entry["source_original_split"], "test" if entry["split"] == "holdout" else "train")
            self.assertTrue(entry["ground_truth_source"].startswith("mirror:"))
            self.assertNotIn("annotations/", entry["ground_truth_source"])
            self.assertNotIn("CLASS_ID", entry["original_annotation"])
            self.assertEqual(entry["original_annotation"]["label_cat_dog"], 1)
            self.assertEqual((self.output / entry["path"]).read_bytes(), expected[entry["id"]]["image"]["bytes"])
            self.assertEqual(entry["label"] is None, entry["split"] == "calibration")
        marker = json.loads((self.output / "prepared.complete.json").read_text())
        self.assertEqual(marker["source_sha256"], digest(self.output / "source.json"))
        benchmark.load_manifest(self.output / "manifest.json")

    def test_supplied_author_initial_and_excluded_cat_capitalization(self):
        self.fixture.metadata["authors"] = ["Omkar M. Parkhi", *self.fixture.metadata["authors"][1:]]
        self.fixture.write_metadata()
        for role, number in (("train", 1), ("test", 101)):
            image_id = f"Abyssinian_{number}"
            self.fixture.rows[role].append({"image_id": image_id, "label": 0, "label_cat_dog": 0,
                "image": {"bytes": b"Ignored excluded cat fixture", "path": f"{image_id}.jpg"}})
            self.fixture.config["mirror"]["files"][role]["row_count"] += 1
        self.run_prepare()
        self.assertEqual(self.source()["mirror"]["authors_as_supplied"][0], "Omkar M. Parkhi")
        self.assertEqual(self.source()["mirror"]["caller_supplied_source_metadata"]["label_names"]["0"], "abyssinian")
        self.assertNotIn("Abyssinian", {entry["breed"] for entry in self.manifest()["images"]})

    def test_sidecar_and_raw_file_mutation_during_stream_are_rejected(self):
        real_spool = prepare._spool_mirror
        original_sidecar = (self.fixture.directory / "mirror-source.json").read_bytes()
        for mutate in ("sidecar", "parquet"):
            with self.subTest(mutate=mutate):
                def mutate_after_spool(*args, **kwargs):
                    records = real_spool(*args, **kwargs)
                    path = self.fixture.directory / ("mirror-source.json" if mutate == "sidecar" else "train.parquet")
                    with path.open("ab") as handle:
                        handle.write(b"synthetic mutation")
                    return records

                with patch.object(prepare, "_spool_mirror", side_effect=mutate_after_spool):
                    with self.assertRaisesRegex(prepare.DatasetPreparationError, "changed during preparation"):
                        self.run_prepare()
                self.assert_unpublished()
                (self.fixture.directory / "mirror-source.json").write_bytes(original_sidecar)
                self.fixture.write()

    def test_full_sha_verification_happens_before_parquet_import(self):
        with (self.fixture.directory / "test.parquet").open("ab") as handle:
            handle.write(b"tampered")
        with patch.object(prepare, "_parquet_reader", side_effect=AssertionError("must not read Parquet")):
            with patch.object(prepare, "load_dataset_config", return_value=self.fixture.config):
                with self.assertRaisesRegex(prepare.DatasetPreparationError, "complete-file LFS SHA256"):
                    prepare.prepare_dataset_from_mirror(self.fixture.directory, self.output, **SMALL)
        self.assert_unpublished()

    def test_revision_license_authors_and_sidecar_sha_reference_are_strict(self):
        original = copy.deepcopy(self.fixture.metadata)
        changes = [{"revision": "0" * 40}, {"dataset_id": "other/dataset"}, {"license": "all-rights-reserved"},
                   {"license_basis": ""}, {"authors": []}, {"publisher_url": "https://example.invalid/"}]
        bad_sha = copy.deepcopy(original)
        bad_sha["files"]["train"]["sha256"] = "0" * 64
        variants = [original | change for change in changes] + [bad_sha]
        for metadata in variants:
            with self.subTest(metadata=metadata):
                self.fixture.metadata = metadata
                self.fixture.write_metadata()
                with self.assertRaises(prepare.DatasetPreparationError):
                    self.run_prepare()
                self.assert_unpublished()
        self.fixture.metadata = original

    def test_sidecar_and_image_paths_cannot_escape_or_be_absolute(self):
        for path in ("../train.parquet", "/train.parquet", "C:/train.parquet", "train\\file.parquet"):
            with self.subTest(path=path):
                self.fixture.metadata["files"]["train"]["path"] = path
                self.fixture.write_metadata()
                with self.assertRaises(prepare.DatasetPreparationError):
                    self.run_prepare()
                self.assert_unpublished()
        self.fixture.metadata["files"]["train"]["path"] = "train.parquet"
        self.fixture.write_metadata()
        original = self.fixture.rows["train"][0]["image"]["path"]
        for path in ("../beagle_1.jpg", "/beagle_1.jpg", "C:/beagle_1.jpg", "folder\\beagle_1.jpg", "wrong_stem.jpg"):
            with self.subTest(image_path=path):
                self.fixture.rows["train"][0]["image"]["path"] = path
                with self.assertRaises(prepare.DatasetPreparationError):
                    self.run_prepare()
                self.assert_unpublished()
        self.fixture.rows["train"][0]["image"]["path"] = original

    def test_image_id_card_label_and_species_are_strict(self):
        original = copy.deepcopy(self.fixture.rows["train"][0])
        variants = [original | {"image_id": "../beagle_1"}, original | {"image_id": 7},
                    original | {"label": True}, original | {"label": 37},
                    original | {"label": original["label"] + 1}, original | {"label_cat_dog": 0},
                    original | {"label_cat_dog": True}, original | {"image_id": "unknown_breed_1"}]
        for row in variants:
            with self.subTest(row={key: value for key, value in row.items() if key != "image"}):
                self.fixture.rows["train"][0] = row
                with self.assertRaises(prepare.DatasetPreparationError):
                    self.run_prepare()
                self.assert_unpublished()
        self.fixture.rows["train"][0] = original

    def test_label_names_cannot_add_a_fuzzy_alias(self):
        key = next(key for key, value in self.fixture.metadata["label_names"].items() if value == "beagle")
        self.fixture.metadata["label_names"][key] = "beagle_mix"
        self.fixture.write_metadata()
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "label_names"):
            self.run_prepare()
        self.assert_unpublished()

    def test_bad_jpeg_and_cross_split_byte_pixel_duplicates_are_replaced(self):
        first = self.fixture.ordered("beagle")[0]
        first["image"]["bytes"] = b"bad JPEG TEST ONLY"
        training = self.fixture.ordered("boxer")[0]
        testing = self.fixture.ordered("boxer", "test")[0]
        testing["image"]["bytes"] = training["image"]["bytes"]
        training_pixel = self.fixture.ordered("pug")[0]
        testing_pixel = self.fixture.ordered("pug", "test")[0]
        testing_pixel["image"]["bytes"] = comment_jpeg(training_pixel["image"]["bytes"])
        self.run_prepare()
        rejects = self.source()["rejections"]
        self.assertEqual({item["reason"] for item in rejects}, {"invalid_jpeg", "duplicate_sha256", "duplicate_rgb_pixels"})
        for item in rejects:
            if item["reason"].startswith("duplicate"):
                self.assertTrue(item["cross_official_split"])
                self.assertEqual(item["duplicate_of_original_split"], "train")

    def test_source_id_overlap_retains_diagnostic_and_never_publishes(self):
        self.fixture.rows["test"][0] = copy.deepcopy(self.fixture.rows["train"][0])
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "source_id overlap") as context:
            self.run_prepare()
        self.assertEqual(context.exception.diagnostics[0]["source_original_splits"], ["train", "test"])
        self.assert_unpublished()

    def test_columns_and_row_count_must_match_the_frozen_mirror(self):
        self.fixture.columns = ["image", "label", "image_id"]
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "columns or frozen row count"):
            self.run_prepare()
        self.fixture.columns.append("label_cat_dog")
        self.fixture.reported_rows["train"] = 1
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "columns or frozen row count"):
            self.run_prepare()
        self.assert_unpublished()

    def test_small_targets_still_fail_actual_benchmark_gates(self):
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "data gates are not met"):
            self.run_prepare(fake_readiness=False)
        self.assert_unpublished()

    def test_order_and_seed_match_shared_sampling_contract(self):
        first = self.run_prepare()
        self.fixture.rows["train"].reverse()
        self.fixture.rows["test"].reverse()
        self.run_prepare(output=self.directory / "second")
        fields = lambda data: [(row["id"], row["split"], row["sha256"]) for row in data["images"]]
        self.assertEqual(fields(self.manifest()), fields(self.manifest(self.directory / "second")))
        self.run_prepare(output=self.directory / "third", seed=19)
        self.assertNotEqual(fields(self.manifest()), fields(self.manifest(self.directory / "third")))
        self.assertEqual(first["counts"], {"calibration": 20, "development": 20, "holdout": 20})

    def test_overwrite_requires_explicit_flag_and_preserves_backup(self):
        self.output.mkdir()
        sentinel = self.output / "user-data.txt"
        sentinel.write_text("keep old data")
        with self.assertRaisesRegex(prepare.DatasetPreparationError, "already exists"):
            self.run_prepare()
        result = self.run_prepare(overwrite=True)
        self.assertEqual((Path(result["backup_dir"]) / "user-data.txt").read_text(), "keep old data")

    def test_missing_pyarrow_is_a_controlled_failure(self):
        with patch.object(prepare, "_parquet_reader", side_effect=prepare.DatasetPreparationError("PyArrow required")):
            with patch.object(prepare, "load_dataset_config", return_value=self.fixture.config):
                with self.assertRaisesRegex(prepare.DatasetPreparationError, "PyArrow required"):
                    prepare.prepare_dataset_from_mirror(self.fixture.directory, self.output, **SMALL)
        self.assert_unpublished()

    def test_real_pyarrow_small_parquet_when_parent_installed_dependency(self):
        try:
            import pyarrow
        except ImportError:
            self.skipTest("PyArrow not installed yet; parent owns dependency installation")
        self.fixture.write(actual_parquet=True)
        self.run_prepare(fake_parquet=False)
        self.assertEqual(len(self.manifest()["images"]), 60)

    def test_mirror_and_tar_cli_inputs_are_mutually_exclusive(self):
        script = ROOT / "tools" / "prepare_classifier_dataset.py"
        for arguments in (["--mirror-dir", "fixture", "--images-archive", "images.tar.gz"],
                          ["--mirror-dir", "fixture", "--annotations-archive", "annotations.tar.gz"],
                          ["--images-archive", "images.tar.gz"]):
            result = subprocess.run([sys.executable, "-B", "-I", "-S", str(script), *arguments],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
