#!/usr/bin/env python3
"""Prepare the approved local Oxford dog subset, without network or model imports.

Only caller-supplied archives or a pinned local Parquet mirror are read.
The production completion marker is
published only after the existing benchmark's unmodified data gates pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import uuid
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "classifier_dataset.json"
SOURCE_LICENSE = "CC BY-SA 4.0"
DEFAULT_SEED = 20261001
DEFAULT_COUNTS = {"calibration": 256, "development": 128, "holdout": 1000}
ARCHIVE_MD5 = {
    "images": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    "annotations": "95a8c909bbe2e81eed6a22bccdf3f68f",
}
MIRROR_DATASET_ID = "timm/oxford-iiit-pet"
MIRROR_REVISION = "089695c834a7deb60505b7cc506672db1c31a6aa"
MIRROR_FILES = {
    "train": {"sha256": "b9ff432b33c77f4b50014684f25686a1cd523090686571743a6b7a257af27422",
              "size_bytes": 377611203, "row_count": 3680},
    "test": {"sha256": "0997d4744ee6740e96f17f56b40b97c70076a4e4dad565bc63d0ffd03906f754",
             "size_bytes": 412654113, "row_count": 3669},
}
MIRROR_BATCH_SIZE = 32
MAX_MIRROR_ROWS = 10000
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_EXPANDED_BYTES = 4 * 1024**3
MAX_MEMBERS = 100000
MAX_JPEG_BYTES = 20 * 1024**2
MAX_TEXT_BYTES = 2 * 1024**2
MAX_PIXELS = 40000000
IMAGE_ID = re.compile(r"^([A-Za-z][A-Za-z_]*)_([1-9][0-9]*)$")
REQUIRED_TEXT = ("annotations/trainval.txt", "annotations/test.txt")

# This manual allow-list must not be expanded by predictions or fuzzy matching.
FROZEN_MAPPING = (
    ("basset_hound", "basset", 161),
    ("beagle", "beagle", 162),
    ("boxer", "boxer", 242),
    ("chihuahua", "Chihuahua", 151),
    ("english_setter", "English setter", 212),
    ("german_shorthaired", "German short-haired pointer", 210),
    ("great_pyrenees", "Great Pyrenees", 257),
    ("japanese_chin", "Japanese spaniel", 152),
    ("keeshond", "keeshond", 261),
    ("leonberger", "Leonberg", 255),
    ("miniature_pinscher", "miniature pinscher", 237),
    ("newfoundland", "Newfoundland", 256),
    ("pomeranian", "Pomeranian", 259),
    ("pug", "pug", 254),
    ("saint_bernard", "Saint Bernard", 247),
    ("samoyed", "Samoyed", 258),
    ("scottish_terrier", "Scotch terrier", 199),
    ("staffordshire_bull_terrier", "Staffordshire bullterrier", 179),
    ("wheaten_terrier", "soft-coated wheaten terrier", 202),
    ("yorkshire_terrier", "Yorkshire terrier", 187),
)

# Official CLASS_ID order; entries are (filename prefix, SPECIES, BREED_ID).
# Non-selected classes are checked as annotations, never mapped to ImageNet.
_OXFORD_CLASS_ORDER = (
    ("Abyssinian", 1, 1), ("american_bulldog", 2, 1),
    ("american_pit_bull_terrier", 2, 2), ("basset_hound", 2, 3),
    ("beagle", 2, 4), ("Bengal", 1, 2), ("Birman", 1, 3),
    ("Bombay", 1, 4), ("boxer", 2, 5), ("British_Shorthair", 1, 5),
    ("chihuahua", 2, 6), ("Egyptian_Mau", 1, 6),
    ("english_cocker_spaniel", 2, 7), ("english_setter", 2, 8),
    ("german_shorthaired", 2, 9), ("great_pyrenees", 2, 10),
    ("havanese", 2, 11), ("japanese_chin", 2, 12),
    ("keeshond", 2, 13), ("leonberger", 2, 14), ("Maine_Coon", 1, 7),
    ("miniature_pinscher", 2, 15), ("newfoundland", 2, 16),
    ("Persian", 1, 8), ("pomeranian", 2, 17), ("pug", 2, 18),
    ("Ragdoll", 1, 9), ("Russian_Blue", 1, 10), ("saint_bernard", 2, 19),
    ("samoyed", 2, 20), ("scottish_terrier", 2, 21), ("shiba_inu", 2, 22),
    ("Siamese", 1, 11), ("Sphynx", 1, 12),
    ("staffordshire_bull_terrier", 2, 23), ("wheaten_terrier", 2, 24),
    ("yorkshire_terrier", 2, 25),
)
OFFICIAL_ANNOTATIONS = {
    prefix: (class_id, species, breed_id)
    for class_id, (prefix, species, breed_id) in enumerate(_OXFORD_CLASS_ORDER, 1)
}
# Exact names in the pinned mirror card; unselected cat filenames may retain capitals.
MIRROR_LABEL_NAMES = {str(index): prefix.lower()
                      for index, (prefix, _, _) in enumerate(_OXFORD_CLASS_ORDER)}


class DatasetPreparationError(ValueError):
    """Preparation failed; diagnostics remain available without a false marker."""

    def __init__(self, message: str, *, diagnostics: list | None = None):
        super().__init__(message)
        self.diagnostics = list(diagnostics or [])


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: dict) -> str:
    payload = _json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _unique_keys(pairs: list) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise DatasetPreparationError(f"Duplicate config key: {key}")
        value[key] = item
    return value


def load_dataset_config() -> dict:
    """Read and validate the frozen mapping; importing torch is not necessary."""
    try:
        raw = CONFIG_PATH.read_bytes()
        config = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys,
                            parse_constant=lambda text: (_ for _ in ()).throw(
                                DatasetPreparationError(f"Non-finite config value: {text}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetPreparationError(f"Cannot read dataset config: {error}") from error
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise DatasetPreparationError("Dataset config requires schema_version=1")
    classes = config.get("classes")
    if not isinstance(classes, list) or len(classes) != len(FROZEN_MAPPING):
        raise DatasetPreparationError("The approved mapping contains exactly 20 classes")
    expected = {prefix: (name, index) for prefix, name, index in FROZEN_MAPPING}
    seen = set()
    for item in classes:
        if not isinstance(item, dict):
            raise DatasetPreparationError("Class mapping entries must be objects")
        prefix = item.get("prefix")
        if not isinstance(prefix, str) or prefix not in expected or prefix in seen:
            raise DatasetPreparationError(f"Unknown or duplicate frozen breed prefix: {prefix!r}")
        if (item.get("canonical_name"), item.get("imagenet_index")) != expected[prefix] or type(item.get("imagenet_index")) is not int:
            raise DatasetPreparationError(f"Frozen ImageNet V1 mapping changed for {prefix}")
        annotation = tuple(item.get(key) for key in ("class_id", "species", "breed_id"))
        if annotation != OFFICIAL_ANNOTATIONS[prefix] or any(type(number) is not int for number in annotation):
            raise DatasetPreparationError(f"Frozen Oxford annotation mapping changed for {prefix}")
        if not isinstance(item.get("alias"), str) or not item["alias"].strip():
            raise DatasetPreparationError(f"Missing manual alias explanation: {prefix}")
        seen.add(prefix)
    if config.get("license") != SOURCE_LICENSE or config.get("weights_version") != "IMAGENET1K_V1":
        raise DatasetPreparationError("Source license or fixed weights version changed")
    if (config.get("dataset") != "Oxford-IIIT Pet Dataset"
            or config.get("homepage") != "https://www.robots.ox.ac.uk/~vgg/data/pets/"
            or config.get("authors") != ["Omkar M Parkhi", "Andrea Vedaldi", "Andrew Zisserman", "C. V. Jawahar"]):
        raise DatasetPreparationError("Frozen source attribution changed")
    for key in ("license_url", "license_notes", "task_usage", "mapping_policy"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise DatasetPreparationError(f"Missing source metadata: {key}")
    if config.get("counts") != DEFAULT_COUNTS or config.get("seed") != DEFAULT_SEED:
        raise DatasetPreparationError("The approved default data budget or seed changed")
    resources = config.get("archive_checksums")
    if not isinstance(resources, dict):
        raise DatasetPreparationError("Missing archive checksum references")
    for role in ARCHIVE_MD5:
        resource = resources.get(role)
        expected_url = config["homepage"] + f"data/{role}.tar.gz"
        if (not isinstance(resource, dict) or resource.get("url") != expected_url
                or resource.get("md5") != ARCHIVE_MD5[role]
                or not isinstance(resource.get("md5_reference"), str)
                or "_RESOURCES" not in resource["md5_reference"]):
            raise DatasetPreparationError(f"Frozen archive MD5 reference changed: {role}")
    if not isinstance(config.get("limitations"), list) or not config["limitations"]:
        raise DatasetPreparationError("Dataset limitations must remain explicit")
    mirror = config.get("mirror")
    if (not isinstance(mirror, dict) or mirror.get("dataset_id") != MIRROR_DATASET_ID
            or mirror.get("revision") != MIRROR_REVISION or mirror.get("license") != "cc-by-sa-4.0"
            or mirror.get("publisher_url") != config["homepage"]
            or mirror.get("files") != MIRROR_FILES or mirror.get("batch_size") != MIRROR_BATCH_SIZE
            or not isinstance(mirror.get("sha256_reference"), str)
            or not isinstance(mirror.get("ground_truth_basis"), str)):
        raise DatasetPreparationError("Frozen mirror identity, revision, license or LFS references changed")
    config["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def validate_category_names(categories: list | tuple) -> None:
    """Optional check of a caller-provided V1 category table, without importing it."""
    if not isinstance(categories, (list, tuple)) or len(categories) != 1000:
        raise DatasetPreparationError("Expected an ImageNet V1 table of exactly 1000 categories")
    for prefix, canonical, index in FROZEN_MAPPING:
        if categories[index] != canonical:
            raise DatasetPreparationError(f"ImageNet V1 canonical name/index mismatch: {prefix}")


def _integer(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise DatasetPreparationError(f"{name} must be an integer in [0, {maximum}]")
    return value


def _raw_digests(handle) -> dict:
    handle.seek(0)
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(chunk)
        if size > MAX_ARCHIVE_BYTES:
            raise DatasetPreparationError("Raw archive exceeds the finite byte budget")
        sha.update(chunk)
        md5.update(chunk)
    handle.seek(0)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(), "size_bytes": size}


def _safe_member_name(member: tarfile.TarInfo, role: str) -> str:
    name = member.name
    while name.startswith("./"):
        name = name[2:]
    if member.isdir():
        name = name.rstrip("/")
    windows = PureWindowsPath(name)
    parts = name.split("/")
    if (not name or "\\" in name or "\x00" in name or PurePosixPath(name).is_absolute()
            or windows.is_absolute() or windows.drive
            or any(part in ("", ".", "..") or ":" in part for part in parts)):
        raise DatasetPreparationError(f"Unsafe tar path: {member.name!r}")
    if parts[0] != role:
        raise DatasetPreparationError(f"Unexpected tar root for {role}: {member.name!r}")
    if (not member.isdir() and not member.isreg()) or member.sparse is not None:
        raise DatasetPreparationError(f"Tar links, devices and sparse members are forbidden: {name}")
    if member.size < 0:
        raise DatasetPreparationError(f"Negative tar member size: {name}")
    return name


def _header_index(archive: tarfile.TarFile, role: str) -> dict:
    """Inspect all headers, retaining only flat JPEG or required text members."""
    index, seen = {}, set()
    expanded = 0
    for number, member in enumerate(archive, 1):
        if number > MAX_MEMBERS:
            raise DatasetPreparationError("Archive has too many headers")
        name = _safe_member_name(member, role)
        if name in seen:
            raise DatasetPreparationError(f"Duplicate tar member: {name}")
        seen.add(name)
        expanded += member.size
        if expanded > MAX_EXPANDED_BYTES:
            raise DatasetPreparationError("Archive exceeds the expanded byte budget")
        if member.isdir():
            continue
        if role == "images" and name.endswith(".jpg"):
            image_id = name[len("images/"):-4]
            if "/" in image_id or not IMAGE_ID.fullmatch(image_id):
                raise DatasetPreparationError(f"JPEG is outside the allowed filename rule: {name}")
            if member.size > MAX_JPEG_BYTES:
                raise DatasetPreparationError(f"JPEG exceeds the finite size budget: {name}")
            index[name] = member
        elif role == "annotations" and name in REQUIRED_TEXT:
            if member.size > MAX_TEXT_BYTES:
                raise DatasetPreparationError(f"Annotation text exceeds the finite size budget: {name}")
            index[name] = member
        # Other regular files (e.g. segmentation PNG/XML) are never extracted.
    if role == "annotations" and any(name not in index for name in REQUIRED_TEXT):
        raise DatasetPreparationError("Archive is missing annotations/trainval.txt or annotations/test.txt")
    return index


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo, maximum: int) -> bytes:
    if member.size > maximum:
        raise DatasetPreparationError(f"Member exceeds the finite read limit: {member.name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise DatasetPreparationError(f"Cannot safely read regular tar member: {member.name}")
    with handle:
        payload = handle.read(maximum + 1)
    if len(payload) != member.size or len(payload) > maximum:
        raise DatasetPreparationError(f"Truncated or oversized tar member: {member.name}")
    return payload


def _annotation_records(payload: bytes, official_split: str) -> list:
    try:
        text = payload.decode("ascii")
    except UnicodeError as error:
        raise DatasetPreparationError("Official annotation lists must be ASCII text") from error
    records, seen = [], set()
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4 or not IMAGE_ID.fullmatch(fields[0]) or any(
                not re.fullmatch(r"[1-9][0-9]*", field) for field in fields[1:]):
            raise DatasetPreparationError(f"Malformed {official_split} annotation at line {line_number}")
        image_id = fields[0]
        prefix = IMAGE_ID.fullmatch(image_id).group(1)
        if prefix not in OFFICIAL_ANNOTATIONS:
            raise DatasetPreparationError(f"Unknown official filename prefix: {prefix}")
        values = tuple(int(field) for field in fields[1:])
        if values != OFFICIAL_ANNOTATIONS[prefix]:
            raise DatasetPreparationError(f"Filename/CLASS_ID/SPECIES/BREED_ID mismatch: {image_id}")
        if image_id in seen:
            raise DatasetPreparationError(f"Duplicate annotation source_id in {official_split}: {image_id}",
                                          diagnostics=[{"reason": "duplicate_source_id", "image_id": image_id,
                                                        "source_original_split": official_split}])
        seen.add(image_id)
        records.append({"image_id": image_id, "breed": prefix,
                        "CLASS_ID": values[0], "SPECIES": values[1], "BREED_ID": values[2],
                        "source_original_split": official_split, "annotation_line": line_number})
    if not records:
        raise DatasetPreparationError(f"Empty annotation list: {official_split}")
    return records


def _read_annotations(archive: tarfile.TarFile, index: dict) -> dict:
    records = {}
    # Physical ordering keeps gzip reads forward where possible.
    for name in sorted(REQUIRED_TEXT, key=lambda key: index[key].offset_data):
        split = Path(name).stem
        records[split] = _annotation_records(_read_member(archive, index[name], MAX_TEXT_BYTES), split)
    train_ids = {item["image_id"] for item in records["trainval"]}
    overlap = sorted(train_ids & {item["image_id"] for item in records["test"]})
    if overlap:
        raise DatasetPreparationError("Official trainval/test source_id overlap; refusing leakage",
                                      diagnostics=[{"reason": "official_split_source_id_overlap",
                                                    "image_id": image_id,
                                                    "source_original_splits": ["trainval", "test"]}
                                                   for image_id in overlap])
    return records


def _rank(seed: int, domain: str, identity: str) -> tuple:
    digest = hashlib.sha256(f"{seed}\0{domain}\0{identity}".encode("ascii")).digest()
    return digest, identity


def _class_quotas(prefixes: list, count: int, seed: int, split: str) -> dict:
    base, remainder = divmod(count, len(prefixes))
    extra = set(sorted(prefixes, key=lambda prefix: _rank(seed, f"quota:{split}", prefix))[:remainder])
    return {prefix: base + int(prefix in extra) for prefix in sorted(prefixes)}


def _spool_candidates(archive: tarfile.TarFile, index: dict, records: dict,
                      prefixes: set, cache: Path) -> None:
    """Copy eligible bytes in physical order, avoiding random gzip rescans.

    The private, bounded cache is removed before publication. No tar path is
    used as a filesystem path, and unrelated archive contents are not read out.
    """
    needed = {f"images/{item['image_id']}.jpg"
              for items in records.values() for item in items if item["breed"] in prefixes}
    cache.mkdir()
    for name in sorted(needed & index.keys(), key=lambda key: index[key].offset_data):
        payload = _read_member(archive, index[name], MAX_JPEG_BYTES)
        destination = cache / Path(name).name
        with destination.open("xb") as handle:
            handle.write(payload)


def _decoded_pixels(payload: bytes) -> dict:
    # Lazy import keeps --help usable even without Pillow, torch or torchvision.
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise DatasetPreparationError("Pillow is required for actual JPEG preparation, not --help") from error
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            verified_image = Image.open(io.BytesIO(payload))
        except Image.DecompressionBombError as error:
            raise ValueError(f"JPEG decompression bomb rejected: {error}") from error
        with verified_image as image:
            if image.format != "JPEG":
                raise ValueError("The .jpg bytes are not a JPEG")
            original_width, original_height = image.size
            if original_width * original_height > MAX_PIXELS or min(image.size) < 1:
                raise ValueError("JPEG exceeds the finite decoded pixel budget")
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            with ImageOps.exif_transpose(image) as oriented:
                with oriented.convert("RGB") as rgb:
                    rgb.load()
                    width, height = rgb.size
                    digest = hashlib.sha256(b"RGB\0" + struct.pack(">II", width, height))
                    digest.update(rgb.tobytes())
    return {"pixel_sha256": digest.hexdigest(), "width": width, "height": height,
            "original_width": original_width, "original_height": original_height}


def _select_images(stage: Path, records: dict, config: dict, seed: int,
                   counts: dict) -> tuple[list, list, dict]:
    mapping = {item["prefix"]: item for item in config["classes"]}
    prefixes = sorted(mapping)
    candidates = {}
    for official_split, items in records.items():
        for prefix in prefixes:
            pool = [item for item in items if item["breed"] == prefix]
            candidates[official_split, prefix] = sorted(
                pool, key=lambda item: _rank(seed, f"candidates:{official_split}", item["image_id"]))
    positions = Counter()
    seen_sha, seen_pixels, seen_source = {}, {}, {}
    accepted, rejections, quotas = [], [], {}
    image_dir = stage / "images"
    image_dir.mkdir()
    for split in ("calibration", "development", "holdout"):
        official_split = "test" if split == "holdout" else "trainval"
        quotas[split] = _class_quotas(prefixes, counts[split], seed, split)
        for prefix in prefixes:
            pool = candidates[official_split, prefix]
            selected = 0
            while selected < quotas[split][prefix]:
                position = positions[official_split, prefix]
                if position >= len(pool):
                    raise DatasetPreparationError(
                        f"Insufficient unique valid {prefix} in official {official_split} for {split}: "
                        f"{selected}/{quotas[split][prefix]}; quotas will not be reduced",
                        diagnostics=rejections)
                positions[official_split, prefix] += 1
                annotation = pool[position]
                original_split = annotation["source_original_split"]
                image_id = annotation["image_id"]
                source_id = f"oxford-iiit-pet:{image_id}"
                cache_file = stage / ".candidate-cache" / f"{image_id}.jpg"
                diagnostic = {"image_id": image_id, "breed": prefix, "requested_split": split,
                              "source_original_split": original_split}
                if not cache_file.is_file():
                    rejections.append({**diagnostic, "reason": "missing_jpeg"})
                    continue
                payload = cache_file.read_bytes()
                sha = hashlib.sha256(payload).hexdigest()
                duplicate = seen_source.get(source_id) or seen_sha.get(sha)
                if duplicate is not None:
                    reason = "duplicate_source_id" if source_id in seen_source else "duplicate_sha256"
                    rejections.append({**diagnostic, "reason": reason, "sha256": sha,
                                       "duplicate_of": duplicate["id"],
                                       "duplicate_of_split": duplicate["split"],
                                       "duplicate_of_original_split": duplicate["source_original_split"],
                                       "cross_official_split": duplicate["source_original_split"] != original_split})
                    continue
                try:
                    pixels = _decoded_pixels(payload)
                except DatasetPreparationError:
                    raise
                except (OSError, ValueError, SyntaxError, Warning, OverflowError, TypeError, KeyError) as error:
                    rejections.append({**diagnostic, "reason": "invalid_jpeg", "sha256": sha,
                                       "detail": f"{type(error).__name__}: {error}"})
                    continue
                duplicate = seen_pixels.get(pixels["pixel_sha256"])
                if duplicate is not None:
                    rejections.append({**diagnostic, "reason": "duplicate_rgb_pixels", "sha256": sha,
                                       "pixel_sha256": pixels["pixel_sha256"],
                                       "duplicate_of": duplicate["id"], "duplicate_of_split": duplicate["split"],
                                       "duplicate_of_original_split": duplicate["source_original_split"],
                                       "cross_official_split": duplicate["source_original_split"] != original_split})
                    continue
                class_mapping = mapping[prefix]
                entry = {"id": image_id, "path": f"images/{image_id}.jpg", "sha256": sha,
                         "split": split, "source_id": source_id,
                         "source": config["dataset"], "source_url": config["homepage"],
                         "license": config["license"], "breed": prefix,
                         "canonical_name": class_mapping["canonical_name"],
                         "label": None if split == "calibration" else class_mapping["imagenet_index"],
                         "source_original_split": original_split,
                         "ground_truth_source": annotation.get("ground_truth_source", f"annotations/{official_split}.txt"),
                         "original_annotation": annotation.copy(), **pixels}
                with (image_dir / f"{image_id}.jpg").open("xb") as handle:
                    handle.write(payload)  # Preserve the source JPEG bytes, including EXIF.
                seen_sha[sha] = seen_pixels[pixels["pixel_sha256"]] = seen_source[source_id] = entry
                accepted.append(entry)
                selected += 1
    return accepted, rejections, quotas


def _target_path(output_dir: str | Path, archive_paths: list[Path], overwrite: bool) -> tuple[Path, tuple | None]:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(output_dir))))
    if target == ROOT or target in ROOT.parents or not target.name:
        raise DatasetPreparationError("Output must be a dedicated dataset directory, not a workspace or filesystem root")
    for path in (target, *target.parents):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise DatasetPreparationError(f"Output symlinks/junctions are not allowed: {path}")
    for archive_path in archive_paths:
        if archive_path == target or target in archive_path.parents:
            raise DatasetPreparationError("Output must not contain either caller-supplied raw archive")
    original_identity = None
    if target.exists():
        if not target.is_dir() or not overwrite:
            raise DatasetPreparationError(f"Output already exists; explicit --overwrite is required: {target}")
        info = target.stat()
        original_identity = (info.st_dev, info.st_ino)
    return target, original_identity


def _validate_prepared(stage: Path, counts: dict) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.vision.classification_benchmark import BenchmarkError, load_manifest, dataset_readiness
    try:
        manifest = load_manifest(stage / "manifest.json", check_files=True)
    except BenchmarkError as error:
        raise DatasetPreparationError(f"Final manifest validation failed: {error}") from error
    images = manifest["images"]
    if dict(Counter(entry["split"] for entry in images)) != counts:
        raise DatasetPreparationError("Prepared image counts do not equal the requested quotas")
    for key in ("sha256", "source_id", "pixel_sha256"):
        if len({entry[key] for entry in images}) != len(images):
            raise DatasetPreparationError(f"Prepared images have duplicate {key}; refusing leakage")
    readiness = dataset_readiness(manifest)
    if not all(readiness[key]["ready"] for key in ("smoke", "formal", "quantization")):
        raise DatasetPreparationError(
            "Existing benchmark data gates are not met; no completion marker or dataset will be published: "
            + json.dumps(readiness, sort_keys=True))
    return readiness


def _publish(stage: Path, target: Path, overwrite: bool, original_identity: tuple | None) -> Path | None:
    backup = None
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir() or not overwrite or original_identity is None:
            raise DatasetPreparationError("Output appeared or changed while preparing; leaving it untouched")
        info = target.stat()
        if (info.st_dev, info.st_ino) != original_identity:
            raise DatasetPreparationError("Output identity changed while preparing; leaving it untouched")
        backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
        os.rename(target, backup)
    elif original_identity is not None:
        raise DatasetPreparationError("Existing output disappeared while preparing; refusing replacement")
    try:
        os.rename(stage, target)
    except OSError:
        if backup is not None and not target.exists():
            os.rename(backup, target)
        raise
    # Preserve the old directory rather than delete user data, even on overwrite.
    return backup


def prepare_dataset(images_archive, annotations_archive, output_dir, *, seed=DEFAULT_SEED,
                    calibration_count=256, development_count=128, holdout_count=1000,
                    overwrite=False) -> dict:
    """Prepare and atomically publish the approved dataset from two local archives.

    Small explicit counts are useful in tests, but cannot pass the production
    gates without a test-only readiness mock. The CLI never offers smaller
    count budgets. Overwrite preserves the previous directory as a backup.
    """
    seed = _integer(seed, "seed", 2**32 - 1)
    counts = {"calibration": _integer(calibration_count, "calibration_count", 100000),
              "development": _integer(development_count, "development_count", 100000),
              "holdout": _integer(holdout_count, "holdout_count", 1000)}
    if min(counts.values()) < 1:
        raise DatasetPreparationError("Each requested split count must be positive")
    if type(overwrite) is not bool:
        raise DatasetPreparationError("overwrite must be bool")
    config = load_dataset_config()
    try:
        paths = {"images": Path(images_archive).expanduser().resolve(strict=True),
                 "annotations": Path(annotations_archive).expanduser().resolve(strict=True)}
    except (OSError, RuntimeError) as error:
        raise DatasetPreparationError(f"Both caller-supplied archives must exist: {error}") from error
    target, original_identity = _target_path(output_dir, list(paths.values()), overwrite)
    stage = None
    rejections = []
    try:
        with ExitStack() as stack:
            archives, indexes, handles, integrity = {}, {}, {}, {}
            for role in ("images", "annotations"):
                path = paths[role]
                if not path.is_file() or not 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES:
                    raise DatasetPreparationError(f"Archive is not a nonempty, finite regular file: {path}")
                handle = stack.enter_context(path.open("rb"))
                digests = _raw_digests(handle)
                reference = config["archive_checksums"][role]
                if digests["md5"] != reference["md5"]:
                    raise DatasetPreparationError(
                        f"{role} archive MD5 mismatch: observed={digests['md5']} expected={reference['md5']}; "
                        "reference comes from torchvision _RESOURCES, not an official SHA256")
                archives[role] = stack.enter_context(tarfile.open(fileobj=handle, mode="r:*"))
                handles[role] = handle
                integrity[role] = {"path": str(path), "expected_url": reference["url"],
                                   "md5_expected": reference["md5"], "md5_observed": digests["md5"],
                                   "md5_reference": reference["md5_reference"], "md5_matches_reference": True,
                                   "sha256_observed": digests["sha256"],
                                   "sha256_origin": "Locally computed from the caller-supplied raw tar archive; not an official published SHA256.",
                                   "size_bytes": digests["size_bytes"]}
                indexes[role] = _header_index(archives[role], role)
            records = _read_annotations(archives["annotations"], indexes["annotations"])
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.prepare-", dir=str(target.parent)))
            _spool_candidates(archives["images"], indexes["images"], records,
                              {item["prefix"] for item in config["classes"]}, stage / ".candidate-cache")
            images, rejections, quotas = _select_images(stage, records, config, seed, counts)
            shutil.rmtree(stage / ".candidate-cache")
            # Rehash the already-open raw files before claiming archive integrity.
            for role, handle in handles.items():
                final = _raw_digests(handle)
                if (final["sha256"] != integrity[role]["sha256_observed"]
                        or final["md5"] != integrity[role]["md5_observed"]
                        or final["size_bytes"] != integrity[role]["size_bytes"]):
                    raise DatasetPreparationError(f"Raw {role} archive changed during preparation")
                integrity[role]["unchanged_during_preparation"] = True
        manifest = {"schema_version": 1, "root": ".", "dataset": config["dataset"],
                    "source_metadata": "source.json", "seed": seed, "images": images}
        source = {"schema_version": 1, "dataset": config["dataset"],
                  "homepage": config["homepage"], "authors": config["authors"],
                  "license": SOURCE_LICENSE, "license_url": config["license_url"],
                  "license_notes": config["license_notes"],
                  "copyright_owner": "Original image owners; ownership is not transferred by the dataset license.",
                  "task_usage": config["task_usage"],
                  "acquisition": {"method": "Caller-supplied existing local archives, consumed offline after MD5 verification.",
                                  "tool_downloaded": False, "tool_uploaded": False,
                                  "original_download_method": "Not independently observed by this tool; expected official URLs and supplied local paths are recorded in archives."},
                  "archives": integrity, "config_sha256": config["config_sha256"],
                  "weights_version": config["weights_version"], "seed": seed,
                  "subset": {"class_count": 20, "species": "dog", "mapping": config["classes"],
                             "mapping_policy": config["mapping_policy"], "counts": counts,
                             "class_quotas": quotas,
                             "ground_truth": "Official CLASS_ID/SPECIES/BREED_ID records; never model predictions.",
                             "original_split_roles": {"trainval": ["calibration", "development"], "test": ["holdout"]}},
                  "sampling": {"algorithm": "Lexically sorted frozen classes; SHA256(seed\\0domain\\0original_image_id) rank with lexical tie-break. Calibration then development consume one trainval order; holdout consumes the official test order.",
                               "replacement": "Reject invalid or duplicate candidates and take the next seeded candidate in the same class and original split. Never lower quotas or inspect predictions.",
                               "pixel_hash": "SHA256 of RGB\\0 + big-endian uint32 width,height + EXIF-transposed decoded RGB bytes. Original JPEG bytes remain unchanged.",
                               "duplicate_scope": "All accepted images and rejected candidates encountered during sampling, not an exhaustive near-duplicate audit of the source dataset."},
                  "safety_limits": {"archive_bytes": MAX_ARCHIVE_BYTES, "expanded_bytes": MAX_EXPANDED_BYTES,
                                    "headers": MAX_MEMBERS, "jpeg_bytes": MAX_JPEG_BYTES,
                                    "annotation_bytes": MAX_TEXT_BYTES, "decoded_pixels": MAX_PIXELS},
                  "rejections": rejections, "limitations": config["limitations"]}
        manifest_sha = _write_json(stage / "manifest.json", manifest)
        source_sha = _write_json(stage / "source.json", source)
        readiness = _validate_prepared(stage, counts)
        _write_json(stage / "prepared.complete.json", {"schema_version": 1, "status": "complete",
                    "manifest_sha256": manifest_sha, "source_sha256": source_sha,
                    "counts": counts, "dataset_readiness": readiness})
        backup = _publish(stage, target, overwrite, original_identity)
        stage = None
        return {"status": "complete", "output_dir": str(target), "manifest": str(target / "manifest.json"),
                "source": str(target / "source.json"), "completion": str(target / "prepared.complete.json"),
                "manifest_sha256": manifest_sha, "source_sha256": source_sha,
                "dataset_readiness": readiness, "counts": counts,
                "rejected_count": len(rejections), "backup_dir": str(backup) if backup else None}
    except DatasetPreparationError as error:
        if rejections and not error.diagnostics:
            error.diagnostics = rejections
        raise
    except (OSError, EOFError, tarfile.TarError) as error:
        raise DatasetPreparationError(f"Local dataset preparation failed: {error}", diagnostics=rejections) from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)  # Only our private, unpublished directory.


def _relative_parts(relative: str) -> tuple:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise DatasetPreparationError(f"Unsafe mirror relative path: {relative!r}")
    parts = relative.split("/")
    windows = PureWindowsPath(relative)
    if (PurePosixPath(relative).is_absolute() or windows.is_absolute() or windows.drive
            or any(part in ("", ".", "..") or ":" in part or part.endswith((" ", "."))
                   or any(character in '<>"|?*' for character in part) for part in parts)):
        raise DatasetPreparationError(f"Unsafe mirror relative path: {relative!r}")
    return tuple(parts)


def _mirror_file(root: Path, relative: str) -> Path:
    parts = _relative_parts(relative)
    if not relative.endswith(".parquet"):
        raise DatasetPreparationError("Mirror input files must be local .parquet files")
    path = root.joinpath(*parts)
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise DatasetPreparationError(f"Mirror input symlinks/junctions are forbidden: {candidate}")
        if candidate == root:
            break
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError, RuntimeError) as error:
        raise DatasetPreparationError(f"Missing or escaping mirror file: {relative}") from error
    if not resolved.is_file():
        raise DatasetPreparationError(f"Mirror input is not a regular file: {relative}")
    return resolved


def _read_mirror_source(mirror_dir, config: dict) -> tuple:
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(mirror_dir))))
    if (not root.is_dir() or root.is_symlink()
            or getattr(root, "is_junction", lambda: False)()):
        raise DatasetPreparationError("mirror_dir must be an existing non-linked local directory")
    sidecar = root / "mirror-source.json"
    if not sidecar.is_file() or sidecar.is_symlink() or sidecar.stat().st_size > MAX_TEXT_BYTES:
        raise DatasetPreparationError("Missing, linked or oversized mirror-source.json")
    try:
        raw = sidecar.read_bytes()
        metadata = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys,
                              parse_constant=lambda text: (_ for _ in ()).throw(
                                  DatasetPreparationError(f"Non-finite mirror metadata: {text}")))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise DatasetPreparationError(f"Cannot read mirror-source.json: {error}") from error
    frozen = config["mirror"]
    if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != 1 or metadata.get("dataset_id") != frozen["dataset_id"]
            or metadata.get("revision") != frozen["revision"]
            or metadata.get("license") not in ("cc-by-sa-4.0", SOURCE_LICENSE)
            or metadata.get("publisher_url") != config["homepage"]
            or metadata.get("authors") not in (config["authors"], ["Omkar M. Parkhi", *config["authors"][1:]])):
        raise DatasetPreparationError("Mirror source identity/revision/license/publisher/attribution mismatch")
    basis = metadata.get("license_basis")
    if not isinstance(basis, (str, dict, list)) or not basis:
        raise DatasetPreparationError("Mirror card license_basis must be explicitly recorded")
    names = metadata.get("label_names")
    if not isinstance(names, dict) or names != MIRROR_LABEL_NAMES:
        raise DatasetPreparationError("Mirror label_names must contain 37 unique safe names including the frozen 20 prefixes")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != {"train", "test"}:
        raise DatasetPreparationError("Mirror metadata must identify exactly train/test Parquet files")
    paths = {}
    for role in ("train", "test"):
        item, reference = files[role], frozen["files"][role]
        if (not isinstance(item, dict) or item.get("split") != role
                or item.get("sha256") != reference["sha256"]
                or type(item.get("size_bytes")) is not int or item["size_bytes"] != reference["size_bytes"]):
            raise DatasetPreparationError(f"Mirror {role} LFS SHA256/size/split reference mismatch")
        paths[role] = _mirror_file(root, item.get("path"))
    if paths["train"] == paths["test"]:
        raise DatasetPreparationError("Mirror train/test cannot reference the same file")
    return root, sidecar, metadata, hashlib.sha256(raw).hexdigest(), paths


def _mirror_digest(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise DatasetPreparationError("Mirror Parquet file exceeds the finite byte budget")
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _parquet_reader(path: Path):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise DatasetPreparationError("PyArrow is required only for actual mirror preparation; no automatic installation") from error
    return parquet.ParquetFile(str(path), memory_map=False, pre_buffer=False)


def _spool_mirror(paths: dict, metadata: dict, config: dict, cache: Path) -> dict:
    """Stream 32 rows at a time; retain small metadata and eligible JPEGs on disk."""
    cache.mkdir()
    records = {"trainval": [], "test": []}
    seen_ids = {}
    selected = {item[0] for item in FROZEN_MAPPING}
    columns = ["image", "label", "image_id", "label_cat_dog"]
    for role in ("train", "test"):
        reader = None
        try:
            reader = _parquet_reader(paths[role])
            expected_rows = config["mirror"]["files"][role]["row_count"]
            if (not set(columns).issubset(set(reader.schema_arrow.names))
                    or reader.metadata.num_rows != expected_rows or not 0 < expected_rows <= MAX_MIRROR_ROWS):
                raise DatasetPreparationError(f"Mirror {role} columns or frozen row count mismatch")
            row_index = 0
            for batch in reader.iter_batches(batch_size=MIRROR_BATCH_SIZE, columns=columns, use_threads=False):
                if batch.num_rows > MIRROR_BATCH_SIZE:
                    raise DatasetPreparationError("Parquet reader exceeded the bounded record batch size")
                for row in batch.to_pylist():
                    if row_index >= expected_rows:
                        raise DatasetPreparationError(f"Too many streamed mirror {role} rows")
                    image_id, label, binary = row.get("image_id"), row.get("label"), row.get("label_cat_dog")
                    if (not isinstance(image_id, str) or not IMAGE_ID.fullmatch(image_id)
                            or type(label) is not int or not 0 <= label < 37
                            or type(binary) is not int or binary not in (0, 1)):
                        raise DatasetPreparationError(f"Unsafe image_id or invalid mirror labels at {role} row {row_index}")
                    prefix = IMAGE_ID.fullmatch(image_id).group(1)
                    card_name = metadata["label_names"][str(label)]
                    frozen_prefix, frozen_species, _ = _OXFORD_CLASS_ORDER[label]
                    allowed_prefixes = (prefix,) if prefix in selected else (frozen_prefix, frozen_prefix.lower())
                    if ((prefix in selected and card_name != prefix)
                            or (prefix not in selected and prefix not in allowed_prefixes)):
                        raise DatasetPreparationError(f"Mirror filename/card label_names/label mismatch: {image_id}")
                    if prefix not in selected and binary != frozen_species - 1:
                        raise DatasetPreparationError(f"Excluded mirror filename/species mismatch: {image_id}")
                    image = row.get("image")
                    if not isinstance(image, dict):
                        raise DatasetPreparationError(f"Mirror image must be a bytes/path struct: {image_id}")
                    original_path = image.get("path")
                    if original_path is not None:
                        parts = _relative_parts(original_path)
                        if parts[-1] != f"{image_id}.jpg":
                            raise DatasetPreparationError(f"Mirror JPEG path/stem/image_id mismatch: {image_id}")
                    if image_id in seen_ids:
                        raise DatasetPreparationError("Mirror train/test or within-split source_id overlap; refusing leakage",
                            diagnostics=[{"reason": "mirror_source_id_overlap", "image_id": image_id,
                                          "source_original_splits": [seen_ids[image_id], role]}])
                    seen_ids[image_id] = role
                    if prefix in selected:
                        if binary != 1:
                            raise DatasetPreparationError(f"Selected mirror breed must have label_cat_dog=1: {image_id}")
                        payload = image.get("bytes")
                        if not isinstance(payload, bytes) or len(payload) > MAX_JPEG_BYTES:
                            raise DatasetPreparationError(f"Mirror JPEG bytes missing or oversized: {image_id}")
                        record = {"image_id": image_id, "breed": prefix, "source_original_split": role,
                                  "label": label, "label_name": metadata["label_names"][str(label)],
                                  "label_cat_dog": binary, "original_image_path": original_path,
                                  "mirror_row_index": row_index,
                                  "ground_truth_source": f"mirror:{metadata['dataset_id']}@{metadata['revision']}:{metadata['files'][role]['path']}#row={row_index}"}
                        records["trainval" if role == "train" else "test"].append(record)
                        with (cache / f"{image_id}.jpg").open("xb") as handle:
                            handle.write(payload)
                    row_index += 1
            if row_index != expected_rows:
                raise DatasetPreparationError(f"Incomplete mirror {role} row stream: {row_index}/{expected_rows}")
        except DatasetPreparationError:
            raise
        except Exception as error:
            raise DatasetPreparationError(f"Cannot stream local mirror {role} Parquet: {type(error).__name__}: {error}") from error
        finally:
            if reader is not None:
                reader.close()
    return records


def prepare_dataset_from_mirror(mirror_dir, output_dir, *, seed=DEFAULT_SEED,
                                calibration_count=256, development_count=128, holdout_count=1000,
                                overwrite=False) -> dict:
    """Prepare the same frozen subset from caller-supplied, pinned local Parquet."""
    seed = _integer(seed, "seed", 2**32 - 1)
    counts = {"calibration": _integer(calibration_count, "calibration_count", 100000),
              "development": _integer(development_count, "development_count", 100000),
              "holdout": _integer(holdout_count, "holdout_count", 1000)}
    if min(counts.values()) < 1 or type(overwrite) is not bool:
        raise DatasetPreparationError("Positive split counts and a bool overwrite are required")
    config = load_dataset_config()
    stage, rejections = None, []
    try:
        root, sidecar, metadata, sidecar_sha, paths = _read_mirror_source(mirror_dir, config)
        target, original_identity = _target_path(output_dir, [sidecar, *paths.values()], overwrite)
        integrity = {}
        for role, path in paths.items():
            observed = _mirror_digest(path)
            reference = config["mirror"]["files"][role]
            if observed["sha256"] != reference["sha256"] or observed["size_bytes"] != reference["size_bytes"]:
                raise DatasetPreparationError(f"Mirror {role} complete-file LFS SHA256 or size mismatch")
            integrity[role] = {"path": str(path), "mirror_relative_path": metadata["files"][role]["path"],
                               "split": role, "sha256_expected": reference["sha256"],
                               "sha256_observed": observed["sha256"], "size_bytes": observed["size_bytes"],
                               "row_count": reference["row_count"], "sha256_matches_lfs_reference": True,
                               "sha256_reference": config["mirror"]["sha256_reference"]}
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.prepare-", dir=str(target.parent)))
        records = _spool_mirror(paths, metadata, config, stage / ".candidate-cache")
        images, rejections, quotas = _select_images(stage, records, config, seed, counts)
        shutil.rmtree(stage / ".candidate-cache")
        for role, path in paths.items():
            final = _mirror_digest(path)
            if final["sha256"] != integrity[role]["sha256_observed"] or final["size_bytes"] != integrity[role]["size_bytes"]:
                raise DatasetPreparationError(f"Mirror {role} raw Parquet changed during preparation")
            integrity[role]["unchanged_during_preparation"] = True
        if hashlib.sha256(sidecar.read_bytes()).hexdigest() != sidecar_sha:
            raise DatasetPreparationError("mirror-source.json changed during preparation")
        manifest = {"schema_version": 1, "root": ".", "dataset": config["dataset"],
                    "source_metadata": "source.json", "seed": seed, "images": images}
        archives_not_read = {
            role: {"expected_url": item["url"], "md5_expected": item["md5"],
                   "md5_reference": item["md5_reference"], "md5_verified": False,
                   "md5_observed": None, "sha256_observed": None,
                   "status": "not_read_not_verified; preparation used the pinned Parquet mirror"}
            for role, item in config["archive_checksums"].items()
        }
        source = {"schema_version": 1, "dataset": config["dataset"], "homepage": config["homepage"],
                  "authors": config["authors"], "license": SOURCE_LICENSE,
                  "license_url": config["license_url"], "license_notes": config["license_notes"],
                  "copyright_owner": "Original image owners; ownership is not transferred by the dataset license.",
                  "task_usage": config["task_usage"], "config_sha256": config["config_sha256"],
                  "acquisition": {"method": "Caller-supplied local timm/oxford-iiit-pet Parquet mirror at the frozen revision; complete files verified against the frozen LFS SHA256 references.",
                                  "tool_downloaded": False, "tool_uploaded": False,
                                  "original_download_method": "Not independently observed by this tool; mirror-source.json retains caller acquisition/license evidence."},
                  "mirror": {"dataset_id": metadata["dataset_id"], "revision": metadata["revision"],
                             "publisher_url": metadata["publisher_url"], "card_license": metadata["license"],
                             "authors_as_supplied": metadata["authors"],
                             "license_basis": metadata["license_basis"], "label_names": metadata["label_names"],
                             "caller_supplied_source_metadata": metadata,
                             "sidecar_path": str(sidecar), "sidecar_sha256_observed": sidecar_sha,
                             "files": integrity, "batch_size": MIRROR_BATCH_SIZE},
                  "archives": archives_not_read, "weights_version": config["weights_version"], "seed": seed,
                  "subset": {"class_count": 20, "species": "dog", "mapping": config["classes"],
                             "mapping_policy": config["mapping_policy"], "counts": counts, "class_quotas": quotas,
                             "ground_truth": config["mirror"]["ground_truth_basis"],
                             "original_split_roles": {"train": ["calibration", "development"], "test": ["holdout"]}},
                  "sampling": {"algorithm": "Frozen SHA256 seeded ranking and balanced quotas, shared with the tar path; mirror train uses the trainval ranking domain but original split/provenance remain mirror train.",
                               "replacement": "Next seeded candidate in the same breed and mirror split after invalid JPEG/byte/pixel duplicate rejection; no prediction filtering.",
                               "pixel_hash": "SHA256 of RGB\\0 + big-endian uint32 width,height + EXIF-transposed RGB bytes; original mirrored JPEG bytes unchanged.",
                               "duplicate_scope": "Accepted images and sampled rejected candidates; not a near-duplicate audit."},
                  "rejections": rejections,
                  "limitations": config["limitations"] + [
                      "Official Oxford tar MD5s were not verified and original annotations/*.txt were not read in this mirror path.",
                      "Mirror train/test and card breed annotations are the provenance actually consumed; mirrored JPEG byte identity with Oxford images.tar.gz has not been independently established."]}
        manifest_sha = _write_json(stage / "manifest.json", manifest)
        source_sha = _write_json(stage / "source.json", source)
        readiness = _validate_prepared(stage, counts)
        _write_json(stage / "prepared.complete.json", {"schema_version": 1, "status": "complete",
                    "manifest_sha256": manifest_sha, "source_sha256": source_sha,
                    "counts": counts, "dataset_readiness": readiness})
        backup = _publish(stage, target, overwrite, original_identity)
        stage = None
        return {"status": "complete", "output_dir": str(target), "manifest": str(target / "manifest.json"),
                "source": str(target / "source.json"), "completion": str(target / "prepared.complete.json"),
                "manifest_sha256": manifest_sha, "source_sha256": source_sha,
                "dataset_readiness": readiness, "counts": counts, "mirror_revision": metadata["revision"],
                "rejected_count": len(rejections), "backup_dir": str(backup) if backup else None}
    except DatasetPreparationError as error:
        if rejections and not error.diagnostics:
            error.diagnostics = rejections
        raise
    except (OSError, ValueError, RuntimeError) as error:
        raise DatasetPreparationError(f"Local mirror preparation failed: {error}", diagnostics=rejections) from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images-archive", help="Existing official images.tar.gz; no download")
    inputs.add_argument("--mirror-dir", help="Existing pinned train/test Parquet files plus mirror-source.json; no download")
    parser.add_argument("--annotations-archive", help="Required with --images-archive; forbidden with --mirror-dir")
    parser.add_argument("--output-dir", default="data/classifier-pets", help="Dedicated local dataset directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this target, preserving its previous directory as a sibling backup")
    args = parser.parse_args(argv)
    if args.mirror_dir and args.annotations_archive:
        parser.error("--annotations-archive cannot be combined with --mirror-dir")
    if args.images_archive and not args.annotations_archive:
        parser.error("--images-archive requires --annotations-archive")
    try:
        if args.mirror_dir:
            result = prepare_dataset_from_mirror(args.mirror_dir, args.output_dir,
                                                 seed=args.seed, overwrite=args.overwrite)
        else:
            result = prepare_dataset(args.images_archive, args.annotations_archive, args.output_dir,
                                     seed=args.seed, overwrite=args.overwrite)
    except DatasetPreparationError as error:
        print(json.dumps({"status": "failed", "error": str(error), "diagnostics": error.diagnostics},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
