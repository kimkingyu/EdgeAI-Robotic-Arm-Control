#!/usr/bin/env python3
"""Fixed, licensed Oxford-IIIT Pet mirror; no remote dataset code is executed.

This verifies mirror LFS SHA256, not the unpublished local Oxford archive MD5.
Original images remain local and original image-owner copyrights are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "timm/oxford-iiit-pet"
REVISION = "089695c834a7deb60505b7cc506672db1c31a6aa"
RESOURCES = {
    "train": {"remote_path": "data/train-00000-of-00001.parquet", "path": "train.parquet",
              "sha256": "b9ff432b33c77f4b50014684f25686a1cd523090686571743a6b7a257af27422", "size_bytes": 377611203},
    "test": {"remote_path": "data/test-00000-of-00001.parquet", "path": "test.parquet",
             "sha256": "0997d4744ee6740e96f17f56b40b97c70076a4e4dad565bc63d0ffd03906f754", "size_bytes": 412654113},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_card():
    url = f"https://huggingface.co/api/datasets/{DATASET_ID}/revision/{REVISION}"
    with urllib.request.urlopen(url, timeout=30) as response:
        info = json.load(response)
    if info["sha"] != REVISION or info["cardData"]["license"] != "cc-by-sa-4.0":
        raise ValueError("Mirror revision or declared license mismatch")
    features = info["cardData"]["dataset_info"]["features"]
    names = next(feature["dtype"]["class_label"]["names"] for feature in features if feature["name"] == "label")
    if len(names) != 37 or {int(key) for key in names} != set(range(37)):
        raise ValueError("Mirror breed-class mapping is not the expected 37-class annotation")
    splits = {entry["name"]: entry["num_examples"] for entry in info["cardData"]["dataset_info"]["splits"]}
    if splits != {"train": 3680, "test": 3669}:
        raise ValueError("Unexpected mirror split counts")
    return names, info["cardData"]


def download_file(split, directory):
    resource = RESOURCES[split]
    target = Path(directory) / resource["path"]
    if target.exists():
        if target.stat().st_size != resource["size_bytes"] or sha256(target) != resource["sha256"]:
            raise ValueError(f"Existing mirror shard is invalid; preserving: {target}")
        return {**resource, "split": split, "cached": True}
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{resource['remote_path']}"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.part")
    started = time.perf_counter()
    total = 0
    next_progress = 50 * 1024 * 1024
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "EdgeAI-Classifier-Research/1.0", "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as output:
            final = urllib.parse.urlparse(response.geturl())
            if final.scheme != "https" or not (final.hostname == "huggingface.co" or
                                               final.hostname.endswith(".hf.co") or
                                               final.hostname.endswith(".huggingface.co")):
                raise ValueError("Unexpected mirror CDN redirect")
            first = True
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if first and not chunk.startswith(b"PAR1"):
                    raise ValueError("Mirror response is not Parquet")
                first = False
                total += len(chunk)
                if total > resource["size_bytes"]:
                    raise ValueError("Mirror response exceeded fixed shard size")
                digest.update(chunk)
                output.write(chunk)
                if total >= next_progress:
                    print(f"{split}: {total / 1024 ** 2:.0f} MiB downloaded", flush=True)
                    next_progress += 50 * 1024 * 1024
        if total != resource["size_bytes"] or digest.hexdigest() != resource["sha256"]:
            raise ValueError("Mirror LFS size/SHA256 mismatch")
        if target.exists():
            raise ValueError("Another process created the target; refusing overwrite")
        os.replace(temporary, target)
        return {**resource, "split": split, "cached": False, "url": url,
                "download_seconds": time.perf_counter() - started,
                "final_host": final.hostname, "hash_source": "Pinned Hugging Face LFS SHA256"}
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "classifier-mirror-cache")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if not args.download:
        print(json.dumps({"dataset_id": DATASET_ID, "revision": REVISION, "resources": RESOURCES,
                          "no_network_or_write": True}, ensure_ascii=False, indent=2))
        return 0
    try:
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(directory).free < 3 * 1024 ** 3:
            raise ValueError("Need 3 GiB free for the fixed shards and prepared subset")
        names, card = verify_card()
        files = {split: download_file(split, directory) for split in RESOURCES}
        source = {"schema_version": 1, "dataset_id": DATASET_ID, "revision": REVISION,
                  "license": "CC BY-SA 4.0", "license_basis": "Oxford publisher declaration and pinned mirror card agree",
                  "publisher_url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
                  "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                  "authors": ["Omkar M. Parkhi", "Andrea Vedaldi", "Andrew Zisserman", "C. V. Jawahar"],
                  "copyright_note": "Original image-owner copyrights retained; no public image redistribution",
                  "label_names": names, "files": files, "dataset_card": card,
                  "original_oxford_archive_md5_verified": False,
                  "integrity_method": "Fixed mirror revision plus exact LFS SHA256 and size",
                  "label_source": "Independent Oxford breed annotations carried by the mirror, never model predictions",
                  "checked_at": datetime.now().astimezone().isoformat()}
        output = directory / "mirror-source.json"
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(source, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        print(json.dumps(source, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
