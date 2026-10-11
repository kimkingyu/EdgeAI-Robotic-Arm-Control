#!/usr/bin/env python3
"""Download the Oxford publisher archives only with explicit authorization.

No model execution, registration, account creation, extraction or third-party mirror.
Publisher license declarations do not remove original image-owner copyrights.
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
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HOMEPAGE = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
RESOURCES = {
    "images.tar.gz": (HOMEPAGE + "data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
    "annotations.tar.gz": (HOMEPAGE + "data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
}
MAX_BYTES_PER_ARCHIVE = 1200 * 1024 * 1024


def checksums(path):
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "size_bytes": Path(path).stat().st_size}


def download_archive(name, directory):
    if name not in RESOURCES:
        raise ValueError("Only the allowlisted Oxford publisher archives are supported")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    url, expected_md5 = RESOURCES[name]
    if target.exists():
        result = checksums(target)
        if result["md5"] != expected_md5:
            raise ValueError(f"Existing archive checksum is wrong; preserving it: {target}")
        return {"path": str(target), "url": url, "expected_md5": expected_md5,
                "md5_source": "torchvision 0.29.0 OxfordIIITPet._RESOURCES", "cached": True, **result}
    if shutil.disk_usage(directory).free < 3 * 1024 ** 3:
        raise ValueError("Need at least 3 GiB free for archives, subset and temporary outputs")
    temporary = directory / f".{name}.{os.getpid()}.{time.time_ns()}.part"
    started = time.perf_counter()
    total = 0
    next_progress = 50 * 1024 * 1024
    request = urllib.request.Request(url, headers={"User-Agent": "EdgeAI-Classifier-Research/1.0",
                                                   "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as handle:
            final_url = response.geturl()
            allowed_endpoints = {url, f"https://thor.robots.ox.ac.uk/pets/{name}"}
            if final_url not in allowed_endpoints:
                raise ValueError(f"Unexpected download redirect: {final_url}")
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_BYTES_PER_ARCHIVE:
                raise ValueError("Archive exceeds the fixed download-size limit")
            first = True
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if first and not chunk.startswith(b"\x1f\x8b"):
                    raise ValueError("Publisher response is not a gzip archive")
                first = False
                total += len(chunk)
                if total > MAX_BYTES_PER_ARCHIVE:
                    raise ValueError("Archive exceeds the fixed download-size limit")
                handle.write(chunk)
                if total >= next_progress:
                    print(f"{name}: {total / (1024 ** 2):.0f} MiB downloaded", flush=True)
                    next_progress += 50 * 1024 * 1024
        result = checksums(temporary)
        if result["md5"] != expected_md5:
            raise ValueError(f"Publisher archive checksum mismatch: {name}")
        if target.exists():
            raise ValueError("Another process created the target; refusing to overwrite")
        os.replace(temporary, target)
        return {"path": str(target), "url": url, "final_url": final_url,
                "expected_md5": expected_md5,
                "md5_source": "torchvision 0.29.0 OxfordIIITPet._RESOURCES",
                "local_sha256_note": "Computed locally; not claimed to be publisher-supplied SHA256",
                "cached": False, "download_seconds": time.perf_counter() - started, **result}
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "classifier-downloads")
    parser.add_argument("--download", action="store_true", help="Explicitly permit the two official downloads")
    args = parser.parse_args(argv)
    if not args.download:
        print(json.dumps({"publisher": HOMEPAGE, "resources": RESOURCES,
                          "license": "CC BY-SA 4.0; retain attribution and original-owner copyrights",
                          "scope": "Local research only; original images are not published by this tool",
                          "no_network_or_write": True}, ensure_ascii=False, indent=2))
        return 0
    try:
        archives = {name: download_archive(name, args.output_dir) for name in RESOURCES}
        source = {"schema_version": 1, "dataset": "Oxford-IIIT Pet",
                  "publisher": HOMEPAGE, "license": "CC BY-SA 4.0",
                  "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                  "license_basis": "Publisher page explicitly permits commercial/research use under CC BY-SA 4.0",
                  "copyright_note": "Original image owners retain copyright",
                  "attribution": ["Omkar M. Parkhi", "Andrea Vedaldi", "Andrew Zisserman", "C. V. Jawahar"],
                  "use": "Local model-optimization research; no public image redistribution",
                  "acquired_at": datetime.now().astimezone().isoformat(), "archives": archives}
        output = args.output_dir / "download-source.json"
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
