#!/usr/bin/env python3
"""Fetch the locked official LLVM source via gh, with bounded parallel ranges.

A partial gh-release download can supply the prefix. Only a matching complete
SHA-256 is published as usable. No third-party mirrors, credentials or installs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

CHUNK = 4 * 1024 * 1024
REQUEST_BYTES = 256 * 1024


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            result.update(block)
    return result.hexdigest()


def parse_range_response(response, start, end, total):
    # gh --include uses CRLF headers on the supported gh release.
    header, separator, body = response.partition(b"\r\n\r\n")
    if not separator:
        header, separator, body = response.partition(b"\n\n")
    if not separator or not re.match(rb"HTTP/\S+ 206\b", header):
        raise ValueError("expected HTTP 206 with a separate response body")
    match = re.search(rb"(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$", header)
    if not match or tuple(map(int, match.groups())) != (start, end, total):
        raise ValueError("Content-Range does not match the requested locked asset")
    if len(body) != end - start + 1:
        raise ValueError("truncated range body")
    return body


def request_range(command, start, end, total):
    # Retry one transient transport failure with a new connection, never bad data/auth.
    for attempt in range(2):
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if completed.returncode == 0:
                return parse_range_response(completed.stdout, start, end, total)
            error = re.sub(r"https?://\S+", "[redacted URL]",
                           completed.stderr.decode("utf-8", errors="replace"))[-1000:]
            transient = any(word in error.lower() for word in
                            ("timeout", "connection reset", "wsarecv", "unexpected eof"))
        except subprocess.TimeoutExpired:
            error, transient = "transfer timeout after 120 seconds", True
        if attempt == 1 or not transient:
            raise RuntimeError("gh range {}-{} failed: {}".format(start, end, error))
        print("Reconnect once for transient range failure at {}".format(start), flush=True)


def fetch_piece(repository, asset_id, start, end, total, output):
    if output.exists() and output.stat().st_size == end - start + 1:
        return "cached"
    # Smaller requests survive this connection's stalls; retain completed subranges.
    fragments = []
    for lower in range(start, end + 1, REQUEST_BYTES):
        upper = min(lower + REQUEST_BYTES - 1, end)
        fragment = output.with_name(output.name + ".{}.range".format(lower))
        if not fragment.exists() or fragment.stat().st_size != upper - lower + 1:
            command = ["gh", "api", "--method", "GET", "--include",
                       "-H", "Accept: application/octet-stream",
                       "-H", "Range: bytes={}-{}".format(lower, upper),
                       "repos/{}/releases/assets/{}".format(repository, asset_id)]
            body = request_range(command, lower, upper, total)
            with tempfile.NamedTemporaryFile(dir=str(output.parent), delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(body)
            temporary.replace(fragment)
        fragments.append(fragment)
    with tempfile.NamedTemporaryFile(dir=str(output.parent), delete=False) as handle:
        temporary = Path(handle.name)
        for fragment in fragments:
            with fragment.open("rb") as stream:
                shutil.copyfileobj(stream, handle)
    temporary.replace(output)
    return "downloaded"


def fetch(lock_path, directory, workers):
    if not directory.is_dir():
        raise ValueError("download directory must already exist")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))["llvm"]
    archive = directory / lock["source_archive"]
    total, checksum = lock["source_bytes"], lock["source_sha256"]
    existing = archive.stat().st_size if archive.exists() else 0
    if existing == total and digest(archive) == checksum:
        print("Verified existing archive: " + str(archive), flush=True)
        return archive
    if existing >= total:
        raise ValueError("existing archive has wrong size/hash; keep it for investigation, do not overwrite")
    if shutil.which("gh") is None:
        raise RuntimeError("gh is required; no automatic installation")
    # Reuse only whole chunks from the interrupted single-stream download.
    prefix = existing // CHUNK * CHUNK
    pieces_dir = directory / (archive.name + ".parts")
    pieces_dir.mkdir(exist_ok=True)
    pieces = [(start, min(start + CHUNK, total) - 1,
               pieces_dir / ("{:012d}.part".format(start))) for start in range(prefix, total, CHUNK)]
    print("Locked asset: {} bytes; reusable prefix: {}; remaining chunks: {}".format(
        total, prefix, len(pieces)), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_piece, lock["repository"], lock["release_asset_id"],
                                   start, end, total, path): start for start, end, path in pieces}
        for future in as_completed(futures):
            print("{} range at {}".format(future.result(), futures[future]), flush=True)
    with tempfile.NamedTemporaryFile(dir=str(directory), delete=False) as assembled:
        temporary = Path(assembled.name)
        if prefix:
            with archive.open("rb") as old:
                remaining = prefix
                while remaining:
                    block = old.read(min(CHUNK, remaining))
                    if not block:
                        raise ValueError("partial archive changed during assembly")
                    assembled.write(block)
                    remaining -= len(block)
        for _, _, path in pieces:
            with path.open("rb") as part:
                shutil.copyfileobj(part, assembled)
    if temporary.stat().st_size != total or digest(temporary) != checksum:
        raise ValueError("assembled archive SHA-256 mismatch; candidate retained at " + str(temporary))
    temporary.replace(archive)
    print("Verified SHA-256: " + checksum, flush=True)
    print("Ready: " + str(archive), flush=True)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[2] / "compiler/toolchain.lock.json")
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args(argv)
    try:
        fetch(args.lock, args.directory, args.workers)
    except Exception as error:
        print("fetch_llvm_source: {}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
