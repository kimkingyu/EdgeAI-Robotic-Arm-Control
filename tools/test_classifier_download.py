#!/usr/bin/env python3
"""Offline download-boundary tests; fake archives are test data only."""
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import download_classifier_dataset as downloader


class Response(io.BytesIO):
    def __init__(self, payload, url, length=None):
        super().__init__(payload)
        self.url = url
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}

    def geturl(self):
        return self.url


class DownloadTests(unittest.TestCase):
    def fixture(self, directory, *, endpoint=None, payload=None, expected=None, length=None):
        data = payload if payload is not None else gzip.compress(b"synthetic-test-only")
        url = downloader.HOMEPAGE + "data/images.tar.gz"
        digest = expected or hashlib.md5(data, usedforsecurity=False).hexdigest()
        return (patch.object(downloader, "RESOURCES", {"images.tar.gz": (url, digest)}),
                patch.object(downloader.urllib.request, "urlopen",
                             return_value=Response(data, endpoint or url, length=length)),
                patch.object(downloader.shutil, "disk_usage", return_value=SimpleNamespace(free=4 * 1024 ** 3)))

    def test_help_and_dry_run_have_no_network_or_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(downloader.urllib.request, "urlopen") as network:
            absent = Path(directory) / "not-created"
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                self.assertEqual(downloader.main(["--output-dir", str(absent)]), 0)
            self.assertTrue(json.loads(capture.getvalue())["no_network_or_write"])
            self.assertFalse(absent.exists())
            network.assert_not_called()
        result = subprocess.run([sys.executable, "-S", str(ROOT / "tools/download_classifier_dataset.py"), "--help"],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_verified_archive_is_published_and_cached_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            resource, network, disk = self.fixture(directory)
            with resource, network as mocked, disk:
                result = downloader.download_archive("images.tar.gz", directory)
                self.assertFalse(result["cached"])
                self.assertEqual(len(result["sha256"]), 64)
                again = downloader.download_archive("images.tar.gz", directory)
                self.assertTrue(again["cached"])
                self.assertEqual(mocked.call_count, 1)
                self.assertFalse(list(Path(directory).glob("*.part")))

    def test_oxford_thor_redirect_is_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            resource, network, disk = self.fixture(directory, endpoint="https://thor.robots.ox.ac.uk/pets/images.tar.gz")
            with resource, network, disk:
                result = downloader.download_archive("images.tar.gz", directory)
                self.assertEqual(result["final_url"], "https://thor.robots.ox.ac.uk/pets/images.tar.gz")

    def test_external_or_plain_http_redirect_is_rejected(self):
        for endpoint in ("https://example.invalid/images.tar.gz", "http://thor.robots.ox.ac.uk/pets/images.tar.gz"):
            with self.subTest(endpoint=endpoint), tempfile.TemporaryDirectory() as directory:
                resource, network, disk = self.fixture(directory, endpoint=endpoint)
                with resource, network, disk, self.assertRaisesRegex(ValueError, "Unexpected download redirect"):
                    downloader.download_archive("images.tar.gz", directory)
                self.assertFalse((Path(directory) / "images.tar.gz").exists())
                self.assertFalse(list(Path(directory).iterdir()))

    def test_bad_checksum_html_and_size_limit_are_not_published(self):
        for arguments in ({"expected": "0" * 32}, {"payload": b"<html>error</html>"},
                          {"length": downloader.MAX_BYTES_PER_ARCHIVE + 1}):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                resource, network, disk = self.fixture(directory, **arguments)
                with resource, network, disk, self.assertRaises(ValueError):
                    downloader.download_archive("images.tar.gz", directory)
                self.assertFalse(list(Path(directory).iterdir()))

    def test_existing_invalid_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "images.tar.gz"
            existing.write_bytes(b"user-owned-invalid-file")
            resource, network, disk = self.fixture(directory)
            with resource, network as mocked, disk, self.assertRaisesRegex(ValueError, "preserving"):
                downloader.download_archive("images.tar.gz", directory)
            self.assertEqual(existing.read_bytes(), b"user-owned-invalid-file")
            mocked.assert_not_called()

    def test_unknown_resource_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            downloader.download_archive("other.tar.gz", directory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
