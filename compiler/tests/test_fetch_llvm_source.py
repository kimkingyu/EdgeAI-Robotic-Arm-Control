#!/usr/bin/env python3
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mlir/fetch_llvm_source.py"
SPEC = importlib.util.spec_from_file_location("fetch_llvm_source", SCRIPT)
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


class SourceFetchTests(unittest.TestCase):
    def test_exact_response(self):
        response = b"HTTP/2.0 206 Partial Content\r\nContent-Range: bytes 4-7/12\r\n\r\nABCD"
        self.assertEqual(fetcher.parse_range_response(response, 4, 7, 12), b"ABCD")

    def test_bad_status_range_total_and_length(self):
        template = b"HTTP/2.0 %s\r\nContent-Range: bytes %s\r\n\r\n%s"
        for status, span, body in [(b"200 OK", b"4-7/12", b"ABCD"),
                                    (b"206 Partial", b"0-3/12", b"ABCD"),
                                    (b"206 Partial", b"4-7/13", b"ABCD"),
                                    (b"206 Partial", b"4-7/12", b"ABC")]:
            with self.subTest(status=status, span=span, body=body), self.assertRaises(ValueError):
                fetcher.parse_range_response(template % (status, span, body), 4, 7, 12)

    def test_subranges_resume_after_failure_without_repeating_success(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "piece.part"
            first = mock.Mock(returncode=0, stderr=b"", stdout=b"HTTP/2.0 206 OK\r\nContent-Range: bytes 4-7/12\r\n\r\nefgh")
            failed = mock.Mock(returncode=1, stderr=b"Get https://example.test/asset?sig=secret : timeout")
            second = mock.Mock(returncode=0, stderr=b"", stdout=b"HTTP/2.0 206 OK\r\nContent-Range: bytes 8-11/12\r\n\r\nijkl")
            with mock.patch.object(fetcher, "REQUEST_BYTES", 4):
                with mock.patch.object(fetcher.subprocess, "run", side_effect=[first, failed, failed]):
                    with self.assertRaises(RuntimeError) as error:
                        fetcher.fetch_piece("test/test", 1, 4, 11, 12, output)
                self.assertIn("timeout", str(error.exception))
                self.assertNotIn("secret", str(error.exception))
                self.assertFalse(output.exists())
                with mock.patch.object(fetcher.subprocess, "run", return_value=second) as run:
                    fetcher.fetch_piece("test/test", 1, 4, 11, 12, output)
                    run.assert_called_once()
                    self.assertIn("Range: bytes=8-11", run.call_args.args[0])
            self.assertEqual(output.read_bytes(), b"efghijkl")

    def test_one_transport_reconnect_but_no_retry_for_auth_or_bad_ranges(self):
        failed = mock.Mock(returncode=1, stderr=b"net/http: TLS handshake timeout")
        accepted = mock.Mock(returncode=0, stdout=b"HTTP/1.1 206 OK\r\nContent-Range: bytes 4-7/12\r\n\r\nABCD")
        with mock.patch.object(fetcher.subprocess, "run", side_effect=[failed, accepted]) as run:
            self.assertEqual(fetcher.request_range(["gh"], 4, 7, 12), b"ABCD")
            self.assertEqual(run.call_count, 2)
        denied = mock.Mock(returncode=1, stderr=b"HTTP 403 forbidden")
        with mock.patch.object(fetcher.subprocess, "run", return_value=denied) as run:
            with self.assertRaises(RuntimeError):
                fetcher.request_range(["gh"], 4, 7, 12)
            run.assert_called_once()
        with mock.patch.object(fetcher.subprocess, "run", return_value=accepted) as run:
            with self.assertRaises(ValueError):
                fetcher.request_range(["gh"], 5, 8, 12)
            run.assert_called_once()

    def fixture(self, root, payload):
        lock = root / "lock.json"
        lock.write_text(json.dumps({"llvm": {"source_archive": "source.xz", "source_bytes": len(payload),
            "source_sha256": hashlib.sha256(payload).hexdigest(), "repository": "test/test", "release_asset_id": 1}}))
        return lock, root / "source.xz"

    def test_verified_archive_needs_no_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock, archive = self.fixture(root, b"known")
            archive.write_bytes(b"known")
            with mock.patch.object(fetcher.subprocess, "run") as run:
                self.assertEqual(fetcher.fetch(lock, root, 1), archive)
                run.assert_not_called()

    def test_corrupt_existing_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock, archive = self.fixture(root, b"known")
            archive.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                fetcher.fetch(lock, root, 1)
            self.assertEqual(archive.read_bytes(), b"wrong")

    def test_resume_and_full_hash_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = b"abcdefghijkl"
            lock, archive = self.fixture(root, payload)
            archive.write_bytes(payload[:5])
            seen = []
            def fake_piece(repository, asset_id, start, end, total, output):
                seen.append((start, end))
                output.write_bytes(payload[start:end + 1])
                return "fixture"
            with mock.patch.object(fetcher, "CHUNK", 4), mock.patch.object(fetcher, "fetch_piece", fake_piece):
                with mock.patch.object(fetcher.shutil, "which", return_value="gh"):
                    fetcher.fetch(lock, root, 1)
            self.assertEqual(seen, [(4, 7), (8, 11)])
            self.assertEqual(archive.read_bytes(), payload)

    def test_wrong_download_cannot_replace_partial_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock, archive = self.fixture(root, b"abcdefghijkl")
            archive.write_bytes(b"abcd")
            def corrupt_piece(repository, asset_id, start, end, total, output):
                output.write_bytes(b"x" * (end - start + 1))
                return "corrupt fixture"
            with mock.patch.object(fetcher, "CHUNK", 4), mock.patch.object(fetcher, "fetch_piece", corrupt_piece):
                with mock.patch.object(fetcher.shutil, "which", return_value="gh"), self.assertRaises(ValueError):
                    fetcher.fetch(lock, root, 1)
            self.assertEqual(archive.read_bytes(), b"abcd")


if __name__ == "__main__":
    unittest.main(verbosity=2)
