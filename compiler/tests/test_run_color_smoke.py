#!/usr/bin/env python3
"""Mock orchestration tests only; these are not real generated-kernel evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mlir/run_color_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_color_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeRunnerTests(unittest.TestCase):
    def fixture(self, root):
        toolchain = root / "llvm"
        (toolchain / "bin").mkdir(parents=True)
        for name in ("mlir-opt", "mlir-translate", "llc"):
            (toolchain / "bin" / name).write_bytes(b"mock tool")
        edge = root / "edge-opt"
        edge.write_bytes(b"mock tool")
        return toolchain, edge, root / "result"

    def invoke(self, paths, process):
        with mock.patch.object(smoke.platform, "system", return_value="Linux"):
            with mock.patch.object(smoke.platform, "machine", return_value="aarch64"):
                with mock.patch.object(smoke.subprocess, "run", side_effect=process):
                    return smoke.run_smoke(*paths)

    def test_host_gate_never_creates_output(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            with mock.patch.object(smoke.platform, "system", return_value="Windows"):
                with self.assertRaises(RuntimeError):
                    smoke.run_smoke(*paths)
            self.assertFalse(paths[2].exists())

    def test_missing_tool_fails_instead_of_skipping(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            (paths[0] / "bin/llc").unlink()
            with self.assertRaises(FileNotFoundError):
                self.invoke(paths, mock.Mock())
            self.assertFalse(paths[2].exists())

    def test_wrong_version_cannot_start_lowering(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            def process(args, **kwargs):
                kwargs["stdout"].write(b"LLVM version 20.1.80\n")
                return mock.Mock(returncode=0)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                self.invoke(paths, process)
            report = (paths[2] / "smoke.json").read_text(encoding="utf-8")
            self.assertIn('"status": "failed"', report)
            self.assertFalse((paths[2] / "01-color.mlir").exists())

    def test_success_requires_real_command_stages_in_correct_order(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            calls = []
            def process(args, **kwargs):
                calls.append(args)
                if "--version" in args:
                    kwargs["stdout"].write(b"LLVM version 20.1.8\n")
                if "-o" in args:
                    Path(args[args.index("-o") + 1]).write_bytes(b"orchestration fixture, NOT generated code")
                return mock.Mock(returncode=0)
            report = self.invoke(paths, process)
            labels = [row["label"] for row in report["commands"]]
            self.assertEqual(labels[4:], ["lower", "translate", "object", "assembly", "link-library",
                                          "link-cpp-test", "cpp-test", "python-test", "dynamic-dependencies"])
            self.assertIn("-mtriple=aarch64-unknown-linux-gnu", calls[6])
            self.assertIn("-Wl,-z,defs", calls[8])
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["production_resize_implemented"])
            self.assertEqual(len(report["artifacts"]), 7)

    def test_nonzero_lowering_cannot_run_translation_or_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            calls = []
            def process(args, **kwargs):
                calls.append(args)
                kwargs["stdout"].write(b"LLVM version 20.1.8\n")
                return mock.Mock(returncode=0 if "--version" in args else 1)
            with self.assertRaisesRegex(RuntimeError, "lower failed"):
                self.invoke(paths, process)
            self.assertEqual(len(calls), 5)
            self.assertFalse((paths[2] / "03-color.ll").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
