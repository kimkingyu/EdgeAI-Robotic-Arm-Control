#!/usr/bin/env python3
"""Offline, standard-library checks; never run an inventory command or hardware API."""

import contextlib
from datetime import datetime, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

if __package__:
    from . import collect_mlir_environment as inventory
else:
    import collect_mlir_environment as inventory


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        # An accidental real child process fails every test instead of touching tools.
        self.run = self.contexts.enter_context(mock.patch.object(
            inventory.subprocess, "run", side_effect=AssertionError("Unexpected command execution")))


class ToolTests(OfflineTestCase):
    def test_missing_tool_does_not_execute(self):
        with mock.patch.object(inventory.shutil, "which", return_value=None):
            result = inventory.probe_tool("mlir-opt")
        self.assertEqual(result["status"], "missing")
        self.assertIsNone(result["which"])
        self.assertIsNone(result["returncode"])
        self.run.assert_not_called()

    def test_version_success_uses_allowlisted_arguments_and_bounded_timeout(self):
        executable = os.path.abspath("tool directory/python")
        self.run.side_effect = None
        self.run.return_value = subprocess.CompletedProcess([], 0, "Python 3.10.12\n", "")
        with mock.patch.object(inventory.shutil, "which", return_value=executable):
            result = inventory.probe_tool("python")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["which"], executable)
        self.assertEqual(result["stdout"], "Python 3.10.12\n")
        args, kwargs = self.run.call_args
        self.assertEqual(args[0], [executable, "--version"])
        self.assertGreater(kwargs["timeout"], 0)
        self.assertLessEqual(kwargs["timeout"], 10)
        self.assertFalse(kwargs["shell"])
        self.assertFalse(kwargs["check"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs["capture_output"])
        self.assertEqual(kwargs["errors"], "replace")

    def test_nonzero_exit_preserves_both_streams(self):
        self.run.side_effect = None
        self.run.return_value = subprocess.CompletedProcess([], 2, "partial output\n", "bad option\n")
        with mock.patch.object(inventory.shutil, "which", return_value="clang"):
            result = inventory.probe_tool("clang")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 2)
        self.assertEqual(result["stdout"], "partial output\n")
        self.assertEqual(result["stderr"], "bad option\n")

    def test_timeout_preserves_partial_bytes_as_json_text(self):
        self.run.side_effect = subprocess.TimeoutExpired(
            ["llc", "--version"], 5, output=b"partial\xff", stderr=b"waiting")
        with mock.patch.object(inventory.shutil, "which", return_value="llc"):
            result = inventory.probe_tool("llc")
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["stdout"], "partial\ufffd")
        self.assertEqual(result["stderr"], "waiting")
        self.assertIsNone(result["returncode"])
        json.dumps(result)

    def test_timeout_without_output_and_text_normalization(self):
        self.run.side_effect = subprocess.TimeoutExpired(["ninja", "--version"], 5)
        with mock.patch.object(inventory.shutil, "which", return_value="ninja"):
            result = inventory.probe_tool("ninja")
        self.assertEqual((result["stdout"], result["stderr"]), ("", ""))
        self.assertEqual(inventory.output_text("unchanged"), "unchanged")

    def test_launch_error_and_which_error_are_recorded(self):
        for lookup_error in (False, True):
            with self.subTest(lookup_error=lookup_error):
                self.run.side_effect = PermissionError("denied")
                with mock.patch.object(inventory.shutil, "which", return_value="gcc",
                                       side_effect=OSError("lookup denied") if lookup_error else None):
                    result = inventory.probe_tool("gcc")
                self.assertEqual(result["status"], "error")
                self.assertIn("denied", result["reason"])

    def test_mixed_failures_do_not_hide_later_tools(self):
        def run(command, **kwargs):
            name = Path(command[0]).name
            self.assertEqual(command[1:], ["--version"])
            self.assertLessEqual(kwargs["timeout"], 10)
            if name == "ninja":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, 1 if name == "clang" else 0, name, "")

        self.run.side_effect = run
        with mock.patch.object(inventory.shutil, "which",
                               side_effect=lambda name: None if name == "mlir-opt" else name):
            result = inventory.collect_tools()
        self.assertEqual(set(result), set(inventory.TOOLS))
        self.assertEqual(len(result), 10)
        self.assertEqual(result["mlir-opt"]["status"], "missing")
        self.assertEqual(result["ninja"]["status"], "timeout")
        self.assertEqual(result["clang"]["status"], "failed")
        self.assertEqual(result["git"]["status"], "ok")
        self.assertEqual(self.run.call_count, 9)

    def test_non_allowlisted_command_is_rejected_before_lookup(self):
        with mock.patch.object(inventory.shutil, "which") as which:
            with self.assertRaises(ValueError):
                inventory.probe_tool("pip")
        which.assert_not_called()
        self.run.assert_not_called()


class PackageTests(OfflineTestCase):
    def metadata_version(self, versions):
        def version(name):
            if name not in versions:
                raise inventory.metadata.PackageNotFoundError(name)
            if isinstance(versions[name], Exception):
                raise versions[name]
            return versions[name]
        return version

    def test_versions_use_distribution_aliases_without_module_imports(self):
        versions = {"numpy": "1.26.4", "opencv-python-headless": "4.9.0.80",
                    "rknn-toolkit-lite2": "2.0.0"}
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "cv2", "rknnlite", "rknn"}:
                self.fail("Inspected package was imported: " + name)
            return original_import(name, *args, **kwargs)

        with mock.patch.object(inventory.metadata, "version", side_effect=self.metadata_version(versions)):
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                result = inventory.collect_packages()
        for name, expected in (("numpy", "1.26.4"), ("cv2", "4.9.0.80"), ("rknnlite", "2.0.0")):
            self.assertEqual(result[name]["status"], "ok")
            self.assertEqual(result[name]["version"], expected)
        self.assertEqual(result["rknnlite"]["distributions"], {"rknn-toolkit-lite2": "2.0.0"})
        self.run.assert_not_called()

    def test_missing_metadata_is_explicit_and_not_importability_claim(self):
        with mock.patch.object(inventory.metadata, "version", side_effect=self.metadata_version({})):
            result = inventory.collect_packages()
        self.assertEqual(set(result), {"numpy", "cv2", "rknnlite"})
        for record in result.values():
            self.assertEqual(record["status"], "missing")
            self.assertEqual(record["version"], "unknown")
            self.assertIn("does not establish module importability", record["note"])

    def test_conflicting_opencv_metadata_does_not_choose_an_active_provider(self):
        versions = {"opencv-python": "4.8.0", "opencv-contrib-python": "4.9.0"}
        with mock.patch.object(inventory.metadata, "version", side_effect=self.metadata_version(versions)):
            result = inventory.collect_packages()
        self.assertEqual(result["cv2"]["status"], "ambiguous")
        self.assertEqual(result["cv2"]["version"], "unknown")
        self.assertEqual(result["cv2"]["distributions"], versions)

    def test_broken_metadata_isolated_to_distribution(self):
        versions = {"numpy": PermissionError("denied"), "opencv-python": ValueError("broken"),
                    "opencv-python-headless": "4.9.0", "rknn-toolkit-lite2": "2.0.0"}
        with mock.patch.object(inventory.metadata, "version", side_effect=self.metadata_version(versions)):
            result = inventory.collect_packages()
        self.assertEqual(result["numpy"]["status"], "error")
        self.assertEqual(result["cv2"]["status"], "partial")
        self.assertEqual(result["rknnlite"]["status"], "ok")
        self.assertIn("opencv-python", result["cv2"]["errors"])


class SystemTests(OfflineTestCase):
    def linux_fixture(self, failure=None):
        files = {
            "meminfo": "MemTotal:       16384 kB\nMemAvailable: 8192 kB\nHugePages_Total: 0\n",
            "os_release": '# comment\nNAME="Ubuntu Linux"\nVERSION_ID="22.04"\n',
            "cpuinfo": "processor : 0\nmodel name : Example CPU\nCPU architecture : 8\n"
                       "Serial : SECRET-SERIAL\nmodel name : Example CPU\nCPU part : 0xd05\n",
        }
        with mock.patch.object(Path, "read_text", side_effect=[
            failure if failure is not None else files["meminfo"], files["os_release"], files["cpuinfo"],
        ]) as reader:
            result = inventory.collect_linux_info("Linux")
        self.assertEqual(reader.call_count, 3)
        return result

    def test_linux_fields_units_and_cpu_serial_exclusion(self):
        result = self.linux_fixture()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["meminfo"]["values"]["MemTotal"], "16384 kB")
        self.assertEqual(result["meminfo"]["values"]["HugePages_Total"], "0")
        self.assertEqual(result["os_release"]["values"]["NAME"], "Ubuntu Linux")
        self.assertEqual(result["cpuinfo"]["values"]["model name"], ["Example CPU"])
        self.assertEqual(result["cpuinfo"]["values"]["CPU part"], ["0xd05"])
        self.assertNotIn("SECRET-SERIAL", json.dumps(result))
        self.assertNotIn("processor", result["cpuinfo"]["values"])
        self.assertEqual(set(result["cpuinfo"]["values"]) - set(inventory.CPU_FIELDS), set())

    def test_non_linux_does_not_read_linux_paths(self):
        with mock.patch.object(Path, "read_text") as reader:
            result = inventory.collect_linux_info("Windows")
        reader.assert_not_called()
        self.assertEqual(result["status"], "not_applicable")
        for name in inventory.LINUX_FILES:
            self.assertEqual(result[name]["status"], "unknown")

    def test_unreadable_meminfo_keeps_release_and_cpu_fields(self):
        result = self.linux_fixture(PermissionError("denied"))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["meminfo"]["status"], "unknown")
        self.assertIn("denied", result["meminfo"]["reason"])
        self.assertEqual(result["os_release"]["status"], "ok")
        self.assertEqual(result["cpuinfo"]["status"], "ok")

    def test_malformed_release_is_unknown_without_losing_other_sources(self):
        with mock.patch.object(Path, "read_text", side_effect=[
            "MemTotal: 42 kB\n", 'NAME="unterminated', "model name: example\n",
        ]):
            result = inventory.collect_linux_info("Linux")
        self.assertEqual(result["os_release"]["status"], "unknown")
        self.assertEqual(result["cpuinfo"]["status"], "ok")

    def test_release_parser_never_evaluates_shell_text(self):
        result = inventory.parse_os_release(
            'NAME="Linux test" # comment\nID=linux\nVALUE="$(touch forbidden) $HOME"\nignored\n')
        self.assertEqual(result, {"NAME": "Linux test", "ID": "linux",
                                  "VALUE": "$(touch forbidden) $HOME"})
        self.run.assert_not_called()

    def test_empty_linux_sources_are_unknown(self):
        with mock.patch.object(Path, "read_text", return_value=""):
            result = inventory.collect_linux_info("Linux")
        self.assertEqual(result["status"], "partial")
        for name in inventory.LINUX_FILES:
            self.assertEqual(result[name]["status"], "unknown")

    def test_cpu_count_affinity_architecture_and_model(self):
        linux = self.linux_fixture()
        with mock.patch.object(inventory.os, "cpu_count", return_value=8):
            with mock.patch.object(inventory.os, "sched_getaffinity", create=True, return_value={5, 2}) as affinity:
                result = inventory.collect_cpu_info("aarch64", linux)
        self.assertEqual(result["logical_count"], {"status": "ok", "value": 8})
        self.assertEqual(result["affinity"], {"status": "ok", "cpus": [2, 5], "count": 2})
        self.assertEqual(result["architecture"]["value"], "aarch64")
        self.assertEqual(result["model"]["values"], ["Example CPU"])
        affinity.assert_called_once_with(0)

    def test_unavailable_cpu_count_affinity_and_model_are_unknown(self):
        linux = inventory.collect_linux_info("Windows")
        with mock.patch.object(inventory.os, "cpu_count", return_value=None):
            with mock.patch.object(inventory.os, "sched_getaffinity", None, create=True):
                result = inventory.collect_cpu_info("", linux)
        for name in ("logical_count", "affinity", "architecture", "model"):
            self.assertEqual(result[name]["status"], "unknown")
        self.assertEqual(result["affinity"]["cpus"], "unknown")
        self.assertEqual(result["logical_count"]["value"], "unknown")

    def test_denied_cpu_apis_are_unknown(self):
        linux = inventory.collect_linux_info("Windows")
        with mock.patch.object(inventory.os, "cpu_count", side_effect=OSError("count denied")):
            with mock.patch.object(inventory.os, "sched_getaffinity", create=True,
                                   side_effect=PermissionError("affinity denied")):
                result = inventory.collect_cpu_info("AMD64", linux)
        self.assertEqual(result["logical_count"]["status"], "unknown")
        self.assertEqual(result["affinity"]["status"], "unknown")
        self.assertIn("affinity denied", result["affinity"]["reason"])

    def test_disk_bytes_and_unavailable_status(self):
        root = Path("example")
        with mock.patch.object(inventory.shutil, "disk_usage",
                               return_value=SimpleNamespace(total=100, used=40, free=60)) as usage:
            result = inventory.collect_disk_usage(root)
        usage.assert_called_once_with(root)
        self.assertEqual(result, {"path": str(root), "status": "ok", "unit": "bytes",
                                  "total": 100, "used": 40, "free": 60})
        with mock.patch.object(inventory.shutil, "disk_usage", side_effect=OSError("denied")):
            result = inventory.collect_disk_usage(root)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["free"], "unknown")


class PlatformTests(OfflineTestCase):
    def test_windows_uses_native_version_and_only_architecture_environment_keys(self):
        version = SimpleNamespace(platform_version=(10, 0, 22631), major=10, minor=0, build=22631)
        environment = mock.Mock()
        environment.get.side_effect = [None, "AMD64"]
        with mock.patch.object(inventory.sys, "platform", "win32"):
            with mock.patch.object(inventory.sys, "getwindowsversion", create=True, return_value=version):
                with mock.patch.object(inventory.os, "environ", environment):
                    result = inventory.collect_platform_info()
        self.assertEqual(result["system"], "Windows")
        self.assertEqual(result["version"], "10.0.22631")
        self.assertEqual(result["architecture"], "AMD64")
        self.assertEqual(environment.get.call_args_list,
                         [mock.call("PROCESSOR_ARCHITEW6432"), mock.call("PROCESSOR_ARCHITECTURE")])
        self.run.assert_not_called()

    def test_windows_prefers_native_architecture_hint_over_process_architecture(self):
        version = SimpleNamespace(platform_version=(), major=10, minor=0, build=19045)
        with mock.patch.object(inventory.sys, "platform", "win32"):
            with mock.patch.object(inventory.sys, "getwindowsversion", create=True, return_value=version):
                with mock.patch.object(inventory.os, "environ", {"PROCESSOR_ARCHITEW6432": "ARM64",
                                                                "PROCESSOR_ARCHITECTURE": "x86",
                                                                "SECRET_TOKEN": "not-for-report"}):
                    result = inventory.collect_platform_info()
        self.assertEqual(result["architecture"], "ARM64")
        self.assertEqual(result["version"], "10.0.19045")
        self.assertNotIn("not-for-report", json.dumps(result))

    def test_linux_uses_uname_without_hostname_or_commands(self):
        native = SimpleNamespace(sysname="Linux", release="5.10-test", version="fixture",
                                 machine="aarch64", nodename="private-host")
        with mock.patch.object(inventory.sys, "platform", "linux"):
            with mock.patch.object(inventory.os, "uname", create=True, return_value=native):
                result = inventory.collect_platform_info()
        self.assertEqual(result["system"], "Linux")
        self.assertEqual(result["architecture"], "aarch64")
        self.assertEqual(result["release"], "5.10-test")
        self.assertNotIn("private-host", json.dumps(result))
        self.run.assert_not_called()

    def test_unavailable_native_platform_api_does_not_run_fallback_commands(self):
        with mock.patch.object(inventory.sys, "platform", "linux"):
            with mock.patch.object(inventory.os, "uname", create=True, side_effect=OSError("denied")):
                result = inventory.collect_platform_info()
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["architecture"], "unknown")
        self.run.assert_not_called()


class ReportTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        directory = self.contexts.enter_context(tempfile.TemporaryDirectory(prefix="mlir-env-test-"))
        self.root = Path(directory)
        self.report = self.root / "report.json"

    def test_cli_requires_report_and_defaults_to_script_project_root(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                inventory.parse_args([])
        self.assertNotEqual(error.exception.code, 0)
        args = inventory.parse_args(["--report", str(self.report)])
        self.assertEqual(args.project_root, Path(inventory.__file__).resolve().parents[1])
        self.assertEqual(args.report, self.report)

    def test_new_report_roundtrips_unicode_json(self):
        inventory.write_report(self.report, {"name": "环境清单", "value": 7})
        self.assertEqual(json.loads(self.report.read_text(encoding="utf-8")),
                         {"name": "环境清单", "value": 7})
        self.assertEqual(list(self.root.iterdir()), [self.report])

    def test_existing_report_refused_without_collecting_or_changing_bytes(self):
        self.report.write_bytes(b"original report\x00\n")
        with mock.patch.object(inventory, "collect_environment") as collect:
            with contextlib.redirect_stderr(io.StringIO()):
                code = inventory.main(["--report", str(self.report)])
        self.assertEqual(code, 1)
        self.assertEqual(self.report.read_bytes(), b"original report\x00\n")
        collect.assert_not_called()
        with self.assertRaises(FileExistsError):
            inventory.write_report(self.report, {"new": True})

    def test_missing_parent_refused_without_creating_directories_or_collecting(self):
        path = self.root / "missing" / "report.json"
        with mock.patch.object(inventory, "collect_environment") as collect:
            with contextlib.redirect_stderr(io.StringIO()):
                code = inventory.main(["--report", str(path)])
        self.assertEqual(code, 1)
        self.assertFalse(path.parent.exists())
        collect.assert_not_called()

    def test_directory_target_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            inventory.write_report(self.root, {"new": True})
        self.assertTrue(self.root.is_dir())

    def test_dangling_symlink_is_rejected_even_if_target_does_not_exist(self):
        with mock.patch.object(Path, "exists", return_value=False):
            with mock.patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaises(FileExistsError):
                    inventory.validate_report_path(self.report)
        self.assertFalse(self.report.exists())

    def test_exclusive_creation_rejects_target_appearing_after_preflight(self):
        self.report.write_text("concurrent writer", encoding="utf-8")
        with mock.patch.object(inventory, "validate_report_path", return_value=self.report):
            with self.assertRaises(FileExistsError):
                inventory.write_report(self.report, {"replacement": True})
        self.assertEqual(self.report.read_text(encoding="utf-8"), "concurrent writer")

    def test_json_serialization_failure_does_not_create_report(self):
        with self.assertRaises(ValueError):
            inventory.write_report(self.report, {"invalid": float("nan")})
        self.assertFalse(self.report.exists())

    def test_invalid_project_root_fails_before_collection(self):
        file_root = self.root / "not-directory"
        file_root.write_text("fixture", encoding="utf-8")
        for root in (self.root / "missing", file_root):
            with self.subTest(root=root), mock.patch.object(inventory, "collect_environment") as collect:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = inventory.main(["--report", str(self.report), "--project-root", str(root)])
                self.assertEqual(code, 1)
                collect.assert_not_called()
                self.assertFalse(self.report.exists())

    def test_cli_writes_new_report_using_requested_root(self):
        with mock.patch.object(inventory, "collect_environment", return_value={"mock": True}) as collect:
            with contextlib.redirect_stdout(io.StringIO()):
                code = inventory.main(["--report", str(self.report), "--project-root", str(self.root)])
        self.assertEqual(code, 0)
        collect.assert_called_once_with(self.root.resolve())
        self.assertEqual(json.loads(self.report.read_text(encoding="utf-8")), {"mock": True})
        self.run.assert_not_called()

    def test_all_required_fields_and_current_python_are_independent_of_path_python(self):
        self.run.side_effect = None
        self.run.return_value = subprocess.CompletedProcess([], 0, "Python 3.10.12\n", "")
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(inventory, "collect_platform_info", return_value={
                "system": "Windows", "architecture": "AMD64", "release": "mock-release",
                "version": "mock-version", "status": "ok",
            }))
            stack.enter_context(mock.patch.object(inventory.shutil, "which",
                                                 side_effect=lambda name: "path-python" if name == "python" else None))
            stack.enter_context(mock.patch.object(inventory.metadata, "version",
                                                 side_effect=inventory.metadata.PackageNotFoundError("fixture")))
            stack.enter_context(mock.patch.object(inventory.os, "cpu_count", return_value=8))
            stack.enter_context(mock.patch.object(inventory.os, "sched_getaffinity", None, create=True))
            stack.enter_context(mock.patch.object(inventory.shutil, "disk_usage",
                                                 return_value=SimpleNamespace(total=100, used=40, free=60)))
            result = inventory.collect_environment(self.root)
        expected = {"schema_version", "created_utc", "platform", "current_python", "tools",
                    "packages", "disk_usage", "cpu", "linux", "script_sha256", "project_root"}
        self.assertTrue(expected <= set(result))
        self.assertEqual(datetime.fromisoformat(result["created_utc"]).utcoffset(), timedelta(0))
        self.assertEqual(result["current_python"]["executable"], sys.executable)
        self.assertEqual(result["current_python"]["version"], sys.version)
        self.assertEqual(result["tools"]["python"]["which"], "path-python")
        self.assertEqual(result["tools"]["python"]["stdout"], "Python 3.10.12\n")
        self.assertEqual(set(result["tools"]), set(inventory.TOOLS))
        for tool in result["tools"].values():
            self.assertTrue({"which", "stdout", "stderr", "status", "returncode"} <= set(tool))
        self.assertEqual(set(result["packages"]), {"numpy", "cv2", "rknnlite"})
        self.assertEqual(result["script_sha256"], hashlib.sha256(Path(inventory.__file__).read_bytes()).hexdigest())
        self.assertEqual(result["cpu"]["affinity"]["status"], "unknown")
        self.assertEqual(result["linux"]["status"], "not_applicable")
        self.assertFalse({"env", "environ", "credentials", "ssh_config"} & set(result))
        inventory.write_report(self.report, result)
        self.assertEqual(json.loads(self.report.read_text(encoding="utf-8")), result)
        self.assertEqual(self.run.call_count, 1)

    def test_sha256_uses_file_bytes(self):
        fixture = self.root / "source.py"
        fixture.write_bytes(b"source\r\n")
        self.assertEqual(inventory.file_sha256(fixture), hashlib.sha256(b"source\r\n").hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
