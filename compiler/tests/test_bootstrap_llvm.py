#!/usr/bin/env python3
"""Offline, native-Windows-compatible tests; subprocesses are always mocked."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mlir/bootstrap_llvm.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_llvm", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name).resolve() / "home"
        self.home.mkdir()
        self.root = self.home / "llvm isolated"
        self.archive = self.home / "source.tar.xz"
        self.lock_path = self.home / "lock.json"
        self.document = json.loads(builder.DEFAULT_LOCK.read_text(encoding="utf-8"))
        self.source_name = self.document["llvm"]["source_directory"]
        self.create_archive()
        self.calls = []
        self.failure = None
        self.missing_system_tool = None
        self.missing_built_tool = None
        self.missing_package = None
        self.versions = {}
        self.output = b"complete mock output\n"
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(Path, "home", return_value=self.home).start()
        mock.patch.object(builder.platform, "system", return_value="Linux").start()
        mock.patch.object(builder.platform, "machine", return_value="aarch64").start()
        mock.patch.object(builder.shutil, "which", side_effect=self.which).start()
        self.run_mock = mock.patch.object(builder.subprocess, "run", side_effect=self.fake_run).start()

    def save_lock(self):
        self.lock_path.write_text(json.dumps(self.document), encoding="utf-8")

    def create_archive(self, extra=()):
        with tarfile.open(self.archive, "w:xz") as handle:
            body = b"# mock LLVM source\n"
            member = tarfile.TarInfo(self.source_name + "/llvm/CMakeLists.txt")
            member.size = len(body)
            handle.addfile(member, io.BytesIO(body))
            for member in extra:
                handle.addfile(member)
        payload = self.archive.read_bytes()
        self.document["llvm"]["source_bytes"] = len(payload)
        self.document["llvm"]["source_sha256"] = hashlib.sha256(payload).hexdigest()
        self.save_lock()

    def lock(self):
        return builder.load_lock(self.lock_path)

    def which(self, executable, **kwargs):
        return None if executable == self.missing_system_tool else executable

    def fake_run(self, command, **kwargs):
        self.calls.append(command)
        code, output = 0, self.output
        if self.failure is not None:
            outcome = self.failure(command)
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome:
                code, output = outcome, b"mock subprocess failed\n"
        if not code:
            if "--version" in command:
                executable = command[0]
                defaults = {
                    "/usr/bin/gcc": "gcc (Ubuntu 11.4.0) 11.4.0\n",
                    "/usr/bin/g++": "g++ (Ubuntu 11.4.0) 11.4.0\n",
                    "/usr/bin/cmake": "cmake version 3.22.1\n",
                    "/usr/bin/make": "GNU Make 4.3\n",
                    "/usr/bin/python3": "Python 3.10.12\n",
                }
                version = self.versions.get(executable, defaults.get(executable,
                    "LLVM (https://llvm.org/):\n  LLVM version 20.1.8\n  Optimized build with assertions.\n"))
                output = version.encode("utf-8")
            elif "--build" in command:
                self.create_build_artifacts()
            elif command[0] == "/usr/bin/du":
                output = ("2048\t" + str(self.root) + "\n").encode("utf-8")
        kwargs["stdout"].write(output)
        return subprocess.CompletedProcess(command, code)

    def create_build_artifacts(self):
        binary = self.root / "build/bin"
        binary.mkdir(parents=True, exist_ok=True)
        for name in builder.REQUIRED_TOOLS:
            if name != self.missing_built_tool:
                (binary / name).write_bytes(b"mock executable")
        for name in ("LLVM", "MLIR"):
            directory = self.root / "build/lib/cmake" / name.lower()
            directory.mkdir(parents=True, exist_ok=True)
            if name != self.missing_package:
                (directory / (name + "Config.cmake")).write_text("# mock package", encoding="utf-8")

    def bootstrap(self, configure_only=False):
        return builder.bootstrap(self.archive, self.root, self.lock_path, configure_only)

    def summary(self):
        return json.loads((self.root / "toolchain.json").read_text(encoding="utf-8"))

    def command_records(self):
        return [json.loads(Path(path).read_text(encoding="utf-8")) for path in self.summary()["commands"]]

    def assert_failed(self):
        self.assertEqual(self.summary()["status"], "failed")
        self.assertFalse(self.summary()["toolchain_complete"])
        self.assertIn("error", self.summary())
        self.assertIn("ended_utc", self.summary())

    def member(self, suffix, kind=tarfile.REGTYPE, target=""):
        member = tarfile.TarInfo(self.source_name + "/" + suffix)
        member.type, member.linkname = kind, target
        return member

    def test_default_lock_is_repository_lock_and_version_comes_from_tag(self):
        self.assertEqual(builder.DEFAULT_LOCK, SCRIPT.parents[2] / "compiler/toolchain.lock.json")
        self.assertEqual(self.lock()["version"], "20.1.8")
        self.document["llvm"]["tag"] = "llvmorg-20.1.9"
        self.save_lock()
        self.assertEqual(self.lock()["version"], "20.1.9")

    def test_invalid_lock_policy_is_rejected(self):
        for key, value in [("generator", "Ninja"), ("parallel_jobs", 4), ("build_type", "Debug"),
                           ("projects", ["clang"]), ("targets", ["X86"]), ("tools", ["llc"]),
                           ("tag", "main"), ("source_bytes", True), ("source_sha256", "not-a-hash")]:
            with self.subTest(key=key):
                original = self.document["llvm"][key]
                self.document["llvm"][key] = value
                self.save_lock()
                with self.assertRaises(ValueError):
                    self.lock()
                self.document["llvm"][key] = original
        self.run_mock.assert_not_called()

    def test_source_directory_cannot_escape_or_collide(self):
        for name in ("../source", "/source", "C:/source", "source/child", "source\\child", ".", "..",
                     "build", "logs", "link.lock", "toolchain.json", builder.SOURCE_MARKER, "source;bad"):
            with self.subTest(name=name):
                self.document["llvm"]["source_directory"] = name
                self.save_lock()
                with self.assertRaises(ValueError):
                    self.lock()

    def test_hash_mismatch_is_rejected_before_tar_is_opened(self):
        payload = self.archive.read_bytes()
        self.archive.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        with mock.patch.object(builder.tarfile, "open") as tar_open, self.assertRaisesRegex(ValueError, "SHA-256"):
            self.bootstrap()
        tar_open.assert_not_called()
        self.run_mock.assert_not_called()
        self.assertFalse((self.root / self.source_name).exists())
        self.assert_failed()

    def test_size_mismatch_is_rejected_before_tar_is_opened(self):
        self.archive.write_bytes(b"short")
        with mock.patch.object(builder.tarfile, "open") as tar_open, self.assertRaisesRegex(ValueError, "size"):
            self.bootstrap()
        tar_open.assert_not_called()
        self.run_mock.assert_not_called()
        self.assert_failed()

    def test_missing_archive_is_a_recorded_failure(self):
        self.archive.unlink()
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.bootstrap()
        self.assert_failed()
        self.run_mock.assert_not_called()

    def test_host_gate_rejects_windows_and_non_aarch64_without_writes(self):
        for system, machine in [("Windows", "ARM64"), ("Linux", "x86_64"), ("Darwin", "arm64")]:
            with self.subTest(system=system, machine=machine):
                with mock.patch.object(builder.platform, "system", return_value=system), \
                        mock.patch.object(builder.platform, "machine", return_value=machine), \
                        self.assertRaisesRegex(RuntimeError, "native Linux aarch64"):
                    self.bootstrap()
        self.assertFalse(self.root.exists())
        self.run_mock.assert_not_called()

    def test_root_must_be_strictly_below_home_not_a_prefix_lookalike(self):
        for root in (self.home, self.home.parent, self.home.parent / "home-other" / "llvm"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "below"):
                builder.validate_root(root)

    def test_root_parent_must_preexist(self):
        root = self.home / "not-created" / "llvm"
        with self.assertRaisesRegex(ValueError, "parent must already exist"):
            builder.validate_root(root)
        self.assertFalse(root.parent.exists())

    def test_root_rejects_cmake_list_separator(self):
        with self.assertRaisesRegex(ValueError, "list separators"):
            builder.validate_root(self.home / "llvm;unsafe")

    def test_root_rejects_symlink(self):
        with mock.patch.object(Path, "is_symlink", return_value=True), self.assertRaisesRegex(ValueError, "symlink"):
            builder.validate_root(self.root)

    def test_root_cannot_overlap_workspace_or_current_directory(self):
        protected = self.home / "rknn-project"
        protected.mkdir()
        for target in (protected, protected / "llvm"):
            with self.subTest(target=target), mock.patch.object(builder, "WORKSPACE", protected), \
                    self.assertRaisesRegex(ValueError, "independent"):
                builder.validate_root(target)
        with mock.patch.object(Path, "cwd", return_value=protected), \
                self.assertRaisesRegex(ValueError, "independent"):
            builder.validate_root(protected / "llvm")
        with mock.patch.object(Path, "cwd", return_value=protected / "child"), \
                self.assertRaisesRegex(ValueError, "independent"):
            builder.validate_root(protected)

    def test_root_cannot_overlap_active_venv_or_conda(self):
        environment = self.home / "rknn-venv"
        environment.mkdir()
        for name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: str(environment)}), \
                    self.assertRaisesRegex(ValueError, "independent"):
                builder.validate_root(environment / "llvm")
        with mock.patch.object(builder.sys, "prefix", str(environment)), \
                self.assertRaisesRegex(ValueError, "independent"):
            builder.validate_root(environment / "llvm")

    def test_nonempty_unowned_root_is_untouched(self):
        self.root.mkdir()
        sentinel = self.root / "existing-rknn-file"
        sentinel.write_bytes(b"do not change")
        with self.assertRaisesRegex(ValueError, "owned marker"):
            self.bootstrap()
        self.assertEqual(list(self.root.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b"do not change")
        self.run_mock.assert_not_called()

    def test_preexisting_empty_root_is_supported(self):
        self.root.mkdir()
        result = self.bootstrap(configure_only=True)
        self.assertEqual(result["status"], "configured")
        self.assertTrue((self.root / builder.ROOT_MARKER).is_file())

    def test_owned_root_marker_must_match_lock(self):
        builder.initialize_root(self.root, self.lock())
        self.document["llvm"]["commit"] = "different"
        self.save_lock()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.bootstrap()
        self.run_mock.assert_not_called()

    def test_existing_source_without_marker_is_not_accepted(self):
        builder.initialize_root(self.root, self.lock())
        source = self.root / self.source_name
        source.mkdir()
        sentinel = source / "old-source"
        sentinel.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "owned marker"):
            self.bootstrap()
        self.assertEqual(sentinel.read_bytes(), b"keep")
        self.assertFalse((source / builder.SOURCE_MARKER).exists())
        self.assert_failed()

    def test_source_marker_written_after_extraction_and_checked_on_reuse(self):
        self.bootstrap(configure_only=True)
        marker = self.root / self.source_name / builder.SOURCE_MARKER
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), builder.provenance(self.lock()))
        with mock.patch.object(builder.tarfile, "open") as tar_open:
            self.bootstrap(configure_only=True)
        tar_open.assert_not_called()
        marker.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.bootstrap(configure_only=True)
        self.assert_failed()

    def test_source_directory_is_taken_from_lock_not_hardcoded(self):
        self.source_name = "alternate-locked-source"
        self.document["llvm"]["source_directory"] = self.source_name
        self.create_archive()
        result = self.bootstrap(configure_only=True)
        self.assertEqual(result["source"], str(self.root / self.source_name))
        self.assertTrue((self.root / self.source_name / builder.SOURCE_MARKER).is_file())

    def test_python310_without_tar_filter_api_uses_complete_custom_validation(self):
        members = [self.member("llvm/CMakeLists.txt")]
        archive_file = mock.MagicMock()
        archive_file.__enter__.return_value = archive_file
        archive_file.getmembers.return_value = members
        def extract(path, members):
            source = path / self.source_name / "llvm"
            source.mkdir(parents=True)
            (source / "CMakeLists.txt").write_text("# mock source", encoding="utf-8")
        archive_file.extractall.side_effect = extract
        legacy_tar = SimpleNamespace(open=mock.Mock(return_value=archive_file))
        with mock.patch.object(builder, "tarfile", legacy_tar), \
                mock.patch.object(builder, "validate_members", wraps=builder.validate_members) as validate:
            self.bootstrap(configure_only=True)
        validate.assert_called_once_with(members, self.source_name)
        archive_file.extractall.assert_called_once_with(path=self.root, members=members)

    def test_reused_source_still_requires_full_archive_hash(self):
        self.bootstrap(configure_only=True)
        self.archive.write_bytes(b"x" * self.document["llvm"]["source_bytes"])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.bootstrap(configure_only=True)
        self.assert_failed()

    def test_all_members_are_checked_before_any_extraction(self):
        self.create_archive([tarfile.TarInfo("../escape")])
        with mock.patch.object(builder.tarfile.TarFile, "extractall") as extract, \
                self.assertRaisesRegex(ValueError, "escapes"):
            self.bootstrap()
        extract.assert_not_called()
        self.assertFalse((self.root / self.source_name).exists())
        self.assert_failed()

    def test_rejects_absolute_foreign_and_traversing_member_paths(self):
        for name in ("/absolute", "C:/absolute", "C:\\absolute", "../escape",
                     self.source_name + "/../escape", "other/file"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.validate_members([tarfile.TarInfo(name)], self.source_name)

    def test_archive_cannot_supply_marker_duplicates_or_special_files(self):
        cases = [[self.member(builder.SOURCE_MARKER)],
                 [self.member("file"), self.member("file")],
                 [self.member("fifo", tarfile.FIFOTYPE)],
                 [self.member("device", tarfile.CHRTYPE)], []]
        for members in cases:
            with self.subTest(members=members), self.assertRaises(ValueError):
                builder.validate_members(members, self.source_name)

    def test_relative_symlink_filter_uses_containing_directory_and_strips_ownership(self):
        member = self.member("clang/test/Driver/Inputs/CUDA-symlinks/usr/bin/ptxas",
                             tarfile.SYMTYPE, "../../opt/cuda/bin/ptxas")
        member.uid, member.gid, member.mode = 123, 456, 0o4777
        with mock.patch.object(builder.tarfile, "data_filter", side_effect=AssertionError("wrong-base legacy filter")):
            result = builder.checked_data_filter(member, self.root)
        self.assertEqual(result.name, member.name)
        self.assertEqual(result.linkname, "../../opt/cuda/bin/ptxas")
        self.assertTrue(result.issym())
        for name in ("uid", "gid", "uname", "gname", "mode"):
            self.assertIsNone(getattr(result, name))
        self.assertEqual((member.uid, member.gid, member.mode), (123, 456, 0o4777))

    def test_symlink_filter_still_rejects_foreign_and_source_escape_paths(self):
        for target in ("/etc/passwd", "C:/escape", "../../escape", "../../../escape"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                builder.checked_data_filter(self.member("llvm/link", tarfile.SYMTYPE, target), self.root)
        with self.assertRaises(ValueError):
            builder.checked_data_filter(self.member("../link", tarfile.SYMTYPE, "target"), self.root)

    def test_symlink_filter_checks_live_parent_resolution(self):
        member = self.member("llvm/link", tarfile.SYMTYPE, "file")
        original = Path.resolve
        def resolve(path, *args, **kwargs):
            return self.home if path == self.root / self.source_name / "llvm" else original(path, *args, **kwargs)
        with mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve), self.assertRaises(ValueError):
            builder.checked_data_filter(member, self.root)

    def test_non_symlink_entries_keep_standard_data_filter(self):
        for kind in (tarfile.REGTYPE, tarfile.LNKTYPE):
            member = self.member("llvm/file", kind, self.source_name + "/llvm/target")
            with mock.patch.object(builder.tarfile, "data_filter", return_value=mock.sentinel.filtered) as filter_fn:
                self.assertIs(builder.checked_data_filter(member, self.root), mock.sentinel.filtered)
                filter_fn.assert_called_once_with(member, self.root)

    def test_safe_internal_symlinks_and_hardlinks_are_allowed(self):
        members = [self.member("llvm/file"), self.member("llvm/alias", tarfile.SYMTYPE, "file"),
                   self.member("llvm/alias2", tarfile.SYMTYPE, "../llvm/alias"),
                   self.member("llvm/hard", tarfile.LNKTYPE, self.source_name + "/llvm/file")]
        builder.validate_members(members, self.source_name)

    def test_absolute_and_escaping_link_targets_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            for target in ("/etc/passwd", "C:/outside", "../../escape", "../../../escape"):
                with self.subTest(kind=kind, target=target), self.assertRaises(ValueError):
                    builder.validate_members([self.member("llvm/link", kind, target)], self.source_name)

    def test_link_chain_dotdot_escape_is_not_hidden_by_lexical_normalization(self):
        members = [self.member("up", tarfile.SYMTYPE, "."),
                   self.member("bad", tarfile.SYMTYPE, "up/../outside")]
        with self.assertRaisesRegex(ValueError, "escapes"):
            builder.validate_members(members, self.source_name)

    def test_archive_cannot_write_through_links_in_either_order(self):
        members = [self.member("alias", tarfile.SYMTYPE, "llvm"), self.member("alias/file")]
        for order in (members, list(reversed(members))):
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, "through a link"):
                builder.validate_members(order, self.source_name)

    def test_link_cycles_and_missing_hardlink_targets_are_rejected(self):
        cases = [[self.member("one", tarfile.SYMTYPE, "two"), self.member("two", tarfile.SYMTYPE, "one")],
                 [self.member("hard", tarfile.LNKTYPE, self.source_name + "/missing")]]
        for members in cases:
            with self.subTest(members=members), self.assertRaises(ValueError):
                builder.validate_members(members, self.source_name)

    def test_partial_extraction_keeps_evidence_but_never_writes_source_marker(self):
        source = self.root / self.source_name
        def incomplete(**kwargs):
            source.mkdir()
            (source / "partial").write_bytes(b"preserve")
            raise OSError("mock extraction interrupted")
        with mock.patch.object(builder.tarfile.TarFile, "extractall", side_effect=incomplete), \
                self.assertRaisesRegex(OSError, "interrupted"):
            self.bootstrap()
        self.assertFalse((source / builder.SOURCE_MARKER).exists())
        self.assert_failed()
        with self.assertRaisesRegex(ValueError, "owned marker"):
            self.bootstrap()
        self.assertEqual((source / "partial").read_bytes(), b"preserve")

    def test_configure_flags_compilers_and_shared_flock_launcher(self):
        self.bootstrap(configure_only=True)
        command = next(command for command in self.calls if "-S" in command)
        self.assertEqual(command[:8], ["/usr/bin/cmake", "-S", str(self.root / self.source_name / "llvm"),
                                      "-B", str(self.root / "build"), "-G", "Unix Makefiles",
                                      "-DCMAKE_BUILD_TYPE=Release"])
        options = dict(option[2:].split("=", 1) for option in command if option.startswith("-D"))
        expected = {
            "LLVM_ENABLE_PROJECTS": "mlir;clang", "LLVM_TARGETS_TO_BUILD": "AArch64",
            "LLVM_ENABLE_ASSERTIONS": "ON", "LLVM_ENABLE_RTTI": "ON", "LLVM_BUILD_EXAMPLES": "OFF",
            "LLVM_INCLUDE_TESTS": "ON", "MLIR_ENABLE_BINDINGS_PYTHON": "OFF",
            "CMAKE_C_COMPILER": "/usr/bin/gcc", "CMAKE_CXX_COMPILER": "/usr/bin/g++",
            "Python3_EXECUTABLE": "/usr/bin/python3", "CMAKE_MAKE_PROGRAM": "/usr/bin/make",
            "LLVM_ENABLE_ZLIB": "OFF", "LLVM_ENABLE_ZSTD": "OFF", "LLVM_ENABLE_LIBXML2": "OFF",
        }
        for key, value in expected.items():
            self.assertEqual(options[key], value)
        for language in ("C", "CXX"):
            self.assertEqual(options["CMAKE_" + language + "_LINKER_LAUNCHER"].split(";"),
                             ["/usr/bin/flock", str(self.root / "link.lock")])
        self.assertNotIn("LLVM_PARALLEL_LINK_JOBS", options)
        self.assertNotIn("Ninja", command)

    def test_clean_environment_does_not_mutate_or_use_rknn_settings(self):
        contaminated = {"VIRTUAL_ENV": str(self.home / "rknn"), "PATH": "rknn/bin",
                        "LD_LIBRARY_PATH": "rknn/lib", "PYTHONPATH": "rknn/packages",
                        "CC": "wrong", "CXXFLAGS": "bad", "CMAKE_PREFIX_PATH": "rknn", "MAKEFLAGS": "-j64"}
        with mock.patch.dict(os.environ, contaminated):
            before = dict(os.environ)
            self.bootstrap(configure_only=True)
            self.assertEqual(dict(os.environ), before)
        for call in self.run_mock.call_args_list:
            env = call.kwargs["env"]
            self.assertEqual(env["PATH"], "/usr/bin:/bin")
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            for key in contaminated:
                if key != "PATH":
                    self.assertNotIn(key, env)
            self.assertEqual(call.kwargs["cwd"], str(self.root))
            self.assertEqual(call.kwargs["stderr"], subprocess.STDOUT)
            self.assertNotIn("shell", call.kwargs)

    def test_missing_system_tool_fails_without_install_or_configure(self):
        self.missing_system_tool = "/usr/bin/flock"
        with self.assertRaisesRegex(RuntimeError, "missing existing system tool"):
            self.bootstrap()
        self.run_mock.assert_not_called()
        self.assert_failed()

    def test_wrong_system_baseline_is_rejected(self):
        self.versions["/usr/bin/cmake"] = "cmake version 3.22.10\n"
        with self.assertRaisesRegex(RuntimeError, "baseline"):
            self.bootstrap()
        self.assertFalse(any("-S" in command for command in self.calls))
        self.assert_failed()

    def test_subprocess_failure_preserves_command_log_timing_and_exit_code(self):
        self.failure = lambda command: 23 if "-S" in command else 0
        with self.assertRaisesRegex(RuntimeError, "exit code 23"):
            self.bootstrap()
        self.assert_failed()
        records = self.command_records()
        failed = records[-1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["returncode"], 23)
        self.assertEqual(failed["command"], self.calls[-1])
        self.assertEqual(Path(failed["log"]).read_text(encoding="utf-8"), "mock subprocess failed\n")
        self.assertTrue(failed["started_utc"].endswith("+00:00"))
        self.assertTrue(failed["ended_utc"].endswith("+00:00"))
        self.assertGreaterEqual(failed["elapsed_seconds"], 0)
        self.assertEqual(sum("-S" in command for command in self.calls), 1)
        self.assertFalse(any("--build" in command for command in self.calls))

    def test_subprocess_launch_error_is_recorded_without_retry(self):
        self.failure = lambda command: OSError("mock executable vanished") if "-S" in command else 0
        with self.assertRaisesRegex(OSError, "vanished"):
            self.bootstrap()
        self.assert_failed()
        record = self.command_records()[-1]
        self.assertIsNone(record["returncode"])
        self.assertIn("OSError", record["error"])
        self.assertEqual(sum("-S" in command for command in self.calls), 1)

    def test_build_failure_does_not_run_tool_versions(self):
        self.failure = lambda command: 9 if "--build" in command else 0
        with self.assertRaisesRegex(RuntimeError, "exit code 9"):
            self.bootstrap()
        self.assert_failed()
        self.assertEqual(sum("--build" in command for command in self.calls), 1)
        self.assertFalse(any(str(self.root / "build/bin") in command[0] for command in self.calls))

    def test_success_requires_all_five_version_checks_and_cmake_packages(self):
        result = self.bootstrap()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["toolchain_complete"])
        self.assertEqual(set(result["tools"]), set(builder.REQUIRED_TOOLS))
        self.assertEqual(result["LLVM_DIR"], str(self.root / "build/lib/cmake/llvm"))
        self.assertEqual(result["MLIR_DIR"], str(self.root / "build/lib/cmake/mlir"))
        for name in builder.REQUIRED_TOOLS:
            path = str(self.root / "build/bin" / name)
            self.assertEqual(result["tools"][name], {"path": path, "version": "20.1.8"})
            self.assertIn([path, "--version"], self.calls)
        build = next(command for command in self.calls if "--build" in command)
        self.assertEqual(build, ["/usr/bin/cmake", "--build", str(self.root / "build"), "--parallel", "2",
                                 "--target", "mlir-opt", "mlir-translate", "llc", "FileCheck", "mlir-tblgen"])
        self.assertFalse(any("install" in command for command in self.calls))
        self.assertEqual(result["disk_usage_kib"], 2048)
        self.assertIn(["/usr/bin/du", "-sk", "--", str(self.root)], self.calls)

    def test_missing_built_tool_is_not_success(self):
        self.missing_built_tool = "mlir-tblgen"
        with self.assertRaisesRegex(RuntimeError, "missing/unsafe built tool"):
            self.bootstrap()
        self.assert_failed()

    def test_wrong_tool_version_including_prefix_collision_is_not_success(self):
        executable = str(self.root / "build/bin/FileCheck")
        for version in ("20.1.7", "20.1.80", "20.1.8git", "20.1.8-rc1"):
            with self.subTest(version=version):
                self.versions[executable] = "LLVM version " + version
                with self.assertRaisesRegex(RuntimeError, "FileCheck did not report"):
                    self.bootstrap()
                self.assert_failed()

    def test_built_tool_version_subprocess_failure_is_not_success(self):
        executable = str(self.root / "build/bin/mlir-opt")
        self.failure = lambda command: 4 if command[0] == executable else 0
        with self.assertRaisesRegex(RuntimeError, "exit code 4"):
            self.bootstrap()
        self.assert_failed()

    def test_missing_cmake_package_is_not_success(self):
        self.missing_package = "MLIR"
        with self.assertRaisesRegex(RuntimeError, "CMake package"):
            self.bootstrap()
        self.assert_failed()

    def test_configure_only_never_claims_completed_toolchain(self):
        result = self.bootstrap(configure_only=True)
        self.assertEqual(result["status"], "configured")
        self.assertFalse(result["toolchain_complete"])
        self.assertNotIn("tools", result)
        self.assertNotIn("LLVM_DIR", result)
        self.assertFalse(any("--build" in command for command in self.calls))
        self.assertFalse(any(str(self.root / "build/bin") in command[0] for command in self.calls))

    def test_each_run_preserves_history_and_configure_only_clears_stale_success(self):
        successful = self.bootstrap()
        historical = Path(successful["evidence_directory"]) / "run.json"
        first = historical.read_bytes()
        configured = self.bootstrap(configure_only=True)
        self.assertNotEqual(successful["evidence_directory"], configured["evidence_directory"])
        self.assertEqual(historical.read_bytes(), first)
        self.assertFalse(self.summary()["toolchain_complete"])
        self.assertNotIn("tools", self.summary())
        self.failure = lambda command: 7 if "--build" in command else 0
        with self.assertRaises(RuntimeError):
            self.bootstrap()
        self.assert_failed()
        self.assertEqual(historical.read_bytes(), first)

    def test_complete_logs_are_not_truncated(self):
        self.output = (b"entire build output\n" * 10000) + b"final line\n"
        self.bootstrap()
        for record in self.command_records():
            if "-S" in record["command"] or "--build" in record["command"]:
                self.assertEqual(Path(record["log"]).read_bytes(), self.output)

    def test_rss_metric_is_labeled_as_single_child_not_concurrent_peak(self):
        resource = SimpleNamespace(RUSAGE_CHILDREN=-1, getrusage=mock.Mock(
            return_value=SimpleNamespace(ru_maxrss=12345)))
        with mock.patch.object(builder, "resource", resource):
            result = self.bootstrap(configure_only=True)
        self.assertEqual(result["maximum_single_child_process_rss_kib"], 12345)
        self.assertIn("maximum single child process RSS", result["rss_note"])
        self.assertIn("not concurrent total memory peak", result["rss_note"])
        self.assertTrue(all(record["maximum_single_child_process_rss_kib"] == 12345
                            for record in self.command_records()))
        resource.getrusage.assert_called_with(resource.RUSAGE_CHILDREN)

    def test_windows_resource_import_fallback_is_safe(self):
        with mock.patch.object(builder, "resource", None):
            result = self.bootstrap(configure_only=True)
        self.assertIsNone(result["maximum_single_child_process_rss_kib"])

    def test_optional_du_absence_does_not_install_anything(self):
        self.missing_system_tool = "/usr/bin/du"
        result = self.bootstrap(configure_only=True)
        self.assertIsNone(result["disk_usage_kib"])
        self.assertFalse(any(command[0] == "/usr/bin/du" for command in self.calls))

    def test_cli_default_lock_configure_only_and_nonzero_failure(self):
        result = {"status": "configured", "toolchain_complete": False, "evidence_directory": "mock"}
        args = ["--archive", str(self.archive), "--root", str(self.root), "--configure-only"]
        with mock.patch.object(builder, "bootstrap", return_value=result) as bootstrap, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(builder.main(args), 0)
        bootstrap.assert_called_once_with(self.archive, self.root, builder.DEFAULT_LOCK, True)
        with mock.patch.object(builder, "bootstrap", side_effect=RuntimeError("mock failure")), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(builder.main(args + ["--lock", str(self.lock_path)]), 1)
        self.assertIn("mock failure", stderr.getvalue())

    def test_interrupt_is_nonzero_and_preserves_evidence(self):
        self.failure = lambda command: KeyboardInterrupt() if "-S" in command else 0
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(builder.main(["--archive", str(self.archive), "--root", str(self.root),
                                           "--lock", str(self.lock_path)]), 1)
        self.assert_failed()
        self.assertEqual(self.command_records()[-1]["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
