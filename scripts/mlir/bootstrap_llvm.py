#!/usr/bin/env python3
"""Build the locked LLVM/MLIR release in an owned directory below the user's home.

Native Linux aarch64 only; uses existing system tools, never downloads, installs,
creates a venv, or modifies the calling environment. Windows can run mock tests.
Run from outside the build root. A failed/partial source tree is retained and is
not silently reused. Each invocation retains its own logs and JSON evidence.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import uuid

try:
    import resource
except ImportError:  # Native Windows supports importing this module for mocks.
    resource = None

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = WORKSPACE / "compiler/toolchain.lock.json"
OWNER = "edgeai-bootstrap-llvm-v1"
ROOT_MARKER = ".edgeai-llvm-root.json"
SOURCE_MARKER = ".edgeai-llvm-source.json"
REQUIRED_TOOLS = ("mlir-opt", "mlir-translate", "llc", "FileCheck", "mlir-tblgen")
SYSTEM_PATH = "/usr/bin:/bin"
SYSTEM_TOOLS = {
    "cmake": "/usr/bin/cmake", "cc": "/usr/bin/gcc", "cxx": "/usr/bin/g++",
    "make": "/usr/bin/make", "python": "/usr/bin/python3", "flock": "/usr/bin/flock",
}
RSS_NOTE = ("Linux RUSAGE_CHILDREN.ru_maxrss: maximum single child process RSS, "
            "in KiB, across this bootstrap process's waited-for children; "
            "not a per-command peak and not concurrent total memory peak.")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("refusing unsafe JSON destination: " + str(path))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def under(path, directory):
    return path == directory or directory in path.parents


def overlaps(left, right):
    return under(left, right) or under(right, left)


def load_lock(path):
    lock = json.loads(path.read_text(encoding="utf-8"))["llvm"]
    tag = re.fullmatch(r"llvmorg-(\d+\.\d+\.\d+)", lock["tag"])
    if not tag:
        raise ValueError("lock tag must identify an exact LLVM release")
    source = lock["source_directory"]
    if (not isinstance(source, str) or source in ("", ".", "..", "build", "logs", "link.lock", "toolchain.json")
            or source.startswith(".") or any(c in source for c in "/\\:;\r\n\x00")
            or PureWindowsPath(source).drive):
        raise ValueError("lock.source_directory must be one safe directory name")
    if type(lock["source_bytes"]) is not int or lock["source_bytes"] <= 0:
        raise ValueError("lock.source_bytes must be a positive integer")
    if not re.fullmatch(r"[0-9a-f]{64}", lock["source_sha256"]):
        raise ValueError("lock.source_sha256 must be a lowercase SHA-256")
    expected = {"generator": "Unix Makefiles", "build_type": "Release",
                "projects": ["mlir", "clang"], "targets": ["AArch64"],
                "parallel_jobs": 2, "target_triple": "aarch64-unknown-linux-gnu"}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError("unsupported locked build policy: " + key)
    if lock.get("tools") not in (list(REQUIRED_TOOLS[:-1]), list(REQUIRED_TOOLS)):
        raise ValueError("lock.tools must name the required LLVM/MLIR tools")
    return dict(lock, version=tag.group(1))


def require_native_host():
    if platform.system() != "Linux" or platform.machine().lower() != "aarch64":
        raise RuntimeError("actual configuration/build requires native Linux aarch64; "
                           "Windows is supported for mock tests only")


def validate_root(value):
    raw = value.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError("root must not be a symlink")
    root, home = raw.resolve(), Path.home().resolve()
    if root == home or not under(root, home):
        raise ValueError("root must be a dedicated directory strictly below the current user's home")
    if any(c in str(root) for c in ";\r\n"):
        raise ValueError("root cannot contain CMake list separators or line breaks")
    if not root.parent.is_dir():
        raise ValueError("root parent must already exist")
    protected = [WORKSPACE.resolve()]
    cwd = Path.cwd().resolve()
    if cwd != home and cwd != Path(cwd.anchor):
        protected.append(cwd)
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if os.environ.get(key):
            protected.append(Path(os.environ[key]).expanduser().resolve())
    if sys.prefix != sys.base_prefix:
        protected.append(Path(sys.prefix).resolve())
    if any(overlaps(root, path) for path in protected):
        raise ValueError("root must be independent of the workspace, working directory and active environments")
    if root.exists() and not root.is_dir():
        raise ValueError("root must be a directory")
    return root


def provenance(lock):
    return {"owner": OWNER, "tag": lock["tag"], "commit": lock["commit"],
            "source_directory": lock["source_directory"], "source_bytes": lock["source_bytes"],
            "source_sha256": lock["source_sha256"]}


def require_marker(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing owned marker: " + str(path))
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ValueError("owned marker does not match this locked source: " + str(path))


def initialize_root(root, lock):
    expected = dict(provenance(lock), root=str(root))
    marker = root / ROOT_MARKER
    if root.exists() and any(root.iterdir()):
        require_marker(marker, expected)
    else:
        root.mkdir(exist_ok=True)  # Deliberately never creates the parent.
        write_json(marker, expected)
    for name in ("build", "logs", lock["source_directory"]):
        child = root / name
        if child.is_symlink() or (child.exists() and not child.is_dir()):
            raise ValueError("refusing unsafe owned directory: " + str(child))
    for name in ("link.lock", "toolchain.json"):
        child = root / name
        if child.is_symlink() or (child.exists() and not child.is_file()):
            raise ValueError("refusing unsafe owned file: " + str(child))


def child_rss():
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def build_environment():
    # Pass a clean copy, never change os.environ or activate/deactivate any venv.
    # A minimal environment also excludes RKNN library paths and compiler flags.
    env = {key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "TZ") if key in os.environ}
    env.update(PATH=SYSTEM_PATH, LANG="C", LC_ALL="C", PYTHONNOUSERSITE="1")
    return env


class Recorder:
    def __init__(self, root, lock, configure_only):
        self.root = root
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8]
        self.directory = logs / name
        self.directory.mkdir()
        self.started = time.monotonic()
        self.commands = []
        self.environment = build_environment()
        self.summary = {
            "owner": OWNER, "root": str(root), "llvm": provenance(lock),
            "expected_version": lock["version"], "configure_only": configure_only,
            "status": "running", "toolchain_complete": False, "started_utc": utc_now(),
            "evidence_directory": str(self.directory), "commands": self.commands,
            "rss_note": RSS_NOTE,
        }
        self.save()

    def save(self):
        write_json(self.directory / "run.json", self.summary)
        write_json(self.root / "toolchain.json", self.summary)

    def run(self, label, command, cwd=None, read_output=False):
        command = [str(part) for part in command]
        cwd = self.root if cwd is None else cwd
        stem = "{:02d}-{}".format(len(self.commands) + 1, label)
        output = self.directory / (stem + ".log")
        metadata = self.directory / (stem + ".json")
        details = {"command": command, "cwd": str(cwd), "environment": self.environment,
                   "started_utc": utc_now(), "returncode": None, "status": "running",
                   "log": str(output), "rss_note": RSS_NOTE}
        self.commands.append(str(metadata))
        write_json(metadata, details)
        self.save()
        start = time.monotonic()
        try:
            with output.open("wb") as handle:
                result = subprocess.run(command, cwd=str(cwd), env=self.environment,
                                        stdout=handle, stderr=subprocess.STDOUT, check=False)
            details["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError("{} failed with exit code {}; see {}".format(label, result.returncode, output))
            details["status"] = "succeeded"
        except BaseException as error:
            details["status"] = "failed"
            details["error"] = "{}: {}".format(type(error).__name__, error)
            raise
        finally:
            details.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - start,
                           maximum_single_child_process_rss_kib=child_rss())
            write_json(metadata, details)
        return output.read_text(encoding="utf-8", errors="replace") if read_output else None

    def finish(self, status, error=None):
        self.summary.update(status=status, toolchain_complete=(status == "complete"),
                            ended_utc=utc_now(), elapsed_seconds=time.monotonic() - self.started,
                            maximum_single_child_process_rss_kib=child_rss())
        if error is not None:
            self.summary["error"] = "{}: {}".format(type(error).__name__, error)
        self.save()


def check_system_tools(recorder):
    tools = {}
    for name, executable in SYSTEM_TOOLS.items():
        found = shutil.which(executable, path=SYSTEM_PATH)
        if found is None:
            raise RuntimeError("missing existing system tool: " + executable + "; no automatic installation")
        tools[name] = executable
    patterns = {
        "cc": r"\b11\.4\.\d+\b", "cxx": r"\b11\.4\.\d+\b",
        "cmake": r"\bcmake version 3\.22\.1(?![\w.+-])",
        "make": r"\bGNU Make 4\.3(?![\w.+-])",
        "python": r"\bPython 3\.10\.\d+\b",
    }
    for name, pattern in patterns.items():
        output = recorder.run("prerequisite-" + name, [tools[name], "--version"], read_output=True)
        if not re.search(pattern, output):
            raise RuntimeError("existing {} does not match the required baseline; see {}".format(name, recorder.directory))
    return tools


def verify_archive(stream, lock):
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size != lock["source_bytes"]:
        raise ValueError("archive size does not match lock.source_bytes")
    checksum, total = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        checksum.update(block)
        total += len(block)
    if total != lock["source_bytes"] or checksum.hexdigest() != lock["source_sha256"]:
        raise ValueError("archive SHA-256/size does not match the lock")
    stream.seek(0)


def relative_tar_parts(name):
    if (not name or name.startswith("/") or "\\" in name
            or PureWindowsPath(name).drive or "\x00" in name):
        raise ValueError("unsafe absolute/foreign archive path: " + repr(name))
    return tuple(part for part in name.split("/") if part not in ("", "."))


def validate_members(members, source_name):
    """Validate the complete member graph before extraction, including link chains.

    Files beneath an archive link are forbidden, so extraction cannot use a
    symlink as an output directory. Link targets are resolved component by
    component (before '..'), not just lexically normalized.
    """
    entries, links = {}, {}
    for member in members:
        parts = relative_tar_parts(member.name)
        if not parts or parts[0] != source_name or ".." in parts:
            raise ValueError("archive member escapes the locked source directory: " + member.name)
        if parts in entries:
            raise ValueError("duplicate archive member: " + member.name)
        if len(parts) == 1 and not member.isdir():
            raise ValueError("archive source root must be a directory")
        if SOURCE_MARKER in parts:
            raise ValueError("archive must not supply the bootstrap source marker")
        if not (member.isdir() or member.isreg() or member.issym() or member.islnk()):
            raise ValueError("unsupported archive special file: " + member.name)
        entries[parts] = member
        if member.issym() or member.islnk():
            links[parts] = relative_tar_parts(member.linkname)

    def resolve_target(parts, active=frozenset()):
        result = []
        for part in parts:
            if part == "..":
                if len(result) <= 1:
                    raise ValueError("archive link target escapes source directory")
                result.pop()
                continue
            result.append(part)
            if result[0] != source_name:
                raise ValueError("archive link target is outside locked source directory")
            key = tuple(result)
            if key in links:
                if key in active or len(active) >= 40:
                    raise ValueError("cyclic or excessively nested archive link")
                target = links[key]
                if entries[key].issym():
                    target = key[:-1] + target
                result = list(resolve_target(target, active | {key}))
        return tuple(result)

    for parts, member in entries.items():
        if any(parts[:index] in links for index in range(1, len(parts))):
            raise ValueError("archive writes through a link: " + member.name)
        if member.issym() or member.islnk():
            target = links[parts]
            if member.issym():
                target = parts[:-1] + target
            resolved = resolve_target(target, frozenset({parts}))
            if member.islnk() and (resolved not in entries or not entries[resolved].isreg()):
                raise ValueError("archive hardlink must resolve to a regular archive file")
    if not entries:
        raise ValueError("empty source archive")


def checked_data_filter(member, destination):
    """Keep data filtering, with correct relative-symlink semantics on Jammy.

    The deployed Python 3.10 backport resolves symbolic links relative to the
    extraction root, as if they were hardlinks. Validate their actual containing
    directory instead; full archive-graph validation still runs before extraction.
    """
    if not member.issym():
        return tarfile.data_filter(member, destination)
    parts = relative_tar_parts(member.name)
    if not parts or ".." in parts:
        raise ValueError("unsafe symbolic link member path")
    relative_tar_parts(member.linkname)  # Reject absolute/foreign link targets.
    root = Path(destination).resolve()
    source_root = root / parts[0]
    location = root.joinpath(*parts)
    if not under(location.parent.resolve(), source_root) or location.is_symlink():
        raise ValueError("symbolic link location is outside source or already a link")
    target = (location.parent / member.linkname).resolve()
    if not under(target, source_root):
        raise ValueError("symbolic link target is outside source")
    filtered = copy.copy(member)
    # Match data_filter's symlink ownership/mode policy, without altering the link.
    for name in ("uid", "gid", "uname", "gname", "mode"):
        setattr(filtered, name, None)
    return filtered


def prepare_source(archive, root, lock):
    source = root / lock["source_directory"]
    if not archive.is_file():
        raise ValueError("archive must be an existing regular file")
    # Keep the same file descriptor for hashing and tar reading; a replaced path
    # cannot substitute an unchecked archive between verification and extraction.
    with archive.open("rb") as stream:
        verify_archive(stream, lock)
        if source.exists() or source.is_symlink():
            if source.is_symlink() or not source.is_dir():
                raise ValueError("source must be an owned real directory")
            require_marker(source / SOURCE_MARKER, provenance(lock))
        else:
            with tarfile.open(fileobj=stream, mode="r:xz") as archive_file:
                members = archive_file.getmembers()
                validate_members(members, lock["source_directory"])
                # Keep graph and live-filesystem checks; never use fully_trusted.
                filters = {"filter": checked_data_filter} if hasattr(tarfile, "data_filter") else {}
                archive_file.extractall(path=root, members=members, **filters)
            cmake_source = source / "llvm/CMakeLists.txt"
            if not cmake_source.is_file() or not under(cmake_source.resolve(), source):
                raise ValueError("verified archive has no usable llvm/CMakeLists.txt")
            write_json(source / SOURCE_MARKER, provenance(lock))
    cmake_source = source / "llvm/CMakeLists.txt"
    if not cmake_source.is_file() or not under(cmake_source.resolve(), source):
        raise ValueError("owned source has no usable llvm/CMakeLists.txt")
    return source


def configure_command(root, source, tools):
    # CMake >= 3.21 Makefile generators honor these launchers. Both languages
    # share one flock; this does not rely on Ninja-only linker job pools.
    launcher = tools["flock"] + ";" + str(root / "link.lock")
    options = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_C_COMPILER": tools["cc"],
        "CMAKE_CXX_COMPILER": tools["cxx"], "CMAKE_MAKE_PROGRAM": tools["make"],
        "Python3_EXECUTABLE": tools["python"],
        "CMAKE_C_LINKER_LAUNCHER": launcher, "CMAKE_CXX_LINKER_LAUNCHER": launcher,
        "LLVM_ENABLE_PROJECTS": "mlir;clang", "LLVM_TARGETS_TO_BUILD": "AArch64",
        "LLVM_ENABLE_ASSERTIONS": "ON", "LLVM_ENABLE_RTTI": "ON",
        "LLVM_BUILD_EXAMPLES": "OFF", "LLVM_INCLUDE_TESTS": "ON",
        "MLIR_ENABLE_BINDINGS_PYTHON": "OFF", "LLVM_ENABLE_ZLIB": "OFF",
        "LLVM_ENABLE_ZSTD": "OFF", "LLVM_ENABLE_LIBXML2": "OFF",
    }
    return [tools["cmake"], "-S", str(source / "llvm"), "-B", str(root / "build"),
            "-G", "Unix Makefiles"] + ["-D{}={}".format(key, value) for key, value in options.items()]


def build_command(root, tools):
    return [tools["cmake"], "--build", str(root / "build"), "--parallel", "2",
            "--target"] + list(REQUIRED_TOOLS)


def verify_built_tools(root, lock, recorder):
    tools = {}
    pattern = r"\bLLVM\s+version\s+" + re.escape(lock["version"]) + r"(?![\w.+-])"
    for name in REQUIRED_TOOLS:
        executable = root / "build/bin" / name
        if not executable.is_file() or not under(executable.resolve(), root / "build"):
            raise RuntimeError("missing/unsafe built tool: " + str(executable))
        output = recorder.run("version-" + name, [str(executable), "--version"], read_output=True)
        if not re.search(pattern, output):
            raise RuntimeError("{} did not report LLVM version {}".format(name, lock["version"]))
        tools[name] = {"path": str(executable), "version": lock["version"]}
    packages = {}
    for name in ("LLVM", "MLIR"):
        directory = root / "build/lib/cmake" / name.lower()
        config = directory / (name + "Config.cmake")
        if not config.is_file() or not under(config.resolve(), root / "build"):
            raise RuntimeError("missing/unsafe CMake package: " + str(config))
        packages[name + "_DIR"] = str(directory)
    return {"tools": tools, **packages}


def bootstrap(archive, root, lock_path=DEFAULT_LOCK, configure_only=False):
    require_native_host()
    lock = load_lock(lock_path)
    root = validate_root(root)
    initialize_root(root, lock)
    recorder = Recorder(root, lock, configure_only)
    try:
        source = prepare_source(archive.expanduser().resolve(), root, lock)
        recorder.summary["source"] = str(source)
        tools = check_system_tools(recorder)
        (root / "build").mkdir(exist_ok=True)
        recorder.run("configure", configure_command(root, source, tools))
        if not configure_only:
            recorder.run("build", build_command(root, tools))
            recorder.summary.update(verify_built_tools(root, lock, recorder))
        du = shutil.which("/usr/bin/du", path=SYSTEM_PATH)
        if du:
            output = recorder.run("disk-usage", ["/usr/bin/du", "-sk", "--", str(root)], read_output=True)
            recorder.summary["disk_usage_kib"] = int(output.split()[0])
        else:
            recorder.summary["disk_usage_kib"] = None
        recorder.finish("configured" if configure_only else "complete")
    except BaseException as error:
        recorder.finish("failed", error)
        raise
    return recorder.summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--configure-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = bootstrap(args.archive, args.root, args.lock, args.configure_only)
    except (Exception, KeyboardInterrupt) as error:
        print("bootstrap_llvm: {}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1
    print("{}; toolchain_complete={}; evidence={}".format(
        result["status"], result["toolchain_complete"], result["evidence_directory"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
