#!/usr/bin/env python3
"""Collect an M0 environment inventory without installing or initializing anything.

Only the explicitly requested, new JSON report is written. Commands are limited
to the allowlisted executables' --version calls; package metadata belongs to the
running interpreter, not necessarily the Python found on PATH. No package under
inspection is imported. Metadata absence does not prove that a module is absent.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


TOOLS = ("python", "ninja", "clang", "mlir-opt", "mlir-translate", "llc",
         "cmake", "gcc", "g++", "git")
COMMAND_TIMEOUT_SECONDS = 5
PACKAGE_DISTRIBUTIONS = {
    "numpy": ("numpy",),
    "cv2": ("opencv-python", "opencv-python-headless", "opencv-contrib-python",
            "opencv-contrib-python-headless", "opencv", "cv2"),
    "rknnlite": ("rknn-toolkit-lite2", "rknn-toolkit-lite", "rknnlite"),
}
LINUX_FILES = {
    "meminfo": Path("/proc/meminfo"),
    "os_release": Path("/etc/os-release"),
    "cpuinfo": Path("/proc/cpuinfo"),
}
CPU_FIELDS = ("model name", "Processor", "Hardware", "CPU architecture",
              "CPU implementer", "CPU part", "CPU variant", "CPU revision")
UNKNOWN = "unknown"


def output_text(value):
    """TimeoutExpired may carry bytes even when subprocess.run uses text=True."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if value is not None else ""


def probe_tool(name):
    """Probe one allowlisted PATH tool; failures never suppress other tools."""
    if name not in TOOLS:
        raise ValueError("Tool is not allowlisted: " + name)
    result = {"which": None, "command": None, "status": "missing",
              "returncode": None, "stdout": "", "stderr": "",
              "timeout_seconds": COMMAND_TIMEOUT_SECONDS}
    try:
        executable = shutil.which(name)
        result["which"] = executable
        if executable is None:
            result["reason"] = "Executable not found on PATH"
            return result
        result["command"] = [os.path.abspath(executable), "--version"]
        process = subprocess.run(
            result["command"], stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace", shell=False,
            check=False, timeout=COMMAND_TIMEOUT_SECONDS,
        )
        result.update(status="ok" if process.returncode == 0 else "failed",
                      returncode=process.returncode, stdout=process.stdout,
                      stderr=process.stderr)
    except subprocess.TimeoutExpired as error:
        result.update(status="timeout", stdout=output_text(error.stdout),
                      stderr=output_text(error.stderr),
                      reason="Version command exceeded its timeout")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result.update(status="error", reason=type(error).__name__ + ": " + str(error))
    return result


def collect_tools():
    return {name: probe_tool(name) for name in TOOLS}


def collect_packages():
    """Read distribution metadata only; never import numpy/cv2/rknnlite."""
    result = {}
    for module, candidates in PACKAGE_DISTRIBUTIONS.items():
        versions, errors = {}, {}
        for distribution in candidates:
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
            except Exception as error:
                # Broken metadata in one distribution must not hide the others.
                errors[distribution] = type(error).__name__ + ": " + str(error)
        unique = set(versions.values())
        status = "ok" if versions else "missing"
        if len(unique) > 1:
            status = "ambiguous"
        if errors:
            status = "partial" if versions else "error"
        result[module] = {
            "status": status,
            "version": next(iter(unique)) if len(unique) == 1 else UNKNOWN,
            "distributions": versions, "errors": errors,
            "checked_distributions": list(candidates),
            "source": "importlib.metadata (current Python)",
            "note": "Metadata only; does not establish module importability or active provider. "
                    "OS-managed packages without distribution metadata may be unreported.",
        }
    return result


def parse_meminfo(text):
    """Keep the kernel's units rather than assuming every value is in kB."""
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            values[key.strip()] = value.strip()
    return values


def parse_os_release(text):
    """Parse shell-style quoting without evaluating variables or commands."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip():
            values[key.strip()] = " ".join(shlex.split(value, comments=True, posix=True))
    return values


def parse_cpuinfo(text):
    """Select model/architecture fields; never retain raw cpuinfo or serials."""
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if separator and key in CPU_FIELDS and value:
            entries = values.setdefault(key, [])
            if value not in entries:
                entries.append(value)
    return values


def collect_linux_info(system):
    result = {"status": "ok" if system == "Linux" else "not_applicable"}
    parsers = {"meminfo": parse_meminfo, "os_release": parse_os_release,
               "cpuinfo": parse_cpuinfo}
    for name, path in LINUX_FILES.items():
        record = {"path": str(path), "status": UNKNOWN, "values": {}}
        if system != "Linux":
            record["reason"] = "Not a Linux host"
        else:
            try:
                record["values"] = parsers[name](path.read_text(encoding="utf-8", errors="replace"))
                record["status"] = "ok" if record["values"] else UNKNOWN
                if not record["values"]:
                    record["reason"] = "No recognized fields"
            except (OSError, ValueError) as error:
                record["reason"] = type(error).__name__ + ": " + str(error)
        result[name] = record
    if system == "Linux" and any(result[name]["status"] != "ok" for name in LINUX_FILES):
        result["status"] = "partial"
    return result


def collect_cpu_info(architecture, linux_info):
    count = {"status": UNKNOWN, "value": UNKNOWN}
    try:
        value = os.cpu_count()
        if value is not None:
            count.update(status="ok", value=value)
        else:
            count["reason"] = "os.cpu_count returned None"
    except (OSError, NotImplementedError) as error:
        count["reason"] = type(error).__name__ + ": " + str(error)

    affinity = {"status": UNKNOWN, "cpus": UNKNOWN, "count": UNKNOWN}
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        affinity["reason"] = "os.sched_getaffinity unavailable on this platform"
    else:
        try:
            cpus = sorted(get_affinity(0))
            affinity.update(status="ok", cpus=cpus, count=len(cpus))
        except (OSError, NotImplementedError) as error:
            affinity["reason"] = type(error).__name__ + ": " + str(error)

    fields = linux_info["cpuinfo"]["values"]
    models = list(dict.fromkeys(value for key in ("model name", "Processor", "Hardware")
                               for value in fields.get(key, [])))
    model = {"status": "ok" if models else UNKNOWN,
             "values": models if models else UNKNOWN, "source": "/proc/cpuinfo"}
    if not models:
        model["reason"] = "No model text available; ARM identifiers may be in linux.cpuinfo.values"
    return {"logical_count": count, "affinity": affinity, "model": model,
            "architecture": {"status": "ok" if architecture and architecture != UNKNOWN else UNKNOWN,
                             "value": architecture or UNKNOWN}}


def collect_disk_usage(project_root):
    result = {"path": str(project_root), "status": UNKNOWN, "unit": "bytes",
              "total": UNKNOWN, "used": UNKNOWN, "free": UNKNOWN}
    try:
        usage = shutil.disk_usage(project_root)
        result.update(status="ok", total=usage.total, used=usage.used, free=usage.free)
    except OSError as error:
        result["reason"] = type(error).__name__ + ": " + str(error)
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_platform_info():
    """Avoid platform.uname/processor: Windows implementations can spawn commands/WMI."""
    system = {"win32": "Windows", "linux": "Linux", "darwin": "Darwin"}.get(sys.platform, sys.platform)
    result = {"system": system,
              "release": UNKNOWN, "version": UNKNOWN, "architecture": UNKNOWN,
              "status": UNKNOWN}
    try:
        if sys.platform == "win32":
            # Read only these architecture hints, not the full environment.
            result["architecture"] = (os.environ.get("PROCESSOR_ARCHITEW6432") or
                                      os.environ.get("PROCESSOR_ARCHITECTURE") or UNKNOWN)
            result["architecture_source"] = "PROCESSOR_ARCHITEW6432/PROCESSOR_ARCHITECTURE"
            version = sys.getwindowsversion()
            native = version.platform_version or (version.major, version.minor, version.build)
            result.update(release=".".join(map(str, native[:2])),
                          version=".".join(map(str, native)), status="ok",
                          version_source="sys.getwindowsversion (OS kernel version)")
        else:
            native = os.uname()
            # Deliberately omit the hostname from uname.
            result.update(system=native.sysname, release=native.release, version=native.version,
                          architecture=native.machine or UNKNOWN, status="ok",
                          architecture_source="os.uname", version_source="os.uname")
    except (AttributeError, OSError, NotImplementedError) as error:
        result["reason"] = type(error).__name__ + ": " + str(error)
    return result


def collect_environment(project_root):
    """Return JSON-ready facts; unavailable capabilities are not a global failure."""
    platform_info = collect_platform_info()
    linux_info = collect_linux_info(platform_info["system"])
    architecture = platform_info["architecture"]
    return {
        "schema_version": 1,
        "purpose": "M0 read-only MLIR preprocessing environment inventory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "platform": platform_info,
        "current_python": {"executable": sys.executable, "version": sys.version,
                           "version_info": list(sys.version_info[:3]),
                           "implementation": sys.implementation.name},
        "path_python_note": "tools.python is PATH's default Python; current_python is the "
                            "running interpreter. packages describes only current_python.",
        "tools": collect_tools(), "packages": collect_packages(),
        "disk_usage": collect_disk_usage(project_root),
        "cpu": collect_cpu_info(architecture, linux_info), "linux": linux_info,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(__file__),
        "limitations": [
            "Version strings and package metadata do not verify compiler or RKNN usability.",
            "No hardware initialization, imports of inspected packages, build, installation, "
            "benchmark, environment dump, credentials, or SSH configuration collection.",
            "CPU model is unknown without Linux model fields; affinity is unknown when unsupported.",
            "PATH executables are trusted to honor --version; each call has a 5-second timeout.",
        ],
    }


def validate_report_path(path):
    path = Path(path).expanduser().absolute()
    if not path.parent.is_dir():
        raise FileNotFoundError("Report parent directory must already exist: " + str(path.parent))
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to overwrite existing report: " + str(path))
    return path


def write_report(path, report):
    """Exclusive creation also rejects a target created after the preflight check."""
    path = validate_report_path(path)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="New JSON report; parent must exist, existing targets are refused")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Existing project root (default: this script's repository root)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report_path = validate_report_path(args.report)
        project_root = args.project_root.expanduser().resolve(strict=True)
        if not project_root.is_dir():
            raise NotADirectoryError("Project root is not a directory: " + str(project_root))
        report = collect_environment(project_root)
        write_report(report_path, report)
    except (OSError, ValueError) as error:
        print("Environment report failed: " + str(error), file=sys.stderr)
        return 1
    print("Environment report written: " + str(report_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
