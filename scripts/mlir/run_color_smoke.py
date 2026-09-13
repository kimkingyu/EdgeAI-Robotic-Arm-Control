#!/usr/bin/env python3
"""Execute the real M1 MLIR -> LLVM IR -> AArch64 object -> C/Python ABI chain.

This is a color-only smoke test, not the production resize preprocessor. No
camera, RKNN or controller imports. Every intermediate and command is retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
LOWERING = ["--convert-scf-to-cf", "--convert-cf-to-llvm", "--finalize-memref-to-llvm",
            "--convert-arith-to-llvm", "--convert-func-to-llvm", "--reconcile-unrealized-casts"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_smoke(toolchain, edge_opt, output):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("real M1 execution requires the native aarch64 board")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output must be a new directory with an existing parent")
    tools = {name: toolchain / "bin" / name for name in ("mlir-opt", "mlir-translate", "llc")}
    tools["edge-opt"] = edge_opt
    for name, path in tools.items():
        if not path.is_file():
            raise FileNotFoundError("missing locked tool {}: {}".format(name, path))
    source = WORKSPACE / "compiler/tests/color_smoke.mlir"
    wrapper = WORKSPACE / "compiler/tests/color_smoke_abi.cpp"
    tests = WORKSPACE / "compiler/tests/color_smoke_checks.py"
    output.mkdir()
    report = {"schema_version": 1, "stage": "M1 color-only AOT smoke", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "commands": [],
              "source_sha256": {str(path.relative_to(WORKSPACE)): sha256(path) for path in (source, wrapper, tests)},
              "production_resize_implemented": False}
    report_path = output / "smoke.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def command(label, args):
        args = [str(arg) for arg in args]
        log = output / (label + ".log")
        row = {"label": label, "argv": args, "log": str(log), "status": "running"}
        report["commands"].append(row)
        save()
        with log.open("wb") as handle:
            result = subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT, timeout=600)
        row.update(returncode=result.returncode, status="passed" if result.returncode == 0 else "failed")
        save()
        if result.returncode:
            raise RuntimeError("{} failed with {}; see {}".format(label, result.returncode, log))
        return log.read_text(encoding="utf-8", errors="replace")

    try:
        for name, path in tools.items():
            version = command("version-" + name, [path, "--version"])
            if not re.search(r"\bLLVM\s+version\s+20\.1\.8(?![\w.+-])", version):
                raise RuntimeError("tool does not match LLVM 20.1.8: " + name)
        initial, lowered, llvm_ir = output / "01-color.mlir", output / "02-llvm.mlir", output / "03-color.ll"
        initial.write_bytes(source.read_bytes())
        command("lower", [edge_opt, initial, *LOWERING, "-o", lowered])
        command("translate", [tools["mlir-translate"], "--mlir-to-llvmir", lowered, "-o", llvm_ir])
        machine_object, assembly = output / "04-color.o", output / "04-color.s"
        target = ["-mtriple=aarch64-unknown-linux-gnu", "-mcpu=generic", "-relocation-model=pic", "-O2"]
        command("object", [tools["llc"], *target, "-filetype=obj", llvm_ir, "-o", machine_object])
        command("assembly", [tools["llc"], *target, "-filetype=asm", llvm_ir, "-o", assembly])
        library, executable = output / "libedge_color_smoke.so", output / "color_smoke"
        cxx = ["/usr/bin/g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"]
        command("link-library", [*cxx, "-fPIC", "-shared", wrapper, machine_object, "-Wl,-z,defs", "-o", library])
        command("link-cpp-test", [*cxx, "-DEDGEAI_COLOR_SMOKE_MAIN", wrapper, machine_object, "-o", executable])
        command("cpp-test", [executable])
        command("python-test", [sys.executable, "-B", tests, "--library", library])
        dynamic = command("dynamic-dependencies", ["/usr/bin/readelf", "-d", library])
        if re.search(r"NEEDED.*(?:MLIR|LLVM)", dynamic, flags=re.IGNORECASE):
            raise RuntimeError("runtime smoke library unexpectedly depends on LLVM/MLIR")
        report["artifacts"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                               for p in (initial, lowered, llvm_ir, machine_object, assembly, library, executable)}
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error="{}: {}".format(type(error).__name__, error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", required=True, type=Path, help="LLVM build directory, containing bin/")
    parser.add_argument("--edge-opt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_smoke(args.toolchain.resolve(), args.edge_opt.resolve(), args.output.resolve())
    except Exception as error:
        print("run_color_smoke: " + str(error), file=sys.stderr)
        return 1
    print("{}: C++ and Python exercised the real generated AArch64 color kernel".format(report["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
