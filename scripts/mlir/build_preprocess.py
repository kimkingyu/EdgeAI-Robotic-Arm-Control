#!/usr/bin/env python3
"""Build four M2/M3 AOT variants from the SAME generated IR, retaining all stages.

Build completion is not numerical/performance acceptance. Native aarch64 only.
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

ROOT = Path(__file__).resolve().parents[2]
BASE = ["--edge-fuse-resize-color", "--edge-prepare-preprocess-destination"]
TILE = "--edge-tile-preprocess=rows={} columns={}"
VECTOR = "--edge-vectorize-preprocess=allow-packed-source-gather=true"
VARIANTS = {
    "unfused": [],
    "fused": ["--edge-fuse-resize-color"],
    "destination_only": ["--edge-prepare-preprocess-destination"],
    "fused_destination": BASE,
    # M4 schedules. Vectorization needs the iteration space to be tiled first.
    "tile_1x16": BASE + [TILE.format(1, 16)],
    "tile_4x32": BASE + [TILE.format(4, 32)],
    "tile_8x64": BASE + [TILE.format(8, 64)],
    "tile_1x128": BASE + [TILE.format(1, 128)],
    # Experimental only: the vectorizer emits gathers with densely packed
    # offsets, so these kernels require a source without row padding.
    "tile_1x16_vector_packed_source": BASE + [TILE.format(1, 16), VECTOR],
    "tile_4x32_vector_packed_source": BASE + [TILE.format(4, 32), VECTOR],
}
PACKED_SOURCE_ONLY = {name for name in VARIANTS if "packed_source" in name}
# Vectorization needs a tile strictly smaller than the output, otherwise tiling
# degenerates to the whole image and the iteration space stays too large.
TILE_SHAPE = {"tile_1x16_vector_packed_source": (1, 16),
              "tile_4x32_vector_packed_source": (4, 32)}
FUSED = {name for name, flags in VARIANTS.items() if "--edge-fuse-resize-color" in flags}
DESTINATION = {name for name, flags in VARIANTS.items()
               if "--edge-prepare-preprocess-destination" in flags}
BUFFERIZE = ["--empty-tensor-to-alloc-tensor", "--one-shot-bufferize", "--canonicalize",
             "--buffer-deallocation-pipeline", "--canonicalize"]
LOWER = ["--convert-linalg-to-loops", "--convert-vector-to-scf", "--lower-affine",
         "--arith-expand", "--canonicalize", "--convert-scf-to-cf", "--convert-cf-to-llvm",
         "--convert-vector-to-llvm", "--convert-bufferization-to-memref",
         # Tiling introduces memref.subview, which finalize-memref-to-llvm does
         # not lower on its own; expand strided metadata first.
         "--expand-strided-metadata", "--lower-affine", "--finalize-memref-to-llvm",
         "--convert-arith-to-llvm", "--convert-func-to-llvm", "--reconcile-unrealized-casts"]
UNLOWERED = re.compile(r"^\s+(?!llvm\.)[a-z_]+\.[a-z_.]+", re.MULTILINE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_shape(text):
    if not re.fullmatch(r"[0-9]+x[0-9]+", text):
        raise argparse.ArgumentTypeError("shape must be HxW")
    h, w = map(int, text.split("x"))
    if not 1 <= h <= 8192 or not 1 <= w <= 8192:
        raise argparse.ArgumentTypeError("dimensions must be 1..8192")
    return h, w


def counts(text):
    names = ("tensor.generate", "linalg.generic", "memref.alloc", "memref.dealloc",
             "memref.copy", "scf.for", "vector.load", "vector.store", "vector.transfer_read",
             "vector.transfer_write", "vector.gather", "vector.mask")
    return {name: len(re.findall(r"\b" + re.escape(name) + r"\b", text)) for name in names}


def build(args):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("requires native Linux aarch64")
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output directory must be new; parent must exist")
    args.output.mkdir()
    report = {"schema_version": 1, "status": "building", "abi_version": 1,
              "semantic_version": "linear_half_pixel_q11_v1", "numerical_validation": "pending",
              "target": "aarch64-unknown-linux-gnu", "created_utc": datetime.now(timezone.utc).isoformat(),
              "variants": [], "commands": [], "script_sha256": sha(Path(__file__))}
    path = args.output / "manifest.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run(label, command, stdout_path=None):
        command = [str(p) for p in command]
        log = args.output / (label + ".log")
        row = {"label": label, "argv": command, "status": "running", "log": str(log)}
        report["commands"].append(row)
        save()
        try:
            with log.open("wb") as err:
                if stdout_path is None:
                    p = subprocess.run(command, stdout=err, stderr=subprocess.STDOUT, timeout=600)
                else:
                    with stdout_path.open("wb") as out:
                        p = subprocess.run(command, stdout=out, stderr=err, timeout=600)
            row.update(returncode=p.returncode, status="passed" if p.returncode == 0 else "failed")
            if p.returncode:
                raise RuntimeError("{} failed; see {}".format(label, log))
        except Exception as exc:
            row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            save()
        return log.read_text(errors="replace")

    try:
        opt = args.compiler_build / "bin/edge-opt"
        generator = args.compiler_build / "bin/edge-preprocess-gen"
        translate, llc = (args.toolchain / "bin" / name for name in ("mlir-translate", "llc"))
        for tool in (opt, translate, llc):
            version = run("version-" + tool.name, [tool, "--version"])
            if not re.search(r"\bLLVM\s+version\s+20\.1\.8(?![\w.+-])", version):
                raise RuntimeError("tool version differs from lock: " + str(tool))
        report["compiler_sha256"] = {str(p): sha(p) for p in (opt, generator, translate, llc)}
        runtime = ROOT / "compiler/runtime/src/preprocess_runtime.cpp"
        header = ROOT / "compiler/runtime/include/edgeai_preprocess.h"
        report["runtime_source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in (runtime, header)}
        for h, w in args.shapes:
            profile = args.output / ("{}x{}".format(h, w))
            profile.mkdir()
            source = profile / "01-input.mlir"
            run(profile.name + "-generate", [generator, "--height", h, "--width", w], source)
            for name, flags in VARIANTS.items():
                tile = TILE_SHAPE.get(name)
                if tile and (h <= tile[0] or w <= tile[1]):
                    report["variants"].append({"profile": [h, w], "variant": name,
                        "status": "not_applicable",
                        "reason": "output {}x{} is not larger than tile {}x{}".format(h, w, *tile)})
                    print("skipped {} {}: output not larger than the tile".format(profile.name, name), flush=True)
                    continue
                folder = profile / name
                folder.mkdir()
                prefix = profile.name + "-" + name + "-"
                transformed, buffered, lowered, ir, obj, assembly, library = [folder / filename for filename in
                    ("02-transformed.mlir", "03-buffered.mlir", "04-lowered.mlir", "05-kernel.ll",
                     "06-kernel.o", "06-kernel.s", "libedgeai_preprocess.so")]
                # No canonicalize before our conservative structural matcher.
                run(prefix + "transform", [opt, source, *flags, "-o", transformed])
                text = transformed.read_text()
                tc = counts(text)
                # Vectorization replaces the tagged linalg.generic with vector
                # ops, so check the fusion marker before that stage instead.
                vectorized = "vector" in name
                if name in FUSED and not vectorized and "edge.fused" not in text:
                    raise ValueError("fusion did not actually modify this profile")
                if name in FUSED and vectorized:
                    stage = folder / "02a-prevector.mlir"
                    run(prefix + "prevector", [opt, source, *flags[:-1], "-o", stage])
                    if "edge.fused" not in stage.read_text():
                        raise ValueError("fusion did not run before vectorization")
                # Without fusion the resize stays a tensor.generate producer;
                # only the fused destination form must consume it entirely.
                if name in DESTINATION and name in FUSED and tc["tensor.generate"] != 0:
                    raise ValueError("fused destination conversion left tensor.generate")
                if vectorized and tc["linalg.generic"] != 0:
                    raise ValueError("vectorization left a scalar linalg.generic")
                if "vector" in name and not any(tc[k] for k in tc if k.startswith("vector.")):
                    raise ValueError("vectorization produced no vector operations")
                if "tile" in name and tc["scf.for"] == 0 and h * w > 1:
                    raise ValueError("tiling produced no loop nest")
                run(prefix + "bufferize", [opt, transformed, *BUFFERIZE, "-o", buffered])
                bc = counts(buffered.read_text())
                allocating = name in DESTINATION and name in FUSED
                if allocating and any(bc[x] for x in ("memref.alloc", "memref.copy")):
                    raise ValueError("fused destination variant still allocates/copies an image")
                run(prefix + "lower", [opt, buffered, *LOWER, "-o", lowered])
                leftovers = sorted({match.strip() for match in UNLOWERED.findall(lowered.read_text())})
                if leftovers:
                    raise ValueError("unlowered operations remain: " + ", ".join(leftovers[:8]))
                run(prefix + "translate", [translate, "--mlir-to-llvmir", lowered, "-o", ir])
                target = ["-mtriple=aarch64-unknown-linux-gnu", "-mcpu=generic", "-relocation-model=pic", "-O2"]
                run(prefix + "object", [llc, *target, "-filetype=obj", ir, "-o", obj])
                run(prefix + "assembly", [llc, *target, "-filetype=asm", ir, "-o", assembly])
                run(prefix + "link", ["/usr/bin/g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-fPIC", "-shared", "-fvisibility=hidden", "-I" + str(header.parent),
                    "-DEDGEAI_OUT_HEIGHT=" + str(h), "-DEDGEAI_OUT_WIDTH=" + str(w),
                    '-DEDGEAI_VARIANT="' + name + '"', "-DEDGEAI_ALLOCATION_AUDIT=1", runtime, obj,
                    "-Wl,-z,defs", "-Wl,-Bsymbolic", "-Wl,--wrap=malloc,--wrap=aligned_alloc,--wrap=free", "-o", library])
                dynamic = run(prefix + "dependencies", ["/usr/bin/readelf", "-d", library])
                needed = re.findall(r"Shared library: \[([^\]]+)\]", dynamic)
                if any("llvm" in dep.lower() or "mlir" in dep.lower() for dep in needed):
                    raise ValueError("unexpected compiler runtime dependency")
                report["variants"].append({"profile": [h, w], "variant": name, "status": "built",
                    "requires_packed_source": name in PACKED_SOURCE_ONLY,
                    "library": str(library.relative_to(args.output)), "source_sha256": sha(source),
                    "transformed_counts": tc, "bufferized_counts": bc, "runtime_needed": needed,
                    "artifacts": {str(p.relative_to(args.output)): {"sha256": sha(p), "bytes": p.stat().st_size}
                                  for p in (source, transformed, buffered, lowered, ir, obj, assembly, library)}})
                save()
                print("built {} {}; alloc/copy IR: {}/{} (numerical checks pending)".format(
                    profile.name, name, bc["memref.alloc"], bc["memref.copy"]), flush=True)
        report["status"] = "built"
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", required=True, type=Path)
    parser.add_argument("--compiler-build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shapes", nargs="+", type=parse_shape,
                        default=[(640, 640), (448, 448), (5, 7), (1, 1), (1, 9), (9, 1)])
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    try:
        build(args)
    except Exception as exc:
        print("build_preprocess: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
