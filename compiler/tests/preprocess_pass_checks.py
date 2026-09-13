#!/usr/bin/env python3
"""Run real edge-opt and FileCheck on positive and negative pass fixtures.

Negative cases must leave the parsed IR unchanged: rejecting an unsupported
pattern is the required behavior, not a missing feature to work around.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

FUSE = "--edge-fuse-resize-color"
DESTINATION = "--edge-prepare-preprocess-destination"
BUFFERIZE = ["--empty-tensor-to-alloc-tensor", "--one-shot-bufferize", "--canonicalize",
             "--buffer-deallocation-pipeline", "--canonicalize"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit(text, replacements):
    for old, new in replacements:
        if old not in text:
            raise ValueError("fixture edit did not apply: " + old[:70])
        text = text.replace(old, new, 1)
    return text


def negatives(seed):
    """Each edit keeps valid IR but breaks one required legality condition.

    Cases are grouped per pass: an extra resize reader forbids fusion (it would
    interpolate twice) while still permitting destination rewriting, so it is
    not evidence of a destination-pass defect.
    """
    color = 'attrs =  {edge.color = "bgr_to_rgb"}'
    anchor = "bufferization.materialize_in_destination %4 in"
    extract = "    %{} = tensor.extract %{}[%c0, %c0, %c0] : tensor<5x7x3xi8>\n"
    fusion_only = {
        "multiple_resize_consumers": [(anchor, extract.format("resize_reader", "generated") + "    " + anchor)],
    }
    destination_only = {
        # An extra reader of the color result blocks writing into the caller
        # buffer, but the resize/color pair itself is still fusable.
        "multiple_color_consumers": [(anchor, extract.format("color_reader", "4") + "    " + anchor)],
        "side_effect_between_producer_and_anchor": [
            ("bufferization.materialize_in_destination",
             "memref.store %cstore, %arg1[%c0, %c0, %c0] : memref<5x7x3xi8>\n"
             "    bufferization.materialize_in_destination"),
            ("%0 = bufferization.to_tensor", "%cstore = arith.constant 7 : i8\n    %0 = bufferization.to_tensor")],
        "destination_without_restrict": [("materialize_in_destination %4 in restrict writable",
                                          "materialize_in_destination %4 in writable")],
    }
    cases = {
        "channel_map_identity": [("affine_map<(d0, d1, d2) -> (d0, d1, -d2 + 2)>",
                                  "affine_map<(d0, d1, d2) -> (d0, d1, d2)>")],
        "normalizing_body": [("linalg.yield %in : i8",
                              "%scaled = arith.shrui %in, %cshift : i8\n      linalg.yield %scaled : i8"),
                             ("^bb0(%in: i8, %out: i8):",
                              "^bb0(%in: i8, %out: i8):\n      %cshift = arith.constant 1 : i8")],
        "yield_destination": [("linalg.yield %in : i8", "linalg.yield %out : i8")],
        "reduction_iterator": [('iterator_types = ["parallel", "parallel", "parallel"]} ins(%generated',
                                'iterator_types = ["parallel", "parallel", "reduction"]} ins(%generated')],
        # Single-tap sampling scaled to Q22: valid IR, but not this bilinear rule.
        "nearest_neighbor_resize": [
            ("%c2097152_i32 = arith.constant 2097152 : i32",
             "%c4194304_i32 = arith.constant 4194304 : i32\n"
             "      %nn = arith.muli %49, %c4194304_i32 : i32\n"
             "      %c2097152_i32 = arith.constant 2097152 : i32"),
            ("%62 = arith.addi %61, %c2097152_i32 : i32", "%62 = arith.addi %nn, %c2097152_i32 : i32")],
        # Rounding the first axis early changes results; one final rounding only.
        "double_rounding": [
            ("%59 = arith.muli %55, %25 : i32",
             "%cearly = arith.constant 11 : i32\n"
             "      %early = arith.shrui %55, %cearly : i32\n"
             "      %59 = arith.muli %early, %25 : i32")],
        "truncating_division": [("%10 = arith.floordivsi %9, %c10_i64 : i64", "%10 = arith.divsi %9, %c10_i64 : i64")],
        "signed_pixel_extension": [("%49 = arith.extui %extracted : i8 to i32", "%49 = arith.extsi %extracted : i8 to i32")],
        "wrong_semantic_version": [('{edge.resize = "linear_half_pixel_q11_v1"}', '{edge.resize = "area_v2"}')],
        "missing_color_attribute": [(color, 'attrs =  {edge.color = "rgb_to_bgr"}')],
        "unknown_color_attribute": [(color, color[:-1] + ', edge.normalize = "mean_std"}')],
        "writable_input_tensor": [("%0 = bufferization.to_tensor %arg0 restrict :",
                                   "%0 = bufferization.to_tensor %arg0 restrict writable :")],
    }
    groups = {"both": cases, "fuse": {**cases, **fusion_only},
              "destination": {**cases, **destination_only}}
    return {flags: {name: edit(seed, changes) for name, changes in group.items()}
            for flags, group in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-opt", required=True, type=Path)
    parser.add_argument("--filecheck", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--checks", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists() or not args.report.parent.is_dir():
        print("report must be new with an existing parent", file=sys.stderr)
        return 1
    result = {"schema_version": 1, "status": "running", "scope": "M3 pass legality; not performance",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "seed_sha256": sha(args.seed), "checks_sha256": sha(args.checks),
              "script_sha256": sha(Path(__file__)), "positive": [], "negative": []}
    seed = args.seed.read_text(encoding="utf-8")

    def opt(text, flags, directory, name):
        source = Path(directory) / (name + ".mlir")
        source.write_text(text, encoding="utf-8")
        process = subprocess.run([str(args.edge_opt), str(source), *flags],
                                 capture_output=True, text=True, timeout=300)
        return process

    try:
        with tempfile.TemporaryDirectory(prefix="edgeai-pass-") as directory:
            for name, flags, prefix in (("fuse", [FUSE], "FUSE"),
                                        ("fused_destination", [FUSE, DESTINATION], "DPS"),
                                        ("destination_only", [DESTINATION], "DEST")):
                process = opt(seed, flags, directory, name)
                if process.returncode:
                    raise RuntimeError("edge-opt failed for " + name + ": " + process.stderr[-800:])
                check = subprocess.run([str(args.filecheck), str(args.checks),
                                        "--check-prefix=" + prefix, "--input-file=-"],
                                       input=process.stdout, capture_output=True, text=True, timeout=300)
                entry = {"case": name, "flags": flags, "prefix": prefix,
                         "filecheck_returncode": check.returncode, "filecheck_stderr": check.stderr[-800:]}
                result["positive"].append(entry)
                if check.returncode:
                    raise RuntimeError("FileCheck failed for " + name + ": " + check.stderr[-800:])
                if name == "fused_destination":
                    buffered = opt(process.stdout, BUFFERIZE, directory, "buffered")
                    if buffered.returncode:
                        raise RuntimeError("bufferization failed: " + buffered.stderr[-800:])
                    memory = subprocess.run([str(args.filecheck), str(args.checks),
                                             "--check-prefix=BUFFER", "--input-file=-"],
                                            input=buffered.stdout, capture_output=True, text=True, timeout=300)
                    entry["bufferized_filecheck_returncode"] = memory.returncode
                    if memory.returncode:
                        raise RuntimeError("bufferized IR still allocates/copies: " + memory.stderr[-800:])

            baseline = opt(seed, [], directory, "parse-only")
            if baseline.returncode:
                raise RuntimeError("seed IR does not parse")
            groups = negatives(seed)
            for label, flags in (("fuse", [FUSE]), ("destination", [DESTINATION]),
                                 ("both", [FUSE, DESTINATION])):
                for name, text in groups[label].items():
                    reference = opt(text, [], directory, name + "-parse-" + label)
                    if reference.returncode:
                        raise RuntimeError("negative fixture is not valid IR: " + name + " " + reference.stderr[-400:])
                    process = opt(text, flags, directory, name + "-" + label)
                    if process.returncode:
                        raise RuntimeError("edge-opt failed on rejected fixture: " + name)
                    changed = process.stdout != reference.stdout
                    result["negative"].append({"case": name, "flags": flags, "ir_changed": changed})
                    if changed:
                        raise RuntimeError("pass transformed an unsupported pattern: " + name + " " + label)
        result["status"] = "passed"
        print("pass legality: {} positive, {} rejected-unchanged".format(
            len(result["positive"]), len(result["negative"])))
    except Exception as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
        print("preprocess_pass_checks: " + str(error), file=sys.stderr)
        return 1
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
