#!/usr/bin/env python3
"""Real generated-kernel validation. Missing libraries/dependencies are failures.

Synthetic patterns are correctness fixtures, not performance or model evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import mmap
from pathlib import Path
import subprocess
import sys

import numpy as np
import cv2
from preprocess_reference import reference_resize_rgb, difference_statistics


class Spec(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("reserved", C.c_uint32),
                ("input_height", C.c_uint64), ("input_width", C.c_uint64),
                ("output_height", C.c_uint64), ("output_width", C.c_uint64)]


class Stats(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ("allocation_calls", "free_calls", "requested_bytes", "outstanding_allocations")]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Kernel:
    def __init__(self, library, height, width, variant):
        self.lib = C.CDLL(str(library.resolve(strict=True)))
        self.height, self.width, self.variant = height, width, variant
        for name in ("abi_version", "allocation_audit_enabled"):
            fn = getattr(self.lib, "edge_preprocess_" + name)
            fn.argtypes, fn.restype = [], C.c_uint32
        for name in ("semantic_version", "variant"):
            fn = getattr(self.lib, "edge_preprocess_" + name)
            fn.argtypes, fn.restype = [], C.c_char_p
        self.create = self.lib.edge_preprocess_create
        self.create.argtypes, self.create.restype = [C.POINTER(Spec), C.POINTER(C.c_void_p)], C.c_int
        self.run = self.lib.edge_preprocess_run
        self.run.argtypes, self.run.restype = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_uint64,
                                             C.c_void_p, C.c_uint64, C.c_uint64], C.c_int
        self.destroy = self.lib.edge_preprocess_destroy
        self.destroy.argtypes, self.destroy.restype = [C.c_void_p], None
        self.stats = self.lib.edge_preprocess_last_run_stats
        self.stats.argtypes, self.stats.restype = [C.c_void_p, C.POINTER(Stats)], C.c_int
        require(self.lib.edge_preprocess_abi_version() == 1, "ABI version mismatch")
        require(self.lib.edge_preprocess_semantic_version() == b"linear_half_pixel_q11_v1", "semantic mismatch")
        require(self.lib.edge_preprocess_variant().decode() == variant, "variant mismatch")
        require(self.lib.edge_preprocess_allocation_audit_enabled() == 1, "allocation audit must be active")

    def context(self, h, w):
        spec = Spec(1, 0, h, w, self.height, self.width)
        ctx = C.c_void_p()
        require(self.create(C.byref(spec), C.byref(ctx)) == 0 and ctx.value, "create failed")
        return ctx

    def get_stats(self, ctx):
        stats = Stats()
        require(self.stats(ctx, C.byref(stats)) == 0, "stats call failed")
        result = {name: getattr(stats, name) for name, _ in stats._fields_}
        # Every schedule derived from the fused destination form must keep the
        # zero-allocation property; an unknown variant is an error, not a skip.
        baseline = {"unfused": 2, "fused": 1, "destination_only": 1, "fused_destination": 0}
        if self.variant in baseline:
            expected = baseline[self.variant]
        elif self.variant.startswith("tile_"):
            expected = 0
        else:
            raise AssertionError("no allocation expectation for variant " + self.variant)
        require(result["allocation_calls"] == expected, "unexpected measured allocations: " + str(result))
        require(result["free_calls"] == expected and result["outstanding_allocations"] == 0,
                "unbalanced generated-code allocations: " + str(result))
        return result

    def check(self, source, expected, repeats=1, padding=5):
        h, w = source.shape[:2]
        src_stride = w * 3 + padding
        src_span = (h - 1) * src_stride + w * 3
        guard = 32
        backing = np.full(src_span + guard * 2, 0xA5, dtype=np.uint8)
        view = np.ndarray((h, w, 3), dtype=np.uint8, buffer=backing, offset=guard,
                          strides=(src_stride, 3, 1))
        view[:] = source
        before = backing.copy()
        dst_size = self.height * self.width * 3
        output = np.full(dst_size + guard * 2, 0xCD, dtype=np.uint8)
        rgb = output[guard:-guard].reshape(self.height, self.width, 3)
        ctx = self.context(h, w)
        try:
            for _ in range(repeats):
                rgb.fill(0xCD)
                status = self.run(ctx, view.ctypes.data, src_span, src_stride,
                                  rgb.ctypes.data, dst_size, self.width * 3)
                require(status == 0, "real generated kernel returned " + str(status))
                require(np.array_equal(rgb, expected), "MLIR output differs from independent Q11 reference: " +
                        str(difference_statistics(rgb, expected)))
                require(np.array_equal(backing, before), "input or input padding/guards modified")
                require(np.all(output[:guard] == 0xCD) and np.all(output[-guard:] == 0xCD), "output guards changed")
                stats = self.get_stats(ctx)
            return {"repeats": repeats, "input_shape": [h, w, 3], "input_row_stride": src_stride,
                    "sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "allocation_stats": stats}
        finally:
            self.destroy(ctx)

    def invalid_arguments(self):
        source = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
        output = np.full((self.height, self.width, 3), 0xCC, dtype=np.uint8)
        ctx = self.context(4, 7)
        valid = [ctx, source.ctypes.data, source.nbytes, 21, output.ctypes.data, output.nbytes, self.width * 3]
        cases = [({0: None}, 1), ({1: None}, 1), ({4: None}, 1), ({3: 0}, 3), ({3: 20}, 3),
                 ({3: 2**64 - 1}, 3), ({6: self.width * 3 + 1}, 3), ({2: 83}, 4),
                 ({5: output.nbytes - 1}, 4), ({3: 2**63 - 1}, 4),
                 ({1: 2**64 - 16}, 4), ({4: 2**64 - 1}, 4),
                 ({4: source.ctypes.data}, 5), ({4: source.ctypes.data + 1}, 5)]
        before, original = output.copy(), source.copy()
        try:
            for changes, expected_status in cases:
                args = valid.copy()
                for pos, value in changes.items():
                    args[pos] = value
                require(self.run(*args) == expected_status, "wrong rejection for " + str(changes))
                require(np.array_equal(output, before) and np.array_equal(source, original), "invalid call wrote memory")
            stats = Stats()
            require(self.stats(ctx, C.byref(stats)) == 0 and bytes(stats) == bytes(Stats()), "invalid run changed stats")
        finally:
            self.destroy(ctx)
        spec = Spec(1, 0, 4, 7, self.height, self.width)
        for field, value, expected in [("abi_version", 2, 1), ("reserved", 1, 1),
            ("input_height", 0, 2), ("input_width", 0, 2), ("input_height", 8193, 2),
            ("input_width", 8193, 2), ("output_height", self.height + 1, 2),
            ("output_width", self.width + 1, 2)]:
            bad = Spec.from_buffer_copy(bytes(spec))
            setattr(bad, field, value)
            result = C.c_void_p(123)
            require(self.create(C.byref(bad), C.byref(result)) == expected and not result.value,
                    "invalid spec accepted: " + field)
        self.destroy(None)
        return len(cases) + 8


def fixtures(images_dir):
    rng = np.random.default_rng(20260913)
    values = []
    for h, w in ((1, 1), (1, 17), (19, 1), (3, 5), (5, 7), (13, 11), (1, 8192), (8192, 1)):
        values.append(("random_{}x{}".format(h, w), rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "synthetic"))
    for name, color in (("black", [0, 0, 0]), ("white", [255, 255, 255]), ("channel_marker", [3, 17, 251])):
        image = np.empty((9, 13, 3), dtype=np.uint8)
        image[:] = color
        values.append((name, image, "synthetic"))
    y, x = np.indices((17, 19))
    checker = np.repeat((((y + x) % 2) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    gradient = np.stack(((x * 11 + y * 7) % 256, (x * 3 + y * 17) % 256, (x * 13 + y * 5) % 256), axis=2).astype(np.uint8)
    impulse = np.zeros((17, 19, 3), dtype=np.uint8)
    impulse[8, 9] = [255, 127, 31]
    values.extend((name, image, "synthetic") for name, image in (("checker", checker), ("gradient", gradient), ("impulse", impulse)))
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:16]
    require(len(files) == 16, "16 real images required; no synthetic substitution")
    provenance = []
    for path in files:
        encoded = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(image is not None, "cannot decode " + str(path))
        provenance.append({"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(), "shape": list(image.shape)})
        values.append((path.name, image, "real_decoded"))
    original = values[-16][1]
    for h, w in ((480, 640), (426, 640), (720, 1280), (1080, 1920), (479, 641)):
        values.append(("derived_{}x{}".format(h, w), cv2.resize(original, (w, h), interpolation=cv2.INTER_LINEAR), "real_derived"))
    return values, provenance


def guard_child(library, h, w, variant):
    """Run in a child: PROT_NONE adjacent pages catch read/write overruns.

    Input middle pages become read-only before invocation, catching input writes.
    Right- and left-aligned buffers tested separately; all touched bytes are real.
    """
    kernel = Kernel(library, h, w, variant)
    libc = C.CDLL(None, use_errno=True)
    libc.mprotect.argtypes, libc.mprotect.restype = [C.c_void_p, C.c_size_t, C.c_int], C.c_int
    page = mmap.PAGESIZE
    for ih, iw in ((1, 1), (1, 17), (19, 1), (5, 7)):
        image = np.arange(ih * iw * 3, dtype=np.uint8).reshape(ih, iw, 3)
        expected = reference_resize_rgb(image, h, w).tobytes()
        for at_end in (True, False):
            areas = []
            try:
                for size in (image.nbytes, len(expected)):
                    middle = (size + page - 1) // page * page
                    region = mmap.mmap(-1, middle + 2 * page, prot=mmap.PROT_READ | mmap.PROT_WRITE)
                    base = C.addressof(C.c_char.from_buffer(region))
                    require(libc.mprotect(base, page, 0) == 0, "mprotect prefix failed")
                    require(libc.mprotect(base + page + middle, page, 0) == 0, "mprotect suffix failed")
                    pointer = base + page + (middle - size if at_end else 0)
                    areas.append((region, base, middle, pointer))
                src, dst = areas
                C.memmove(src[3], image.ctypes.data, image.nbytes)
                require(libc.mprotect(src[1] + page, src[2], mmap.PROT_READ) == 0, "read-only input failed")
                ctx = kernel.context(ih, iw)
                try:
                    require(kernel.run(ctx, src[3], image.nbytes, iw * 3, dst[3], len(expected), w * 3) == 0, "guard call failed")
                    require(C.string_at(dst[3], len(expected)) == expected, "guard pixels mismatch")
                    kernel.get_stats(ctx)
                finally:
                    kernel.destroy(ctx)
            finally:
                for region, base, middle, _ in areas:
                    libc.mprotect(base, middle + 2 * page, mmap.PROT_READ | mmap.PROT_WRITE)
                    region.close()
    print("guard pages and read-only input passed: " + variant)


def validate(args):
    require(not args.report.exists() and args.report.parent.is_dir(), "report must be new with existing parent")
    manifest = json.loads(args.manifest.read_text())
    require(manifest["status"] == "built", "build manifest is not complete")
    base = args.manifest.parent
    data, provenance = fixtures(args.images_dir)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    result = {"schema_version": 1, "status": "running", "scope": "M2/M3 correctness only; no performance/model accuracy claim",
              "created_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha(args.manifest),
              "reference_sha256": sha(Path(__file__).with_name("preprocess_reference.py")),
              "checker_sha256": sha(Path(__file__)), "python": sys.version,
              "numpy": np.__version__, "opencv": cv2.__version__, "real_images": provenance,
              "reference_vs_opencv_max_abs_error_limit": 1, "variants": [], "comparisons": []}
    refs = {}
    try:
        for item in manifest["variants"]:
            if item.get("status") == "not_applicable":
                result["variants"].append({**item, "cases": []})
                print("skipped {}: {}".format(item["variant"], item["reason"]), flush=True)
                continue
            h, w = item["profile"]
            name = item["variant"]
            library = base / item["library"]
            for path, evidence in item["artifacts"].items():
                require(sha(base / path) == evidence["sha256"], "artifact hash mismatch: " + path)
            kernel = Kernel(library, h, w, name)
            # Packed-source kernels are only valid without row padding; that
            # restriction is recorded, not hidden by copying the input.
            packed_only = bool(item.get("requires_packed_source"))
            row_padding = 0 if packed_only else 5
            record = {"profile": [h, w], "variant": name, "cases": [], "status": "running",
                      "requires_packed_source": packed_only, "row_padding": row_padding}
            result["variants"].append(record)
            # Same-size test is tied to each generated output profile.
            same = np.random.default_rng(h * 8192 + w).integers(0, 256, (h, w, 3), dtype=np.uint8)
            for label, image, origin in data + [("same_size", same, "synthetic")]:
                key = (h, w, label)
                if key not in refs:
                    reference = reference_resize_rgb(image, h, w)
                    opencv = cv2.cvtColor(cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                    diff = difference_statistics(reference, opencv, max_worst=3)
                    result["comparisons"].append({"profile": [h, w], "input": label, "origin": origin, **diff})
                    require(diff["max_abs_error"] <= 1, "fixed OpenCV tolerance exceeded: " + str(diff))
                    refs[key] = reference
                check = kernel.check(image, refs[key], repeats=3,
                                     padding=0 if label == "same_size" else row_padding)
                record["cases"].append({"input": label, "origin": origin, **check})
            record["invalid_argument_cases"] = kernel.invalid_arguments()
            if packed_only:
                # Document the measured failure mode instead of only asserting it.
                padded = kernel.check.__self__
                image = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
                try:
                    padded.check(image, reference_resize_rgb(image, h, w), padding=4)
                    record["row_padded_behavior"] = "unexpectedly matched"
                except AssertionError:
                    record["row_padded_behavior"] = "rejected: row-padded input produces wrong pixels"
                require(record["row_padded_behavior"].startswith("rejected"),
                        "packed-source kernel unexpectedly handled row padding")
            tiny = np.arange(15 * 3, dtype=np.uint8).reshape(3, 5, 3)
            reference = reference_resize_rgb(tiny, h, w)
            record["repeated_buffer_reuse"] = kernel.check(tiny, reference,
                                                           repeats=100 if h * w < 100 else 10,
                                                           padding=row_padding)
            with ThreadPoolExecutor(max_workers=4) as pool:
                record["independent_context_workers"] = list(pool.map(
                    lambda _: kernel.check(tiny, reference, repeats=5, padding=row_padding), range(4)))
            child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--guard-library", str(library.resolve()),
                                    "--guard-shape", str(h), str(w), "--guard-variant", name], capture_output=True, text=True, timeout=120)
            record["guard_process"] = {"returncode": child.returncode, "stdout": child.stdout, "stderr": child.stderr}
            require(child.returncode == 0, "guard-page child failed: " + str(record["guard_process"]))
            record["status"] = "passed"
            print("passed {}x{} {}: {} images, real allocation audit and guard pages".format(h, w, name, len(record["cases"])), flush=True)
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--guard-library", type=Path)
    parser.add_argument("--guard-shape", type=int, nargs=2)
    parser.add_argument("--guard-variant")
    args = parser.parse_args(argv)
    try:
        if args.guard_library:
            guard_child(args.guard_library, *args.guard_shape, args.guard_variant)
        else:
            if not all((args.manifest, args.images_dir, args.report)):
                parser.error("--manifest, --images-dir and --report are required")
            validate(args)
    except Exception as error:
        print("preprocess_aot_checks: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
