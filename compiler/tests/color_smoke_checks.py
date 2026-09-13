#!/usr/bin/env python3
"""Real generated-library M1 tests. Missing/unloadable library is a failure, not skip."""
import argparse
import ctypes as C
from pathlib import Path
import unittest

RUN = None
GUARD = 16


def storage(size, value):
    array = (C.c_uint8 * (size + 2 * GUARD))()
    C.memset(array, value, len(array))
    return array, C.addressof(array) + GUARD


class ColorSmokeTests(unittest.TestCase):
    def test_exact_pixels_strides_guards_and_repeat(self):
        for height, width in ((1, 1), (1, 7), (5, 1), (4, 7), (8, 8)):
            for padding in (0, 5):
                src_stride, dst_stride = width * 3 + padding, width * 3 + padding + 2
                src, src_ptr = storage(height * src_stride, 0xA5)
                dst, dst_ptr = storage(height * dst_stride, 0xCC)
                for y in range(height):
                    for x in range(width):
                        for channel in range(3):
                            src[GUARD + y * src_stride + x * 3 + channel] = (y * 39 + x * 7 + channel * 61) % 256
                initial = bytes(src)
                expected = bytearray(bytes(dst))
                for y in range(height):
                    for x in range(width):
                        for channel in range(3):
                            expected[GUARD + y * dst_stride + x * 3 + channel] = src[GUARD + y * src_stride + x * 3 + 2 - channel]
                with self.subTest(height=height, width=width, padding=padding):
                    # Capacity is the precise touched span, not all allocated padding.
                    src_span = (height - 1) * src_stride + width * 3
                    dst_span = (height - 1) * dst_stride + width * 3
                    for _ in range(25):
                        self.assertEqual(RUN(src_ptr, src_span, height, width, src_stride,
                                             dst_ptr, dst_span, dst_stride), 0)
                        self.assertEqual(bytes(src), initial)
                        self.assertEqual(bytes(dst), bytes(expected))

    def test_errors_never_write_output(self):
        src, a = storage(128, 0xA5)
        dst, b = storage(128, 0xCC)
        valid = [a, 128, 4, 7, 26, b, 128, 24]
        failures = [({0: None}, 1), ({5: None}, 1), ({2: 0}, 2), ({3: 0}, 2),
                    ({2: 8193}, 2), ({3: 8193}, 2), ({4: 20}, 3), ({7: 20}, 3),
                    ({4: 2**64 - 1}, 3), ({7: 2**64 - 1}, 3),
                    ({1: 98}, 4), ({6: 92}, 4), ({4: 2**63 - 1}, 4),
                    ({5: a}, 5), ({5: a + 1}, 5), ({0: 2**64 - 16}, 5)]
        source_before, destination_before = bytes(src), bytes(dst)
        for changes, expected_code in failures:
            args = valid.copy()
            for position, value in changes.items():
                args[position] = value
            with self.subTest(changes=changes):
                self.assertEqual(RUN(*args), expected_code)
                self.assertEqual(bytes(src), source_before)
                self.assertEqual(bytes(dst), destination_before)


def main(argv=None):
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args(argv)
    library = C.CDLL(str(args.library.resolve(strict=True)))
    RUN = library.edge_color_smoke_run
    RUN.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64,
                    C.c_uint64, C.c_void_p, C.c_uint64, C.c_uint64]
    RUN.restype = C.c_int
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ColorSmokeTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
