#!/usr/bin/env python3
"""Run with py -3.12 -B compiler/tests/test_preprocess_reference.py -v.

Standard-library tests always run. NumPy and OpenCV checks explicitly skip when
unavailable; skips are not evidence of array execution or generated-MLIR parity.
OpenCV is never used to construct the numerical oracle.
"""

import builtins
from fractions import Fraction
import importlib.util
from pathlib import Path
import random
import unittest
from unittest import mock

REFERENCE_PATH = Path(__file__).with_name("preprocess_reference.py")
SPEC = importlib.util.spec_from_file_location("preprocess_reference", REFERENCE_PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)
np = reference.np

if np is not None:
    try:
        import cv2
    except ImportError:
        cv2 = None
else:
    cv2 = None

NUMPY_SKIP = "NumPy unavailable: actual ndarray reference/differential tests NOT run"
OPENCV_SKIP = ("NumPy unavailable: OpenCV differential test NOT run" if np is None
               else "cv2 unavailable: OpenCV differential test NOT run")


def gradient(height, width):
    return [[((y * 37 + x * 11) % 256,
              (y * 17 + x * 29 + 31) % 256,
              (y * 53 + x * 7 + 89) % 256)
             for x in range(width)] for y in range(height)]


class AxisWeightsTests(unittest.TestCase):
    def test_negative_half_pixel_is_floored_before_clamping(self):
        self.assertEqual(reference.axis_weights(2, 4), [
            (0, 0, 512, 1536), (0, 1, 1536, 512),
            (0, 1, 512, 1536), (1, 1, 1536, 512),
        ])

    def test_fractional_odd_downsample_weights(self):
        self.assertEqual(reference.axis_weights(5, 3), [
            (0, 1, 1365, 683), (2, 3, 2048, 0), (3, 4, 683, 1365),
        ])

    def test_weight_half_tie_rounds_up_not_to_even(self):
        # alpha = 2049/4096; alpha * 2048 = 1024.5, not a float approximation.
        self.assertEqual(reference.axis_weights(1, 2048)[0], (0, 0, 1023, 1025))

    def test_same_size_has_unit_left_weight(self):
        self.assertEqual(reference.axis_weights(4, 4), [
            (0, 1, 2048, 0), (1, 2, 2048, 0),
            (2, 3, 2048, 0), (3, 3, 2048, 0),
        ])

    def test_single_source_indices_always_replicate(self):
        for weights in reference.axis_weights(1, 17):
            self.assertEqual((weights.i0, weights.i1), (0, 0))
            self.assertEqual(weights.w0 + weights.w1, reference.Q)

    def test_all_small_axes_have_convex_quantized_weights(self):
        for source in range(1, 18):
            for target in range(1, 18):
                weights = reference.axis_weights(source, target)
                self.assertEqual(len(weights), target)
                for i0, i1, w0, w1 in weights:
                    self.assertTrue(0 <= i0 <= i1 < source)
                    self.assertTrue(0 <= w0 <= reference.Q)
                    self.assertTrue(0 <= w1 <= reference.Q)
                    self.assertEqual(w0 + w1, reference.Q)

    def test_coordinate_and_weight_floors_receive_fractions(self):
        with mock.patch.object(reference, "floor", wraps=reference.floor) as floor_spy:
            reference.axis_weights(7, 11)
        self.assertEqual(floor_spy.call_count, 22)
        self.assertTrue(all(isinstance(call.args[0], Fraction)
                            for call in floor_spy.call_args_list))

    def test_dimension_limit_is_inclusive(self):
        self.assertEqual(len(reference.axis_weights(1, 8192)), 8192)
        self.assertEqual(reference.axis_weights(8192, 1), [(4095, 4096, 1024, 1024)])

    def test_invalid_axis_sizes_rejected(self):
        for value in (0, -1, 8193, 10**30):
            for args in ((value, 1), (1, value)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    reference.axis_weights(*args)
        for value in (True, False, 1.0, "1", None, Fraction(1)):
            for args in ((value, 1), (1, value)):
                with self.subTest(args=args), self.assertRaises(TypeError):
                    reference.axis_weights(*args)

    def test_biased_accumulator_bound_fits_signed_int32(self):
        self.assertEqual(reference.Q, 2048)
        self.assertEqual(reference.MAX_BIASED_ACCUMULATOR, 1071644672)
        self.assertEqual(255 * 2048**2 + 2048**2 // 2, 1071644672)
        self.assertLessEqual(reference.MAX_BIASED_ACCUMULATOR, 2**31 - 1)
        self.assertEqual(reference.MAX_BIASED_ACCUMULATOR // reference.Q_SQUARED, 255)
        # Sum of four nonnegative products is exactly Q**2, including borders.
        for yw in reference.axis_weights(3, 7):
            for xw in reference.axis_weights(5, 9):
                four_weights = (yw.w0 * xw.w0, yw.w0 * xw.w1,
                                yw.w1 * xw.w0, yw.w1 * xw.w1)
                self.assertEqual(sum(four_weights), reference.Q_SQUARED)
                self.assertLessEqual(255 * sum(four_weights) + reference.Q_SQUARED // 2,
                                     2**31 - 1)


class ScalarReferenceTests(unittest.TestCase):
    def test_single_pixel_replicates_and_swaps_rgb(self):
        for out_height, out_width in ((1, 1), (1, 7), (5, 1), (5, 9)):
            with self.subTest(shape=(out_height, out_width)):
                self.assertEqual(reference.reference_resize_rgb_scalar([[(3, 17, 251)]],
                                                                      out_height, out_width),
                                 [[(251, 17, 3)] * out_width for _ in range(out_height)])

    def test_single_row_upsample_exact_golden(self):
        image = [[(0, 0, 0), (10, 20, 30)]]
        self.assertEqual(reference.reference_resize_rgb_scalar(image, 1, 4),
                         [[(0, 0, 0), (8, 5, 3), (23, 15, 8), (30, 20, 10)]])

    def test_single_column_upsample_exact_golden(self):
        image = [[(0, 0, 0)], [(10, 20, 30)]]
        self.assertEqual(reference.reference_resize_rgb_scalar(image, 4, 1),
                         [[(0, 0, 0)], [(8, 5, 3)], [(23, 15, 8)], [(30, 20, 10)]])

    def test_same_size_is_only_rgb_swap(self):
        image = gradient(3, 5)
        self.assertEqual(reference.reference_resize_rgb_scalar(image, 3, 5),
                         [[tuple(reversed(pixel)) for pixel in row] for row in image])

    def test_two_dimensional_odd_upsample_exact_golden(self):
        image = [[(0, 0, 0), (10, 20, 30)], [(20, 40, 60), (30, 60, 90)]]
        self.assertEqual(reference.reference_resize_rgb_scalar(image, 3, 3), [
            [(0, 0, 0), (15, 10, 5), (30, 20, 10)],
            [(30, 20, 10), (45, 30, 15), (60, 40, 20)],
            [(60, 40, 20), (75, 50, 25), (90, 60, 30)],
        ])

    def test_downsample_exact_golden(self):
        image = [[tuple(y * 40 + x * 8 + c * 32 for c in range(3))
                  for x in range(4)] for y in range(4)]
        self.assertEqual(reference.reference_resize_rgb_scalar(image, 2, 2), [
            [(88, 56, 24), (104, 72, 40)], [(168, 136, 104), (184, 152, 120)],
        ])

    def test_odd_downsample_gradient_exact_golden(self):
        image = [[(x * 30 + y * 60,) * 3 for x in range(5)] for y in range(3)]
        # Half-pixel y = 1/4, 7/4; x = 1/3, 2, 11/3. Q-quantized x
        # changes these exact linear values by < 1/2, hence these integer results.
        self.assertEqual(reference.reference_resize_rgb_scalar(image, 2, 3), [
            [(25, 25, 25), (75, 75, 75), (125, 125, 125)],
            [(115, 115, 115), (165, 165, 165), (215, 215, 215)],
        ])

    def test_zero_full_scale_and_constant_colors(self):
        for color in ((0, 0, 0), (255, 255, 255), (13, 117, 243)):
            image = [[color] * 5 for _ in range(3)]
            for target in ((1, 1), (2, 3), (7, 9)):
                with self.subTest(color=color, target=target):
                    self.assertEqual(reference.reference_resize_rgb_scalar(image, *target),
                                     [[tuple(reversed(color))] * target[1]
                                      for _ in range(target[0])])

    def test_output_half_tie_rounds_up(self):
        self.assertEqual(reference.reference_resize_rgb_scalar(
            [[(0, 0, 0), (1, 1, 1)]], 1, 1), [[(1, 1, 1)]])

    def test_four_neighbors_round_once_not_after_each_axis(self):
        # Horizontal-then-vertical intermediate rounding would incorrectly give 1.
        image = [[(0, 0, 0), (1, 1, 1)], [(0, 0, 0), (0, 0, 0)]]
        self.assertEqual(reference.reference_resize_rgb_scalar(image, 1, 1), [[(0, 0, 0)]])

    def test_pixel_api_agrees_at_every_small_output_coordinate(self):
        image = gradient(3, 5)
        resized = reference.reference_resize_rgb_scalar(image, 5, 7)
        for y in range(5):
            for x in range(7):
                self.assertEqual(reference.reference_pixel_rgb(image, 5, 7, y, x),
                                 resized[y][x])

    def test_scalar_oracle_is_independent_of_axis_weights_and_uses_fraction(self):
        with mock.patch.object(reference, "axis_weights", side_effect=AssertionError(
                "scalar oracle must not reuse the array axis table")):
            with mock.patch.object(reference, "floor", wraps=reference.floor) as floor_spy:
                result = reference.reference_resize_rgb_scalar(gradient(3, 5), 2, 7)
        self.assertEqual((len(result), len(result[0])), (2, 7))
        self.assertTrue(floor_spy.call_args_list)
        self.assertTrue(all(isinstance(call.args[0], Fraction)
                            for call in floor_spy.call_args_list))

    def test_seeded_small_random_results_remain_uint8(self):
        rng = random.Random(2026)
        for height, width, oh, ow in ((1, 5, 7, 3), (7, 1, 3, 9), (5, 7, 3, 11)):
            image = [[tuple(rng.randrange(256) for _ in range(3))
                      for _ in range(width)] for _ in range(height)]
            output = reference.reference_resize_rgb_scalar(image, oh, ow)
            self.assertEqual((len(output), len(output[0])), (oh, ow))
            self.assertTrue(all(type(value) is int and 0 <= value <= 255
                                for row in output for pixel in row for value in pixel))

    def test_no_input_mutation_or_shared_output_rows_or_calls(self):
        image = [[(1, 2, 3)]]
        first = reference.reference_resize_rgb_scalar(image, 2, 3)
        second = reference.reference_resize_rgb_scalar(image, 2, 3)
        first[0][0] = (0, 0, 0)
        self.assertEqual(first[1][0], (3, 2, 1))
        self.assertEqual(second[0][0], (3, 2, 1))
        self.assertEqual(image, [[(1, 2, 3)]])

    def test_scalar_input_shapes_and_values_are_strict(self):
        malformed = ([], [[]], [[(1, 2, 3)], []], [[7]], [[(1, 2)]],
                     [[(1, 2, 3, 4)]], [[(-1, 0, 0)]], [[(256, 0, 0)]])
        for image in malformed:
            with self.subTest(image=image), self.assertRaises(ValueError):
                reference.reference_resize_rgb_scalar(image, 1, 1)
        for image in (None, "abc", [1], [[(1.0, 2, 3)]], [[(True, 2, 3)]]):
            with self.subTest(image=image), self.assertRaises(TypeError):
                reference.reference_resize_rgb_scalar(image, 1, 1)

    def test_input_dimension_limits(self):
        pixel = (1, 2, 3)
        self.assertEqual(reference.reference_pixel_rgb([[pixel] * 8192], 1, 1, 0, 0),
                         (3, 2, 1))
        self.assertEqual(reference.reference_pixel_rgb([[pixel]] * 8192, 1, 1, 0, 0),
                         (3, 2, 1))
        for image in ([[pixel] * 8193], [[pixel]] * 8193):
            with self.assertRaises(ValueError):
                reference.reference_resize_rgb_scalar(image, 1, 1)

    def test_output_dimension_limits_and_types(self):
        image = [[(1, 2, 3)]]
        self.assertEqual(reference.reference_pixel_rgb(image, 8192, 8192, 8191, 8191),
                         (3, 2, 1))
        for value in (0, -1, 8193, 10**30):
            for target in ((value, 1), (1, value)):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    reference.reference_resize_rgb_scalar(image, *target)
        for value in (True, 1.5, "2", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                reference.reference_resize_rgb_scalar(image, value, 1)

    def test_pixel_destination_coordinate_validation(self):
        for y, x in ((-1, 0), (0, -1), (2, 0), (0, 3)):
            with self.subTest(y=y, x=x), self.assertRaises(ValueError):
                reference.reference_pixel_rgb([[(0, 0, 0)]], 2, 3, y, x)
        for y, x in ((0.5, 0), (0, True)):
            with self.subTest(y=y, x=x), self.assertRaises(TypeError):
                reference.reference_pixel_rgb([[(0, 0, 0)]], 2, 3, y, x)


class OptionalDependencyTests(unittest.TestCase):
    def test_module_really_imports_and_scalar_runs_when_numpy_import_fails(self):
        original_import = builtins.__import__

        def without_numpy(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "cv2"):
                raise ModuleNotFoundError("simulated missing NumPy", name="numpy")
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("preprocess_without_numpy", REFERENCE_PATH)
        isolated = importlib.util.module_from_spec(spec)
        with mock.patch("builtins.__import__", side_effect=without_numpy):
            spec.loader.exec_module(isolated)
        self.assertIsNone(isolated.np)
        self.assertEqual(isolated.axis_weights(1, 1), [(0, 0, 2048, 0)])
        self.assertEqual(isolated.reference_resize_rgb_scalar([[(1, 2, 3)]], 1, 1),
                         [[(3, 2, 1)]])
        self.assertEqual(isolated.reference_pixel_rgb([[(1, 2, 3)]], 1, 1, 0, 0),
                         (3, 2, 1))
        with self.assertRaisesRegex(RuntimeError, "NumPy is unavailable.*cannot run"):
            isolated.reference_resize_rgb([[(1, 2, 3)]], 1, 1)
        with self.assertRaisesRegex(RuntimeError, "NumPy is unavailable.*cannot run"):
            isolated.difference_statistics(None, None)


@unittest.skipUnless(np is not None, NUMPY_SKIP)
class NumpyReferenceTests(unittest.TestCase):
    def assert_matches_scalar(self, image, out_height, out_width):
        before = image.copy()
        expected = np.array(reference.reference_resize_rgb_scalar(
            image.tolist(), out_height, out_width), dtype=np.uint8)
        actual = reference.reference_resize_rgb(image, out_height, out_width)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(image, before)
        self.assertEqual(actual.dtype, np.dtype(np.uint8))
        self.assertEqual(actual.shape, (out_height, out_width, 3))
        self.assertEqual(actual.strides, (out_width * 3, 3, 1))
        self.assertTrue(actual.flags.c_contiguous)
        self.assertTrue(actual.flags.owndata)
        self.assertFalse(np.shares_memory(actual, image))
        return actual

    def test_1x1_and_single_axis_images(self):
        for height, width in ((1, 1), (1, 7), (5, 1)):
            image = np.array(gradient(height, width), dtype=np.uint8)
            for target in ((1, 1), (1, 11), (9, 1), (7, 9), (2, 3)):
                with self.subTest(source=(height, width), target=target):
                    self.assert_matches_scalar(image, *target)

    def test_same_size_only_swaps_rgb(self):
        image = np.array(gradient(5, 7), dtype=np.uint8)
        result = self.assert_matches_scalar(image, 5, 7)
        np.testing.assert_array_equal(result, image[:, :, ::-1])

    def test_up_down_and_odd_size_gradients(self):
        for shape, target in (((3, 5), (7, 9)), ((7, 9), (3, 5)),
                              ((5, 7), (2, 11)), ((3, 3), (8, 8)), ((8, 8), (3, 3))):
            with self.subTest(shape=shape, target=target):
                self.assert_matches_scalar(np.array(gradient(*shape), dtype=np.uint8), *target)

    def test_seeded_random_arrays_exactly_match_independent_scalar(self):
        rng = random.Random(61541)
        for case in range(60):
            h, w, oh, ow = (rng.randrange(1, 10) for _ in range(4))
            image = np.array([[[rng.randrange(256) for _ in range(3)]
                               for _ in range(w)] for _ in range(h)], dtype=np.uint8)
            with self.subTest(case=case, source=(h, w), target=(oh, ow)):
                self.assert_matches_scalar(image, oh, ow)

    def test_pure_colors_zero_and_255(self):
        for color in ((0, 0, 0), (255, 255, 255), (13, 117, 243)):
            image = np.empty((3, 5, 3), dtype=np.uint8)
            image[:] = color
            for target in ((1, 1), (2, 3), (7, 11)):
                with self.subTest(color=color, target=target):
                    result = self.assert_matches_scalar(image, *target)
                    self.assertTrue(np.all(result == np.array(color[::-1], dtype=np.uint8)))

    def test_checkerboard_uses_four_taps_and_single_final_rounding(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 1] = 1
        np.testing.assert_array_equal(self.assert_matches_scalar(image, 1, 1),
                                      np.zeros((1, 1, 3), dtype=np.uint8))
        image[0, 1] = 255
        image[1, 0] = 255
        self.assert_matches_scalar(image, 7, 9)

    def test_output_half_tie_rounds_up(self):
        image = np.array([[(0, 0, 0), (1, 1, 1)]], dtype=np.uint8)
        self.assertTrue(np.all(self.assert_matches_scalar(image, 1, 1) == 1))

    def test_positive_row_stride_offset_roi_and_skipped_rows(self):
        parent = np.array(gradient(9, 11), dtype=np.uint8)
        before = parent.copy()
        image = parent[1:8:2, 2:9, :]
        self.assertGreater(image.strides[0], image.shape[1] * 3)
        self.assertFalse(image.flags.c_contiguous)
        self.assert_matches_scalar(image, 7, 5)
        np.testing.assert_array_equal(parent, before)

    def test_odd_byte_row_padding_and_canaries(self):
        height, width, stride, guard = 3, 5, 20, 7
        backing = bytearray([0xA5] * (2 * guard + (height - 1) * stride + width * 3))
        image = np.ndarray((height, width, 3), dtype=np.uint8, buffer=backing,
                           offset=guard, strides=(stride, 3, 1))
        image[:] = np.array(gradient(height, width), dtype=np.uint8)
        before = bytes(backing)
        self.assert_matches_scalar(image, 7, 9)
        self.assertEqual(bytes(backing), before)

    def test_read_only_input_is_accepted(self):
        image = np.array(gradient(3, 5), dtype=np.uint8)
        image.flags.writeable = False
        self.assert_matches_scalar(image, 7, 3)

    def test_output_allocations_are_independent_between_calls(self):
        image = np.array(gradient(3, 5), dtype=np.uint8)
        before = image.copy()
        first = reference.reference_resize_rgb(image, 3, 5)
        second = reference.reference_resize_rgb(image, 3, 5)
        expected_second = second.copy()
        self.assertFalse(np.shares_memory(first, second))
        first[:] = 0
        np.testing.assert_array_equal(second, expected_second)
        np.testing.assert_array_equal(image, before)

    def test_array_axis_weights_come_from_fraction_table(self):
        image = np.array(gradient(3, 5), dtype=np.uint8)
        with mock.patch.object(reference, "axis_weights", wraps=reference.axis_weights) as spy:
            reference.reference_resize_rgb(image, 7, 9)
        self.assertEqual(spy.call_args_list, [mock.call(3, 7), mock.call(5, 9)])

    def test_multiple_vectorized_row_tiles(self):
        with mock.patch.object(reference, "_MAX_TILE_PIXELS", 14):
            self.assert_matches_scalar(np.array(gradient(3, 5), dtype=np.uint8), 9, 7)

    def test_maximum_valid_input_and_output_axis_size(self):
        for shape in ((1, 8192, 3), (8192, 1, 3)):
            image = np.full(shape, 255, dtype=np.uint8)
            self.assertTrue(np.all(reference.reference_resize_rgb(image, 1, 1) == 255))
        image = np.array([[(1, 2, 3)]], dtype=np.uint8)
        for oh, ow in ((1, 8192), (8192, 1)):
            output = reference.reference_resize_rgb(image, oh, ow)
            self.assertEqual(output.shape, (oh, ow, 3))
            self.assertTrue(np.all(output == np.array([3, 2, 1], dtype=np.uint8)))

    def test_numpy_integral_dimensions_are_accepted_but_boolean_is_not(self):
        image = np.array([[(1, 2, 3)]], dtype=np.uint8)
        self.assertEqual(reference.reference_resize_rgb(image, np.int64(2), np.int32(3)).shape,
                         (2, 3, 3))
        with self.assertRaises(TypeError):
            reference.reference_resize_rgb(image, np.bool_(True), 1)

    def test_reject_non_array_without_implicit_conversion(self):
        for image in ([[[1, 2, 3]]], None, bytearray(3)):
            with self.subTest(value=image), self.assertRaises(TypeError):
                reference.reference_resize_rgb(image, 1, 1)

    def test_reject_gray_rgba_chw_and_wrong_rank(self):
        for shape in ((3, 5), (3, 5, 4), (3, 5, 1), (3, 4, 5), (1, 3, 5, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                reference.reference_resize_rgb(np.zeros(shape, dtype=np.uint8), 1, 1)

    def test_reject_float_signed_and_other_dtypes(self):
        for dtype in (np.float32, np.float64, np.int8, np.uint16, np.bool_, object):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                reference.reference_resize_rgb(np.zeros((2, 3, 3), dtype=dtype), 1, 1)

    def test_reject_negative_sparse_transposed_and_fortran_layouts(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        layouts = (image[::-1], image[:, ::-1], image[:, :, ::-1], image[:, ::2],
                   image.transpose(1, 0, 2), np.asfortranarray(image),
                   np.zeros((4, 6, 6), dtype=np.uint8)[:, :, ::2])
        for bad in layouts:
            with self.subTest(strides=bad.strides), self.assertRaises(ValueError):
                reference.reference_resize_rgb(bad, 1, 1)

    def test_reject_zero_and_overlapping_strides(self):
        for strides in ((0, 3, 1), (14, 3, 1), (15, 0, 1), (15, 3, 0)):
            image = np.ndarray((3, 5, 3), dtype=np.uint8, buffer=bytearray(128),
                               strides=strides)
            with self.subTest(strides=strides), self.assertRaises(ValueError):
                reference.reference_resize_rgb(image, 1, 1)

    def test_reject_empty_and_oversized_input_dimensions(self):
        for shape in ((0, 3, 3), (3, 0, 3), (8193, 1, 3), (1, 8193, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                reference.reference_resize_rgb(np.zeros(shape, dtype=np.uint8), 1, 1)

    def test_reject_invalid_output_sizes_before_allocating(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        for value in (0, -1, 8193, 10**30):
            for target in ((value, 1), (1, value)):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    reference.reference_resize_rgb(image, *target)
        for value in (True, False, 1.0, "1", None):
            for target in ((value, 1), (1, value)):
                with self.subTest(target=target), self.assertRaises(TypeError):
                    reference.reference_resize_rgb(image, *target)


@unittest.skipUnless(np is not None, NUMPY_SKIP)
class DifferenceStatisticsTests(unittest.TestCase):
    def test_equal_outputs_have_zero_statistics_and_no_coordinates(self):
        image = np.array(gradient(3, 5), dtype=np.uint8)
        self.assertEqual(reference.difference_statistics(image, image.copy()), {
            "max_abs_error": 0, "mae": 0.0,
            "different_component_ratio": 0.0, "different_pixel_ratio": 0.0,
            "worst_coordinates": [], "error_bounding_box": None,
        })

    def test_component_and_pixel_ratios_have_different_denominators(self):
        expected = np.zeros((2, 2, 3), dtype=np.uint8)
        actual = expected.copy()
        actual[0, 1] = [2, 0, 1]
        actual[1, 0] = [0, 3, 0]
        stats = reference.difference_statistics(actual, expected, max_worst=2)
        self.assertEqual(stats["max_abs_error"], 3)
        self.assertEqual(stats["mae"], 0.5)
        self.assertEqual(stats["different_component_ratio"], 3 / 12)
        self.assertEqual(stats["different_pixel_ratio"], 2 / 4)
        self.assertEqual([(entry["y"], entry["x"], entry["channel"], entry["abs_error"])
                          for entry in stats["worst_coordinates"]],
                         [(1, 0, 1, 3), (0, 1, 0, 2)])
        self.assertEqual(stats["error_bounding_box"],
                         {"y_min": 0, "x_min": 0, "y_max": 1, "x_max": 1})

    def test_signed_subtraction_cannot_wrap_uint8(self):
        expected = np.full((1, 1, 3), 255, dtype=np.uint8)
        actual = np.zeros_like(expected)
        stats = reference.difference_statistics(actual, expected)
        self.assertEqual(stats["max_abs_error"], 255)
        self.assertEqual(stats["mae"], 255.0)
        self.assertEqual(stats["different_component_ratio"], 1.0)
        self.assertEqual(stats["different_pixel_ratio"], 1.0)
        for entry in stats["worst_coordinates"]:
            self.assertEqual((entry["signed_error"], entry["abs_error"]), (-255, 255))

    def test_worst_ties_are_deterministic_and_capped(self):
        actual = np.ones((2, 2, 3), dtype=np.uint8)
        expected = np.zeros_like(actual)
        stats = reference.difference_statistics(actual, expected, max_worst=4)
        self.assertEqual([(entry["y"], entry["x"], entry["channel"])
                          for entry in stats["worst_coordinates"]],
                         [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)])
        self.assertEqual(len(reference.difference_statistics(actual, expected, 100)
                             ["worst_coordinates"]), 12)

    def test_bounding_box_includes_unreported_low_error_pixels(self):
        expected = np.zeros((5, 7, 3), dtype=np.uint8)
        actual = expected.copy()
        actual[1, 2, 0] = 1
        actual[2, 3, 1] = 200
        actual[4, 6, 2] = 2
        stats = reference.difference_statistics(actual, expected, max_worst=1)
        self.assertEqual(stats["worst_coordinates"][0]["abs_error"], 200)
        self.assertEqual(stats["error_bounding_box"],
                         {"y_min": 1, "x_min": 2, "y_max": 4, "x_max": 6})
        no_records = reference.difference_statistics(actual, expected, max_worst=0)
        self.assertEqual(no_records["worst_coordinates"], [])
        self.assertEqual(no_records["error_bounding_box"], stats["error_bounding_box"])
        self.assertEqual(no_records["max_abs_error"], 200)

    def test_worst_coordinates_across_multiple_bounded_scan_chunks(self):
        actual = np.zeros((3, 5, 3), dtype=np.uint8)
        expected = np.zeros_like(actual)
        actual[0, 0, 0] = 2
        actual[2, 4, 2] = 7
        with mock.patch.object(reference, "_MAX_TILE_PIXELS", 4):
            stats = reference.difference_statistics(actual, expected, max_worst=2)
        self.assertEqual([(entry["y"], entry["x"], entry["channel"])
                          for entry in stats["worst_coordinates"]], [(2, 4, 2), (0, 0, 0)])

    def test_statistics_reject_shape_dtype_and_limit_errors(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            reference.difference_statistics(image, np.zeros((3, 2, 3), dtype=np.uint8))
        with self.assertRaises(TypeError):
            reference.difference_statistics(image, image.astype(np.float32))
        with self.assertRaises(ValueError):
            reference.difference_statistics(image, image, max_worst=-1)
        for limit in (True, 1.5, "2"):
            with self.subTest(limit=limit), self.assertRaises(TypeError):
                reference.difference_statistics(image, image, max_worst=limit)


@unittest.skipUnless(np is not None and cv2 is not None, OPENCV_SKIP)
class OptionalOpenCVDifferentialTests(unittest.TestCase):
    def test_inter_linear_max_abs_error_is_at_most_one_not_bit_exact_claim(self):
        rng = random.Random(8539)
        fixtures = [(np.array(gradient(h, w), dtype=np.uint8), oh, ow)
                    for h, w, oh, ow in ((1, 1, 7, 9), (1, 7, 3, 11), (5, 1, 9, 3),
                                         (3, 5, 7, 9), (7, 9, 3, 5), (5, 7, 5, 7))]
        for _ in range(30):
            h, w, oh, ow = (rng.randrange(1, 18) for _ in range(4))
            image = np.array([[[rng.randrange(256) for _ in range(3)]
                               for _ in range(w)] for _ in range(h)], dtype=np.uint8)
            fixtures.append((image, oh, ow))
        for image, oh, ow in fixtures:
            with self.subTest(source=image.shape, target=(oh, ow)):
                independent = reference.reference_resize_rgb(image, oh, ow)
                opencv = cv2.cvtColor(cv2.resize(image, (ow, oh),
                                                interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                stats = reference.difference_statistics(opencv, independent)
                self.assertLessEqual(stats["max_abs_error"], 1, stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
