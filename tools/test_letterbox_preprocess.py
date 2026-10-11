#!/usr/bin/env python3
"""UNITTEST：合成像素/张量验证几何契约，不代表模型精度或性能。"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.vision.preprocess import Preprocessor, UnsupportedInput, from_config
from src.vision.rknn_detector import RKNNObjectDetector
from src.vision.yolo_postprocess import decode
from tools.test_yolo_postprocess import make_output, unittest_forward_box

TRANSFORM_FIELDS = {
    "mode", "source_width", "source_height", "target_width", "target_height",
    "resized_width", "resized_height", "pad_left", "pad_top", "scale_x", "scale_y",
}


def unittest_bgr(height, width):
    """UNITTEST 固定彩色图，不读取相机、模型或外部数据。"""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = (11, 37, 203)
    return image


class LetterboxPreprocessTests(unittest.TestCase):
    def test_default_resize_pixels_and_transform_are_legacy_compatible(self):
        source = unittest_bgr(3, 5)
        source[1, 2] = (200, 17, 1)
        with Preprocessor(output_size=(8, 10)) as pre:
            rgb, transform = pre.run_with_transform(source)
            expected = cv2.cvtColor(cv2.resize(source, (10, 8), interpolation=cv2.INTER_LINEAR),
                                    cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(rgb, expected)
            np.testing.assert_array_equal(pre.run(source), expected)
            np.testing.assert_array_equal(pre.run_batch1(source), expected[np.newaxis])
            self.assertEqual(pre.geometry, "resize")
        self.assertEqual(transform, {
            "mode": "resize", "source_width": 5, "source_height": 3,
            "target_width": 10, "target_height": 8, "resized_width": 10,
            "resized_height": 8, "pad_left": 0, "pad_top": 0,
            "scale_x": 2.0, "scale_y": 8 / 3,
        })

    def test_odd_vertical_padding_fills_remainder_at_bottom(self):
        with Preprocessor(output_size=(8, 8), geometry="letterbox") as pre:
            rgb, transform = pre.run_with_transform(unittest_bgr(3, 5))
        self.assertEqual(set(transform), TRANSFORM_FIELDS)
        self.assertEqual((transform["resized_width"], transform["resized_height"],
                          transform["pad_left"], transform["pad_top"]), (8, 5, 0, 1))
        self.assertEqual(transform["scale_x"], 8 / 5)
        self.assertEqual(transform["scale_y"], 5 / 3)
        np.testing.assert_array_equal(rgb[0], np.full((8, 3), 114, dtype=np.uint8))
        np.testing.assert_array_equal(rgb[6:], np.full((2, 8, 3), 114, dtype=np.uint8))
        np.testing.assert_array_equal(rgb[1:6], np.broadcast_to([203, 37, 11], (5, 8, 3)))

    def test_odd_horizontal_padding_fills_remainder_at_right(self):
        with Preprocessor(output_size=(8, 8), geometry="letterbox") as pre:
            rgb, transform = pre.run_with_transform(unittest_bgr(5, 3))
        self.assertEqual((transform["resized_width"], transform["resized_height"],
                          transform["pad_left"], transform["pad_top"]), (5, 8, 1, 0))
        np.testing.assert_array_equal(rgb[:, 0], np.full((8, 3), 114, dtype=np.uint8))
        np.testing.assert_array_equal(rgb[:, 6:], np.full((8, 2, 3), 114, dtype=np.uint8))
        np.testing.assert_array_equal(rgb[:, 1:6], np.broadcast_to([203, 37, 11], (8, 5, 3)))

    def test_non_square_target_uses_height_width_api(self):
        with Preprocessor(output_size=(12, 10), geometry="letterbox") as pre:
            rgb, transform = pre.run_with_transform(unittest_bgr(7, 3))
        self.assertEqual(rgb.shape, (12, 10, 3))
        self.assertEqual((transform["target_width"], transform["target_height"]), (10, 12))
        self.assertEqual((transform["resized_width"], transform["resized_height"]), (5, 12))
        self.assertEqual((transform["pad_left"], transform["pad_top"]), (2, 0))
        self.assertEqual(transform["scale_x"], 5 / 3)
        self.assertEqual(transform["scale_y"], 12 / 7)

    def test_positive_half_ties_round_up_in_both_orientations(self):
        for shape, expected in (((5, 8), (4, 3)), ((8, 5), (3, 4))):
            with self.subTest(shape=shape), Preprocessor((4, 4), geometry="letterbox") as pre:
                _, transform = pre.run_with_transform(unittest_bgr(*shape))
                self.assertEqual((transform["resized_width"], transform["resized_height"]),
                                 expected)
                self.assertEqual(transform["scale_x"], expected[0] / shape[1])
                self.assertEqual(transform["scale_y"], expected[1] / shape[0])

    def test_pixels_match_inter_linear_resize_and_rgb_conversion(self):
        source = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        with Preprocessor((8, 8), geometry="letterbox") as pre:
            rgb, _ = pre.run_with_transform(source)
        expected = np.full((8, 8, 3), 114, dtype=np.uint8)
        expected[1:6] = cv2.cvtColor(cv2.resize(source, (8, 5), interpolation=cv2.INTER_LINEAR),
                                    cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(rgb, expected)

    def test_uint8_contiguous_rgb_without_normalization(self):
        with Preprocessor((12, 10), geometry="letterbox") as pre:
            rgb, _ = pre.run_with_transform(unittest_bgr(7, 3))
            batch = pre.run_batch1(unittest_bgr(7, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb.strides, (30, 3, 1))
        self.assertTrue(rgb.flags["C_CONTIGUOUS"] and rgb.flags["OWNDATA"])
        self.assertEqual(batch.shape, (1, 12, 10, 3))
        self.assertEqual(batch.dtype, np.uint8)
        self.assertTrue(batch.flags["C_CONTIGUOUS"])
        self.assertEqual(int(rgb[0, 2, 0]), 203)

    def test_positive_row_stride_roi_matches_packed_input(self):
        source = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        roi = source[2:5, 3:8]
        self.assertGreater(roi.strides[0], roi.shape[1] * 3)
        with Preprocessor((8, 8), geometry="letterbox") as pre:
            actual, transform = pre.run_with_transform(roi)
            packed, packed_transform = pre.run_with_transform(np.ascontiguousarray(roi))
        np.testing.assert_array_equal(actual, packed)
        self.assertEqual(transform, packed_transform)

    def test_bad_dtype_and_strides_are_rejected_for_both_modes(self):
        image = unittest_bgr(6, 8)
        bad = (image.astype(np.float32), image[::-1], image[:, ::2], image[:, :, ::-1],
               np.zeros((0, 8, 3), dtype=np.uint8), np.zeros((6, 8), dtype=np.uint8))
        for geometry in ("resize", "letterbox"):
            with Preprocessor((8, 8), geometry=geometry) as pre:
                for source in bad:
                    with self.subTest(mode=geometry, shape=source.shape), \
                            self.assertRaises(UnsupportedInput):
                        pre.run_with_transform(source)

    def test_legacy_output_buffer_and_batch_view_survive_letterbox(self):
        with Preprocessor((8, 8), geometry="letterbox") as pre:
            output = pre.new_output()
            self.assertIs(pre.run(unittest_bgr(3, 5), out=output), output)
            batch = pre.run_batch1(unittest_bgr(3, 5), out=output)
            self.assertTrue(np.shares_memory(batch, output))
            self.assertEqual(batch.shape, (1, 8, 8, 3))
            for bad in (np.empty((8, 8, 3), dtype=np.float32),
                        np.empty((8, 16, 3), dtype=np.uint8)[:, ::2]):
                with self.assertRaises(UnsupportedInput):
                    pre.run(unittest_bgr(3, 5), out=bad)

    def test_each_transform_and_output_belongs_to_its_frame(self):
        with Preprocessor((8, 8), geometry="letterbox") as pre:
            first, first_transform = pre.run_with_transform(unittest_bgr(3, 5))
            second, second_transform = pre.run_with_transform(unittest_bgr(5, 3))
        self.assertFalse(np.shares_memory(first, second))
        self.assertEqual(first_transform["pad_top"], 1)
        self.assertEqual(second_transform["pad_left"], 1)
        first_transform["pad_top"] = 999
        self.assertEqual(second_transform["pad_top"], 0)

    def test_unknown_geometry_and_empty_rounded_dimension_rejected(self):
        with self.assertRaises(ValueError):
            Preprocessor((8, 8), geometry="auto")
        with Preprocessor((1, 1), geometry="letterbox") as pre:
            with self.assertRaisesRegex(UnsupportedInput, "舍入后尺寸为空"):
                pre.run_with_transform(unittest_bgr(8, 1))

    def test_config_geometry_is_explicit_and_default_unchanged(self):
        with from_config({}, (8, 8)) as pre:
            self.assertEqual(pre.geometry, "resize")
        with from_config({"preprocess": {"geometry": "letterbox"}}, (8, 8)) as pre:
            self.assertEqual(pre.geometry, "letterbox")
            self.assertEqual(pre.describe()["geometry"], "letterbox")


class UnittestRKNN:
    """UNITTEST 推理夹具，仅记录输入并返回人工张量，不加载任何SDK/模型。"""
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = None

    def inference(self, inputs):
        self.inputs = inputs
        return self.outputs


class DetectorGeometryTests(unittest.TestCase):
    def test_detector_default_and_old_preprocess_postprocess_entrypoints(self):
        detector = RKNNObjectDetector("UNITTEST-do-not-load.rknn", target_size=(10, 8))
        source = unittest_bgr(3, 5)
        expected = cv2.cvtColor(cv2.resize(source, (10, 8)), cv2.COLOR_BGR2RGB)
        self.assertIsNone(detector.preprocess(None))
        np.testing.assert_array_equal(detector.preprocess(source), expected)
        tensor = make_output([(5, 4, 4, 4)], [(0, 0.9)])
        self.assertEqual(detector._postprocess(tensor, source.shape),
                         decode(tensor, source.shape, input_size=(8, 10), conf_thresh=0.5))
        detector.release()

    def test_detect_passes_one_transform_and_contiguous_uint8_batch(self):
        source = unittest_bgr(7, 3)
        detector = RKNNObjectDetector("UNITTEST-do-not-load.rknn", target_size=(10, 12),
                                      geometry="letterbox")
        pre = detector.preprocessor
        with Preprocessor((12, 10), geometry="letterbox") as fixture_pre:
            expected_rgb, transform = fixture_pre.run_with_transform(source)
        tensor = make_output([unittest_forward_box((0, 1, 2, 5), transform)], [(0, 0.9)])
        detector.rknn = UnittestRKNN([tensor[:, :4], tensor[:, 4:]])
        detector.is_loaded = True
        with mock.patch.object(pre, "run_with_transform", wraps=pre.run_with_transform) as run, \
                mock.patch("src.vision.yolo_postprocess.decode", wraps=decode) as wrapped_decode:
            result = detector.detect(source)
            run.assert_called_once_with(source)
            self.assertEqual(wrapped_decode.call_args.kwargs["transform"], transform)
            self.assertEqual(wrapped_decode.call_args.kwargs["input_size"], (12, 10))
        self.assertEqual(result[0]["bbox"], [0, 1, 2, 5])
        batch = detector.rknn.inputs[0]
        self.assertEqual(batch.shape, (1, 12, 10, 3))
        self.assertEqual(batch.dtype, np.uint8)
        self.assertTrue(batch.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(batch[0], expected_rgb)
        detector.rknn = None
        detector.release()

    def test_detect_default_resize_matches_old_decode(self):
        detector = RKNNObjectDetector("UNITTEST-do-not-load.rknn", target_size=(10, 8))
        source = unittest_bgr(3, 5)
        tensor = make_output([(5, 4, 4, 4)], [(0, 0.9)])
        detector.rknn = UnittestRKNN([tensor])
        detector.is_loaded = True
        self.assertEqual(detector.detect(source),
                         decode(tensor, source.shape, input_size=(8, 10), conf_thresh=0.5))
        detector.rknn = None
        detector.release()

    def test_explicit_mlir_letterbox_is_not_silently_resized(self):
        detector = RKNNObjectDetector("UNITTEST-do-not-load.rknn", preprocess_backend="mlir",
                                      geometry="letterbox")
        with self.assertRaisesRegex(UnsupportedInput, "MLIR.*letterbox"):
            detector.preprocess(unittest_bgr(3, 5))


if __name__ == "__main__":
    unittest.main()
