#!/usr/bin/env python3
"""YOLOv8 后处理解码的边界测试。

UNITTEST 构造的张量只验证解码逻辑，不构成检测精度证据 —— 真实模型验证见
build/mlir-results/verify_yolo_decode.py。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.vision.yolo_postprocess import (
    COCO_CLASSES, PostprocessError, decode, nms,
)


def make_output(boxes, scores_by_class, num_classes=80, num_anchors=8400):
    """UNITTEST 合成 [1, 4+num_classes, num_anchors] 输出。

    boxes           : [(cx, cy, w, h), ...]
    scores_by_class : [(class_id, score), ...]，与 boxes 一一对应
    """
    tensor = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)
    for index, ((cx, cy, w, h), (class_id, score)) in enumerate(zip(boxes, scores_by_class)):
        tensor[0, 0, index] = cx
        tensor[0, 1, index] = cy
        tensor[0, 2, index] = w
        tensor[0, 3, index] = h
        tensor[0, 4 + class_id, index] = score
    return tensor


class LayoutTests(unittest.TestCase):
    def test_accepts_transposed_layout(self):
        """有些导出会给 (anchors, attrs)，两种都要能解。"""
        tensor = make_output([(320, 320, 100, 100)], [(0, 0.9)])
        straight = decode(tensor, (640, 640))
        flipped = decode(np.transpose(tensor, (0, 2, 1)), (640, 640))
        self.assertEqual(len(straight), 1)
        self.assertEqual(straight, flipped)

    def test_accepts_bare_two_dim_input(self):
        tensor = make_output([(320, 320, 100, 100)], [(0, 0.9)])[0]
        self.assertEqual(len(decode(tensor, (640, 640))), 1)

    def test_accepts_split_coordinate_and_score_outputs_in_either_order(self):
        combined = make_output([(320, 320, 100, 100)], [(0, 0.9)])[0]
        coords = combined[:4]
        scores = combined[4:]
        expected = decode(combined, (640, 640))
        self.assertEqual(decode([coords, scores], (640, 640)), expected)
        self.assertEqual(decode([scores, coords], (640, 640)), expected)

    def test_rejects_invalid_split_output_contract(self):
        coords = np.zeros((1, 4, 8400), dtype=np.float32)
        wrong = np.zeros((1, 79, 8400), dtype=np.float32)
        with self.assertRaises(PostprocessError):
            decode([coords, wrong], (640, 640))

    def test_rejects_split_anchor_mismatch(self):
        coords = np.zeros((1, 4, 8400), dtype=np.float32)
        scores = np.zeros((1, 80, 8399), dtype=np.float32)
        with self.assertRaises(PostprocessError):
            decode([coords, scores], (640, 640))

    def test_rejects_malformed_instead_of_returning_empty(self):
        """布局不对必须报错。静默返回空列表会被当成图里没东西。"""
        cases = {
            "none": None,
            "batch_gt_1": np.zeros((2, 84, 8400), dtype=np.float32),
            "four_dim": np.zeros((1, 1, 84, 8400), dtype=np.float32),
            "attrs_too_few": np.zeros((1, 3, 8400), dtype=np.float32),
        }
        for name, value in cases.items():
            with self.subTest(case=name), self.assertRaises(PostprocessError):
                decode(value, (640, 640))

    def test_rejects_nonpositive_shapes(self):
        tensor = make_output([(320, 320, 100, 100)], [(0, 0.9)])
        for orig, size in (((0, 640), (640, 640)), ((640, 0), (640, 640)),
                           ((640, 640), (0, 640)), ((640, 640), (640, 0))):
            with self.subTest(orig=orig, size=size), self.assertRaises(PostprocessError):
                decode(tensor, orig, input_size=size)


class ThresholdTests(unittest.TestCase):
    def test_all_below_threshold_returns_empty(self):
        tensor = make_output([(320, 320, 100, 100)], [(0, 0.10)])
        self.assertEqual(decode(tensor, (640, 640), conf_thresh=0.25), [])

    def test_zero_scores_returns_empty(self):
        """INT8 量化损坏时的真实形态：坐标正常但分数全零。"""
        tensor = make_output([(320, 320, 100, 100)], [(0, 0.0)])
        self.assertEqual(decode(tensor, (640, 640)), [])

    def test_threshold_is_inclusive(self):
        tensor = make_output([(320, 320, 100, 100)], [(0, 0.25)])
        self.assertEqual(len(decode(tensor, (640, 640), conf_thresh=0.25)), 1)

    def test_max_detections_caps_and_keeps_highest(self):
        boxes = [(50 + i * 60, 320, 40, 40) for i in range(10)]
        scores = [(i % 80, 0.3 + i * 0.05) for i in range(10)]
        result = decode(make_output(boxes, scores), (640, 640), max_detections=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, sorted(result, key=lambda d: -d["score"]))
        self.assertAlmostEqual(result[0]["score"], 0.75, places=5)


class NmsTests(unittest.TestCase):
    def test_suppresses_overlapping_same_class(self):
        boxes = [(320, 320, 100, 100), (325, 325, 100, 100), (330, 330, 100, 100)]
        scores = [(0, 0.9), (0, 0.8), (0, 0.7)]
        result = decode(make_output(boxes, scores), (640, 640), nms_thresh=0.45)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["score"], 0.9, places=5)

    def test_keeps_overlapping_different_classes(self):
        """逐类别 NMS：重叠的不同类目标不应互相抑制。"""
        boxes = [(320, 320, 100, 100), (322, 322, 100, 100)]
        scores = [(0, 0.9), (5, 0.85)]
        result = decode(make_output(boxes, scores), (640, 640), nms_thresh=0.45)
        self.assertEqual(len(result), 2)
        self.assertEqual({d["class_id"] for d in result}, {0, 5})

    def test_keeps_distant_same_class(self):
        boxes = [(100, 100, 50, 50), (500, 500, 50, 50)]
        scores = [(0, 0.9), (0, 0.85)]
        self.assertEqual(len(decode(make_output(boxes, scores), (640, 640))), 2)

    def test_nms_handles_empty_and_zero_area(self):
        self.assertEqual(nms(np.zeros((0, 4)), np.zeros((0,)), 0.5), [])
        boxes = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]])
        self.assertEqual(len(nms(boxes, np.array([0.9, 0.8]), 0.5)), 2)


class CoordinateTests(unittest.TestCase):
    def test_scales_back_to_original_size(self):
        """预处理是直接 resize，x/y 独立缩放。"""
        tensor = make_output([(320, 320, 64, 64)], [(0, 0.9)])
        result = decode(tensor, (480, 640), input_size=(640, 640))[0]
        self.assertEqual(result["center"], (320, 240))
        x1, y1, x2, y2 = result["bbox"]
        self.assertAlmostEqual(x2 - x1, 64, delta=1)
        self.assertAlmostEqual(y2 - y1, 48, delta=1)

    def test_clamps_out_of_range_boxes(self):
        tensor = make_output([(10, 10, 400, 400)], [(0, 0.9)])
        x1, y1, x2, y2 = decode(tensor, (640, 640))[0]["bbox"]
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLess(x2, 640)
        self.assertLess(y2, 640)

    def test_drops_degenerate_after_clamp(self):
        """完全落在画面外的框 clamp 后退化，必须丢弃而不是返回零面积框。"""
        tensor = make_output([(-500, -500, 10, 10)], [(0, 0.9)])
        self.assertEqual(decode(tensor, (640, 640)), [])

    def test_center_is_inside_bbox(self):
        boxes = [(200, 300, 80, 120), (450, 150, 60, 60)]
        scores = [(1, 0.8), (2, 0.7)]
        for item in decode(make_output(boxes, scores), (426, 640)):
            x1, y1, x2, y2 = item["bbox"]
            cx, cy = item["center"]
            self.assertTrue(x1 <= cx <= x2 and y1 <= cy <= y2)


def unittest_transform(mode="letterbox", source=(3, 5), target=(8, 8)):
    """UNITTEST 元数据由真实预处理接口生成，不模拟实际检测结果。"""
    from src.vision.preprocess import Preprocessor
    with Preprocessor(output_size=target, geometry=mode) as pre:
        return pre.run_with_transform(np.zeros((*source, 3), dtype=np.uint8))[1]


def unittest_forward_box(box, transform):
    """UNITTEST 原图 xyxy -> 网络 cxcywh，用于正/逆变换闭环。"""
    x1, y1, x2, y2 = box
    sx, sy = transform["scale_x"], transform["scale_y"]
    left, top = transform["pad_left"], transform["pad_top"]
    return ((x1 + x2) * sx / 2 + left, (y1 + y2) * sy / 2 + top,
            (x2 - x1) * sx, (y2 - y1) * sy)


class TransformTests(unittest.TestCase):
    def test_letterbox_forward_inverse_on_odd_padding(self):
        transform = unittest_transform(source=(3, 5), target=(8, 8))
        self.assertEqual((transform["resized_width"], transform["resized_height"],
                          transform["pad_top"]), (8, 5, 1))
        self.assertNotEqual(transform["scale_x"], transform["scale_y"])
        box = (1, 0, 4, 2)
        tensor = make_output([unittest_forward_box(box, transform)], [(0, 0.9)])
        result = decode(tensor, (3, 5, 3), (8, 8), transform=transform)[0]
        self.assertEqual(result["bbox"], list(box))
        self.assertEqual(result["center"], (2, 1))

    def test_letterbox_non_square_target_and_horizontal_padding(self):
        transform = unittest_transform(source=(7, 3), target=(12, 10))
        box = (0, 1, 2, 5)
        tensor = make_output([unittest_forward_box(box, transform)], [(2, 0.85)])
        result = decode(tensor, (7, 3), (12, 10), transform=transform)[0]
        self.assertEqual(result["bbox"], list(box))
        self.assertEqual(transform["pad_left"], 2)

    def test_letterbox_half_up_not_ties_to_even(self):
        transform = unittest_transform(source=(5, 8), target=(4, 4))
        self.assertEqual(transform["resized_height"], 3)  # 5 * 0.5 = 2.5
        tensor = make_output([unittest_forward_box((1, 1, 6, 4), transform)], [(0, 0.9)])
        self.assertEqual(decode(tensor, (5, 8), (4, 4), transform=transform)[0]["bbox"],
                         [1, 1, 6, 4])

    def test_letterbox_clips_edges_and_drops_padding_only_box(self):
        transform = unittest_transform(source=(3, 5), target=(8, 8))
        tensor = make_output([(4, 4, 20, 20), (4, 0.25, 2, 0.25)],
                             [(0, 0.95), (1, 0.9)])
        result = decode(tensor, (3, 5), (8, 8), transform=transform)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bbox"], [0, 0, 4, 2])

    def test_split2_letterbox_in_both_orders(self):
        transform = unittest_transform(source=(9, 16), target=(10, 14))
        tensor = make_output([unittest_forward_box((1, 2, 12, 7), transform)], [(0, 0.9)])
        expected = decode(tensor, (9, 16), (10, 14), transform=transform)
        for parts in ([tensor[:, :4], tensor[:, 4:]], [tensor[:, 4:], tensor[:, :4]]):
            self.assertEqual(decode(parts, (9, 16), (10, 14), transform=transform), expected)

    def test_resize_transform_is_legacy_compatible(self):
        transform = unittest_transform("resize", source=(426, 640), target=(448, 640))
        tensor = make_output([(320, 224, 128, 96), (10, 20, 80, 100)],
                             [(0, 0.95), (1, 0.9)])
        self.assertEqual(decode(tensor, (426, 640), (448, 640)),
                         decode(tensor, (426, 640), (448, 640), transform=transform))

    def test_resize_preserves_legacy_float32_half_pixel_rounding(self):
        transform = unittest_transform("resize", source=(426, 640), target=(448, 640))
        for half_pixel in (9.5, 10.5, 11.5, 27.5, 123.5):
            tensor = make_output([(320, half_pixel * 448 / 426 + 1, 10, 2)], [(0, 0.9)])
            with self.subTest(half_pixel=half_pixel):
                self.assertEqual(decode(tensor, (426, 640), (448, 640)),
                                 decode(tensor, (426, 640), (448, 640), transform=transform))

    def test_nms_keeps_network_semantics_before_clipping(self):
        # 两框原网络 IoU 很小，clip 后相同；旧路径保留两个，不能改成 clip 后 NMS。
        transform = unittest_transform(source=(3, 5), target=(8, 8))
        boxes = [(-4, 0, 1, 2), (0, 0, 1, 2)]
        tensor = make_output([unittest_forward_box(box, transform) for box in boxes],
                             [(0, 0.9), (0, 0.8)])
        self.assertEqual(len(decode(tensor, (3, 5), (8, 8), transform=transform)), 2)

    def test_rejects_transform_schema_and_mode_errors(self):
        valid = unittest_transform()
        missing = dict(valid)
        del missing["pad_top"]
        tensor = make_output([], [])
        for transform in ([], "letterbox", missing, valid | {"extra": 0},
                          valid | {"mode": "auto"}, valid | {"mode": 1}):
            with self.subTest(transform=transform), self.assertRaises(PostprocessError):
                decode(tensor, (3, 5), (8, 8), transform=transform)

    def test_rejects_transform_shape_padding_and_scale_mismatches(self):
        valid = unittest_transform()
        tensor = make_output([], [])
        for key, value in (("source_width", 6), ("source_height", 4),
                           ("target_width", 9), ("target_height", 9),
                           ("resized_width", 7), ("resized_height", 4),
                           ("pad_left", 1), ("pad_top", 2), ("pad_top", -1),
                           ("scale_x", 1.5), ("scale_y", 1.6), ("scale_x", 0.0),
                           ("mode", "resize")):
            with self.subTest(key=key, value=value), self.assertRaises(PostprocessError):
                decode(tensor, (3, 5), (8, 8), transform=valid | {key: value})

    def test_rejects_noninteger_dimensions_and_nonfinite_transform(self):
        valid = unittest_transform()
        tensor = make_output([], [])
        integers = ("source_width", "source_height", "target_width", "target_height",
                    "resized_width", "resized_height", "pad_left", "pad_top")
        for key in integers:
            for value in (float(valid[key]), True, np.bool_(False), "3", np.nan, np.inf):
                with self.subTest(key=key, value=value), self.assertRaises(PostprocessError):
                    decode(tensor, (3, 5), (8, 8), transform=valid | {key: value})
        for key in ("scale_x", "scale_y"):
            for value in (np.nan, np.inf, -np.inf, True, "1.6", -1.0):
                with self.subTest(key=key, value=value), self.assertRaises(PostprocessError):
                    decode(tensor, (3, 5), (8, 8), transform=valid | {key: value})

    def test_rejects_invalid_resize_transform(self):
        valid = unittest_transform("resize")
        tensor = make_output([], [])
        for change in ({"pad_left": 1}, {"pad_top": 1}, {"resized_height": 7},
                       {"resized_width": 7}, {"scale_x": 1.5}):
            with self.subTest(change=change), self.assertRaises(PostprocessError):
                decode(tensor, (3, 5), (8, 8), transform=valid | change)

    def test_shapes_are_validated_even_when_no_detections(self):
        tensor = make_output([], [])
        for orig, size in (((0, 5), (8, 8)), ((3.0, 5), (8, 8)),
                           ((True, 5), (8, 8)), ((3, np.inf), (8, 8)),
                           ((3,), (8, 8)), ((3, 5), (8,)), ((3, 5), (8, 8, 3)),
                           (None, (8, 8)), ((3, 5), None)):
            with self.subTest(orig=orig, size=size), self.assertRaises(PostprocessError):
                decode(tensor, orig, size)

    def test_nonfinite_outputs_are_rejected_single_and_split2(self):
        for value in (np.nan, np.inf, -np.inf):
            for attribute in (0, 4):
                tensor = make_output([], [])
                tensor[0, attribute, -1] = value  # 低分 anchor 也不能藏 NaN/Inf
                for output in (tensor, [tensor[:, :4], tensor[:, 4:]]):
                    with self.subTest(value=value, attribute=attribute), \
                            self.assertRaises(PostprocessError):
                        decode(output, (3, 5), (8, 8))

    def test_rejects_non_numeric_outputs(self):
        for tensor in (np.full((1, 84, 100), "UNITTEST", dtype=object),
                       np.zeros((1, 84, 100), dtype=np.complex64)):
            with self.subTest(dtype=tensor.dtype), self.assertRaises(PostprocessError):
                decode(tensor, (3, 5), (8, 8))


class ClassNameTests(unittest.TestCase):
    def test_uses_coco_names_by_default(self):
        tensor = make_output([(320, 320, 100, 100)], [(50, 0.9)])
        self.assertEqual(decode(tensor, (640, 640))[0]["class_name"], COCO_CLASSES[50])

    def test_custom_names_supported(self):
        tensor = make_output([(320, 320, 100, 100)], [(1, 0.9)], num_classes=3)
        result = decode(tensor, (640, 640), class_names=("阀芯", "法兰", "轴承"))
        self.assertEqual(result[0]["class_name"], "法兰")

    def test_short_name_table_gets_placeholders(self):
        """类别表短于输出时补占位名，不能越界取值。"""
        tensor = make_output([(320, 320, 100, 100)], [(2, 0.9)], num_classes=3)
        result = decode(tensor, (640, 640), class_names=("a",))
        self.assertEqual(result[0]["class_name"], "class_2")


if __name__ == "__main__":
    unittest.main()
