#!/usr/bin/env python3
"""YOLOv8 后处理解码的边界测试。

构造的张量只验证解码逻辑，不构成检测精度证据 —— 真实模型验证见
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
    """造一个 [1, 4+num_classes, num_anchors] 的输出。

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
