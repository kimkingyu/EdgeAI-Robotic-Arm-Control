#!/usr/bin/env python3
"""YOLO 双输出 ONNX 导出器的结构测试。

使用真实 ONNX 图检查输出契约；不会导出 RKNN，也不构成模型精度测试。
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.split_yolo_output import split_model


class SplitOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ.get(
            "EDGEAI_PROJECT_ROOT", "/home/orangepi/project/EdgeAI-Robotic-Arm-Control"))
        cls.source = cls.root / "models/weights/yolov8n_op19.onnx"
        if not cls.source.is_file():
            raise unittest.SkipTest("板端真实 ONNX 不存在：" + str(cls.source))

    def test_real_graph_exposes_two_independent_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "split.onnx"
            result = split_model(self.source, target)
            self.assertTrue(target.is_file())
            self.assertEqual([item["shape"] for item in result["split_outputs"]],
                             [[1, 4, 8400], [1, 80, 8400]])
            self.assertEqual([item["role"] for item in result["split_outputs"]],
                             ["coordinates", "class_scores"])

    def test_missing_source_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            split_model(self.root / "models/weights/not-found.onnx",
                        self.root / "build/not-found.onnx")


if __name__ == "__main__":
    unittest.main()
