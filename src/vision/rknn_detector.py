import os
from typing import List, Dict, Any, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN_LITE = True
except ImportError:
    HAS_RKNN_LITE = False


class RKNNObjectDetector:
    """RK3588 板载 NPU 目标检测器 (基于 rknn-toolkit-lite2)"""

    def __init__(self, model_path: str, target_size=(640, 640), conf_thresh=0.5, nms_thresh=0.45):
        self.model_path = model_path
        self.target_size = target_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.rknn = None
        self.is_loaded = False

    def init_model(self) -> bool:
        if not HAS_RKNN_LITE:
            print("[RKNN] 提示: 当前系统未安装 rknnlite 库，将切换至 Mock 模拟推理模式")
            return True

        if not os.path.exists(self.model_path):
            print(f"[RKNN] 错误: 模型文件不存在 -> {self.model_path}")
            return False

        print(f"[RKNN] 正在加载 RKNN 模型: {self.model_path} ...")
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            print(f"[RKNN] 加载模型失败，返回值: {ret}")
            return False

        # 初始化运行时环境：RK3588 支持多核调度 RKNNLite.NPU_CORE_0_1_2
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            print(f"[RKNN] 初始化 NPU 运行时失败，返回值: {ret}")
            return False

        self.is_loaded = True
        print("[RKNN] NPU 运行时初始化成功 (RK3588 6TOPS 加速)")
        return True

    def preprocess(self, img: Any) -> Any:
        """保持比例或者直接缩放至网络输入尺寸"""
        if not HAS_CV2 or img is None:
            return img
        resized = cv2.resize(img, self.target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def detect(self, frame: Any) -> List[Dict[str, Any]]:
        """输入原始画面，返回检测框及中心像素点坐标"""
        if not self.is_loaded or self.rknn is None:
            # Mock 模式：若检测到画面，返回一个居中的虚拟目标用于跑通控制回路
            w, h = 640, 480
            if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
                h, w = frame.shape[:2]
            return [{
                "class_name": "mock_target",
                "score": 0.99,
                "bbox": [int(w * 0.4), int(h * 0.4), int(w * 0.6), int(h * 0.6)],
                "center": (int(w * 0.5), int(h * 0.5))
            }]

        input_data = self.preprocess(frame)
        input_data = np.expand_dims(input_data, axis=0)

        # 送入 NPU 执行推理
        outputs = self.rknn.inference(inputs=[input_data])

        # 此处根据具体检测模型（如 YOLOv8/YOLOv5/RT-DETR）进行后处理解码
        # 后续接入实际 .rknn 权重时填入对应解码逻辑
        detections = self._postprocess(outputs, frame.shape)
        return detections

    def _postprocess(self, outputs, orig_shape) -> List[Dict[str, Any]]:
        # 预留后处理通道
        return []

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            print("[RKNN] 释放 NPU 资源")
