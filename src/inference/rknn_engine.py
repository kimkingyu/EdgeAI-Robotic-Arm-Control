import os
import time
from typing import List, Dict, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False


class RKNNMultiCoreVisionEngine:
    """RK3588 3核 NPU (Core 0,1,2) 异步负载均衡视觉推理加速引擎"""

    def __init__(self, model_path: str, core_mode: str = "auto"):
        self.model_path = model_path
        self.core_mode = core_mode
        self.rknn = None
        self.is_ready = False

    def init_engine(self) -> bool:
        if not HAS_RKNN:
            print("[RKNN-Vision] 当前环境无 rknnlite，以 Mock 模式运行")
            self.is_ready = True
            return True

        if not os.path.exists(self.model_path):
            print(f"[RKNN-Vision] 模型不存在: {self.model_path}")
            return False

        print(f"[RKNN-Vision] 加载模型: {self.model_path} ...")
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0:
            return False

        # 多核算力调度策略
        if self.core_mode == "all":
            core_mask = RKNNLite.NPU_CORE_0_1_2  # 榨干 3 核满血 6 TOPS
        elif self.core_mode == "core0":
            core_mask = RKNNLite.NPU_CORE_0
        elif self.core_mode == "core1":
            core_mask = RKNNLite.NPU_CORE_1
        else:
            core_mask = RKNNLite.NPU_CORE_AUTO

        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            print(f"[RKNN-Vision] NPU 运行时启动失败: {ret}")
            return False

        self.is_ready = True
        print(f"[RKNN-Vision] 视觉引擎就绪 (调度模式: {self.core_mode})")
        return True

    def infer(self, input_data) -> List[Any]:
        if not HAS_RKNN or self.rknn is None:
            time.sleep(0.01) # 模拟 10ms NPU 延时
            return []
        return self.rknn.inference(inputs=[input_data])

    def release(self):
        if self.rknn:
            self.rknn.release()
