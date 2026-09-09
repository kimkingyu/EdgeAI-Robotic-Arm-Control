"""
RK3588 NPU 视觉推理引擎

设计依据（板端实测结论，见 docs/benchmarks/NPU_BENCHMARK_REPORT.md）：
  RKNNLite.inference() 是同步阻塞调用。单实例即便绑定 NPU_CORE_0_1_2，
  同一时刻仍只有一路任务在跑，实测三核相较单核仅 1.03x —— 近乎无加速。
  真正榨干 6 TOPS 算力的方式是「每核心一个独立 RKNNLite 实例 + 多线程并发」，
  实测 3 路并发达成 2.71x 线性加速（38.5 FPS -> 104.5 FPS）。

因此本模块提供两级接口：
  - RKNNMultiCoreVisionEngine : 单实例引擎，用于低时延单路场景
  - RKNNParallelVisionPool    : 三核实例池，用于高吞吐多路/流水线场景
"""
import os
import queue
import threading
import time
from typing import Any, List, Optional

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


CORE_MASK_MAP = {
    "auto": "NPU_CORE_AUTO",
    "core0": "NPU_CORE_0",
    "core1": "NPU_CORE_1",
    "core2": "NPU_CORE_2",
    "all": "NPU_CORE_0_1_2",
}


class RKNNMultiCoreVisionEngine:
    """单实例 NPU 推理引擎（适用于单路低时延场景）"""

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
            print("[RKNN-Vision] 模型加载失败")
            return False

        attr = CORE_MASK_MAP.get(self.core_mode, "NPU_CORE_AUTO")
        if self.rknn.init_runtime(core_mask=getattr(RKNNLite, attr)) != 0:
            print(f"[RKNN-Vision] NPU 运行时启动失败 (core={attr})")
            return False

        self.is_ready = True
        print(f"[RKNN-Vision] 视觉引擎就绪 (调度模式: {self.core_mode} / {attr})")
        return True

    def infer(self, input_data) -> List[Any]:
        if not HAS_RKNN or self.rknn is None:
            time.sleep(0.023)  # 模拟板端实测 23ms NPU 时延
            return []
        return self.rknn.inference(inputs=[input_data])

    def release(self):
        if self.rknn:
            self.rknn.release()
            self.rknn = None
        self.is_ready = False


class _CoreWorker(threading.Thread):
    """绑定到单一 NPU 物理核心的常驻推理工作线程"""

    def __init__(self, model_path: str, core_attr: str,
                 task_q: "queue.Queue", result_q: "queue.Queue"):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.core_attr = core_attr
        self.task_q = task_q
        self.result_q = result_q
        self.rknn: Optional[Any] = None
        self.ready = threading.Event()
        self.error: Optional[str] = None

    def run(self):
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0:
            self.error = f"{self.core_attr}: load_rknn failed"
            self.ready.set()
            return
        if self.rknn.init_runtime(core_mask=getattr(RKNNLite, self.core_attr)) != 0:
            self.error = f"{self.core_attr}: init_runtime failed"
            self.ready.set()
            return
        self.ready.set()

        while True:
            task = self.task_q.get()
            if task is None:                      # 毒丸信号，优雅退出
                self.task_q.task_done()
                break
            seq, data = task
            t0 = time.perf_counter()
            try:
                out = self.rknn.inference(inputs=[data])
                cost = (time.perf_counter() - t0) * 1000.0
                self.result_q.put((seq, out, cost, self.core_attr))
            except Exception as e:
                self.result_q.put((seq, None, -1.0, f"{self.core_attr}:{e}"))
            finally:
                self.task_q.task_done()

        try:
            self.rknn.release()
        except Exception:
            pass


class RKNNParallelVisionPool:
    """
    三核 NPU 并发推理池

    每个物理核心持有独立 RKNNLite 实例，通过任务队列并发喂帧，
    突破单实例同步阻塞瓶颈，实测系统吞吐可达单核的 2.71 倍。
    """

    def __init__(self, model_path: str, num_cores: int = 3, queue_size: int = 8):
        self.model_path = model_path
        self.num_cores = max(1, min(3, num_cores))
        self.task_q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self.result_q: "queue.Queue" = queue.Queue()
        self.workers: List[_CoreWorker] = []
        self.is_ready = False
        self._seq = 0

    def init_pool(self) -> bool:
        if not HAS_RKNN:
            print("[RKNN-Pool] 当前环境无 rknnlite，以 Mock 模式运行")
            self.is_ready = True
            return True
        if not os.path.exists(self.model_path):
            print(f"[RKNN-Pool] 模型不存在: {self.model_path}")
            return False

        core_attrs = ["NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2"][: self.num_cores]
        for attr in core_attrs:
            w = _CoreWorker(self.model_path, attr, self.task_q, self.result_q)
            w.start()
            self.workers.append(w)

        for w in self.workers:                    # 等待所有核心完成运行时初始化
            w.ready.wait(timeout=30)
            if w.error:
                print(f"[RKNN-Pool] 核心初始化失败 -> {w.error}")
                self.shutdown()
                return False

        self.is_ready = True
        print(f"[RKNN-Pool] {self.num_cores} 核 NPU 并发推理池就绪 ({', '.join(core_attrs)})")
        return True

    def submit(self, data) -> int:
        """投递一帧待推理数据，返回该帧的序号（非阻塞乱序返回）"""
        seq = self._seq
        self._seq += 1
        if not HAS_RKNN or not self.workers:
            self.result_q.put((seq, [], 23.0, "mock"))
            return seq
        self.task_q.put((seq, data))
        return seq

    def fetch(self, timeout: float = 5.0):
        """取回一个已完成的推理结果 (seq, outputs, cost_ms, core)"""
        try:
            return self.result_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self):
        for _ in self.workers:
            self.task_q.put(None)
        for w in self.workers:
            w.join(timeout=5)
        self.workers.clear()
        self.is_ready = False
