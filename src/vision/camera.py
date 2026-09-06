try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import threading
import time
from typing import Optional, Tuple, Any


class USBCamera:
    """USB/免驱摄像头多线程读取器，防止单线程处理慢导致丢帧和画面延迟堆积"""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.running = False
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if not HAS_CV2:
            print("[Camera] 提示: 当前未安装 OpenCV (cv2)，无法启动真实视频捕获")
            return False

        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            print(f"[Camera] 无法打开摄像头设备: /dev/video{self.device_id}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print(f"[Camera] 摄像头 /dev/video{self.device_id} 启动成功 ({self.width}x{self.height} @ {self.fps}fps)")
        return True

    def _update(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Any]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy() if hasattr(self.frame, "copy") else self.frame

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[Camera] 摄像头已关闭")
