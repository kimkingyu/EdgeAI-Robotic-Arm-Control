#!/usr/bin/env python3
"""统一预处理后端：OpenCV 为默认，MLIR 为显式可选。

MLIR 后端不会在失败时悄悄退回 OpenCV —— 缺库、架构不符、输入不受支持都直接
报错。否则就会把 OpenCV 的耗时挂在 MLIR 名下。

语义：uint8 BGR H*W*3 输入 -> 连续 uint8 RGB HWC 输出，双线性 half-pixel、
边界复制。不做 /255、不做 mean/std、不转 NCHW。
"""
import ctypes as C
import hashlib
import json
import platform
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - 由调用方显式处理
    np = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

ABI_VERSION = 1
SEMANTIC_VERSION = b"linear_half_pixel_q11_v1"
MAX_DIMENSION = 8192

STATUS_NAMES = {
    0: "OK", 1: "INVALID_ARGUMENT", 2: "INVALID_DIMENSIONS", 3: "INVALID_STRIDE",
    4: "INVALID_CAPACITY", 5: "OVERLAP", 6: "CONTEXT_ALLOCATION_FAILED", 7: "AUDIT_FAILED",
}


class PreprocessError(RuntimeError):
    """预处理失败的基类。"""


class BackendUnavailable(PreprocessError):
    """后端无法加载：缺库、架构不符、ABI 不符。"""


class UnsupportedInput(PreprocessError):
    """输入不满足冻结的数值契约。"""


class _Spec(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("reserved", C.c_uint32),
                ("input_height", C.c_uint64), ("input_width", C.c_uint64),
                ("output_height", C.c_uint64), ("output_width", C.c_uint64)]


def _require_numpy():
    if np is None:
        raise BackendUnavailable("需要 numpy，但当前解释器未安装")


def buffer_span(array):
    """返回从 array 数据指针起可安全访问的字节数；无法确认时拒绝。

    NumPy 的 nbytes 是逻辑大小，对带 row stride 的 ROI 不等于实际跨度，
    所以这里沿 base 链找到真正的 owner，不靠虚报容量绕过校验。
    """
    _require_numpy()
    pointer = array.__array_interface__["data"][0]
    owner, offset = array, 0
    while True:
        base = owner.base
        if base is None:
            if not owner.flags["C_CONTIGUOUS"]:
                raise UnsupportedInput("无法确认非连续数组的底层可访问范围")
            root_pointer, root_bytes = owner.__array_interface__["data"][0], owner.nbytes
            break
        if isinstance(base, np.ndarray):
            owner = base
            continue
        try:
            flat = np.frombuffer(base, dtype=np.uint8)
        except (TypeError, ValueError, BufferError) as error:
            raise UnsupportedInput("无法确认底层缓冲区的可访问范围: " + str(error))
        root_pointer, root_bytes = flat.__array_interface__["data"][0], flat.nbytes
        break
    offset = pointer - root_pointer
    if offset < 0 or offset > root_bytes:
        raise UnsupportedInput("数组数据指针落在其 owner 缓冲区之外")
    return root_bytes - offset


def check_source(array):
    """核对 BGR 输入契约，返回 (height, width, row_stride, span)。"""
    _require_numpy()
    if not isinstance(array, np.ndarray):
        raise UnsupportedInput("输入必须是 numpy 数组")
    if array.dtype != np.uint8:
        raise UnsupportedInput("输入 dtype 必须是 uint8，收到 " + str(array.dtype))
    if array.ndim != 3 or array.shape[2] != 3:
        raise UnsupportedInput("输入必须是 H*W*3，收到 " + str(array.shape))
    height, width = int(array.shape[0]), int(array.shape[1])
    if height < 1 or width < 1:
        raise UnsupportedInput("输入尺寸不能为空")
    if height > MAX_DIMENSION or width > MAX_DIMENSION:
        raise UnsupportedInput("输入尺寸超过 %d" % MAX_DIMENSION)
    row_stride, pixel_stride, channel_stride = (int(s) for s in array.strides)
    if channel_stride != 1 or pixel_stride != 3:
        raise UnsupportedInput("像素内通道必须连续（strides 需为 (row, 3, 1)），收到 " + str(array.strides))
    if row_stride < width * 3:
        raise UnsupportedInput("row stride 必须为正且不小于 width*3，收到 " + str(row_stride))
    required = (height - 1) * row_stride + width * 3
    span = buffer_span(array)
    if span < required:
        raise UnsupportedInput("可访问范围 %d 字节小于所需 %d 字节" % (span, required))
    return height, width, row_stride, span


class OpenCVBackend:
    """现有流水线的两步：resize 后 BGR->RGB。默认后端。"""

    name = "opencv"

    def __init__(self, output_size):
        if cv2 is None:
            raise BackendUnavailable("OpenCV 后端需要 cv2")
        _require_numpy()
        self.output_size = output_size
        self.details = {"opencv_version": cv2.__version__, "interpolation": "INTER_LINEAR"}

    def run(self, source, out):
        height, width = self.output_size
        resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=out)

    def close(self):
        pass


class MlirBackend:
    """显式可选的 AOT 内核。加载失败即报错，不退回 OpenCV。"""

    name = "mlir"

    def __init__(self, output_size, library, manifest=None):
        _require_numpy()
        if library is None:
            raise BackendUnavailable("mlir 后端必须给出库路径")
        self.output_size = output_size
        self.library_path = Path(library)
        if not self.library_path.is_file():
            raise BackendUnavailable("找不到 MLIR 预处理库: " + str(self.library_path))
        try:
            self.lib = C.CDLL(str(self.library_path.resolve(strict=True)))
        except OSError as error:
            raise BackendUnavailable("无法加载 %s: %s（架构：%s）"
                                     % (self.library_path, error, platform.machine()))
        self._bind()
        if self.lib.edge_preprocess_abi_version() != ABI_VERSION:
            raise BackendUnavailable("ABI 版本不符，期望 %d" % ABI_VERSION)
        semantic = self.lib.edge_preprocess_semantic_version()
        if semantic != SEMANTIC_VERSION:
            raise BackendUnavailable("语义版本不符: " + repr(semantic))
        self.variant = self.lib.edge_preprocess_variant().decode()
        self.details = {"library": str(self.library_path), "variant": self.variant,
                        "semantic_version": SEMANTIC_VERSION.decode(),
                        "abi_version": ABI_VERSION, "machine": platform.machine(),
                        "library_sha256": hashlib.sha256(self.library_path.read_bytes()).hexdigest()}
        if manifest is not None:
            self.details["manifest"] = self._verify_manifest(Path(manifest))
        self._contexts = {}
        self.last_context_create_ns = None

    def _bind(self):
        for name, restype in (("abi_version", C.c_uint32), ("allocation_audit_enabled", C.c_uint32),
                              ("semantic_version", C.c_char_p), ("variant", C.c_char_p)):
            try:
                fn = getattr(self.lib, "edge_preprocess_" + name)
            except AttributeError:
                raise BackendUnavailable("库缺少符号 edge_preprocess_" + name)
            fn.argtypes, fn.restype = [], restype
        self._create = self.lib.edge_preprocess_create
        self._create.argtypes = [C.POINTER(_Spec), C.POINTER(C.c_void_p)]
        self._create.restype = C.c_int
        self._run = self.lib.edge_preprocess_run
        self._run.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_uint64,
                              C.c_void_p, C.c_uint64, C.c_uint64]
        self._run.restype = C.c_int
        self._destroy = self.lib.edge_preprocess_destroy
        self._destroy.argtypes, self._destroy.restype = [C.c_void_p], None

    def _verify_manifest(self, path):
        """按 manifest 核对产物，不靠文件名猜测兼容性。"""
        if not path.is_file():
            raise BackendUnavailable("找不到 manifest: " + str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "built":
            raise BackendUnavailable("manifest 未标记 built")
        resolved = self.library_path.resolve()
        for entry in data.get("variants", []):
            if entry.get("status") != "built":
                continue
            if (path.parent / entry["library"]).resolve() != resolved:
                continue
            if tuple(entry["profile"]) != tuple(self.output_size):
                raise BackendUnavailable("manifest 中该库的 profile 为 %s，与请求的 %s 不符"
                                         % (entry["profile"], list(self.output_size)))
            recorded = entry["artifacts"][entry["library"]]["sha256"]
            if recorded != self.details["library_sha256"]:
                raise BackendUnavailable("库的 SHA256 与 manifest 记录不符")
            return {"path": str(path), "variant": entry["variant"],
                    "requires_packed_source": bool(entry.get("requires_packed_source"))}
        raise BackendUnavailable("manifest 中没有这个库的 built 记录")

    def _context(self, height, width):
        """输入尺寸变化时新建 context；创建耗时单列，不藏进第一帧。"""
        key = (height, width)
        existing = self._contexts.get(key)
        if existing is not None:
            self.last_context_create_ns = None
            return existing
        spec = _Spec(ABI_VERSION, 0, height, width, self.output_size[0], self.output_size[1])
        handle = C.c_void_p()
        start = time.perf_counter_ns()
        status = self._create(C.byref(spec), C.byref(handle))
        self.last_context_create_ns = time.perf_counter_ns() - start
        if status != 0 or not handle.value:
            raise UnsupportedInput("edge_preprocess_create 失败: %s（输入 %dx%d，输出 %s）"
                                   % (STATUS_NAMES.get(status, status), height, width,
                                      list(self.output_size)))
        self._contexts[key] = handle
        return handle

    def run(self, source, out):
        height, width, row_stride, span = check_source(source)
        context = self._context(height, width)
        status = self._run(context, source.ctypes.data, span, row_stride,
                           out.ctypes.data, out.nbytes, self.output_size[1] * 3)
        if status != 0:
            raise UnsupportedInput("edge_preprocess_run 失败: " + STATUS_NAMES.get(status, str(status)))

    def close(self):
        for handle in self._contexts.values():
            self._destroy(handle)
        self._contexts.clear()


class Preprocessor:
    """把一帧 BGR 图变成模型要的 RGB HWC uint8。

    单个实例不是线程安全的：一个 context 只允许顺序使用。每个 worker 建自己的
    实例，输出缓冲各自持有，直到 RKNN 消费完才可复用。
    """

    def __init__(self, output_size=(640, 640), backend="opencv", library=None, manifest=None):
        _require_numpy()
        height, width = (int(v) for v in output_size)
        if height < 1 or width < 1 or height > MAX_DIMENSION or width > MAX_DIMENSION:
            raise ValueError("非法输出尺寸: " + str(output_size))
        self.output_size = (height, width)
        if backend == "opencv":
            self.backend = OpenCVBackend(self.output_size)
        elif backend == "mlir":
            self.backend = MlirBackend(self.output_size, library, manifest)
        else:
            raise ValueError("未知后端 %r，可选 opencv / mlir" % (backend,))
        self.backend_name = self.backend.name
        self.last_timings = {}

    def describe(self):
        info = {"backend": self.backend_name, "output_size": list(self.output_size)}
        info.update(self.backend.details)
        return info

    def new_output(self):
        """新建一块调用方独占的输出缓冲。"""
        return np.empty((self.output_size[0], self.output_size[1], 3), dtype=np.uint8)

    def run(self, source, out=None):
        """返回 RGB HWC uint8。out 为 None 时新分配；分配与内核耗时分列。"""
        allocate_ns = 0
        if out is None:
            start = time.perf_counter_ns()
            out = self.new_output()
            allocate_ns = time.perf_counter_ns() - start
        else:
            if not isinstance(out, np.ndarray) or out.dtype != np.uint8:
                raise UnsupportedInput("输出缓冲必须是 uint8 numpy 数组")
            if out.shape != (self.output_size[0], self.output_size[1], 3):
                raise UnsupportedInput("输出形状必须是 %s，收到 %s"
                                       % ((self.output_size[0], self.output_size[1], 3), out.shape))
            if not out.flags["C_CONTIGUOUS"] or not out.flags["WRITEABLE"]:
                raise UnsupportedInput("输出缓冲必须连续且可写")
        start = time.perf_counter_ns()
        self.backend.run(source, out)
        call_ns = time.perf_counter_ns() - start
        self.last_timings = {
            "backend": self.backend_name, "allocate_ns": allocate_ns, "call_ns": call_ns,
            "total_ns": allocate_ns + call_ns,
            "context_create_ns": getattr(self.backend, "last_context_create_ns", None),
        }
        return out

    def run_batch1(self, source, out=None):
        """返回 [1,H,W,3]。连续输出上 expand_dims 是视图，不是整图拷贝。"""
        return np.expand_dims(self.run(source, out), 0)

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def from_config(config, output_size):
    """从配置里取后端设置；缺省即 OpenCV。

    AOT 内核的输出尺寸是编译期固定的，一个 .so 只服务一个 profile。所以 mlir
    后端的库路径按 profile 查找 `libraries`（键形如 "640x640"），找不到就报错，
    不会拿错尺寸的库硬凑。
    """
    section = (config or {}).get("preprocess", {}) if hasattr(config, "get") else {}
    height, width = (int(v) for v in output_size)
    backend = section.get("backend", "opencv")
    library = section.get("library")
    manifest = section.get("manifest")
    libraries = section.get("libraries") or {}
    key = "%dx%d" % (height, width)
    if key in libraries:
        entry = libraries[key]
        if isinstance(entry, dict):
            library, manifest = entry.get("library", library), entry.get("manifest", manifest)
        else:
            library = entry
    elif backend == "mlir" and libraries:
        raise BackendUnavailable("preprocess.libraries 中没有 profile %s 的库" % key)
    return Preprocessor(output_size=(height, width), backend=backend,
                        library=library, manifest=manifest)
