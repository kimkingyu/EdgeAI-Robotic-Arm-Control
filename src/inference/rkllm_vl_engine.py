"""
RK3588 端侧视觉多模态大模型 (Qwen3-VL) 推理引擎

架构说明：
  VLM 在 RKNPU 上被拆成两个独立模型，通过 embedding 向量衔接：

    [图像] → 视觉编码器 (.rknn) → image_embed 浮点向量 ┐
                                                      ├→ 语言模型 (.rkllm) → 文本
    [文本 prompt] ────────────────────────────────────┘

  语言模型侧用 ctypes 直接绑定 librkllmrt.so（v1.2.3），
  无需编译 C++ 即可在 Python 中驱动，与项目现有工具链保持一致。

参考：airockchip/rknn-llm release-v1.2.3 examples/multimodal_model_demo
"""
import ctypes
import os
import time
from typing import Any, Callable, Dict, List, Optional

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


# ── rkllm.h v1.2.3 结构体映射 ────────────────────────────────────────────

RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

LLM_RUN_NORMAL = 0
LLM_RUN_WAITING = 1
LLM_RUN_FINISH = 2
LLM_RUN_ERROR = 3


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)), ("n_tokens", ctypes.c_size_t)]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_size_t)]


class _RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("_u", _RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [("save_prompt_cache", ctypes.c_int),
                ("prompt_cache_path", ctypes.c_char_p)]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)),
                ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)),
                ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [("prefill_time_ms", ctypes.c_float), ("prefill_tokens", ctypes.c_int),
                ("generate_time_ms", ctypes.c_float), ("generate_tokens", ctypes.c_int),
                ("memory_usage_mb", ctypes.c_float)]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


LLMResultCallback = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)


# ── 视觉编码器 ───────────────────────────────────────────────────────────

class VisionEncoder:
    """Qwen3-VL 视觉编码器：图像 → image_embed 向量"""

    def __init__(self, model_path: str, core_num: int = 3):
        self.model_path = model_path
        self.core_num = core_num
        self.rknn: Optional[Any] = None
        self.is_ready = False
        self.n_image_tokens = 0
        self.embed_size = 0
        self.input_size = (448, 448)

    def init(self) -> bool:
        if not HAS_RKNN:
            print("[VisionEnc] 未安装 rknn-toolkit-lite2")
            return False
        if not os.path.exists(self.model_path):
            print(f"[VisionEnc] 模型不存在: {self.model_path}")
            return False

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0:
            print("[VisionEnc] 加载视觉编码器失败")
            return False
        core_mask = (RKNNLite.NPU_CORE_0_1_2 if self.core_num >= 3
                     else RKNNLite.NPU_CORE_AUTO)
        if self.rknn.init_runtime(core_mask=core_mask) != 0:
            print("[VisionEnc] 视觉编码器运行时启动失败")
            return False

        self.is_ready = True
        print(f"[VisionEnc] 视觉编码器就绪: {os.path.basename(self.model_path)}")
        return True

    def encode(self, image) -> Optional["np.ndarray"]:
        """
        对单张图像编码，返回 (n_image_tokens, embed_dim) 的 float32 特征

        重要：Qwen3-VL 的视觉编码器输出 **4 个不同的特征层**（deepstack 结构），
        每层 (196, 2560)。实测必须沿最后一维拼接成 (196, 10240) 再喂给语言模型；
        若只取 out[0]，语言模型会在首个 token 后立即停止生成。
        """
        if not self.is_ready or self.rknn is None:
            return None
        outputs = self.rknn.inference(inputs=[image])
        if not outputs:
            return None

        if len(outputs) > 1:
            embed = np.concatenate(
                [np.asarray(o, dtype=np.float32) for o in outputs], axis=-1)
        else:
            embed = np.asarray(outputs[0], dtype=np.float32)

        if embed.ndim >= 2:
            self.n_image_tokens = int(embed.shape[-2])
            self.embed_size = int(embed.shape[-1])
        return np.ascontiguousarray(embed)

    def release(self):
        if self.rknn:
            self.rknn.release()
            self.rknn = None
        self.is_ready = False


# ── 视觉语言模型引擎 ─────────────────────────────────────────────────────

class QwenVLEngine:
    """
    Qwen3-VL 端侧视觉多模态推理引擎

    与纯文本引擎的区别：可直接把相机画面喂给模型，
    省去「YOLO 检测 → 坐标 → LLM 规划」的两段式转译。
    """

    def __init__(self, llm_model_path: str, vision_model_path: str,
                 lib_path: str = "/usr/lib/librkllmrt.so",
                 max_context_len: int = 4096, max_new_tokens: int = 512,
                 core_num: int = 3):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.lib_path = lib_path
        self.max_context_len = max_context_len
        self.max_new_tokens = max_new_tokens

        self.lib: Optional[ctypes.CDLL] = None
        self.handle = ctypes.c_void_p()
        self.vision = VisionEncoder(vision_model_path, core_num)
        self.is_ready = False
        self.is_mock = False

        self._buffer: List[str] = []
        self._first_token_t: Optional[float] = None
        self._start_t: Optional[float] = None
        self._token_count = 0
        self._callback_ref = None       # 必须持有引用，否则被 GC 回收导致段错误

        self.last_perf: Dict[str, Any] = {
            "ttft_ms": None, "tps": None, "total_tokens": None,
            "total_time_s": None, "is_mock": None,
        }

    # -- 生命周期 --------------------------------------------------------

    def init_engine(self) -> bool:
        if not os.path.exists(self.lib_path):
            print(f"[QwenVL] 运行时库不存在: {self.lib_path}")
            self.is_mock = True
            return False
        if not os.path.exists(self.llm_model_path):
            print(f"[QwenVL] 语言模型不存在: {self.llm_model_path}")
            self.is_mock = True
            return False

        if not self.vision.init():
            print("[QwenVL] 视觉编码器初始化失败")
            return False

        self.lib = ctypes.CDLL(self.lib_path)
        self.lib.rkllm_createDefaultParam.restype = RKLLMParam
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), LLMResultCallback
        ]
        self.lib.rkllm_init.restype = ctypes.c_int
        self.lib.rkllm_run.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p
        ]
        self.lib.rkllm_run.restype = ctypes.c_int
        self.lib.rkllm_destroy.argtypes = [ctypes.c_void_p]

        param = self.lib.rkllm_createDefaultParam()
        param.model_path = self.llm_model_path.encode("utf-8")
        param.max_context_len = self.max_context_len
        param.max_new_tokens = self.max_new_tokens
        param.top_k = 1                       # 贪心解码：工业指令解析要确定性，不要随机
        param.top_p = 0.9
        param.temperature = 0.1
        param.repeat_penalty = 1.1
        param.skip_special_token = True
        param.is_async = False
        # Qwen-VL 的图像占位符标记
        param.img_start = "<|vision_start|>".encode("utf-8")
        param.img_end = "<|vision_end|>".encode("utf-8")
        param.img_content = "<|image_pad|>".encode("utf-8")
        param.extend_param.base_domain_id = 1
        param.extend_param.embed_flash = 1
        param.extend_param.enabled_cpus_num = 4
        param.extend_param.enabled_cpus_mask = 0xF0   # 绑定 A76 大核 (CPU 4-7)

        self._callback_ref = LLMResultCallback(self._on_token)

        print(f"[QwenVL] 正在加载语言模型（{os.path.basename(self.llm_model_path)}），"
              f"约需 1~2 分钟 ...")
        t0 = time.time()
        ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param),
                                  self._callback_ref)
        if ret != 0:
            print(f"[QwenVL] 语言模型初始化失败，错误码: {ret}")
            return False

        self.is_ready = True
        print(f"[QwenVL] 视觉多模态引擎就绪，加载耗时 {time.time() - t0:.1f}s")
        return True

    def _on_token(self, result, userdata, state):
        if state == LLM_RUN_NORMAL and result:
            if self._first_token_t is None:
                self._first_token_t = time.perf_counter()
            self._token_count += 1
            txt = result.contents.text
            if txt:
                self._buffer.append(txt.decode("utf-8", errors="ignore"))
        return 0

    # -- 推理 ------------------------------------------------------------

    def generate(self, prompt: str, image=None,
                 callback: Optional[Callable[[str], None]] = None) -> str:
        """
        执行推理。传入 image 则走多模态路径，否则退化为纯文本。
        image 需为 uint8 的 (1,H,W,3) RGB 数组。
        """
        if not self.is_ready or self.lib is None:
            print("[QwenVL] 引擎未就绪")
            return ""

        self._buffer.clear()
        self._first_token_t = None
        self._token_count = 0
        self._start_t = time.perf_counter()

        rkllm_input = RKLLMInput()
        ctypes.memset(ctypes.byref(rkllm_input), 0, ctypes.sizeof(RKLLMInput))
        rkllm_input.role = b"user"
        rkllm_input.enable_thinking = False

        img_embed_buf = None
        if image is not None:
            embed = self.vision.encode(image)
            if embed is None:
                print("[QwenVL] 图像编码失败，退化为纯文本推理")
            else:
                # prompt 中必须含 <image> 标记，运行时据此切换到多模态分支
                if "<image>" not in prompt:
                    prompt = "<image>" + prompt
                flat = embed.reshape(-1)
                img_embed_buf = (ctypes.c_float * len(flat))(*flat.tolist())
                rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL
                rkllm_input._u.multimodal_input.prompt = prompt.encode("utf-8")
                rkllm_input._u.multimodal_input.image_embed = img_embed_buf
                rkllm_input._u.multimodal_input.n_image_tokens = self.vision.n_image_tokens
                rkllm_input._u.multimodal_input.n_image = 1
                rkllm_input._u.multimodal_input.image_width = self.vision.input_size[0]
                rkllm_input._u.multimodal_input.image_height = self.vision.input_size[1]

        if img_embed_buf is None:
            rkllm_input.input_type = RKLLM_INPUT_PROMPT
            rkllm_input._u.prompt_input = prompt.encode("utf-8")

        infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        infer_param.mode = 0
        infer_param.keep_history = 0

        self.lib.rkllm_run(self.handle, ctypes.byref(rkllm_input),
                           ctypes.byref(infer_param), None)

        total = time.perf_counter() - self._start_t
        ttft = ((self._first_token_t - self._start_t) * 1000.0
                if self._first_token_t else None)
        decode_s = total - (ttft / 1000.0) if ttft else total
        tps = ((self._token_count - 1) / decode_s
               if decode_s > 0 and self._token_count > 1 else None)

        self.last_perf = {
            "ttft_ms": round(ttft, 2) if ttft else None,
            "tps": round(tps, 2) if tps else None,
            "total_tokens": self._token_count,
            "total_time_s": round(total, 3),
            "is_mock": False,
        }

        out = "".join(self._buffer)
        if callback:
            callback(out)
        return out

    def get_benchmark_report(self) -> str:
        p = self.last_perf
        if p.get("ttft_ms") is None:
            return "[Benchmark] 尚未执行推理或未采集到 token。"
        return (f"[Benchmark] TTFT {p['ttft_ms']} ms | "
                f"{p['tps']} tok/s | {p['total_tokens']} tokens | "
                f"{p['total_time_s']} s")

    def release(self):
        if self.lib and self.handle:
            self.lib.rkllm_destroy(self.handle)
            self.handle = ctypes.c_void_p()
        self.vision.release()
        self.is_ready = False
