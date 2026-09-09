import time
import os
from typing import Optional, Callable

try:
    from rkllm.api import RKLLM
    HAS_RKLLM = True
except ImportError:
    HAS_RKLLM = False


class RKLLMInferenceEngine:
    """RK3588 板载端侧大模型 (Qwen) 推理加速引擎 (基于瑞芯微 RKLLM 运行时)"""

    def __init__(self, model_path: str, max_context_len: int = 1024, max_new_tokens: int = 256):
        self.model_path = model_path
        self.max_context_len = max_context_len
        self.max_new_tokens = max_new_tokens
        self.rkllm_handle = None
        self.is_ready = False
        self.is_mock = False      # True 表示未加载真实模型，输出为预置文本
        self.last_perf = {
            "ttft_ms": None,      # 首 Token 时延 (Prefill)，未真实推理时为 None
            "tps": None,          # 每秒生成 Token 吞吐 (Decode)
            "total_tokens": None,
            "total_time_s": None,
            "is_mock": None,      # True 表示该组数据来自 Mock，不可用于性能结论
        }

    def load_model(self) -> bool:
        """加载量化后的 .rkllm 格式大模型"""
        if not HAS_RKLLM:
            print("[RKLLM] 警告: 未安装 rkllm 运行时，进入 Mock 模式。")
            print("        Mock 模式仅用于打通上层链路，其输出为预置文本，"
                  "不代表任何真实模型能力，严禁据此得出性能或准确率结论。")
            self.is_ready = True
            self.is_mock = True
            return True

        if not os.path.exists(self.model_path):
            print(f"[RKLLM] 错误: 模型文件不存在 -> {self.model_path}")
            return False

        print(f"[RKLLM] 正在将大模型加载至 RK3588 NPU (6 TOPS)...")
        start_t = time.time()
        try:
            self.rkllm_handle = RKLLM()
            ret = self.rkllm_handle.init(
                model_path=self.model_path,
                max_context_len=self.max_context_len,
                max_new_tokens=self.max_new_tokens
            )
            if ret != 0:
                print(f"[RKLLM] 模型初始化失败，错误码: {ret}")
                return False

            load_time = time.time() - start_t
            self.is_ready = True
            print(f"[RKLLM] 大模型加载成功！耗时: {load_time:.2f}s, 最大上下文: {self.max_context_len}")
            return True
        except Exception as e:
            print(f"[RKLLM] 初始化异常: {e}")
            return False

    def generate(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """执行流式/阻塞端侧推理，并统计 TTFT 与吞吐性能"""
        if not self.is_ready:
            print("[RKLLM] 错误: 模型尚未就绪")
            return ""

        if not HAS_RKLLM or self.rkllm_handle is None:
            # 纯仿真/模拟推理
            time.sleep(0.12) # 模拟 NPU 延迟
            import re
            u_match = re.search(r"<\|im_start\|>user\n([\s\S]*?)<\|im_end\|>", prompt)
            user_text = u_match.group(1).lower() if u_match else prompt.lower()

            if "急停" in user_text or "紧急" in user_text or "stop" in user_text:
                mock_res = (
                    '{\n'
                    '  "intent": "触发紧急安全制动",\n'
                    '  "priority": 0,\n'
                    '  "actions": [\n'
                    '    {"action": "emergency_stop", "params": {}}\n'
                    '  ]\n'
                    '}'
                )
            elif "巡检" in user_text or "inspect" in user_text or "检查" in user_text:
                mock_res = (
                    '{\n'
                    '  "intent": "工件外观缺陷巡检与分拣",\n'
                    '  "priority": 1,\n'
                    '  "actions": [\n'
                    '    {"action": "move_safe", "params": {}},\n'
                    '    {"action": "inspect", "params": {"target_name": "flange"}},\n'
                    '    {"action": "pick", "params": {"target_name": "flange", "x": 170.0, "y": -10.0, "z": 30.0}},\n'
                    '    {"action": "place", "params": {"x": 200.0, "y": 60.0, "z": 30.0}},\n'
                    '    {"action": "move_safe", "params": {}}\n'
                    '  ]\n'
                    '}'
                )
            elif "原位" in user_text or "收回" in user_text or "复位" in user_text:
                mock_res = (
                    '{\n'
                    '  "intent": "机械臂复位至安全停泊位",\n'
                    '  "priority": 2,\n'
                    '  "actions": [\n'
                    '    {"action": "move_safe", "params": {}}\n'
                    '  ]\n'
                    '}'
                )
            else:
                mock_res = (
                    '{\n'
                    '  "intent": "工业工件自主识别与抓取",\n'
                    '  "priority": 1,\n'
                    '  "actions": [\n'
                    '    {"action": "move_safe", "params": {}},\n'
                    '    {"action": "pick", "params": {"target_name": "valve_core", "x": 160.0, "y": 10.0, "z": 30.0}},\n'
                    '    {"action": "place", "params": {"x": 200.0, "y": 60.0, "z": 30.0}},\n'
                    '    {"action": "move_safe", "params": {}}\n'
                    '  ]\n'
                    '}'
                )
            # Mock 模式不产出任何性能数字：此处没有真实推理发生，
            # 若在此填入 ttft/tps，将污染 benchmark 报告并误导后续决策。
            self.last_perf = {
                "ttft_ms": None,
                "tps": None,
                "total_tokens": None,
                "total_time_s": None,
                "is_mock": True,
            }
            if callback:
                callback(mock_res)
            return mock_res

        # 板端真实推理
        start_t = time.time()
        first_token_t = None
        token_count = 0
        output_buffer = []

        def _inner_callback(token_str, state):
            nonlocal first_token_t, token_count
            now = time.time()
            if first_token_t is None:
                first_token_t = now
            token_count += 1
            output_buffer.append(token_str)
            if callback:
                callback(token_str)

        self.rkllm_handle.run(prompt, _inner_callback)
        total_time = time.time() - start_t

        ttft = (first_token_t - start_t) * 1000.0 if first_token_t else 0.0
        decode_time = total_time - (ttft / 1000.0)
        tps = (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else 0.0

        self.last_perf = {
            "ttft_ms": round(ttft, 2),
            "tps": round(tps, 2),
            "total_tokens": token_count,
            "total_time_s": round(total_time, 3),
            "is_mock": False,
        }
        return "".join(output_buffer)

    def get_benchmark_report(self) -> str:
        p = self.last_perf
        if p.get("is_mock"):
            return ("[Benchmark Report] 当前为 Mock 模式，未发生真实 NPU 推理，"
                    "无性能数据可报告。\n"
                    "  → 需部署 .rkllm 模型并安装 RKLLM 运行时后重测。")
        if p.get("ttft_ms") is None:
            return "[Benchmark Report] 尚未执行任何推理，无数据。"
        return (
            f"[Benchmark Report]\n"
            f"  • 首字时延 (TTFT): {p['ttft_ms']} ms\n"
            f"  • 生成吞吐率: {p['tps']} Tokens/s\n"
            f"  • 总生成 Token: {p['total_tokens']}\n"
            f"  • 端到端耗时: {p['total_time_s']} s"
        )

    def release(self):
        if self.rkllm_handle:
            self.rkllm_handle.release()
            print("[RKLLM] 已释放板载 NPU 内存")
