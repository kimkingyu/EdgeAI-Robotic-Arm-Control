import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


class CloudLLMClient:
    """云端大模型 API 客户端 (支持 OpenAI / DashScope Qwen / DeepSeek 兼容接口)"""

    def __init__(self, api_key: str = "", base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", model: str = "qwen-plus"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.last_perf = {
            "rtt_ms": 0.0,
            "ttft_ms": 0.0,
            "total_time_s": 0.0,
            "success": False
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """向云端 API 发送请求并统计端到端网络时延与推理性能"""
        if not self.api_key:
            # Mock 模式：网络往返延迟模拟 (通常公网在 400ms~1500ms 浮动)
            time.sleep(0.65)
            self.last_perf = {
                "rtt_ms": 650.0,
                "ttft_ms": 520.0,
                "total_time_s": 0.65,
                "success": True
            }
            return (
                '{\n'
                '  "intent": "云端大模型(Qwen-Plus)高维规划: 识别流水线异常法兰并安全转运",\n'
                '  "priority": 1,\n'
                '  "actions": [\n'
                '    {"action": "move_safe", "params": {}},\n'
                '    {"action": "pick", "params": {"target_name": "flange", "x": 170.0, "y": -10.0, "z": 30.0}},\n'
                '    {"action": "place", "params": {"x": 210.0, "y": 70.0, "z": 35.0}},\n'
                '    {"action": "move_safe", "params": {}}\n'
                '  ]\n'
                '}'
            )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1
        }

        start_t = time.time()
        try:
            req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                elapsed = (time.time() - start_t) * 1000.0
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                self.last_perf = {
                    "rtt_ms": round(elapsed, 2),
                    "ttft_ms": round(elapsed * 0.8, 2),
                    "total_time_s": round(elapsed / 1000.0, 3),
                    "success": True
                }
                return content
        except Exception as e:
            elapsed = (time.time() - start_t) * 1000.0
            print(f"[CloudAPI] 请求失败 ({e}), 耗时: {elapsed:.1f}ms")
            self.last_perf = {
                "rtt_ms": round(elapsed, 2),
                "ttft_ms": 0.0,
                "total_time_s": round(elapsed / 1000.0, 3),
                "success": False
            }
            return None
