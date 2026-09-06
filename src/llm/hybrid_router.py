import time
from typing import Dict, Any, Optional

from .planner import IndustrialTaskPlanner
from .cloud_api_client import CloudLLMClient


class HybridDecisionRouter:
    """端云协同智能决策路由器
    根据工业现场的网络连通性、工件数据隐私安全等级 (Privacy Level) 与时延敏感度，
    在【端侧本地 Qwen (RKLLM)】与【云端大模型 API】之间实施自适应动态分发与无感回退。
    """

    def __init__(self, edge_planner: IndustrialTaskPlanner, cloud_client: Optional[CloudLLMClient] = None, routing_mode: str = "auto_hybrid"):
        self.edge_planner = edge_planner
        self.cloud_client = cloud_client or CloudLLMClient()
        self.routing_mode = routing_mode # "edge_only" | "cloud_only" | "auto_hybrid"

    def dispatch(self, user_query: str, privacy_level: str = "normal", current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """决策分发入口，返回统一格式的工业动作规划与执行源追踪"""
        # 1. 强制端侧模式或高密工况（杜绝数据出域）
        if self.routing_mode == "edge_only" or privacy_level.lower() == "high":
            print(f"[Router] 触发高保密/强制离线策略 -> 路由至【端侧 NPU 本地 Qwen】")
            plan = self.edge_planner.plan(user_query, current_state)
            plan["source"] = "edge_npu_qwen"
            return plan

        # 2. 强制云端模式
        if self.routing_mode == "cloud_only":
            print(f"[Router] 路由至【云端全知大模型 API】")
            cloud_res = self._call_cloud(user_query)
            if cloud_res:
                cloud_res["source"] = "cloud_api"
                return cloud_res
            print("[Router] 云端 API 失败，紧急切换本地规则兜底")
            fallback = self.edge_planner.plan(user_query, current_state)
            fallback["source"] = "rule_fallback"
            return fallback

        # 3. 自动混合协同模式 (Auto Hybrid with Seamless Fallback)
        print(f"[Router] 自动混合调度模式 -> 优先尝试【云端高维大模型 API】...")
        cloud_res = self._call_cloud(user_query)
        if cloud_res:
            cloud_res["source"] = "cloud_api"
            return cloud_res

        # 云端超时或断网，毫秒级无感降级到端侧
        print(f"[Router] ⚠️ 监测到云端连接异常/超时 -> 毫秒级无缝降级至【板载 6 TOPS NPU Qwen】")
        edge_res = self.edge_planner.plan(user_query, current_state)
        edge_res["source"] = "edge_fallback_qwen"
        return edge_res

    def _call_cloud(self, query: str) -> Optional[Dict[str, Any]]:
        raw = self.cloud_client.generate(query, system_prompt=self.edge_planner.SYSTEM_PROMPT)
        if raw:
            return self.edge_planner._extract_json(raw)
        return None
