import json
import re
from typing import List, Dict, Any, Optional


class IndustrialTaskPlanner:
    """面向工业自动化场景的端侧大模型 (Qwen) 任务规划与指令解析器"""

    SYSTEM_PROMPT = """你是部署在边缘嵌入式平台上的工业具身智能机械臂核心大脑。
你的任务是将现场操作员下达的自然语言指令，解析并拆解为可供嵌入式执行机构执行的标准 JSON 动作流。

机械臂支持的基础原子动作库：
1. pick(target_name, x, y, z): 抓取指定坐标的工件
2. place(x, y, z): 将夹爪中的工件放置在指定坐标
3. move_safe(): 机械臂归位到安全预备高度
4. inspect(target_name): 调整机位对准指定工件进行细致质检
5. emergency_stop(): 紧急悬停制动

输出格式必须为纯 JSON，严禁输出任何多余闲聊，结构如下：
{
  "intent": "任务意图简述",
  "priority": 1,
  "actions": [
    {"action": "move_safe", "params": {}},
    {"action": "pick", "params": {"target_name": "valve_core", "x": 160.0, "y": -20.0, "z": 30.0}},
    {"action": "place", "params": {"x": 200.0, "y": 80.0, "z": 40.0}},
    {"action": "move_safe", "params": {}}
  ]
}
"""

    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine

    def build_prompt(self, user_query: str, current_state: Optional[Dict[str, Any]] = None) -> str:
        state_desc = ""
        if current_state:
            state_desc = f"\n当前设备与环境状态：{json.dumps(current_state, ensure_ascii=False)}"
        
        prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_query}{state_desc}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def plan(self, user_query: str, current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """将自然语言指令转换为可执行的机械臂动作队列"""
        if self.llm_engine and getattr(self.llm_engine, "is_ready", False):
            prompt = self.build_prompt(user_query, current_state)
            raw_output = self.llm_engine.generate(prompt)
            parsed = self._extract_json(raw_output)
            if parsed:
                return parsed

        # 离线回退规则引擎 (保证在模型未加载或无网络时工业现场 100% 高可靠兜底)
        return self._rule_fallback(user_query)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            # 优先精准正则提取
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"[Planner] JSON 提取失败: {e}")
        return None

    def _rule_fallback(self, query: str) -> Dict[str, Any]:
        """工业高可靠确定性规则兜底"""
        q = query.lower()
        if "急停" in q or "停止" in q or "stop" in q:
            return {
                "intent": "紧急安全制动",
                "priority": 0,
                "actions": [{"action": "emergency_stop", "params": {}}]
            }
        elif "复位" in q or "安全" in q or "reset" in q:
            return {
                "intent": "复位至安全高度",
                "priority": 2,
                "actions": [{"action": "move_safe", "params": {}}]
            }
        elif "抓" in q or "拿" in q or "搬" in q or "pick" in q:
            return {
                "intent": "自主工业抓取与码垛",
                "priority": 1,
                "actions": [
                    {"action": "move_safe", "params": {}},
                    {"action": "pick", "params": {"target_name": "workpiece", "x": 150.0, "y": 0.0, "z": 30.0}},
                    {"action": "place", "params": {"x": 180.0, "y": 60.0, "z": 30.0}},
                    {"action": "move_safe", "params": {}}
                ]
            }
        else:
            return {
                "intent": "通用巡检与定位",
                "priority": 3,
                "actions": [
                    {"action": "move_safe", "params": {}},
                    {"action": "inspect", "params": {"target_name": "workpiece"}}
                ]
            }
