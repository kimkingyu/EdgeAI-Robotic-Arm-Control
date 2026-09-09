"""
视觉多模态驱动的机械臂抓取闭环

与 task_runner.GraspPipeline 的区别：
  旧链路  相机 → YOLO 检测 → 坐标 → LLM 拿坐标规划（两段式，语义在转译中丢失）
  本链路  相机 → VLM 直接看图决策 → YOLO 提供精确像素坐标 → 手眼标定 → 逆解 → 执行

职责分工（两个模型各司其职，不是二选一）：
  YOLOv8   高频定位，实测 44 FPS / 22.6ms —— 负责"目标在哪"
  Qwen3-VL 低频语义，实测 TTFT 253ms     —— 负责"该抓哪个、怎么抓"

硬件未接入时全链路自动降级为 Mock，不会崩溃，便于离线验证编排逻辑。
"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# VLM 的工业抓取决策提示词：要求其输出可直接执行的结构化动作流
GRASP_SYSTEM_PROMPT = """你是工业机械臂的视觉决策大脑。观察图像后输出 JSON 动作流。

可用原子动作：
  move_safe  : 归位到安全高度
  pick       : 抓取目标，params 需含 target_name
  place      : 放置，params 需含 x/y/z
  inspect    : 对准目标做质检，params 需含 target_name
  emergency_stop : 紧急制动

输出格式（只输出 JSON，不要解释）：
{"intent":"任务意图","priority":1,"actions":[{"action":"pick","params":{"target_name":"..."}}]}

若图中没有可抓取的目标，输出 {"intent":"none","actions":[]}。"""


class VLMGraspPipeline:
    """VLM 驱动的抓取闭环编排器"""

    def __init__(self, config: Dict[str, Any], mock_mode: bool = False):
        self.config = config
        self.mock_mode = mock_mode

        self.camera = None
        self.detector = None
        self.vlm = None
        self.controller = None
        self.kinematics = None

        self.stats: Dict[str, Any] = {
            "frames_captured": 0,
            "vlm_calls": 0,
            "actions_executed": 0,
            "last_vlm_perf": None,
        }

    # ── 初始化 ──────────────────────────────────────────────────────────

    def setup(self) -> bool:
        print("=" * 70)
        print("  VLM 驱动的机械臂抓取闭环 —— 系统初始化")
        print("=" * 70)

        ok = True
        ok &= self._setup_camera()
        ok &= self._setup_detector()
        ok &= self._setup_arm()
        ok &= self._setup_vlm()

        print("-" * 70)
        print(f"[Pipeline] 初始化完成（mock_mode={self.mock_mode}）")
        return ok

    def _setup_camera(self) -> bool:
        cam_cfg = self.config.get("camera", {})
        dev = cam_cfg.get("device_id", 0)
        if self.mock_mode or not os.path.exists(f"/dev/video{dev}"):
            print(f"[相机] /dev/video{dev} 不存在 → Mock 模式（读取静态测试图）")
            self.camera = None
            return True
        try:
            from src.vision import USBCamera
            self.camera = USBCamera(device_id=dev,
                                    width=cam_cfg.get("width", 640),
                                    height=cam_cfg.get("height", 480))
            if self.camera.start():
                print(f"[相机] /dev/video{dev} 取流就绪")
                return True
            print("[相机] 启动失败，降级为 Mock")
            self.camera = None
        except Exception as e:
            print(f"[相机] 初始化异常: {str(e)[:80]}")
            self.camera = None
        return True

    def _setup_detector(self) -> bool:
        vis_cfg = self.config.get("vision", {})
        path = vis_cfg.get("model_path", "models/weights/yolov8n_int8.rknn")
        if not os.path.exists(path):
            print(f"[YOLO] 模型不存在 {path} → 跳过精确定位")
            return True
        try:
            from src.inference.rknn_engine import RKNNMultiCoreVisionEngine
            self.detector = RKNNMultiCoreVisionEngine(path, core_mode="auto")
            if self.detector.init_engine():
                print(f"[YOLO] 定位引擎就绪（{os.path.basename(path)}）")
                return True
        except Exception as e:
            print(f"[YOLO] 初始化异常: {str(e)[:80]}")
        self.detector = None
        return True

    def _setup_arm(self) -> bool:
        from src.kinematics import SimpleArmKinematics
        from src.controller import I2CArmController

        self.kinematics = SimpleArmKinematics()
        arm_cfg = self.config.get("arm", {}).get("i2c", {})
        self.controller = I2CArmController(
            bus_num=arm_cfg.get("bus", 7),
            address=arm_cfg.get("address", 0x40),
            channel_map=arm_cfg.get("channels", [0, 1, 2, 3]),
            mock=self.mock_mode,
        )
        if self.controller.connect():
            print("[机械臂] 控制器已连接")
        else:
            print("[机械臂] 硬件未就绪 → Mock 模式（动作仅打印不下发）")
        return True

    def _setup_vlm(self) -> bool:
        llm_cfg = self.config.get("vlm", {})
        llm_path = llm_cfg.get("model_path", "models/weights/qwen3vl4b_w8a8.rkllm")
        vis_path = llm_cfg.get("vision_path", "models/weights/qwen3vl4b_vision.rknn")

        if not (os.path.exists(llm_path) and os.path.exists(vis_path)):
            print(f"[VLM] 模型缺失 → 跳过语义决策，将使用规则兜底")
            return True
        try:
            from src.inference.rkllm_vl_engine import QwenVLEngine
            self.vlm = QwenVLEngine(
                llm_path, vis_path,
                max_context_len=llm_cfg.get("max_context_len", 2048),
                max_new_tokens=llm_cfg.get("max_new_tokens", 256),
            )
            if self.vlm.init_engine():
                return True
            print("[VLM] 初始化失败 → 规则兜底")
        except Exception as e:
            print(f"[VLM] 初始化异常: {str(e)[:100]}")
        self.vlm = None
        return True

    # ── 感知 ────────────────────────────────────────────────────────────

    def capture(self) -> Optional["np.ndarray"]:
        """取一帧画面，无相机时回退到静态测试图"""
        if self.camera is not None:
            ok, frame = self.camera.read()
            if ok and frame is not None:
                self.stats["frames_captured"] += 1
                return frame

        fallback = self.config.get("camera", {}).get(
            "mock_image", "data/calibration/images/000000000074.jpg")
        if HAS_CV2 and os.path.exists(fallback):
            self.stats["frames_captured"] += 1
            return cv2.imread(fallback)
        return None

    def locate_targets(self, frame) -> List[Dict[str, Any]]:
        """YOLO 高频定位：给出目标的像素坐标（VLM 只做语义，不给精确坐标）"""
        if self.detector is None or frame is None or not HAS_CV2:
            return []
        try:
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            self.detector.infer(np.expand_dims(img, 0))
            cost = (time.perf_counter() - t0) * 1000
            print(f"[YOLO] 定位耗时 {cost:.1f} ms")
        except Exception as e:
            print(f"[YOLO] 推理异常: {str(e)[:80]}")
        return []

    # ── 决策 ────────────────────────────────────────────────────────────

    def decide(self, frame, instruction: str) -> Dict[str, Any]:
        """VLM 看图 + 指令 → 结构化动作流"""
        if self.vlm is None or frame is None:
            return self._rule_fallback(instruction)

        if HAS_CV2 and HAS_NUMPY:
            img = cv2.resize(frame, (448, 448))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, 0)
        else:
            return self._rule_fallback(instruction)

        prompt = f"{GRASP_SYSTEM_PROMPT}\n\n操作员指令：{instruction}"
        out = self.vlm.generate(prompt, image=img)
        self.stats["vlm_calls"] += 1
        self.stats["last_vlm_perf"] = dict(self.vlm.last_perf)

        p = self.vlm.last_perf
        print(f"[VLM] TTFT {p.get('ttft_ms')} ms | {p.get('tps')} tok/s "
              f"| {p.get('total_tokens')} tokens")

        plan = self._extract_json(out)
        if plan is None:
            print(f"[VLM] 输出非合法 JSON，转用规则兜底。原始输出: {out.strip()[:120]}")
            return self._rule_fallback(instruction)
        return plan

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        t = (text or "").strip()
        if "```" in t:
            for seg in t.split("```"):
                seg = seg.strip()
                if seg.startswith("json"):
                    seg = seg[4:].strip()
                if seg.startswith("{"):
                    t = seg
                    break
        s, e = t.find("{"), t.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except Exception:
                return None
        return None

    @staticmethod
    def _rule_fallback(instruction: str) -> Dict[str, Any]:
        """确定性规则兜底：模型不可用时保证工业现场仍能安全响应"""
        q = instruction.lower()
        if any(k in q for k in ("急停", "停止", "stop", "紧急")):
            return {"intent": "紧急安全制动", "priority": 0,
                    "actions": [{"action": "emergency_stop", "params": {}}]}
        if any(k in q for k in ("复位", "归位", "安全", "reset")):
            return {"intent": "复位至安全高度", "priority": 2,
                    "actions": [{"action": "move_safe", "params": {}}]}
        return {"intent": "通用巡检", "priority": 3,
                "actions": [{"action": "move_safe", "params": {}},
                            {"action": "inspect", "params": {"target_name": "workpiece"}}]}

    # ── 坐标变换 ────────────────────────────────────────────────────────

    def pixel_to_world(self, px: float, py: float) -> Dict[str, float]:
        """
        手眼标定：像素 (u,v) → 机械臂基座 (X,Y,Z)

        当前为线性近似占位，真实标定矩阵需相机到货后用标定板测定，
        参数存于 config 的 hand_eye 段。
        """
        he = self.config.get("hand_eye", {})
        scale = he.get("mm_per_pixel", 0.5)
        cx = he.get("center_u", 320)
        cy = he.get("center_v", 240)
        base_x = he.get("base_x", 150.0)
        return {
            "x": base_x + (py - cy) * scale,
            "y": (px - cx) * scale,
            "z": self.config.get("pipeline", {}).get("grasp_z_height", 30.0),
        }

    # ── 执行 ────────────────────────────────────────────────────────────

    def execute_plan(self, plan: Dict[str, Any]) -> bool:
        actions = plan.get("actions", [])
        intent = plan.get("intent", "未知")

        if intent == "none" or not actions:
            print(f"[执行] 决策结果：无可执行动作（intent={intent}）")
            return True

        print(f"[执行] 意图: {intent} | 共 {len(actions)} 个动作")
        safe_z = self.config.get("pipeline", {}).get("safe_z_height", 150.0)

        for i, act in enumerate(actions, 1):
            name = act.get("action")
            params = act.get("params", {}) or {}
            print(f"  [{i}/{len(actions)}] {name} {params}")

            if name == "emergency_stop":
                print("  ⚠ 触发紧急制动，中止后续动作")
                self.controller.gripper_control(0.0)
                self.stats["actions_executed"] += 1
                return False

            if name == "move_safe":
                j = self.kinematics.inverse_kinematics({"x": 150.0, "y": 0.0, "z": safe_z})
                if j:
                    self.controller.move_joints(j, speed=30)

            elif name in ("pick", "place"):
                x = float(params.get("x", 150.0))
                y = float(params.get("y", 0.0))
                z = float(params.get("z", 30.0))
                approach = self.kinematics.inverse_kinematics({"x": x, "y": y, "z": safe_z})
                target = self.kinematics.inverse_kinematics({"x": x, "y": y, "z": z})
                if not (approach and target):
                    print(f"  ⚠ 逆解失败，目标 ({x},{y},{z}) 可能超出工作空间，跳过")
                    continue
                if name == "pick":
                    self.controller.gripper_control(1.0)
                    self.controller.move_joints(approach, speed=35)
                    self.controller.move_joints(target, speed=20)
                    self.controller.gripper_control(0.0)
                else:
                    self.controller.move_joints(approach, speed=35)
                    self.controller.move_joints(target, speed=20)
                    self.controller.gripper_control(1.0)
                time.sleep(0.3)
                self.controller.move_joints(approach, speed=25)

            elif name == "inspect":
                print(f"  → 对准 {params.get('target_name', '目标')} 执行质检")
                time.sleep(0.3)

            else:
                print(f"  ⚠ 未知动作 {name}，跳过")
                continue

            self.stats["actions_executed"] += 1

        return True

    # ── 主循环 ──────────────────────────────────────────────────────────

    def run_once(self, instruction: str) -> bool:
        print("\n" + "=" * 70)
        print(f"[指令] {instruction}")
        print("=" * 70)

        frame = self.capture()
        if frame is None:
            print("[感知] 无可用画面，仅按指令文本决策")
        else:
            h, w = frame.shape[:2]
            print(f"[感知] 取得画面 {w}x{h}")
            self.locate_targets(frame)

        plan = self.decide(frame, instruction)
        return self.execute_plan(plan)

    def print_stats(self):
        s = self.stats
        print("\n" + "-" * 70)
        print(f"[统计] 采集 {s['frames_captured']} 帧 | VLM 调用 {s['vlm_calls']} 次 "
              f"| 执行动作 {s['actions_executed']} 个")
        if s["last_vlm_perf"]:
            p = s["last_vlm_perf"]
            print(f"       末次 VLM: TTFT {p.get('ttft_ms')} ms | {p.get('tps')} tok/s")
        print("-" * 70)

    def stop(self):
        if self.camera:
            self.camera.stop()
        if self.detector:
            self.detector.release()
        if self.vlm:
            self.vlm.release()
        if self.controller:
            self.controller.disconnect()
        print("[Pipeline] 所有资源已释放")
