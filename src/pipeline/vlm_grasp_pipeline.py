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
        self.hand_eye = None

        # 640 给 YOLO、448 给 VLM，两处共用同一个后端实现，避免预处理逻辑漂移。
        # 后端默认 OpenCV；配 preprocess.backend=mlir 才走 MLIR，且失败直接报错。
        self._preprocessors: Dict[Tuple[int, int], Any] = {}

        self.stats: Dict[str, Any] = {
            "frames_captured": 0,
            "vlm_calls": 0,
            "actions_executed": 0,
            "last_vlm_perf": None,
            "preprocess_backend": None,
            "last_preprocess": None,
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
        ok &= self._setup_hand_eye()
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

    def _setup_hand_eye(self) -> bool:
        try:
            from src.vision.hand_eye import HandEyeCalibrator
            path = self.config.get("hand_eye", {}).get(
                "calib_path", "configs/hand_eye_calib.json")
            self.hand_eye = HandEyeCalibrator(path)
            if not self.hand_eye.load():
                print(f"[手眼] 未找到标定文件 {path} → 暂用线性近似")
                print("       实机抓取前请先运行标定：tools/calibrate_hand_eye.py")
        except Exception as e:
            print(f"[手眼] 初始化异常: {str(e)[:80]}")
            self.hand_eye = None
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

    def preprocessor(self, output_size: Tuple[int, int]):
        """按输出尺寸取预处理器；显式 mlir 后端失败时抛错，不悄悄退回 OpenCV。"""
        existing = self._preprocessors.get(output_size)
        if existing is not None:
            return existing
        from src.vision.preprocess import from_config
        created = from_config(self.config, output_size)
        self._preprocessors[output_size] = created
        self.stats["preprocess_backend"] = created.describe()
        return created

    def locate_targets(self, frame) -> List[Dict[str, Any]]:
        """YOLO 高频定位：给出目标的像素坐标（VLM 只做语义，不给精确坐标）"""
        if self.detector is None or frame is None or not HAS_CV2:
            return []
        try:
            pre = self.preprocessor((640, 640))
            batch = pre.run_batch1(frame)
            self.stats["last_preprocess"] = dict(pre.last_timings)
            # 预处理单独计时；不把它算进推理耗时，也不拿旧的"定位耗时"当预处理基线。
            print(f"[YOLO] 预处理 {pre.last_timings['total_ns'] / 1e6:.1f} ms"
                  f"（{pre.backend_name}）")
            t0 = time.perf_counter()
            outputs = self.detector.infer(batch)
            cost = (time.perf_counter() - t0) * 1000
            print(f"[YOLO] 推理耗时 {cost:.1f} ms")

            if not outputs:
                print("[YOLO] 推理未返回输出，无法解码")
                return []
            from src.vision.yolo_postprocess import decode
            detections = decode(outputs, frame.shape, input_size=(640, 640),
                                conf_thresh=self.config.get("vision", {}).get("conf_thresh", 0.25))
            self.stats["last_detections"] = len(detections)
            if not detections:
                print("[YOLO] 未检出目标")
            else:
                print(f"[YOLO] 检出 {len(detections)} 个目标: "
                      + ", ".join("%s(%.2f)@%s" % (d["class_name"], d["score"], d["center"])
                                  for d in detections[:3]))
            return detections
        except Exception as e:
            print(f"[YOLO] 推理异常: {type(e).__name__}: {str(e)[:80]}")
        return []

    # ── 决策 ────────────────────────────────────────────────────────────

    def decide(self, frame, instruction: str) -> Dict[str, Any]:
        """VLM 看图 + 指令 → 结构化动作流"""
        if self.vlm is None or frame is None:
            return self._rule_fallback(instruction)

        if HAS_CV2 and HAS_NUMPY:
            pre = self.preprocessor((448, 448))
            img = pre.run_batch1(frame)
            self.stats["last_preprocess"] = dict(pre.last_timings)
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

    def pixel_to_world(self, px: float, py: float) -> Optional[Dict[str, float]]:
        """手眼标定：像素 (u,v) → 机械臂基座 (X,Y,Z)。未标定返回 None。

        这里不提供线性近似兜底。无标定依据的坐标会让机械臂稳定抓偏，而且因为
        "看起来在工作"极难察觉 —— C++ 侧（src/main.cpp）已按同样理由改为拒绝
        解算，Python 侧保持一致。
        """
        if self.hand_eye is None or not self.hand_eye.is_calibrated:
            return None
        return self.hand_eye.pixel_to_robot(px, py)

    def resolve_target(self, detections: List[Dict[str, Any]],
                       target_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """按名称（或最高分）挑一个检测目标，并解算其基座坐标。

        返回 None 的两种情况必须区分对待，调用方不应把它们当成同一件事：
        没找到匹配目标，或找到了但未标定无法解算。日志会说明是哪一种。
        """
        if not detections:
            return None
        picked = None
        if target_name:
            needle = str(target_name).strip().lower()
            matches = [d for d in detections if needle in d["class_name"].lower()]
            if not matches:
                print(f"[定位] 画面中没有匹配 \"{target_name}\" 的目标"
                      f"（检出的是：{', '.join(d['class_name'] for d in detections[:5])}）")
                return None
            picked = max(matches, key=lambda d: d["score"])
        else:
            picked = max(detections, key=lambda d: d["score"])

        px, py = picked["center"]
        world = self.pixel_to_world(px, py)
        if world is None:
            print(f"[定位] 已锁定 {picked['class_name']}@({px},{py})，"
                  f"但手眼标定未载入，拒绝解算物理坐标")
            return None
        return {"detection": picked, "pixel": (px, py), "world": world}

    @staticmethod
    def _finite_coordinate(params: Dict[str, Any], key: str) -> Optional[float]:
        """取一个必须存在的有限数值坐标；缺失或非法返回 None，不用默认值填充。"""
        if key not in params:
            return None
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        # NaN 与 inf 能通过 float() 但会让逆解产生无意义结果。
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value

    # ── 执行 ────────────────────────────────────────────────────────────

    def execute_plan(self, plan: Dict[str, Any],
                     detections: Optional[List[Dict[str, Any]]] = None) -> bool:
        actions = plan.get("actions", [])
        intent = plan.get("intent", "未知")
        detections = detections or []

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
                # 坐标必须由调用方显式给全且合法。曾经缺一个就套默认 (150,0,30)，
                # 那等于朝一个无依据的固定点下探。
                coords = {k: self._finite_coordinate(params, k) for k in ("x", "y", "z")}
                missing = [k for k, v in coords.items() if v is None]
                # VLM 通常只给语义不给坐标。此时用检测结果 + 手眼标定补齐，
                # 这是"检测 → 像素 → 基座坐标"链路的实际接入点。
                if missing and name == "pick":
                    located = self.resolve_target(detections, params.get("target_name"))
                    if located is not None:
                        coords = dict(located["world"])
                        missing = [k for k in ("x", "y", "z")
                                   if self._finite_coordinate(coords, k) is None]
                        if not missing:
                            print(f"  → 由检测结果解算坐标: {located['detection']['class_name']}"
                                  f"@{located['pixel']} → "
                                  f"({coords['x']:.1f}, {coords['y']:.1f}, {coords['z']:.1f})")
                if missing:
                    print(f"  ⚠ {name} 缺少或非法的坐标 {missing}，拒绝执行"
                          f"（不使用默认坐标，避免无依据下探）")
                    continue
                x, y, z = coords["x"], coords["y"], coords["z"]
                approach = self.kinematics.inverse_kinematics({"x": x, "y": y, "z": safe_z})
                target = self.kinematics.inverse_kinematics({"x": x, "y": y, "z": z})
                if not (approach and target):
                    why = getattr(self.kinematics, "last_reject_reason", None)
                    print(f"  ⚠ 逆解失败，目标 ({x},{y},{z}) 不可达"
                          f"{'：' + why if why else ''}，跳过该动作")
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
        detections: List[Dict[str, Any]] = []
        if frame is None:
            print("[感知] 无可用画面，仅按指令文本决策")
        else:
            h, w = frame.shape[:2]
            print(f"[感知] 取得画面 {w}x{h}")
            detections = self.locate_targets(frame)

        plan = self.decide(frame, instruction)
        # 检测结果传给执行环节，供 VLM 未给坐标时解算
        return self.execute_plan(plan, detections)

    def print_stats(self):
        s = self.stats
        print("\n" + "-" * 70)
        print(f"[统计] 采集 {s['frames_captured']} 帧 | VLM 调用 {s['vlm_calls']} 次 "
              f"| 执行动作 {s['actions_executed']} 个")
        if s["last_vlm_perf"]:
            p = s["last_vlm_perf"]
            print(f"       末次 VLM: TTFT {p.get('ttft_ms')} ms | {p.get('tps')} tok/s")
        if s["preprocess_backend"]:
            b = s["preprocess_backend"]
            print(f"       预处理后端: {b['backend']}"
                  + (f"（variant {b['variant']}）" if b.get("variant") else ""))
        print("-" * 70)

    def stop(self):
        for pre in self._preprocessors.values():
            pre.close()
        self._preprocessors.clear()
        if self.camera:
            self.camera.stop()
        if self.detector:
            self.detector.release()
        if self.vlm:
            self.vlm.release()
        if self.controller:
            self.controller.disconnect()
        print("[Pipeline] 所有资源已释放")
