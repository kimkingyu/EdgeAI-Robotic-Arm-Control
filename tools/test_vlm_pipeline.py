#!/usr/bin/env python3
"""VLM 抓取编排器的安全边界测试。

这是阶段 4.4 实机抓取的核心路径，硬件到货后直接跑它。这里先用假控制器把
"未标定时会不会下发动作"这类问题钉死 —— 等舵机通电再发现就晚了。

合成输入只验证编排逻辑，不构成任何实机精度证据。
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.vlm_grasp_pipeline import VLMGraspPipeline


class RecordingController:
    """记录所有下发，不碰真实总线。"""

    def __init__(self):
        self.calls = []

    def move_joints(self, angles, speed=30):
        self.calls.append(("move_joints", tuple(angles) if angles else None, speed))
        return True

    def gripper_control(self, value):
        self.calls.append(("gripper_control", value))
        return True

    def disconnect(self):
        self.calls.append(("disconnect",))

    @property
    def motions(self):
        return [c for c in self.calls if c[0] == "move_joints"]


class StubKinematics:
    """可控的逆解桩：按需返回可达或不可达。"""

    def __init__(self, reachable=True):
        self.reachable = reachable
        self.requests = []
        self.last_reject_reason = "超出工作空间" if not reachable else None

    def inverse_kinematics(self, target):
        self.requests.append(dict(target))
        return [10.0, 20.0, 30.0] if self.reachable else None


def build(config=None, reachable=True):
    pipeline = VLMGraspPipeline(config or {}, mock_mode=True)
    pipeline.controller = RecordingController()
    pipeline.kinematics = StubKinematics(reachable)
    return pipeline


class JsonExtractionTests(unittest.TestCase):
    """VLM 输出未必规整，解析失败必须走兜底而不是崩。"""

    def test_accepts_plain_and_fenced_json(self):
        plan = {"intent": "抓取", "actions": [{"action": "pick", "params": {}}]}
        raw = json.dumps(plan, ensure_ascii=False)
        for text in (raw, "```json\n%s\n```" % raw, "好的：\n```\n%s\n```" % raw,
                     "说明文字 %s 后缀" % raw):
            with self.subTest(text=text[:30]):
                self.assertEqual(VLMGraspPipeline._extract_json(text), plan)

    def test_rejects_malformed_without_raising(self):
        for text in ("", None, "没有大括号", "{不是合法json}", "{'单引号':1}", "{", "}"):
            with self.subTest(text=text):
                self.assertIsNone(VLMGraspPipeline._extract_json(text))


class RuleFallbackTests(unittest.TestCase):
    """模型不可用时兜底必须确定、且急停优先。"""

    def test_emergency_takes_priority(self):
        for text in ("急停", "立即停止", "EMERGENCY STOP", "紧急情况"):
            plan = VLMGraspPipeline._rule_fallback(text)
            self.assertEqual(plan["priority"], 0)
            self.assertEqual(plan["actions"][0]["action"], "emergency_stop")

    def test_reset_and_default(self):
        self.assertEqual(
            VLMGraspPipeline._rule_fallback("复位")["actions"][0]["action"], "move_safe")
        default = VLMGraspPipeline._rule_fallback("看看有什么")
        self.assertEqual([a["action"] for a in default["actions"]], ["move_safe", "inspect"])


class ExecutionSafetyTests(unittest.TestCase):
    def test_emergency_stop_aborts_and_opens_nothing(self):
        pipeline = build()
        ok = pipeline.execute_plan({"intent": "急停", "actions": [
            {"action": "emergency_stop", "params": {}},
            {"action": "pick", "params": {"x": 200, "y": 0, "z": 30}}]})
        self.assertFalse(ok)
        # 急停之后不得再有任何关节下发。
        self.assertEqual(pipeline.controller.motions, [])
        self.assertEqual(pipeline.stats["actions_executed"], 1)

    def test_unreachable_target_is_skipped_not_forced(self):
        pipeline = build(reachable=False)
        ok = pipeline.execute_plan({"intent": "抓取", "actions": [
            {"action": "pick", "params": {"x": 9999, "y": 9999, "z": 30}}]})
        self.assertTrue(ok)
        self.assertEqual(pipeline.controller.motions, [])
        self.assertEqual(pipeline.stats["actions_executed"], 0)

    def test_none_intent_executes_nothing(self):
        pipeline = build()
        self.assertTrue(pipeline.execute_plan({"intent": "none", "actions": []}))
        self.assertEqual(pipeline.controller.calls, [])

    def test_unknown_action_is_skipped(self):
        pipeline = build()
        self.assertTrue(pipeline.execute_plan(
            {"intent": "x", "actions": [{"action": "自毁", "params": {}}]}))
        self.assertEqual(pipeline.controller.calls, [])
        self.assertEqual(pipeline.stats["actions_executed"], 0)

    def test_pick_sequence_order(self):
        """抓取顺序必须是张爪→接近→下探→闭爪→抬升，顺序错会撞件。"""
        pipeline = build()
        pipeline.execute_plan({"intent": "抓取", "actions": [
            {"action": "pick", "params": {"x": 180, "y": 20, "z": 30}}]})
        names = [c[0] for c in pipeline.controller.calls]
        self.assertEqual(names, ["gripper_control", "move_joints", "move_joints",
                                 "gripper_control", "move_joints"])
        self.assertEqual(pipeline.controller.calls[0][1], 1.0)
        self.assertEqual(pipeline.controller.calls[3][1], 0.0)
        heights = [r["z"] for r in pipeline.kinematics.requests]
        self.assertGreater(heights[0], heights[1], "必须先到安全高度再下探")


class Calibrated:
    """假标定：像素按固定偏移映射到基座坐标，便于断言链路是否真的走通。"""

    is_calibrated = True

    @staticmethod
    def pixel_to_robot(px, py):
        return {"x": 100.0 + px * 0.1, "y": py * 0.1, "z": 30.0}


def detection(name, score, center):
    cx, cy = center
    return {"class_id": 0, "class_name": name, "score": score,
            "bbox": [cx - 10, cy - 10, cx + 10, cy + 10], "center": (cx, cy)}


class TargetResolutionTests(unittest.TestCase):
    """检测结果 → 像素 → 基座坐标，这条链路此前完全没接。"""

    def test_picks_highest_score_without_name(self):
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        located = pipeline.resolve_target(
            [detection("bowl", 0.4, (100, 200)), detection("vase", 0.9, (300, 400))])
        self.assertEqual(located["detection"]["class_name"], "vase")
        self.assertEqual(located["pixel"], (300, 400))
        self.assertAlmostEqual(located["world"]["x"], 130.0, places=5)

    def test_matches_by_name_case_insensitively(self):
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        items = [detection("bowl", 0.9, (100, 100)), detection("Vase", 0.5, (200, 200))]
        self.assertEqual(
            pipeline.resolve_target(items, "vase")["detection"]["class_name"], "Vase")

    def test_no_match_returns_none(self):
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        self.assertIsNone(pipeline.resolve_target([detection("bowl", 0.9, (10, 10))], "阀芯"))

    def test_empty_detections_returns_none(self):
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        self.assertIsNone(pipeline.resolve_target([], "vase"))

    def test_uncalibrated_refuses_even_with_detection(self):
        """有检测但未标定时必须拒绝，不能退回近似坐标。"""
        pipeline = build()
        self.assertIsNone(pipeline.resolve_target([detection("vase", 0.9, (300, 400))]))


class DetectionToGraspTests(unittest.TestCase):
    def test_pick_without_coords_uses_detection(self):
        """VLM 只给 target_name 时，坐标由检测加标定补齐。"""
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        ok = pipeline.execute_plan(
            {"intent": "抓取", "actions": [{"action": "pick", "params": {"target_name": "vase"}}]},
            [detection("vase", 0.9, (300, 400))])
        self.assertTrue(ok)
        self.assertEqual(len(pipeline.controller.motions), 3)
        self.assertEqual(pipeline.stats["actions_executed"], 1)
        requested = pipeline.kinematics.requests[0]
        self.assertAlmostEqual(requested["x"], 130.0, places=5)
        self.assertAlmostEqual(requested["y"], 40.0, places=5)

    def test_explicit_coords_take_precedence(self):
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        pipeline.execute_plan(
            {"intent": "抓取", "actions": [
                {"action": "pick", "params": {"x": 200, "y": 50, "z": 30, "target_name": "vase"}}]},
            [detection("vase", 0.9, (300, 400))])
        self.assertAlmostEqual(pipeline.kinematics.requests[0]["x"], 200.0, places=5)

    def test_pick_without_coords_and_without_detection_refuses(self):
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        pipeline.execute_plan(
            {"intent": "抓取", "actions": [{"action": "pick", "params": {"target_name": "vase"}}]}, [])
        self.assertEqual(pipeline.controller.motions, [])

    def test_place_does_not_fall_back_to_detection(self):
        """放置目标位置不能由检测推断 —— 检测的是工件而非放置点。"""
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        pipeline.execute_plan(
            {"intent": "放置", "actions": [{"action": "place", "params": {"target_name": "vase"}}]},
            [detection("vase", 0.9, (300, 400))])
        self.assertEqual(pipeline.controller.motions, [])


class CoordinateSafetyTests(unittest.TestCase):
    """未标定时不得给出无依据的坐标 —— 这类缺陷会稳定抓偏且看起来在工作。"""

    def test_uncalibrated_pixel_to_world_refuses(self):
        pipeline = build()
        self.assertIsNone(pipeline.pixel_to_world(320, 240),
                          "未标定必须拒绝解算，不能退回线性近似")

    def test_pick_without_coordinates_is_refused(self):
        """VLM 只给语义不给坐标时，不能用默认点硬抓。"""
        pipeline = build()
        ok = pipeline.execute_plan({"intent": "抓取", "actions": [
            {"action": "pick", "params": {"target_name": "阀芯"}}]})
        self.assertTrue(ok)
        self.assertEqual(pipeline.controller.motions, [],
                         "缺坐标时不得下发任何动作")
        self.assertEqual(pipeline.stats["actions_executed"], 0)

    def test_partial_coordinates_are_refused(self):
        pipeline = build()
        for params in ({"x": 180}, {"y": 20}, {"x": 180, "y": 20}):
            with self.subTest(params=params):
                pipeline.controller.calls.clear()
                pipeline.execute_plan({"intent": "抓取",
                                       "actions": [{"action": "pick", "params": params}]})
                self.assertEqual(pipeline.controller.motions, [])

    def test_non_numeric_coordinates_are_refused(self):
        pipeline = build()
        for params in ({"x": "近处", "y": 0, "z": 30}, {"x": None, "y": 0, "z": 30},
                       {"x": float("nan"), "y": 0, "z": 30},
                       {"x": float("inf"), "y": 0, "z": 30}):
            with self.subTest(params=params):
                pipeline.controller.calls.clear()
                pipeline.execute_plan({"intent": "抓取",
                                       "actions": [{"action": "pick", "params": params}]})
                self.assertEqual(pipeline.controller.motions, [],
                                 "非法坐标必须拒绝而不是转成默认值")

    def test_calibrated_pixel_to_world_uses_homography(self):
        pipeline = build()
        pipeline.hand_eye = Calibrated()
        self.assertEqual(pipeline.pixel_to_world(320, 240),
                         {"x": 132.0, "y": 24.0, "z": 30.0})


if __name__ == "__main__":
    unittest.main()
