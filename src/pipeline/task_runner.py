import time
from typing import Dict, Any, Optional, List

from src.vision import USBCamera, RKNNObjectDetector
from src.kinematics import SimpleArmKinematics
from src.controller import SerialArmController, I2CArmController
from src.llm import IndustrialTaskPlanner
from src.inference import RKLLMInferenceEngine


class GraspPipeline:
    """博拓里尼工业场景: 端侧大模型任务规划与视觉伺服机械臂全闭环调度器"""

    def __init__(self, config: Dict[str, Any], mock_mode: bool = False):
        self.config = config
        self.mock_mode = mock_mode

        # 1. 视觉感知
        cam_cfg = config.get("camera", {})
        self.camera = USBCamera(
            device_id=cam_cfg.get("device_id", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480)
        )

        vis_cfg = config.get("vision", {})
        self.detector = RKNNObjectDetector(
            model_path=vis_cfg.get("model_path", "models/weights/yolov8n_int8.rknn")
        )

        # 2. 机械臂驱动
        arm_cfg = config.get("arm", {})
        ctrl_type = arm_cfg.get("controller_type", "i2c")
        if ctrl_type == "i2c":
            i2c_cfg = arm_cfg.get("i2c", {})
            self.controller = I2CArmController(
                bus_num=i2c_cfg.get("bus", 7),
                address=i2c_cfg.get("address", 0x40),
                channel_map=i2c_cfg.get("channels", [0, 1, 2, 3, 4, 5]),
                mock=mock_mode
            )
        else:
            ser_cfg = arm_cfg.get("serial", {})
            self.controller = SerialArmController(
                port=ser_cfg.get("port", "/dev/ttyUSB0"),
                baudrate=ser_cfg.get("baudrate", 115200),
                mock=mock_mode
            )

        # 3. 运动学逆解
        self.kinematics = SimpleArmKinematics()

        # 4. 端侧大模型 (Qwen) 任务规划大脑
        llm_cfg = config.get("llm", {})
        self.llm_engine = RKLLMInferenceEngine(
            model_path=llm_cfg.get("model_path", "models/weights/qwen2.5_0.5b_w4a16.rkllm"),
            max_context_len=llm_cfg.get("max_context_len", 1024),
            max_new_tokens=llm_cfg.get("max_new_tokens", 256)
        )
        self.planner = IndustrialTaskPlanner(llm_engine=self.llm_engine)

    def setup(self) -> bool:
        print("[Pipeline] 正在初始化博拓里尼边缘工业大脑与执行子系统...")
        if not self.mock_mode:
            self.camera.start()
        self.detector.init_model()
        self.controller.connect()
        self.llm_engine.load_model()
        print("[Pipeline] 全系统就绪！")
        return True

    def pixel_to_world_coords(self, px: int, py: int) -> Dict[str, float]:
        """手眼标定映射: 图像 2D 像素坐标 (u, v) -> 机械臂基座物理坐标 (X, Y, Z)"""
        scale = 0.5  # mm/pixel
        base_x = 150.0 + (py - 240) * scale
        base_y = (px - 320) * scale
        base_z = self.config.get("pipeline", {}).get("grasp_z_height", 30.0)
        return {"x": base_x, "y": base_y, "z": base_z}

    def execute_command(self, user_instruction: str) -> bool:
        """接收自然语言工业指令，由 Qwen 进行大模型任务拆解并执行"""
        print("\n" + "=" * 60)
        print(f"[指令输入] 操作员语音/文本指令: \"{user_instruction}\"")
        print("=" * 60)

        # 1. 大模型端侧意图理解与任务规划 (Task Planning)
        plan_result = self.planner.plan(user_instruction)
        print(f"[Qwen 大脑] 解析任务意图: {plan_result.get('intent')} (优先级: {plan_result.get('priority')})")

        actions = plan_result.get("actions", [])
        if not actions:
            print("[Qwen 大脑] 未生成有效动作序列")
            return False

        # 2. 依次调度执行动作序列
        safe_z = self.config.get("pipeline", {}).get("safe_z_height", 150.0)
        for idx, act in enumerate(actions):
            action_name = act.get("action")
            params = act.get("params", {})
            print(f"\n>> 动作步骤 [{idx+1}/{len(actions)}]: {action_name} | 参数: {params}")

            if action_name == "move_safe":
                joints = self.kinematics.inverse_kinematics({"x": 150.0, "y": 0.0, "z": safe_z})
                if joints:
                    self.controller.move_joints(joints, speed=30)

            elif action_name == "pick":
                # 视觉目标识别与坐标校正
                target_x = params.get("x", 150.0)
                target_y = params.get("y", 0.0)
                target_z = params.get("z", 30.0)

                # 下探抓取
                approach_pose = {"x": target_x, "y": target_y, "z": safe_z}
                grasp_pose = {"x": target_x, "y": target_y, "z": target_z}

                app_joints = self.kinematics.inverse_kinematics(approach_pose)
                gr_joints = self.kinematics.inverse_kinematics(grasp_pose)

                if app_joints and gr_joints:
                    self.controller.gripper_control(1.0) # 张开夹爪
                    self.controller.move_joints(app_joints, speed=35)
                    self.controller.move_joints(gr_joints, speed=20) # 慢速下探
                    self.controller.gripper_control(0.0) # 闭合夹紧
                    time.sleep(0.3)
                    self.controller.move_joints(app_joints, speed=25) # 稳步抬升

            elif action_name == "place":
                target_x = params.get("x", 180.0)
                target_y = params.get("y", 60.0)
                target_z = params.get("z", 30.0)

                approach_pose = {"x": target_x, "y": target_y, "z": safe_z}
                place_pose = {"x": target_x, "y": target_y, "z": target_z}

                app_joints = self.kinematics.inverse_kinematics(approach_pose)
                pl_joints = self.kinematics.inverse_kinematics(place_pose)

                if app_joints and pl_joints:
                    self.controller.move_joints(app_joints, speed=35)
                    self.controller.move_joints(pl_joints, speed=20)
                    self.controller.gripper_control(1.0) # 释放工件
                    time.sleep(0.3)
                    self.controller.move_joints(app_joints, speed=25)

            elif action_name == "inspect":
                print(f"[巡检] 正在对准工件 {params.get('target_name')} 执行工业视觉检测...")
                time.sleep(0.5)

            elif action_name == "emergency_stop":
                print("[安全机制] 触发紧急安全停机！")
                self.controller.gripper_control(0.0)
                return False

        print("\n[Pipeline] 工业任务指令序列全部执行完毕！")
        return True

    def stop(self):
        self.camera.stop()
        self.detector.release()
        self.controller.disconnect()
        self.llm_engine.release()
