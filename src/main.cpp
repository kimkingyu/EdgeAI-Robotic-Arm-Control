#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>

#include "safe_queue.hpp"
#include "camera_v4l2.hpp"
#include "rknn_detector.hpp"
#include "hand_eye_trans.hpp"
#include "kinematics.hpp"
#include "pca9685.hpp"

// Global pipeline stop flag
std::atomic<bool> g_running{true};

// Pipeline Queues
SafeQueue<FramePacket> g_raw_queue(5);
SafeQueue<std::pair<FramePacket, std::vector<DetectionResult>>> g_detection_queue(5);

/**
 * @brief Thread 1: Camera Acquisition Stage
 */
void camera_capture_thread(const std::string& dev, int width, int height) {
    std::cout << "[Thread 1] Camera capture thread started on " << dev << std::endl;
    cv::VideoCapture cap;
    if (!cap.open(0)) {
        std::cout << "[Mock Camera] No physical camera found, running synthetic test pattern." << std::endl;
    }

    uint64_t frame_id = 0;
    while (g_running) {
        cv::Mat frame;
        if (cap.isOpened()) {
            cap >> frame;
        } else {
            // Synthetic test frame
            frame = cv::Mat::zeros(height, width, CV_8UC3);
            cv::circle(frame, cv::Point(width / 2, height / 2), 30, cv::Scalar(0, 0, 255), -1);
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }

        if (frame.empty()) continue;

        FramePacket pkt{
            ++frame_id,
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count(),
            frame
        };

        g_raw_queue.push(pkt);
    }
    std::cout << "[Thread 1] Camera capture thread stopped." << std::endl;
}

/**
 * @brief Thread 2: NPU Hardware Accelerated Inference Stage
 */
void npu_inference_thread(const std::string& model_path) {
    std::cout << "[Thread 2] NPU Inference thread started with model: " << model_path << std::endl;

    while (g_running) {
        auto pkt_opt = g_raw_queue.pop_timeout(std::chrono::milliseconds(100));
        if (!pkt_opt.has_value()) continue;

        FramePacket pkt = pkt_opt.value();
        std::vector<DetectionResult> results;

        // Mock detection if running on host x86 without NPU
        DetectionResult dummy;
        dummy.class_id = 0;
        dummy.confidence = 0.92f;
        dummy.box = cv::Rect2f(300, 200, 80, 80);
        dummy.center = cv::Point2f(340.0f, 240.0f);
        results.push_back(dummy);

        g_detection_queue.push({pkt, results});
    }
    std::cout << "[Thread 2] NPU Inference thread stopped." << std::endl;
}

/**
 * @brief Thread 3: Spatial Coordinate Mapping, IK Solving & Motor Control
 */
/**
 * @brief Thread 3: 像素→基座坐标变换、逆解与舵机下发
 *
 * @param calib_path  手眼标定文件（Python 侧 calibrate_hand_eye.py 产出）
 * @param i2c_dev     PCA9685 所在 I2C 设备；为空则完全不碰总线
 * @param allow_drive 未显式允许时只解算不下发。默认不驱动舵机 ——
 *                    误启动主程序就让机械臂动起来是不可接受的。
 */
void control_dispatch_thread(const std::string& calib_path,
                             const std::string& i2c_dev,
                             bool allow_drive) {
    std::cout << "[Thread 3] Control and Kinematics thread started." << std::endl;
    ArmKinematics ik_solver(4);

    // 手眼标定：未载入则拒绝解算，不退回线性近似。
    // 曾经的硬编码 (u-320)*0.5+150 是占位实现，其斜率与偏移没有任何
    // 标定依据，实机上会稳定地抓偏到错误位置。
    HandEyeTransformer hand_eye;
    if (!calib_path.empty() && hand_eye.loadCalibration(calib_path)) {
        std::cout << "[HandEye] 已载入标定参数 " << calib_path
                  << " (工作平面 z=" << hand_eye.zPlane() << "mm)" << std::endl;
    } else {
        std::cout << "[HandEye] 未载入标定参数，本线程只做感知不解算坐标。\n"
                  << "          请先运行 tools/calibrate_hand_eye.py 完成标定。" << std::endl;
    }

    // 舵机驱动：默认不打开总线，更不下发指令
    PCA9685Driver servo(i2c_dev.empty() ? "/dev/null" : i2c_dev);
    bool servo_ready = false;
    if (allow_drive && !i2c_dev.empty()) {
        servo_ready = servo.open();
        std::cout << (servo_ready
                      ? "[Servo] PCA9685 已连接，将下发关节指令"
                      : "[Servo] PCA9685 打开失败，仅解算不下发") << std::endl;
    } else {
        std::cout << "[Servo] 未启用驱动（需 --i2c 与 --drive 同时指定），仅解算" << std::endl;
    }

    uint64_t solved = 0, rejected = 0, dispatched = 0;
    while (g_running) {
        auto item_opt = g_detection_queue.pop_timeout(std::chrono::milliseconds(100));
        if (!item_opt.has_value()) continue;

        auto [pkt, detections] = item_opt.value();
        (void)pkt;

        for (const auto& det : detections) {
            if (!hand_eye.isCalibrated()) continue;

            // 1. 像素 → 基座坐标（射影变换，非线性占位）
            const RobotTargetPose target = hand_eye.pixelToRobotBase(det.center);
            if (!target.is_valid) {
                ++rejected;
                continue;
            }

            // 2. 逆运动学。不可达时不得下发任何关节角
            const JointAngles ik = ik_solver.solveIK(target.x, target.y, target.z);
            if (!ik.reachable) {
                ++rejected;
                continue;
            }
            ++solved;

            // 3. 下发。setJointAngles 内部会先校验全部关节再一次性写寄存器，
            //    任一关节非法则零写入，避免机械臂进入半解姿态
            if (servo_ready && servo.setJointAngles(ik)) ++dispatched;
        }
    }

    std::cout << "[Thread 3] 停止 | 解算成功 " << solved
              << " 次，拒绝 " << rejected << " 次，下发 " << dispatched << " 次" << std::endl;
    // 注意：关闭 I2C 句柄不会停止 PCA9685 的 PWM 输出，舵机仍保持力矩
    servo.close();
}

int main(int argc, char** argv) {
    std::cout << "========================================================" << std::endl;
    std::cout << "  EdgeAI-Robotic-Arm-Control (RK3588 NPU Visual Servoing)" << std::endl;
    std::cout << "  Author: kimkingyu" << std::endl;
    std::cout << "========================================================" << std::endl;

    std::string camera_dev = "/dev/video0";
    std::string model_path = "model/yolov8n_int8.rknn";
    std::string calib_path = "configs/hand_eye_calib.json";
    std::string i2c_dev;              // 留空则完全不碰 I2C 总线
    bool allow_drive = false;         // 必须显式开启才会驱动舵机

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_next = (i + 1 < argc);
        if (arg == "--calib" && has_next) calib_path = argv[++i];
        else if (arg == "--i2c" && has_next) i2c_dev = argv[++i];
        else if (arg == "--camera" && has_next) camera_dev = argv[++i];
        else if (arg == "--model" && has_next) model_path = argv[++i];
        else if (arg == "--drive") allow_drive = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: edge_arm_control [选项]\n"
                      << "  --calib PATH   手眼标定文件 (默认 configs/hand_eye_calib.json)\n"
                      << "  --i2c DEV      PCA9685 所在 I2C 设备，如 /dev/i2c-7\n"
                      << "  --drive        允许下发舵机指令；需与 --i2c 同时使用\n"
                      << "  --camera DEV   摄像头设备 (默认 /dev/video0)\n"
                      << "  --model PATH   RKNN 模型路径\n"
                      << "默认不驱动舵机：仅指定 --i2c 而不加 --drive 时只解算不下发。\n";
            return 0;
        } else {
            std::cerr << "未知参数: " << arg << "（用 --help 查看用法）" << std::endl;
            return 2;
        }
    }
    if (allow_drive && i2c_dev.empty()) {
        std::cerr << "--drive 必须与 --i2c 同时指定，拒绝启动" << std::endl;
        return 2;
    }

    std::thread t1(camera_capture_thread, camera_dev, 640, 480);
    std::thread t2(npu_inference_thread, model_path);
    std::thread t3(control_dispatch_thread, calib_path, i2c_dev, allow_drive);

    std::cout << "[Pipeline] All 3 stages running. Press Enter to exit..." << std::endl;
    std::cin.get();

    g_running = false;
    g_raw_queue.stop();
    g_detection_queue.stop();

    if (t1.joinable()) t1.join();
    if (t2.joinable()) t2.join();
    if (t3.joinable()) t3.join();

    std::cout << "[Pipeline] System safely exited." << std::endl;
    return 0;
}
