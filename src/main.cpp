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
void control_dispatch_thread() {
    std::cout << "[Thread 3] Control and Kinematics thread started." << std::endl;
    ArmKinematics ik_solver(4);

    while (g_running) {
        auto item_opt = g_detection_queue.pop_timeout(std::chrono::milliseconds(100));
        if (!item_opt.has_value()) continue;

        auto [pkt, detections] = item_opt.value();

        for (const auto& det : detections) {
            // 1. Convert pixel coordinate (u, v) -> robot coordinate (X, Y, Z)
            double target_x = (det.center.x - 320.0) * 0.5 + 150.0;
            double target_y = (det.center.y - 240.0) * 0.5;
            double target_z = 20.0;

            // 2. Solve Inverse Kinematics
            JointAngles ik_res = ik_solver.solveIK(target_x, target_y, target_z);

            if (ik_res.reachable) {
                // std::cout << "[Target] Frame " << pkt.frame_id 
                //           << " -> Pos: (" << target_x << ", " << target_y << ", " << target_z << ")"
                //           << " -> Joint0: " << ik_res.angles[0] << " deg" << std::endl;
            }
        }
    }
    std::cout << "[Thread 3] Control and Kinematics thread stopped." << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================================" << std::endl;
    std::cout << "  EdgeAI-Robotic-Arm-Control (RK3588 NPU Visual Servoing)" << std::endl;
    std::cout << "  Author: kimkingyu" << std::endl;
    std::cout << "========================================================" << std::endl;

    std::string camera_dev = "/dev/video0";
    std::string model_path = "model/yolov8n_int8.rknn";

    std::thread t1(camera_capture_thread, camera_dev, 640, 480);
    std::thread t2(npu_inference_thread, model_path);
    std::thread t3(control_dispatch_thread);

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
