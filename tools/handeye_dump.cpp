// 供 test_handeye_parity.py 调用的 C++ 手眼变换导出程序。
// 从命令行取标定文件，从 stdin 逐行读 "u v"，输出 JSON。
// 不访问相机或 I2C 设备。
#include "hand_eye_trans.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: handeye_dump <标定文件>" << std::endl;
        return 2;
    }

    HandEyeTransformer transformer;
    const bool loaded = transformer.loadCalibration(argv[1]);

    std::cout << std::setprecision(17);
    std::cout << "{\"loaded\":" << (loaded ? "true" : "false")
              << ",\"z_plane\":" << transformer.zPlane() << ",\"points\":[";

    std::string line;
    bool first = true;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        double u = 0, v = 0;
        std::istringstream iss(line);
        if (!(iss >> u >> v)) continue;

        const auto pose = transformer.pixelToRobotBase(
            cv::Point2f(static_cast<float>(u), static_cast<float>(v)));
        if (!first) std::cout << ",";
        first = false;
        std::cout << "{\"valid\":" << (pose.is_valid ? "true" : "false");
        if (pose.is_valid) {
            std::cout << ",\"x\":" << pose.x << ",\"y\":" << pose.y << ",\"z\":" << pose.z;
            cv::Point2f back;
            if (transformer.robotBaseToPixel(pose.x, pose.y, back)) {
                std::cout << ",\"back_u\":" << back.x << ",\"back_v\":" << back.y;
            }
        }
        std::cout << "}";
    }

    std::cout << "]}" << std::endl;
    return 0;
}
