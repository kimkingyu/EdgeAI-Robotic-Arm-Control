// 供 test_ik_parity.py 调用的 C++ 逆解导出程序。
// 从标准输入逐行读取 "x y z"，以 JSON 数组输出每个目标点的解。
// 仅做纯数学求解，不访问任何硬件。
#include "kinematics.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string escape(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}
} // namespace

int main() {
    // 与 Python 侧 SimpleArmKinematics 的默认连杆参数保持一致
    ArmKinematics kin(4, 100.0, 150.0, 150.0);

    std::cout << std::setprecision(17);
    std::cout << "[";
    std::string line;
    bool first = true;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        double x = 0, y = 0, z = 0;
        std::istringstream iss(line);
        if (!(iss >> x >> y >> z)) continue;

        const auto result = kin.solveIK(x, y, z);
        if (!first) std::cout << ",";
        first = false;
        std::cout << "{\"reachable\":" << (result.reachable ? "true" : "false");
        if (result.reachable) {
            std::cout << ",\"angles\":[";
            for (std::size_t i = 0; i < result.angles.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << result.angles[i];
            }
            std::cout << "]";
        } else {
            std::cout << ",\"reason\":\"" << escape(result.reject_reason) << "\"";
        }
        std::cout << "}";
    }
    std::cout << "]" << std::endl;
    return 0;
}
