// 供 test_pwm_parity.py 调用的 C++ PWM 换算导出程序。
// 用与 test_cpp_control.cpp 相同的 --wrap 手法接管系统调用，
// 使 PCA9685Driver 在假总线上运行并捕获真实写入的 tick 值。
// 不访问任何真实 I2C 设备。
#include "pca9685.hpp"

#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace {
std::array<uint8_t, 256> g_regs{};
uint8_t g_cursor = 0;
constexpr int kFakeFd = 91;
} // namespace

extern "C" int __wrap_open(const char* path, int, ...) {
    return std::strcmp(path, "dump://pca9685") == 0 ? kFakeFd : -1;
}
extern "C" int __wrap_ioctl(int fd, unsigned long, ...) { return fd == kFakeFd ? 0 : -1; }
extern "C" int __wrap_close(int) { return 0; }
extern "C" int __wrap_usleep(useconds_t) { return 0; }
extern "C" ssize_t __wrap_write(int fd, const void* data, size_t count) {
    if (fd != kFakeFd || count == 0) return -1;
    const auto* bytes = static_cast<const uint8_t*>(data);
    g_cursor = bytes[0];
    for (size_t i = 1; i < count; ++i) g_regs.at(g_cursor + i - 1) = bytes[i];
    return static_cast<ssize_t>(count);
}
extern "C" ssize_t __wrap_read(int fd, void* data, size_t count) {
    if (fd != kFakeFd || count != 1) return -1;
    *static_cast<uint8_t*>(data) = g_regs[g_cursor];
    return 1;
}
extern "C" ssize_t __wrap___read_chk(int fd, void* data, size_t count, size_t size) {
    return count > size ? -1 : __wrap_read(fd, data, count);
}

int main() {
    PCA9685Driver driver("dump://pca9685");
    if (!driver.open()) {
        std::cerr << "假总线初始化失败" << std::endl;
        return 1;
    }

    std::cout << std::setprecision(17);
    std::cout << "{\"period_us\":" << driver.estimatedPeriodUs() << ",\"samples\":[";

    bool first = true;
    for (int step = 0; step <= 720; ++step) {
        const double angle = step * 0.25; // 0~180°，步进 0.25°
        if (!driver.setServoAngle(0, angle)) {
            std::cerr << "角度 " << angle << " 被拒绝" << std::endl;
            return 1;
        }
        const int reg = PCA9685Driver::LED0_OFF_L;
        const int ticks = g_regs[reg] | (g_regs[reg + 1] << 8);
        if (!first) std::cout << ",";
        first = false;
        std::cout << "{\"angle\":" << angle << ",\"ticks\":" << ticks << "}";
    }

    std::cout << "]}" << std::endl;
    return 0;
}
