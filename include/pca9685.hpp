#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

/**
 * @brief 现代 C++17 实现的高性能 Linux 原生 I2C (PCA9685) 舵机驱动
 * 参考 Adafruit-PWM-Servo 核心算法与寄存器定义规范
 */
class PCA9685Driver {
public:
    static constexpr uint8_t MODE1 = 0x00;
    static constexpr uint8_t PRESCALE = 0xFE;
    static constexpr uint8_t LED0_ON_L = 0x06;
    static constexpr uint8_t LED0_ON_H = 0x07;
    static constexpr uint8_t LED0_OFF_L = 0x08;
    static constexpr uint8_t LED0_OFF_H = 0x09;

    explicit PCA9685Driver(const std::string& i2c_dev = "/dev/i2c-7", uint8_t address = 0x40)
        : i2c_dev_(i2c_dev), address_(address), fd_(-1), is_open_(false) {}

    ~PCA9685Driver() {
        close();
    }

    bool open() {
        fd_ = ::open(i2c_dev_.c_str(), O_RDWR);
        if (fd_ < 0) {
            std::cout << "[PCA9685-C++] (Mock) 无法打开硬件总线 " << i2c_dev_ << "，转入 Mock 模拟模式" << std::endl;
            is_open_ = false;
            return false;
        }

        if (ioctl(fd_, I2C_SLAVE, address_) < 0) {
            std::cerr << "[PCA9685-C++] 无法获取 I2C 从机地址 0x" << std::hex << (int)address_ << std::dec << std::endl;
            ::close(fd_);
            fd_ = -1;
            is_open_ = false;
            return false;
        }

        is_open_ = true;
        reset();
        setPWMFreq(50.0f); // 舵机标准 50Hz
        std::cout << "[PCA9685-C++] 成功打开 I2C 总线 " << i2c_dev_ << " (从机地址: 0x" << std::hex << (int)address_ << std::dec << ")" << std::endl;
        return true;
    }

    void reset() {
        if (is_open_) {
            write8(MODE1, 0x00);
        }
    }

    void setPWMFreq(float freq_hz) {
        if (!is_open_) return;

        // Adafruit 官方校准算法: 乘以 0.9 修正 PCA9685 内部 25MHz RC 振荡器频率过冲 (Issue #11)
        float freq = freq_hz * 0.9f;
        float prescaleval = 25000000.0f; // 25MHz 内部时钟
        prescaleval /= 4096.0f;          // 12-bit 分辨率
        prescaleval /= freq;
        prescaleval -= 1.0f;

        uint8_t prescale = static_cast<uint8_t>(std::floor(prescaleval + 0.5f));

        uint8_t oldmode = read8(MODE1);
        uint8_t newmode = (oldmode & 0x7F) | 0x10; // 进入 Sleep 模式以修改预分频器
        write8(MODE1, newmode);
        write8(PRESCALE, prescale);
        write8(MODE1, oldmode);
        usleep(5000);
        write8(MODE1, oldmode | 0xa1); // 恢复并开启 Auto-Increment
    }

    void setPWM(uint8_t channel, uint16_t on, uint16_t off) {
        if (channel > 15) return;
        if (!is_open_) {
            // Mock 输出
            return;
        }

        uint8_t reg = LED0_ON_L + 4 * channel;
        uint8_t buf[5] = {
            reg,
            static_cast<uint8_t>(on & 0xFF),
            static_cast<uint8_t>(on >> 8),
            static_cast<uint8_t>(off & 0xFF),
            static_cast<uint8_t>(off >> 8)
        };
        ::write(fd_, buf, 5);
    }

    void setServoAngle(uint8_t channel, double angle_deg, double min_us = 500.0, double max_us = 2500.0) {
        angle_deg = std::clamp(angle_deg, 0.0, 180.0);
        double pulse_us = min_us + (angle_deg / 180.0) * (max_us - min_us);
        // 50Hz 下，周期为 20000 us, 映射到 4096 ticks
        uint16_t off_ticks = static_cast<uint16_t>(std::round(pulse_us * 4096.0 / 20000.0));
        setPWM(channel, 0, off_ticks);
    }

    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
            is_open_ = false;
        }
    }

    bool isOpen() const { return is_open_; }

private:
    uint8_t read8(uint8_t reg) {
        if (fd_ < 0) return 0;
        ::write(fd_, &reg, 1);
        uint8_t val = 0;
        ::read(fd_, &val, 1);
        return val;
    }

    void write8(uint8_t reg, uint8_t val) {
        if (fd_ < 0) return;
        uint8_t buf[2] = {reg, val};
        ::write(fd_, buf, 2);
    }

    std::string i2c_dev_;
    uint8_t address_;
    int fd_;
    bool is_open_;
};
