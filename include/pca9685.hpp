#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include "kinematics.hpp"

// Linux PCA9685 driver. Angles here are SERVO degrees, not geometric IK angles.
// Default pulse endpoints are examples, not measured mechanical limits.
class PCA9685Driver {
public:
    static constexpr uint8_t MODE1 = 0x00;
    static constexpr uint8_t PRESCALE = 0xFE;
    static constexpr uint8_t LED0_ON_L = 0x06;
    static constexpr uint8_t LED0_ON_H = 0x07;
    static constexpr uint8_t LED0_OFF_L = 0x08;
    static constexpr uint8_t LED0_OFF_H = 0x09;

    explicit PCA9685Driver(const std::string& i2c_dev = "/dev/i2c-7",
                           uint8_t address = 0x40,
                           double oscillator_hz = 25000000.0,
                           double frequency_correction = 0.9)
        : i2c_dev_(i2c_dev), address_(address), oscillator_hz_(oscillator_hz),
          frequency_correction_(frequency_correction) {
        // Preserve the project's legacy correction by default. It is NOT a
        // measured universal calibration. With a measured oscillator, use 1.0.
        if (address_ > 0x7f || !std::isfinite(oscillator_hz_) || oscillator_hz_ <= 0.0 ||
            !std::isfinite(frequency_correction_) || frequency_correction_ <= 0.0) {
            throw std::invalid_argument("Invalid PCA9685 address/clock configuration");
        }
    }

    PCA9685Driver(const PCA9685Driver&) = delete;
    PCA9685Driver& operator=(const PCA9685Driver&) = delete;
    ~PCA9685Driver() { close(); }

    bool open() {
        if (is_open_) return true;
        fd_ = ::open(i2c_dev_.c_str(), O_RDWR);
        if (fd_ < 0) return false;
        if (::ioctl(fd_, I2C_SLAVE, address_) < 0) return failIO();
        is_open_ = true;
        if (!reset() || !setPWMFreq(50.0f)) return failIO();
        return true;
    }

    // Reinitializes MODE1 only; this is not a chip-wide reset or emergency stop.
    bool reset() {
        period_us_ = 0.0;
        if (!is_open_) return false;
        if (!write8(MODE1, 0x00)) return failIO();
        return true;
    }

    bool setPWMFreq(float freq_hz) {
        if (!std::isfinite(freq_hz) || freq_hz <= 0.0f) return false;
        const double freq = static_cast<double>(freq_hz) * frequency_correction_;
        if (!std::isfinite(freq) || freq <= 0.0) return false;
        const double raw = oscillator_hz_ / (4096.0 * freq) - 1.0;
        if (!std::isfinite(raw)) return false;
        const double rounded = std::round(raw);
        // NXP PRE_SCALE accepts 3..255. Reject instead of wrapping/clamping.
        if (rounded < 3.0 || rounded > 255.0 || !is_open_) return false;
        const uint8_t prescale = static_cast<uint8_t>(rounded);

        uint8_t oldmode = 0;
        if (!read8(MODE1, oldmode)) return failIO();
        const uint8_t sleepmode = (oldmode & 0x7f) | 0x10;
        // Wake even if the previous MODE1 state had SLEEP set.
        const uint8_t wakemode = oldmode & 0x6f;
        if (!write8(MODE1, sleepmode) || !write8(PRESCALE, prescale) ||
            !write8(MODE1, wakemode)) return failIO();
        ::usleep(5000);
        if (!write8(MODE1, wakemode | 0xa0)) return failIO();

        // Period ESTIMATE from the configured oscillator and written prescaler.
        // This is not an oscilloscope measurement; both conversions share it.
        period_us_ = 4096.0 * (prescale + 1.0) * 1000000.0 / oscillator_hz_;
        return true;
    }

    bool setPWM(int channel, int on, int off) {
        // Full-on/off control bits are not exposed by this counter-only API.
        if (channel < 0 || channel > 15 || on < 0 || on > 4095 || off < 0 || off > 4095 ||
            !is_open_ || period_us_ <= 0.0) return false;
        const uint8_t buf[5] = {
            static_cast<uint8_t>(LED0_ON_L + 4 * channel),
            static_cast<uint8_t>(on & 0xff),
            static_cast<uint8_t>((on >> 8) & 0x0f),
            static_cast<uint8_t>(off & 0xff),
            static_cast<uint8_t>((off >> 8) & 0x0f)
        };
        if (::write(fd_, buf, sizeof(buf)) != static_cast<ssize_t>(sizeof(buf))) return failIO();
        return true;
    }

    // Reject invalid requests before conversion or register writes. A missing
    // bus is a failure, not a successful mock movement.
    bool setServoAngle(int channel, double angle_deg,
                       double min_us = 500.0, double max_us = 2500.0) {
        uint16_t ticks = 0;
        if (channel < 0 || channel > 15 || !servoTicks(angle_deg, min_us, max_us, ticks)) return false;
        return setPWM(channel, 0, ticks);
    }

    // Default arm mapping: geometric joints 0..N-1 -> channels 0..N-1.
    // Validate ALL joints and pulse widths before the first register write.
    // Not called by main.cpp: installation calibration and hardware integration
    // are still pending. A failed I2C transfer cannot promise atomic rollback.
    bool setJointAngles(const JointAngles& solution,
                        double min_us = 500.0, double max_us = 2500.0) {
        const auto count = solution.angles.size();
        if (!solution.reachable || (count != 3 && count != 4)) return false;
        uint8_t buf[17] = {LED0_ON_L};
        for (std::size_t i = 0; i < count; ++i) {
            const double angle = solution.angles[i];
            const auto& limit = ArmKinematics::JOINT_LIMITS[i];
            if (!std::isfinite(angle) || angle < limit[0] || angle > limit[1]) return false;
            uint16_t ticks = 0;
            if (!servoTicks(angle + ArmKinematics::SERVO_OFFSETS[i], min_us, max_us, ticks)) return false;
            buf[3 + 4 * i] = static_cast<uint8_t>(ticks & 0xff);
            buf[4 + 4 * i] = static_cast<uint8_t>((ticks >> 8) & 0x0f);
        }
        const std::size_t bytes = 1 + 4 * count;
        if (::write(fd_, buf, bytes) != static_cast<ssize_t>(bytes)) return failIO();
        return true;
    }

    void close() {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        is_open_ = false;
        period_us_ = 0.0;
        // Closing the FD does NOT disable the PCA9685's autonomous PWM.
    }

    bool isOpen() const { return is_open_; }
    double estimatedPeriodUs() const { return period_us_; }

private:
    bool servoTicks(double angle_deg, double min_us, double max_us, uint16_t& out) const {
        if (!std::isfinite(angle_deg) || angle_deg < 0.0 || angle_deg > 180.0 ||
            !std::isfinite(min_us) || !std::isfinite(max_us) || min_us <= 0.0 ||
            max_us <= min_us || !is_open_ || period_us_ <= 0.0 || max_us >= period_us_) return false;
        const double pulse_us = min_us + (angle_deg / 180.0) * (max_us - min_us);
        const double ticks = std::round(pulse_us * 4096.0 / period_us_);
        if (!std::isfinite(ticks) || ticks < 1.0 || ticks > 4095.0) return false;
        out = static_cast<uint16_t>(ticks);
        return true;
    }

    bool failIO() { close(); return false; }

    bool read8(uint8_t reg, uint8_t& value) {
        if (fd_ < 0 || ::write(fd_, &reg, 1) != 1) return false;
        return ::read(fd_, &value, 1) == 1;
    }

    bool write8(uint8_t reg, uint8_t val) {
        if (fd_ < 0) return false;
        const uint8_t buf[2] = {reg, val};
        return ::write(fd_, buf, sizeof(buf)) == static_cast<ssize_t>(sizeof(buf));
    }

    std::string i2c_dev_;
    uint8_t address_;
    double oscillator_hz_;
    double frequency_correction_;
    int fd_ = -1;
    bool is_open_ = false;
    double period_us_ = 0.0;
};
