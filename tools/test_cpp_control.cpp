// Offline regression test. GNU ld wraps every device syscall used by the
// driver; this executable NEVER opens an I2C device or commands real servos.
#include "pca9685.hpp" // First include also checks header self-containment.

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
struct FakeBus {
    std::array<uint8_t, 256> regs{};
    uint8_t cursor = 0;
    int opens = 0, closes = 0, writes = 0;
    int fail_write = 0;
    bool fail_open = false, fail_ioctl = false, fail_read = false;
    std::vector<uint8_t> last_write;
} bus;
constexpr int fake_fd = 77;
int checks = 0, failures = 0;
void check(bool ok, const char* expression, int line) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cerr << "FAIL line " << line << ": " << expression << '\n';
    }
}
#define CHECK(expr) check(static_cast<bool>(expr), #expr, __LINE__)
const double nan = std::numeric_limits<double>::quiet_NaN();
const double inf = std::numeric_limits<double>::infinity();

int capturedTicks(int channel) {
    const int reg = PCA9685Driver::LED0_OFF_L + 4 * channel;
    return bus.regs[reg] | (bus.regs[reg + 1] << 8);
}

void testIK() {
    ArmKinematics kin;
    const auto known = kin.solveIK(150.0, 0.0, 105.0);
    CHECK(known.reachable && known.angles.size() == 4 && known.reject_reason.empty());
    if (known.angles.size() == 4) {
        CHECK(std::abs(known.angles[0]) < 1e-10);
        CHECK(std::abs(known.angles[1] - 60.0) < 1e-10);
        CHECK(std::abs(known.angles[2] + 120.0) < 1e-10);
        CHECK(std::abs(known.angles[3] - 60.0) < 1e-10);
    }
    double x = 0, y = 0, z = 0;
    kin.forwardKinematics({0.0, 60.0, -120.0, 60.0}, x, y, z);
    CHECK(std::abs(x - 150.0) < 1e-10 && std::abs(y) < 1e-10 && std::abs(z - 105.0) < 1e-10);

    // Fixed accept/reject expectations prevent an "always reject" solver
    // from passing merely because no returned solution has a roundtrip error.
    struct Case { double x, y, z, pitch; bool accepted; };
    const Case cases[] = {
        {200, 0, 105, 0, true}, {150, 50, 120, 0, true},
        {180, -60, 80, 0, true}, {250, 0, 100, 0, true},
        {0, 200, 105, 0, true}, {0, -200, 105, 0, true},
        {5, 0, 105, 0, true}, {295, 0, 105, 0, true},
        {4.999, 0, 105, 0, false}, {295.001, 0, 105, 0, false},
        {0, 0, 105, 0, false}, {0, 0, 200, 0, false},
        {500, 0, 105, 0, false}, {120, 0, 200, 0, false},
        {-200, 0, 105, 0, false}, {200, 0, 105, 200, false},
        {nan, 0, 105, 0, false}, {200, nan, 105, 0, false},
        {200, 0, nan, 0, false}, {200, 0, 105, nan, false},
        {inf, 0, 105, 0, false}, {200, 0, 105, inf, false},
        {1e308, 1e308, -1e308, 0, false}
    };
    for (const auto& c : cases) {
        const auto result = kin.solveIK(c.x, c.y, c.z, c.pitch);
        CHECK(result.reachable == c.accepted);
        CHECK(result.reachable ? result.reject_reason.empty()
                               : (!result.reject_reason.empty() && result.angles.empty()));
    }
    const auto pitched = kin.solveIK(200, 0, 105, 10);
    CHECK(pitched.reachable);
    if (pitched.reachable) CHECK(std::abs(pitched.angles[1] + pitched.angles[2] + pitched.angles[3] - 10) < 1e-10);

    double max_error = 0.0;
    int accepted = 0, rejected = 0;
    for (int r = 60; r < 340; r += 20) {
        for (int zz = -20; zz < 300; zz += 20) {
            const auto result = kin.solveIK(r, 0, zz);
            if (!result.reachable) {
                ++rejected;
                CHECK(!result.reject_reason.empty() && result.angles.empty());
                continue;
            }
            ++accepted;
            for (std::size_t i = 0; i < result.angles.size(); ++i) {
                CHECK(std::isfinite(result.angles[i]));
                CHECK(result.angles[i] >= ArmKinematics::JOINT_LIMITS[i][0] &&
                      result.angles[i] <= ArmKinematics::JOINT_LIMITS[i][1]);
            }
            kin.forwardKinematics(result.angles, x, y, z);
            const double error = std::hypot(std::hypot(x - r, y), z - zz);
            CHECK(error < 1e-9);
            max_error = std::max(max_error, error);
        }
    }
    CHECK(accepted > 100 && rejected > 0 && accepted + rejected == 224);
    std::cout << "IK grid: accepted=" << accepted << ", rejected=" << rejected
              << ", max_math_roundtrip_mm=" << std::scientific << max_error << '\n';

    ArmKinematics three(3);
    CHECK(three.solveIK(200, 0, 105).angles.size() == 3);
    CHECK(!three.solveIK(200, 0, 105, 10).reachable);
    ArmKinematics unequal(3, 105, 150, 100);
    CHECK(!unequal.solveIK(54.999, 0, 105).reachable);
    CHECK(unequal.solveIK(55, 0, 105).reachable);
    CHECK(!unequal.solveIK(245.001, 0, 105).reachable);
    for (int dof : {-1, 0, 1, 2, 5, 6}) {
        bool threw = false;
        try { ArmKinematics invalid(dof); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    for (double length : {-150.0, 0.0, 5.0, nan, inf, 1e308}) {
        bool threw = false;
        try { ArmKinematics invalid(4, 105, length, 150); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    for (const std::vector<double>& angles : std::vector<std::vector<double>>{{}, {0, 1}, {0, 1, 2}, {0, nan, 2, 3}}) {
        bool threw = false;
        try { kin.forwardKinematics(angles, x, y, z); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
}

void testPWM() {
    static_assert(!std::is_copy_constructible<PCA9685Driver>::value, "Driver must not duplicate FD ownership");
    bus = {};
    PCA9685Driver driver("test://pca9685");
    CHECK(!driver.setServoAngle(0, 90));
    CHECK(!driver.setPWMFreq(50));
    CHECK(!driver.setPWM(0, 0, 300));
    CHECK(bus.opens == 0 && bus.writes == 0);
    CHECK(driver.open() && driver.isOpen());
    if (!driver.isOpen()) return; // Avoid cascading assertions/metrics on a broken fixture.
    CHECK(driver.open() && bus.opens == 1);
    CHECK(bus.regs[PCA9685Driver::PRESCALE] == 135);
    CHECK((bus.regs[PCA9685Driver::MODE1] & 0x30) == 0x20);
    CHECK(std::abs(driver.estimatedPeriodUs() - 22282.24) < 1e-8);
    std::vector<int> endpoint_ticks;
    for (const auto& endpoint : std::array<std::array<int, 2>, 3>{{{{0, 92}}, {{90, 276}}, {{180, 460}}}}) {
        CHECK(driver.setServoAngle(0, endpoint[0]));
        endpoint_ticks.push_back(capturedTicks(0));
        CHECK(endpoint_ticks.back() == endpoint[1]);
    }
    std::cout << "PWM fake registers: prescale=" << int(bus.regs[PCA9685Driver::PRESCALE])
              << ", estimated_period_us=" << std::fixed << std::setprecision(2) << driver.estimatedPeriodUs()
              << ", endpoint_ticks=" << endpoint_ticks[0] << '/' << endpoint_ticks[1] << '/' << endpoint_ticks[2]
              << " (captured from test doubles; not hardware measurements)\n";

    double max_pulse_error = 0.0;
    for (float frequency : {40.0f, 50.0f, 60.0f}) {
        CHECK(driver.setPWMFreq(frequency));
        // Independent reconstruction from captured PRE_SCALE, not a duplicate
        // invocation of the production conversion or an ideal 20ms assumption.
        const double tick_us = (bus.regs[PCA9685Driver::PRESCALE] + 1.0) / 25.0;
        int previous = -1;
        for (int angle = 0; angle <= 180; ++angle) {
            CHECK(driver.setServoAngle(15, angle));
            const int ticks = capturedTicks(15);
            CHECK(ticks > previous);
            previous = ticks;
            const double target_us = 500.0 + angle * (2000.0 / 180.0);
            const double error = std::abs(ticks * tick_us - target_us);
            CHECK(error <= tick_us / 2.0 + 1e-9);
            max_pulse_error = std::max(max_pulse_error, error);
        }
    }
    std::cout << "PWM max_quantization_error_us=" << std::setprecision(6) << max_pulse_error << '\n';
    CHECK(driver.setPWMFreq(50));
    const int before = bus.writes;
    const double period = driver.estimatedPeriodUs();
    for (double bad : {-1.0, 180.001, nan, inf, -inf}) CHECK(!driver.setServoAngle(0, bad));
    for (int ch : {-1, 16, 256}) CHECK(!driver.setServoAngle(ch, 90));
    for (float bad : {0.0f, -50.0f, 1.0f, 100000.0f,
                      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        CHECK(!driver.setPWMFreq(bad));
    }
    CHECK(!driver.setServoAngle(0, 90, 2500, 500));
    CHECK(!driver.setServoAngle(0, 90, 0, 2500));
    CHECK(!driver.setServoAngle(0, 90, 500, period));
    CHECK(!driver.setServoAngle(0, 90, nan, 2500));
    CHECK(!driver.setServoAngle(0, 90, 500, inf));
    CHECK(!driver.setServoAngle(0, 0, 0.01, 0.02)); // would round to zero ticks
    CHECK(!driver.setServoAngle(0, 180, period - 1, period - 0.01)); // would round to 4096
    CHECK(!driver.setPWM(16, 0, 300));
    CHECK(!driver.setPWM(0, -1, 300));
    CHECK(!driver.setPWM(0, 0, 4096));
    CHECK(bus.writes == before && driver.estimatedPeriodUs() == period && driver.isOpen());

    CHECK(driver.setJointAngles(JointAngles{{0, 0, -120, 0}, true, {}}));
    CHECK(bus.last_write.size() == 17 && bus.last_write[0] == PCA9685Driver::LED0_ON_L);
    CHECK(capturedTicks(2) == 214); // elbow -120 -> servo 60, not clamped to zero
    CHECK(capturedTicks(0) == 276 && capturedTicks(3) == 276);
    CHECK(driver.setJointAngles(JointAngles{{0, 0, -120}, true, {}}));
    CHECK(bus.last_write.size() == 13);
    const int batch_before = bus.writes;
    CHECK(!driver.setJointAngles(JointAngles{{0, 0, -120, 91}, true, {}}));
    CHECK(!driver.setJointAngles(JointAngles{{0, 0, -120, nan}, true, {}}));
    CHECK(!driver.setJointAngles(JointAngles{{0, 0, -181, 0}, true, {}}));
    CHECK(!driver.setJointAngles(JointAngles{{0, 0, 1, 0}, true, {}}));
    CHECK(!driver.setJointAngles(JointAngles{{0, 0}, true, {}}));
    CHECK(!driver.setJointAngles(JointAngles{{0, 0, -120, 0}, false, "rejected"}));
    CHECK(!driver.setJointAngles(JointAngles{{0, 0, -120, 0}, true, {}}, 2500, 500));
    CHECK(bus.writes == batch_before);
    CHECK(driver.reset());
    CHECK(!driver.setServoAngle(0, 90));
    CHECK(driver.setPWMFreq(50) && driver.setServoAngle(0, 90));
    driver.close();
    CHECK(!driver.isOpen() && driver.estimatedPeriodUs() == 0);
    const int closes = bus.closes;
    driver.close();
    CHECK(bus.closes == closes);

    bus = {};
    PCA9685Driver calibrated("test://pca9685", 0x40, 27000000.0, 1.0);
    CHECK(calibrated.open());
    CHECK(bus.regs[PCA9685Driver::PRESCALE] == 131);
    CHECK(calibrated.setServoAngle(0, 90) && capturedTicks(0) == 307);
    calibrated.close();
}

void testIOFailures() {
    for (int failing_write = 1; failing_write <= 6; ++failing_write) {
        bus = {};
        bus.fail_write = failing_write;
        PCA9685Driver driver("test://pca9685");
        CHECK(!driver.open() && !driver.isOpen());
        CHECK(bus.closes == 1 && driver.estimatedPeriodUs() == 0);
    }
    for (int stage = 0; stage < 3; ++stage) {
        bus = {};
        bus.fail_open = stage == 0;
        bus.fail_ioctl = stage == 1;
        bus.fail_read = stage == 2;
        PCA9685Driver driver("test://pca9685");
        CHECK(!driver.open() && !driver.isOpen());
        CHECK(bus.closes == (stage == 0 ? 0 : 1));
    }
    for (int operation = 0; operation < 3; ++operation) {
        bus = {};
        PCA9685Driver driver("test://pca9685");
        CHECK(driver.open());
        bus.fail_write = bus.writes + 1;
        bool ok = operation == 0 ? driver.setServoAngle(0, 90)
                : operation == 1 ? driver.setJointAngles(JointAngles{{0, 0, -120, 0}, true, {}})
                                 : driver.setPWMFreq(60);
        CHECK(!ok && !driver.isOpen() && driver.estimatedPeriodUs() == 0);
        const int writes = bus.writes;
        CHECK(!driver.setServoAngle(0, 90) && bus.writes == writes);
    }
}
} // namespace

// Never forward to __real_*: even an unexpected path/FD fails locally.
extern "C" int __wrap_open(const char* path, int, ...) {
    ++bus.opens;
    return !bus.fail_open && std::strcmp(path, "test://pca9685") == 0 ? fake_fd : -1;
}
extern "C" int __wrap_ioctl(int fd, unsigned long, ...) {
    return fd == fake_fd && !bus.fail_ioctl ? 0 : -1;
}
extern "C" int __wrap_close(int fd) { if (fd == fake_fd) ++bus.closes; return 0; }
extern "C" int __wrap_usleep(useconds_t) { return 0; }
extern "C" ssize_t __wrap_write(int fd, const void* data, size_t count) {
    if (fd != fake_fd || count == 0) return -1;
    ++bus.writes;
    if (bus.writes == bus.fail_write) return static_cast<ssize_t>(count) - 1;
    const auto* bytes = static_cast<const uint8_t*>(data);
    bus.last_write.assign(bytes, bytes + count);
    bus.cursor = bytes[0];
    for (size_t i = 1; i < count; ++i) bus.regs.at(bus.cursor + i - 1) = bytes[i];
    return static_cast<ssize_t>(count);
}
extern "C" ssize_t __wrap_read(int fd, void* data, size_t count) {
    if (fd != fake_fd || count != 1 || bus.fail_read) return -1;
    *static_cast<uint8_t*>(data) = bus.regs[bus.cursor];
    return 1;
}

// Some optimization/sanitizer combinations retain glibc's fortified symbol
// instead of plain read. Keep that route inside the same fake bus as well.
extern "C" ssize_t __wrap___read_chk(int fd, void* data, size_t count, size_t buffer_size) {
    if (count > buffer_size) return -1;
    return __wrap_read(fd, data, count);
}

int main() {
    testIK();
    testPWM();
    testIOFailures();
    std::cout << "Offline C++ checks=" << checks << ", failures=" << failures
              << "; real hardware accesses=0 (syscalls replaced by test doubles)\n";
    return failures ? 1 : 0;
}
