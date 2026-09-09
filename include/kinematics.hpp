#pragma once

#include <array>
#include <string>
#include <vector>

// Geometric joint angles in DEGREES; never pass these directly to setServoAngle.
struct JointAngles {
    std::vector<double> angles;
    bool reachable = false;
    std::string reject_reason;
};

// Three position joints plus an optional orientation-only wrist. The returned
// XYZ is the end of the forearm; a physical tool offset is NOT modeled.
class ArmKinematics {
public:
    // Match Python's default joint mapping. These are configured assumptions,
    // not measured installation limits or a collision-avoidance guarantee.
    inline static constexpr std::array<double, 4> SERVO_OFFSETS{90.0, 90.0, 180.0, 90.0};
    inline static constexpr std::array<std::array<double, 2>, 4> JOINT_LIMITS{{
        {{-90.0, 90.0}}, {{-90.0, 90.0}}, {{-180.0, 0.0}}, {{-90.0, 90.0}}
    }};
    static constexpr double REACH_MARGIN_MM = 5.0;

    // Keep the existing C++ 105mm base default (Python defaults to 100mm).
    // Use identical explicit dimensions when comparing implementations.
    explicit ArmKinematics(int dof = 4, double base_height_mm = 105.0,
                           double upper_arm_mm = 150.0, double forearm_mm = 150.0);

    JointAngles solveIK(double x, double y, double z, double pitch = 0.0);
    // Invalid FK input explicitly throws instead of leaving stale output values.
    void forwardKinematics(const std::vector<double>& joints, double& out_x, double& out_y, double& out_z);

private:
    int dof_;
    double l1_, l2_, l3_; // mm; no unmodeled tool length in the solver
};
