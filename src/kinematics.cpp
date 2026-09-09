#include "kinematics.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ArmKinematics::ArmKinematics(int dof, double base_height_mm,
                             double upper_arm_mm, double forearm_mm)
    : dof_(dof), l1_(base_height_mm), l2_(upper_arm_mm), l3_(forearm_mm) {
    if ((dof != 3 && dof != 4) || !std::isfinite(l1_) || l1_ < 0.0 ||
        !std::isfinite(l2_) || !std::isfinite(l3_) || l2_ <= REACH_MARGIN_MM ||
        l3_ <= REACH_MARGIN_MM || !std::isfinite(l2_ + l3_) ||
        !std::isfinite(l2_ * l2_ + l3_ * l3_)) {
        throw std::invalid_argument("Expected 3/4 DOF and finite valid link dimensions");
    }
}

JointAngles ArmKinematics::solveIK(double x, double y, double z, double pitch) {
    JointAngles res;
    const auto reject = [](const std::string& reason) {
        return JointAngles{{}, false, reason};
    };
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(pitch)) {
        return reject("Non-finite target coordinate or pitch");
    }
    if (dof_ == 3 && pitch != 0.0) return reject("Pitch requires a fourth orientation joint");
    res.angles.resize(dof_, 0.0);

    // 1. Base yaw angle (degrees)
    double base_deg = std::atan2(y, x) * 180.0 / M_PI;
    res.angles[0] = base_deg;

    // 2. Projected radius and height
    double r = std::hypot(x, y);
    double dz = z - l1_;

    // Hard radial bounds measured from the shoulder, with the Python margin.
    double dist = std::hypot(r, dz);
    double max_reach = l2_ + l3_ - REACH_MARGIN_MM;
    double min_reach = std::max(std::abs(l2_ - l3_) + REACH_MARGIN_MM, 1e-3);

    if (!std::isfinite(dist) || dist > max_reach) return reject("Beyond maximum reach");
    if (dist < min_reach) return reject("Inside minimum reach / near singularity");
    if (r < 1e-9) return reject("Base yaw undefined on the vertical axis");

    // Cosine law for elbow angle (theta2)
    double cos_elbow = (l2_ * l2_ + l3_ * l3_ - dist * dist) / (2.0 * l2_ * l3_);
    cos_elbow = std::clamp(cos_elbow, -1.0, 1.0);
    double elbow_rad = std::acos(cos_elbow);
    // Negative relative turn matches FK's shoulder + elbow convention.
    double elbow_deg = -(180.0 - (elbow_rad * 180.0 / M_PI));

    // Shoulder angle (theta1)
    double alpha1 = std::atan2(dz, r);
    double cos_alpha2 = (l2_ * l2_ + dist * dist - l3_ * l3_) / (2.0 * l2_ * dist);
    cos_alpha2 = std::clamp(cos_alpha2, -1.0, 1.0);
    double alpha2 = std::acos(cos_alpha2);
    double shoulder_deg = (alpha1 + alpha2) * 180.0 / M_PI;

    res.angles[1] = shoulder_deg;
    res.angles[2] = elbow_deg;

    // Wrist pitch compensation
    if (dof_ >= 4) {
        double wrist_deg = -(shoulder_deg + elbow_deg) + pitch;
        res.angles[3] = wrist_deg;
    }

    // Validate the entire solution before exposing ANY commandable angles.
    for (int i = 0; i < dof_; ++i) {
        const double angle = res.angles[i];
        const auto& limit = JOINT_LIMITS[i];
        if (!std::isfinite(angle) || angle < limit[0] || angle > limit[1]) {
            return reject("Joint " + std::to_string(i) + " exceeds configured travel");
        }
    }
    res.reachable = true;
    return res;
}

void ArmKinematics::forwardKinematics(const std::vector<double>& joints, double& out_x, double& out_y, double& out_z) {
    if (joints.size() != static_cast<std::size_t>(dof_) ||
        !std::all_of(joints.begin(), joints.end(), [](double a) { return std::isfinite(a); })) {
        throw std::invalid_argument("FK expects exactly the configured DOF and finite angles");
    }

    double rad0 = joints[0] * M_PI / 180.0;
    double rad1 = joints[1] * M_PI / 180.0;
    double rad2 = joints[2] * M_PI / 180.0;

    double r = l2_ * std::cos(rad1) + l3_ * std::cos(rad1 + rad2);
    out_x = r * std::cos(rad0);
    out_y = r * std::sin(rad0);
    out_z = l1_ + l2_ * std::sin(rad1) + l3_ * std::sin(rad1 + rad2);
}
