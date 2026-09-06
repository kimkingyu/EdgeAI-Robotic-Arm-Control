#include "kinematics.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ArmKinematics::ArmKinematics(int dof)
    : dof_(dof), l1_(105.0), l2_(150.0), l3_(150.0), l4_(80.0) {}

JointAngles ArmKinematics::solveIK(double x, double y, double z, double pitch) {
    JointAngles res;
    res.reachable = false;
    res.angles.resize(dof_, 0.0);

    // 1. Base yaw angle (degrees)
    double base_deg = std::atan2(y, x) * 180.0 / M_PI;
    res.angles[0] = base_deg;

    // 2. Projected radius and height
    double r = std::sqrt(x * x + y * y);
    double dz = z - l1_;

    // Target point distance from shoulder joint
    double dist = std::sqrt(r * r + dz * dz);
    double max_reach = l2_ + l3_;
    double min_reach = std::abs(l2_ - l3_);

    if (dist > max_reach || dist < min_reach || dist < 1e-3) {
        return res; // Unreachable
    }

    // Cosine law for elbow angle (theta2)
    double cos_elbow = (l2_ * l2_ + l3_ * l3_ - dist * dist) / (2.0 * l2_ * l3_);
    cos_elbow = std::clamp(cos_elbow, -1.0, 1.0);
    double elbow_rad = std::acos(cos_elbow);
    double elbow_deg = 180.0 - (elbow_rad * 180.0 / M_PI);

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

    res.reachable = true;
    return res;
}

void ArmKinematics::forwardKinematics(const std::vector<double>& joints, double& out_x, double& out_y, double& out_z) {
    if (joints.size() < 3) return;

    double rad0 = joints[0] * M_PI / 180.0;
    double rad1 = joints[1] * M_PI / 180.0;
    double rad2 = joints[2] * M_PI / 180.0;

    double r = l2_ * std::cos(rad1) + l3_ * std::cos(rad1 + rad2);
    out_x = r * std::cos(rad0);
    out_y = r * std::sin(rad0);
    out_z = l1_ + l2_ * std::sin(rad1) + l3_ * std::sin(rad1 + rad2);
}
