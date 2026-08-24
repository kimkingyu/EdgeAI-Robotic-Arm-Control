#pragma once

#include <vector>
#include <cmath>

/**
 * @brief Joint angles for multi-DOF robot arm (Unit: degrees/radians).
 */
struct JointAngles {
    std::vector<double> angles; // theta_1, theta_2, ..., theta_n
    bool reachable;
};

/**
 * @brief Forward and Inverse Kinematics solver.
 */
class ArmKinematics {
public:
    explicit ArmKinematics(int dof = 4);

    JointAngles solveIK(double x, double y, double z, double pitch = 0.0);
    void forwardKinematics(const std::vector<double>& joints, double& out_x, double& out_y, double& out_z);

private:
    int dof_;
    double l1_, l2_, l3_, l4_; // Link lengths in mm
};
