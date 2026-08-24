#pragma once

#include <opencv2/opencv.hpp>

/**
 * @brief 3D Spatial Position in Robot Base Coordinate System (Unit: mm).
 */
struct RobotTargetPose {
    double x; // Forward
    double y; // Left/Right
    double z; // Height
    double yaw;
    bool is_valid;
};

/**
 * @brief Eye-to-Hand Calibration and PnP 2D-to-3D Spatial Transformer.
 */
class HandEyeTransformer {
public:
    HandEyeTransformer();

    void setCameraIntrinsics(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
    void setExtrinsics(const cv::Mat& rvec, const cv::Mat& tvec);
    
    RobotTargetPose pixelToRobotBase(const cv::Point2f& pixel_pt, double target_height_plane_z = 0.0);

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Mat rvec_;
    cv::Mat tvec_;
    cv::Mat rmat_;
    bool is_calibrated_;
};
