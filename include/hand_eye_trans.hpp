#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// 机械臂基座坐标系下的目标位姿 (mm)
struct RobotTargetPose {
    double x = 0.0;      // 前向
    double y = 0.0;      // 左右
    double z = 0.0;      // 高度
    bool is_valid = false;
    std::string reject_reason;
};

/**
 * @brief Eye-to-Hand 平面单应手眼变换（与 Python 侧 src/vision/hand_eye.py 对齐）
 *
 * 为什么是单应而非完整 PnP：本项目是 3-DOF 机械臂在固定工作平面上抓取，
 * 像素平面到物理平面是射影变换，单应矩阵即可精确描述，且无需标定相机内参
 * 与畸变系数。此处只做变换与载入，标定求解仍由 Python 侧完成 ——
 * 标定是低频离线作业，不必在实时控制路径里重复实现一遍求解器。
 *
 * 本类不访问相机或 I2C 设备。
 */
class HandEyeTransformer {
public:
    HandEyeTransformer() = default;

    /**
     * 直接设置单应矩阵。校验规则必须与 Python 侧 load() 一致：
     * 3x3、全有限、可逆且回乘接近单位阵。
     *
     * 回乘校验不可省：cv::invert 对秩亏矩阵不报错，会返回数值垃圾，
     * 且每个元素都是有限值能通过 isfinite 检查（Python 侧实测
     * [[1,2,3],[2,4,6],[3,6,9]] 的逆元素达 1e16，回乘偏离单位阵 3.0）。
     */
    bool setHomography(const cv::Mat& H, double z_plane = 30.0) {
        if (H.empty() || H.rows != 3 || H.cols != 3) return false;
        cv::Mat h;
        H.convertTo(h, CV_64F);
        if (!cv::checkRange(h, true, nullptr, -1e300, 1e300)) return false;

        cv::Mat inv;
        if (!cv::invert(h, inv, cv::DECOMP_LU)) return false;
        if (!cv::checkRange(inv, true, nullptr, -1e300, 1e300)) return false;
        const double residual = cv::norm(h * inv - cv::Mat::eye(3, 3, CV_64F), cv::NORM_INF);
        if (!(residual < 1e-6)) return false;

        if (!std::isfinite(z_plane)) return false;
        H_ = h;
        H_inv_ = inv;
        z_plane_ = z_plane;
        is_calibrated_ = true;
        return true;
    }

    // 载入 Python 侧 HandEyeCalibrator.save() 产出的标定文件。
    // 字段名与结构必须与其保持一致，否则实机会用错标定参数。
    bool loadCalibration(const std::string& path) {
        cv::FileStorage fs;
        // OpenCV 的 FileStorage 支持 JSON；文件不存在或格式非法均返回 false
        try {
            if (!fs.open(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON)) {
                return false;
            }
        } catch (const cv::Exception&) {
            return false;
        }

        std::vector<std::vector<double>> rows;
        double z_plane = 30.0;
        try {
            cv::FileNode node = fs["homography"];
            if (node.empty() || !node.isSeq() || node.size() != 3) return false;
            for (const auto& row : node) {
                std::vector<double> values;
                row >> values;
                if (values.size() != 3) return false;
                rows.push_back(values);
            }
            if (!fs["z_plane"].empty()) fs["z_plane"] >> z_plane;
        } catch (const cv::Exception&) {
            return false;
        }

        cv::Mat H(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) H.at<double>(r, c) = rows[r][c];
        return setHomography(H, z_plane);
    }

    /**
     * 像素坐标 → 基座坐标。未标定时返回 is_valid=false 而非退回某种
     * "近似映射" —— 悄悄用错误坐标继续执行，比明确失败危险得多。
     */
    RobotTargetPose pixelToRobotBase(const cv::Point2f& pixel) const {
        RobotTargetPose pose;
        if (!is_calibrated_) {
            pose.reject_reason = "未载入手眼标定参数";
            return pose;
        }
        if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)) {
            pose.reject_reason = "像素坐标非有限值";
            return pose;
        }

        const cv::Matx31d p(pixel.x, pixel.y, 1.0);
        const cv::Matx33d h(H_.at<double>(0, 0), H_.at<double>(0, 1), H_.at<double>(0, 2),
                            H_.at<double>(1, 0), H_.at<double>(1, 1), H_.at<double>(1, 2),
                            H_.at<double>(2, 0), H_.at<double>(2, 1), H_.at<double>(2, 2));
        const cv::Matx31d v = h * p;
        // 齐次归一化前必须查分母：射影变换在无穷远线上分母为 0
        if (!std::isfinite(v(2)) || std::abs(v(2)) < 1e-12) {
            pose.reject_reason = "射影分母接近零（点位于无穷远线附近）";
            return pose;
        }
        const double x = v(0) / v(2);
        const double y = v(1) / v(2);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            pose.reject_reason = "变换结果非有限值";
            return pose;
        }

        pose.x = x;
        pose.y = y;
        pose.z = z_plane_;
        pose.is_valid = true;
        return pose;
    }

    // 基座坐标 → 像素坐标，用于把规划结果画回画面做可视化校验
    bool robotBaseToPixel(double x, double y, cv::Point2f& out) const {
        if (!is_calibrated_ || !std::isfinite(x) || !std::isfinite(y)) return false;
        const cv::Matx31d p(x, y, 1.0);
        const cv::Matx33d hi(H_inv_.at<double>(0, 0), H_inv_.at<double>(0, 1), H_inv_.at<double>(0, 2),
                             H_inv_.at<double>(1, 0), H_inv_.at<double>(1, 1), H_inv_.at<double>(1, 2),
                             H_inv_.at<double>(2, 0), H_inv_.at<double>(2, 1), H_inv_.at<double>(2, 2));
        const cv::Matx31d v = hi * p;
        if (!std::isfinite(v(2)) || std::abs(v(2)) < 1e-12) return false;
        const double u = v(0) / v(2);
        const double w = v(1) / v(2);
        if (!std::isfinite(u) || !std::isfinite(w)) return false;
        out = cv::Point2f(static_cast<float>(u), static_cast<float>(w));
        return true;
    }

    bool isCalibrated() const { return is_calibrated_; }
    double zPlane() const { return z_plane_; }

private:
    cv::Mat H_;
    cv::Mat H_inv_;
    double z_plane_ = 30.0;
    bool is_calibrated_ = false;
};
