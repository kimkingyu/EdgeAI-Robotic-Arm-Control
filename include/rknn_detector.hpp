#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @brief Detection bounding box result.
 */
struct DetectionResult {
    int class_id;
    float confidence;
    cv::Rect2f box;
    cv::Point2f center; // (u, v) pixel center
};

/**
 * @brief RKNN-based NPU object detector on RK3588.
 */
class RKNNDetector {
public:
    RKNNDetector();
    ~RKNNDetector();

    bool init(const std::string& model_path, int num_threads = 3);
    bool detect(const cv::Mat& input_image, std::vector<DetectionResult>& results, float conf_threshold = 0.5f, float nms_threshold = 0.45f);
    void release();

private:
    std::string model_path_;
    int model_width_;
    int model_height_;
    int model_channel_;
    void* rknn_ctx_;
    bool is_initialized_;
};
