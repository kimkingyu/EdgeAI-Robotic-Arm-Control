#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

/**
 * @brief Frame data packet passed through the pipeline.
 */
struct FramePacket {
    uint64_t frame_id;
    int64_t timestamp_us;
    cv::Mat image;
};

/**
 * @brief V4L2 Camera capture driver with DMA-BUF zero-copy support.
 */
class CameraV4L2 {
public:
    CameraV4L2(const std::string& device_path, int width, int height, int fps = 60);
    ~CameraV4L2();

    bool open();
    void close();
    bool capture(FramePacket& packet);

private:
    std::string device_path_;
    int width_;
    int height_;
    int fps_;
    int fd_;
    bool is_opened_;
    uint64_t frame_count_;
};
