#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

namespace cv { class Mat; }
namespace edgeai::yolo {
// This interface has no controller, camera, language-model or RKNN SDK dependency.
struct Transform {
    std::string mode;
    int source_width = 0, source_height = 0, target_width = 0, target_height = 0;
    int resized_width = 0, resized_height = 0, pad_left = 0, pad_top = 0;
    double scale_x = 0, scale_y = 0;
};
Transform make_transform(int sw, int sh, int tw, int th, const std::string& mode);
void validate_transform(const Transform& transform);
struct Input {
    Transform transform;
    std::vector<std::uint8_t> rgb; // contiguous uint8 NHWC, batch=1; no normalization
};
Input preprocess(const cv::Mat& bgr, int width, int height, const std::string& mode);
struct Tensor {
    std::vector<std::uint32_t> dims;
    std::vector<float> values; // SDK want_float=1, dense channel-major output
};
std::string validate_output_layout(const std::vector<std::vector<std::uint32_t>>& dims);
// Validates raw SDK AFFINE metadata only; never rescales an input or output tensor.
void validate_affine_metadata(const std::string& type, double scale, int zero_point);
struct Detection {
    int class_id = 0;
    std::string class_name;
    double score = 0;
    std::array<double, 4> source_box{};
    std::array<int, 4> bbox{}; // Python round-to-nearest-even, after clipping to width/height - 1
    std::array<int, 2> center{};
};
std::vector<Detection> decode(const std::vector<Tensor>& outputs, const Transform& transform,
                              double confidence = 0.25, double nms = 0.45,
                              std::size_t max_detections = 100);
struct TensorInfo {
    std::uint32_t index = 0;
    std::string name, format, type, quantization;
    std::vector<std::uint32_t> dims;
    std::uint32_t elements = 0, bytes = 0, width_stride = 0, height_stride = 0, bytes_with_stride = 0;
    int zero_point = 0, fractional_length = 0;
    double scale = 0;
};
struct RuntimeInfo {
    std::string provider, sdk_version, driver_version, layout;
    int core_mask = 0, input_width = 0, input_height = 0;
    TensorInfo input;
    std::vector<TensorInfo> outputs;
};
struct Inference {
    std::vector<Tensor> outputs;
    double inference_call_ms = 0; // inputs_set + synchronous run (not hardware-only latency)
    double output_ms = 0; // outputs_get + copy + outputs_release
};
class Runtime {
public:
    virtual ~Runtime() = default;
    virtual const RuntimeInfo& info() const = 0;
    virtual Inference infer(const Input& input) = 0;
};
std::unique_ptr<Runtime> create_runtime(const std::string& model, int core_mask);
bool sdk_available() noexcept;
struct QueueStats {
    std::size_t capacity = 0, peak = 0, pushes = 0, pops = 0, cancelled_entries = 0;
    double producer_wait_ms = 0, consumer_wait_ms = 0;
    bool cancelled = false;
};
// close() drains queued work; cancel() discards it and wakes both sides.
// Polling also wakes blocked sides on SIGINT/SIGTERM without locking in a signal handler.
template<class T> class BoundedQueue {
public:
    using Clock = std::chrono::steady_clock;
    struct Entry { T value; Clock::time_point enqueued; double producer_wait_ms; };
    explicit BoundedQueue(std::size_t capacity, std::function<bool()> interrupted = {})
        : interrupted_(std::move(interrupted)) {
        if (!capacity) throw std::invalid_argument("Queue capacity must be positive");
        stats_.capacity = capacity;
    }
    ~BoundedQueue() { cancel(); }
    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;
    bool push(T value) {
        const auto start=Clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        while (!closed_ && !stats_.cancelled && queue_.size()==stats_.capacity) {
            if (interrupt_locked()) break;
            changed_.wait_for(lock,std::chrono::milliseconds(25));
        }
        interrupt_locked();
        const auto now=Clock::now();
        const double wait=std::chrono::duration<double,std::milli>(now-start).count();
        stats_.producer_wait_ms+=wait;
        if (closed_ || stats_.cancelled) return false;
        queue_.push_back({std::move(value),now,wait}); ++stats_.pushes;
        if (queue_.size()>stats_.peak) stats_.peak=queue_.size();
        changed_.notify_all(); return true;
    }
    std::optional<Entry> pop() {
        const auto start=Clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        while (!closed_ && !stats_.cancelled && queue_.empty()) {
            if (interrupt_locked()) break;
            changed_.wait_for(lock,std::chrono::milliseconds(25));
        }
        interrupt_locked();
        stats_.consumer_wait_ms+=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        if (stats_.cancelled || queue_.empty()) return std::nullopt;
        Entry item=std::move(queue_.front()); queue_.pop_front(); ++stats_.pops;
        changed_.notify_all(); return item;
    }
    void close() { std::lock_guard<std::mutex> lock(mutex_); closed_=true; changed_.notify_all(); }
    void cancel() { std::lock_guard<std::mutex> lock(mutex_); cancel_locked(); }
    bool cancelled() const { std::lock_guard<std::mutex> lock(mutex_); return stats_.cancelled; }
    QueueStats stats() const { std::lock_guard<std::mutex> lock(mutex_); return stats_; }
private:
    void cancel_locked() {
        stats_.cancelled=true; stats_.cancelled_entries+=queue_.size(); queue_.clear(); changed_.notify_all();
    }
    bool interrupt_locked() {
        if (interrupted_ && interrupted_()) cancel_locked();
        return stats_.cancelled;
    }
    std::function<bool()> interrupted_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Entry> queue_;
    QueueStats stats_;
    bool closed_=false;
};
struct Options {
    std::string model, image, image_list, video, report, geometry = "resize";
    int workers = 1, iterations = 1, warmup = 0, queue_capacity = 4;
    bool overwrite = false, help = false;
};
Options parse_options(const std::vector<std::string>& arguments);
const char* usage() noexcept;
std::string json_quote(const std::string& value);
std::string json_number(double value); // non-finite -> null, locale independent
std::string transform_json(const Transform& transform);
} // namespace edgeai::yolo
