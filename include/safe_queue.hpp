#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>

/**
 * @brief Thread-safe blocking queue for asynchronous multi-thread pipeline.
 * @tparam T Element type stored in queue.
 */
template <typename T>
class SafeQueue {
public:
    explicit SafeQueue(size_t max_capacity = 10) 
        : max_capacity_(max_capacity), stop_flag_(false) {}

    ~SafeQueue() {
        stop();
    }

    void push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_not_full_.wait(lock, [this]() { 
            return queue_.size() < max_capacity_ || stop_flag_; 
        });

        if (stop_flag_) return;

        queue_.push(std::move(value));
        cond_not_empty_.notify_one();
    }

    std::optional<T> pop_timeout(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool acquired = cond_not_empty_.wait_for(lock, timeout, [this]() {
            return !queue_.empty() || stop_flag_;
        });

        if (!acquired || (queue_.empty() && stop_flag_)) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop();
        cond_not_full_.notify_one();
        return value;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_flag_ = true;
        }
        cond_not_empty_.notify_all();
        cond_not_full_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_not_empty_;
    std::condition_variable cond_not_full_;
    size_t max_capacity_;
    bool stop_flag_;
};
