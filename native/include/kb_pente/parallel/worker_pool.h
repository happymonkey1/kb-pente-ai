#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace kb_pente {

// WorkerPoolWaveTelemetry is a detached report of one synchronous worker
// wave. items is the requested/scheduled index count; a canceled wave may
// execute fewer callbacks. Callback busy time is summed across workers, so
// busy_fraction is normalized by wall_seconds multiplied by workers.
struct WorkerPoolWaveTelemetry final {
    std::uint64_t items = 0U;
    std::size_t workers = 0U;
    double wall_seconds = 0.0;
    double callback_busy_seconds = 0.0;
    double busy_fraction = 0.0;
};

[[nodiscard]] inline bool operator==(
    const WorkerPoolWaveTelemetry& left,
    const WorkerPoolWaveTelemetry& right) noexcept {
    return left.items == right.items && left.workers == right.workers &&
           left.wall_seconds == right.wall_seconds &&
           left.callback_busy_seconds == right.callback_busy_seconds &&
           left.busy_fraction == right.busy_fraction;
}

[[nodiscard]] inline bool operator!=(
    const WorkerPoolWaveTelemetry& left,
    const WorkerPoolWaveTelemetry& right) noexcept {
    return !(left == right);
}

// WorkerPoolTelemetry keeps detached reports for the most recent wave and
// the cumulative waves executed by one persistent pool.
struct WorkerPoolTelemetry final {
    WorkerPoolWaveTelemetry cumulative{};
    WorkerPoolWaveTelemetry last_wave{};
};

[[nodiscard]] inline bool operator==(
    const WorkerPoolTelemetry& left,
    const WorkerPoolTelemetry& right) noexcept {
    return left.cumulative == right.cumulative &&
           left.last_wave == right.last_wave;
}

[[nodiscard]] inline bool operator!=(
    const WorkerPoolTelemetry& left,
    const WorkerPoolTelemetry& right) noexcept {
    return !(left == right);
}

// Owns a fixed set of workers and executes synchronous index waves over it.
// One shared callable and an atomic cursor avoid allocating a task per index.
class WorkerPool final {
private:
    using WaveFunction = std::function<void(std::size_t)>;

public:
    explicit WorkerPool(std::size_t thread_count);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    WorkerPool(WorkerPool&&) = delete;
    WorkerPool& operator=(WorkerPool&&) = delete;

    [[nodiscard]] std::size_t thread_count() const noexcept {
        return workers_.size();
    }

    // Return detached reports without exposing mutable pool state.
    [[nodiscard]] WorkerPoolTelemetry telemetry() const;

    // Runs function once for every un-cancelled index in [0, count). Calls
    // from external threads are serialized. A callback cannot synchronously
    // submit another wave to this same pool.
    template <typename Function>
    void parallel_for(std::size_t count, Function&& function) {
        if (count == 0U) {
            return;
        }
        if (current_pool_ == this) {
            throw std::logic_error(
                "WorkerPool does not allow nested parallel_for calls");
        }

        WaveFunction retired_function;
        std::exception_ptr failure;
        {
            std::unique_lock<std::mutex> caller_lock(call_mutex_);
            WaveFunction next_function(std::forward<Function>(function));
            const auto wall_started = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> state_lock(state_mutex_);
                if (stopping_) {
                    throw std::logic_error("WorkerPool is stopping");
                }

                wave_function_ = std::move(next_function);
                wave_count_ = count;
                next_index_.store(0U, std::memory_order_relaxed);
                cancellation_.store(false, std::memory_order_release);
                first_exception_ = nullptr;
                active_workers_ = workers_.size();
                ++wave_generation_;
            }
            work_cv_.notify_all();

            {
                std::unique_lock<std::mutex> state_lock(state_mutex_);
                completion_cv_.wait(
                    state_lock, [this] { return active_workers_ == 0U; });
                publish_wave_telemetry(
                    count,
                    wall_started,
                    std::chrono::steady_clock::now());
                failure = first_exception_;
                first_exception_ = nullptr;
                cancellation_.store(false, std::memory_order_release);
                next_index_.store(0U, std::memory_order_relaxed);
                wave_count_ = 0U;
                retired_function = std::move(wave_function_);
            }
        }

        if (failure) {
            std::rethrow_exception(failure);
        }
    }

private:
    static void add_busy_nanoseconds(
        std::atomic<std::uint64_t>& total,
        std::uint64_t value) noexcept;
    static void add_callback_busy_nanoseconds(
        std::uint64_t& total,
        std::chrono::steady_clock::time_point started,
        std::chrono::steady_clock::time_point finished) noexcept;
    void publish_wave_telemetry(
        std::size_t items,
        std::chrono::steady_clock::time_point started,
        std::chrono::steady_clock::time_point finished) noexcept;

    void worker_loop();

    std::vector<std::thread> workers_;

    std::mutex call_mutex_;
    mutable std::mutex state_mutex_;
    std::condition_variable work_cv_;
    std::condition_variable completion_cv_;

    bool stopping_ = false;
    std::uint64_t wave_generation_ = 0U;
    std::size_t active_workers_ = 0U;
    std::size_t wave_count_ = 0U;
    std::atomic<std::size_t> next_index_{0U};
    std::atomic<bool> cancellation_{false};
    std::atomic<std::uint64_t> wave_busy_nanoseconds_{0U};
    WaveFunction wave_function_;
    std::exception_ptr first_exception_;
    WorkerPoolTelemetry telemetry_{};

    static thread_local WorkerPool* current_pool_;
};

}  // namespace kb_pente
