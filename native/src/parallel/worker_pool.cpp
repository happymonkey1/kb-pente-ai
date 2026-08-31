#include "kb_pente/parallel/worker_pool.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace kb_pente {

namespace {

[[nodiscard]] double elapsed_seconds(
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point finished) noexcept {
    const double seconds =
        std::chrono::duration<double>(finished - started).count();
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return 0.0;
    }
    return seconds;
}

[[nodiscard]] double add_finite_seconds(
    double total,
    double increment) noexcept {
    if (!std::isfinite(total) || total < 0.0) {
        total = 0.0;
    }
    if (!std::isfinite(increment) || increment < 0.0) {
        increment = 0.0;
    }
    if (increment > std::numeric_limits<double>::max() - total) {
        return std::numeric_limits<double>::max();
    }
    return total + increment;
}

[[nodiscard]] std::uint64_t saturating_add(
    std::uint64_t left,
    std::uint64_t right) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return left + right;
}

[[nodiscard]] double busy_fraction(
    double wall_seconds,
    std::size_t workers,
    double callback_busy_seconds) noexcept {
    if (!std::isfinite(wall_seconds) || wall_seconds <= 0.0 ||
        workers == 0U || !std::isfinite(callback_busy_seconds) ||
        callback_busy_seconds <= 0.0) {
        return 0.0;
    }

    const double capacity = wall_seconds * static_cast<double>(workers);
    if (!std::isfinite(capacity) || capacity <= 0.0) {
        return 1.0;
    }
    const double fraction = callback_busy_seconds / capacity;
    if (!std::isfinite(fraction) || fraction >= 1.0) {
        return 1.0;
    }
    return fraction <= 0.0 ? 0.0 : fraction;
}

[[nodiscard]] WorkerPoolWaveTelemetry aggregate_wave_telemetry(
    const WorkerPoolWaveTelemetry& cumulative,
    const WorkerPoolWaveTelemetry& wave) noexcept {
    WorkerPoolWaveTelemetry result = cumulative;
    result.items = saturating_add(result.items, wave.items);
    result.workers = wave.workers;
    result.wall_seconds =
        add_finite_seconds(result.wall_seconds, wave.wall_seconds);
    result.callback_busy_seconds = add_finite_seconds(
        result.callback_busy_seconds,
        wave.callback_busy_seconds);
    result.busy_fraction = busy_fraction(
        result.wall_seconds,
        result.workers,
        result.callback_busy_seconds);
    return result;
}

}  // namespace

thread_local WorkerPool* WorkerPool::current_pool_ = nullptr;

WorkerPool::WorkerPool(std::size_t requested_thread_count) {
    if (requested_thread_count == 0U) {
        throw std::invalid_argument("WorkerPool thread count must be positive");
    }

    try {
        workers_.reserve(requested_thread_count);
        for (std::size_t index = 0U; index < requested_thread_count; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
        telemetry_.cumulative.workers = workers_.size();
        telemetry_.last_wave.workers = workers_.size();
    } catch (...) {
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            stopping_ = true;
        }
        work_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        throw;
    }
}

WorkerPool::~WorkerPool() {
    // Callers must finish member calls before destruction. Order shutdown with
    // a final valid caller so workers are joined only after the pool is idle.
    std::lock_guard<std::mutex> caller_lock(call_mutex_);
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

WorkerPoolTelemetry WorkerPool::telemetry() const {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    return telemetry_;
}

void WorkerPool::add_busy_nanoseconds(
    std::atomic<std::uint64_t>& total,
    std::uint64_t value) noexcept {
    if (value == 0U) {
        return;
    }
    std::uint64_t current = total.load(std::memory_order_relaxed);
    for (;;) {
        const std::uint64_t next =
            value > std::numeric_limits<std::uint64_t>::max() - current
                ? std::numeric_limits<std::uint64_t>::max()
                : current + value;
        if (total.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}

void WorkerPool::add_callback_busy_nanoseconds(
    std::uint64_t& total,
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point finished) noexcept {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started);
    if (elapsed.count() <= 0) {
        return;
    }
    const std::uint64_t value = static_cast<std::uint64_t>(elapsed.count());
    total = value > std::numeric_limits<std::uint64_t>::max() - total
                ? std::numeric_limits<std::uint64_t>::max()
                : total + value;
}

void WorkerPool::publish_wave_telemetry(
    std::size_t items,
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point finished) noexcept {
    const auto item_count = items > std::numeric_limits<std::uint64_t>::max()
                                ? std::numeric_limits<std::uint64_t>::max()
                                : static_cast<std::uint64_t>(items);
    const double wall = elapsed_seconds(started, finished);
    const double callback_busy = add_finite_seconds(
        0.0,
        static_cast<double>(
            wave_busy_nanoseconds_.exchange(0U, std::memory_order_relaxed)) /
            1.0e9);

    WorkerPoolWaveTelemetry wave{};
    wave.items = item_count;
    wave.workers = workers_.size();
    wave.wall_seconds = wall;
    wave.callback_busy_seconds = callback_busy;
    wave.busy_fraction = busy_fraction(
        wave.wall_seconds,
        wave.workers,
        wave.callback_busy_seconds);
    telemetry_.last_wave = wave;
    telemetry_.cumulative = aggregate_wave_telemetry(
        telemetry_.cumulative,
        wave);
}

void WorkerPool::worker_loop() {
    std::uint64_t observed_generation = 0U;

    for (;;) {
        std::size_t count = 0U;
        {
            std::unique_lock<std::mutex> state_lock(state_mutex_);
            work_cv_.wait(state_lock, [this, observed_generation] {
                return stopping_ || wave_generation_ != observed_generation;
            });
            if (stopping_) {
                return;
            }
            observed_generation = wave_generation_;
            count = wave_count_;
        }

        std::uint64_t callback_busy_nanoseconds = 0U;
        while (!cancellation_.load(std::memory_order_acquire)) {
            const std::size_t index =
                next_index_.fetch_add(1U, std::memory_order_relaxed);
            if (index >= count) {
                break;
            }
            if (cancellation_.load(std::memory_order_acquire)) {
                break;
            }

            WorkerPool* previous_pool = current_pool_;
            current_pool_ = this;
            const auto callback_started = std::chrono::steady_clock::now();
            try {
                wave_function_(index);
            } catch (...) {
                add_callback_busy_nanoseconds(
                    callback_busy_nanoseconds,
                    callback_started,
                    std::chrono::steady_clock::now());
                current_pool_ = previous_pool;
                const std::exception_ptr failure = std::current_exception();
                {
                    std::lock_guard<std::mutex> state_lock(state_mutex_);
                    if (!first_exception_) {
                        first_exception_ = failure;
                    }
                    cancellation_.store(true, std::memory_order_release);
                }
                break;
            }
            add_callback_busy_nanoseconds(
                callback_busy_nanoseconds,
                callback_started,
                std::chrono::steady_clock::now());
            current_pool_ = previous_pool;
        }

        add_busy_nanoseconds(wave_busy_nanoseconds_, callback_busy_nanoseconds);
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            --active_workers_;
            if (active_workers_ == 0U) {
                completion_cv_.notify_one();
            }
        }
    }
}

}  // namespace kb_pente
