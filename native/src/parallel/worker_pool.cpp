#include "kb_pente/parallel/worker_pool.h"

#include <stdexcept>

namespace kb_pente {

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
            try {
                wave_function_(index);
            } catch (...) {
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
            current_pool_ = previous_pool;
        }

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
