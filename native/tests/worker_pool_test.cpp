#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "kb_pente/parallel/worker_pool.h"

namespace {

class TestFailure final : public std::runtime_error {
public:
    explicit TestFailure(const std::string& message)
        : std::runtime_error(message) {}
};

void expect(bool condition, const char* message) {
    if (!condition) {
        throw TestFailure(message);
    }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (...) {
        throw TestFailure(message);
    }
    throw TestFailure(message);
}

void test_construction_and_traits() {
    static_assert(!std::is_copy_constructible_v<kb_pente::WorkerPool>);
    static_assert(!std::is_copy_assignable_v<kb_pente::WorkerPool>);
    static_assert(!std::is_move_constructible_v<kb_pente::WorkerPool>);
    static_assert(!std::is_move_assignable_v<kb_pente::WorkerPool>);

    expect_throws<std::invalid_argument>(
        [] { kb_pente::WorkerPool pool(0U); },
        "zero worker pools are rejected");

    kb_pente::WorkerPool pool(3U);
    expect(pool.thread_count() == 3U, "configured worker count is retained");
}

void test_zero_and_exact_once() {
    const auto caller_id = std::this_thread::get_id();
    for (const std::size_t worker_count : {1U, 2U, 4U}) {
        kb_pente::WorkerPool pool(worker_count);
        for (const std::size_t count : {
                 worker_count == 1U ? 0U : worker_count - 1U,
                 worker_count,
                 worker_count + 3U,
             }) {
            std::vector<std::atomic<std::size_t>> calls(count);
            for (auto& call_count : calls) {
                call_count.store(0U, std::memory_order_relaxed);
            }
            std::atomic<bool> called_on_caller{false};

            pool.parallel_for(count, [&](std::size_t index) {
                if (std::this_thread::get_id() == caller_id) {
                    called_on_caller.store(true, std::memory_order_release);
                }
                calls[index].fetch_add(1U, std::memory_order_relaxed);
            });

            for (const auto& call_count : calls) {
                expect(call_count.load(std::memory_order_relaxed) == 1U,
                       "each wave index executes exactly once");
            }
            expect(!called_on_caller.load(std::memory_order_acquire),
                   "callbacks execute on workers, not the calling thread");
        }
    }
}

void test_reuse_and_stable_worker_identity() {
    kb_pente::WorkerPool pool(3U);
    constexpr std::size_t count = 96U;
    constexpr std::size_t wave_count = 8U;
    std::vector<std::atomic<std::size_t>> calls(count);
    for (auto& call_count : calls) {
        call_count.store(0U, std::memory_order_relaxed);
    }

    std::mutex identity_mutex;
    std::condition_variable identity_cv;
    std::set<std::thread::id> initial_identities;
    std::set<std::thread::id> all_identities;
    std::size_t initial_workers_entered = 0U;
    bool initial_workers_released = false;

    pool.parallel_for(count, [&](std::size_t index) {
        calls[index].fetch_add(1U, std::memory_order_relaxed);
        std::unique_lock<std::mutex> lock(identity_mutex);
        const auto identity = std::this_thread::get_id();
        all_identities.insert(identity);
        if (!initial_workers_released) {
            initial_identities.insert(identity);
            ++initial_workers_entered;
            if (initial_workers_entered == pool.thread_count()) {
                initial_workers_released = true;
                identity_cv.notify_all();
            } else {
                identity_cv.wait(lock, [&] { return initial_workers_released; });
            }
        }
    });

    for (std::size_t wave = 1U; wave < wave_count; ++wave) {
        pool.parallel_for(count, [&calls, &identity_mutex, &all_identities](
                                    std::size_t index) {
            calls[index].fetch_add(1U, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(identity_mutex);
            all_identities.insert(std::this_thread::get_id());
        });
    }

    for (const auto& call_count : calls) {
        expect(call_count.load(std::memory_order_relaxed) == wave_count,
               "a pool can be reused for many complete waves");
    }
    expect(initial_identities.size() == pool.thread_count(),
           "the first synchronized wave observes every worker");
    expect(all_identities == initial_identities,
           "reused waves preserve the original worker identities");
}

void test_synchronous_completion() {
    kb_pente::WorkerPool pool(4U);
    constexpr std::size_t count = 32U;
    std::atomic<std::size_t> finished{0U};
    std::atomic<std::size_t> active{0U};
    std::atomic<std::size_t> maximum_active{0U};

    pool.parallel_for(count, [&finished, &active, &maximum_active](
                               std::size_t) {
        const auto current = active.fetch_add(1U) + 1U;
        auto maximum = maximum_active.load(std::memory_order_relaxed);
        while (current > maximum &&
               !maximum_active.compare_exchange_weak(
                   maximum, current, std::memory_order_relaxed)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        active.fetch_sub(1U);
        finished.fetch_add(1U);
    });

    expect(finished.load() == count,
           "parallel_for returns after every callback completes");
    expect(active.load() == 0U, "no callback remains after completion");
    expect(maximum_active.load() > 1U,
           "a nontrivial wave uses multiple workers");
}

void test_concurrent_call_serialization() {
    kb_pente::WorkerPool pool(3U);
    std::mutex start_mutex;
    std::condition_variable start_cv;
    bool first_callback_started = false;
    std::atomic<std::size_t> first_active{0U};
    std::atomic<std::size_t> second_active{0U};
    std::atomic<std::size_t> first_callbacks{0U};
    std::atomic<std::size_t> second_callbacks{0U};
    std::atomic<bool> overlap{false};

    auto first = [&](std::size_t) {
        {
            std::lock_guard<std::mutex> lock(start_mutex);
            first_callback_started = true;
        }
        start_cv.notify_one();
        first_callbacks.fetch_add(1U, std::memory_order_relaxed);
        first_active.fetch_add(1U, std::memory_order_relaxed);
        if (second_active.load(std::memory_order_acquire) != 0U) {
            overlap.store(true, std::memory_order_release);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        first_active.fetch_sub(1U, std::memory_order_relaxed);
    };
    auto second = [&](std::size_t) {
        second_callbacks.fetch_add(1U, std::memory_order_relaxed);
        second_active.fetch_add(1U, std::memory_order_relaxed);
        if (first_active.load(std::memory_order_acquire) != 0U) {
            overlap.store(true, std::memory_order_release);
        }
        second_active.fetch_sub(1U, std::memory_order_relaxed);
    };

    std::thread first_caller([&] { pool.parallel_for(12U, first); });
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_cv.wait(lock, [&] { return first_callback_started; });
    }
    std::thread second_caller([&] { pool.parallel_for(12U, second); });
    first_caller.join();
    second_caller.join();

    expect(!overlap.load(std::memory_order_acquire),
           "concurrent external calls execute serialized waves");
    expect(first_callbacks.load(std::memory_order_relaxed) == 12U,
           "the first concurrent wave completes all callbacks");
    expect(second_callbacks.load(std::memory_order_relaxed) == 12U,
           "the second concurrent wave completes all callbacks");
}

void test_nested_call_rejection() {
    kb_pente::WorkerPool pool(2U);
    std::atomic<bool> rejected{false};
    pool.parallel_for(1U, [&](std::size_t) {
        try {
            pool.parallel_for(1U, [](std::size_t) {});
        } catch (const std::logic_error& error) {
            rejected.store(
                std::string(error.what()).find("nested") != std::string::npos,
                std::memory_order_release);
        }
    });
    expect(rejected.load(std::memory_order_acquire),
           "same-pool nested calls fail promptly");
}

void test_exception_cancellation_and_reuse() {
    kb_pente::WorkerPool pool(4U);
    std::atomic<bool> exception_callback_started{false};
    std::atomic<std::size_t> active{0U};
    std::atomic<std::size_t> calls{0U};
    constexpr std::size_t count = 512U;

    expect_throws<std::runtime_error>(
        [&] {
            pool.parallel_for(count, [&](std::size_t index) {
                active.fetch_add(1U, std::memory_order_relaxed);
                calls.fetch_add(1U, std::memory_order_relaxed);
                if (index == 0U) {
                    exception_callback_started.store(
                        true, std::memory_order_release);
                    active.fetch_sub(1U, std::memory_order_relaxed);
                    throw std::runtime_error("worker callback failure");
                }
                while (!exception_callback_started.load(
                    std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                active.fetch_sub(1U, std::memory_order_relaxed);
            });
        },
        "the first callback exception is rethrown");

    expect(active.load(std::memory_order_relaxed) == 0U,
           "exception waves drain in-flight callbacks");
    expect(calls.load(std::memory_order_relaxed) < count,
           "exception cancellation leaves unclaimed work uncalled");

    constexpr std::size_t reuse_count = 47U;
    std::vector<std::atomic<std::size_t>> reuse_calls(reuse_count);
    for (auto& call_count : reuse_calls) {
        call_count.store(0U, std::memory_order_relaxed);
    }
    pool.parallel_for(reuse_count, [&reuse_calls](std::size_t index) {
        reuse_calls[index].fetch_add(1U, std::memory_order_relaxed);
    });
    for (const auto& call_count : reuse_calls) {
        expect(call_count.load(std::memory_order_relaxed) == 1U,
               "a failed pool remains reusable");
    }
}

}  // namespace

int main() {
    try {
        test_construction_and_traits();
        test_zero_and_exact_once();
        test_reuse_and_stable_worker_identity();
        test_synchronous_completion();
        test_concurrent_call_serialization();
        test_nested_call_rejection();
        test_exception_cancellation_and_reuse();
    } catch (const TestFailure& failure) {
        std::cerr << "worker pool test failure: " << failure.what() << '\n';
        return 1;
    } catch (const std::exception& failure) {
        std::cerr << "unexpected worker pool test failure: " << failure.what()
                  << '\n';
        return 1;
    }

    std::cout << "worker pool tests: PASS\n";
    return 0;
}
