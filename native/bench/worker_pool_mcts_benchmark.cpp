#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "kb_pente/constants.h"
#include "kb_pente/mcts/search_session.h"
#include "kb_pente/parallel/worker_pool.h"

namespace {

constexpr std::size_t kDefaultTrees = 128U;
constexpr std::uint32_t kDefaultSimulations = 400U;
constexpr std::size_t kDefaultRepetitions = 3U;
constexpr std::size_t kDefaultWarmups = 1U;
constexpr std::size_t kMaxTrees = 4096U;
constexpr std::size_t kMaxRepetitions = 100U;
constexpr std::size_t kMaxWarmups = 100U;
constexpr std::size_t kMaxWorkers = 4096U;
constexpr std::array<std::size_t, 5> kDefaultWorkers{{1U, 2U, 4U, 8U, 16U}};

struct BenchmarkConfig final {
    std::size_t trees = kDefaultTrees;
    std::uint32_t simulations = kDefaultSimulations;
    std::uint8_t board_size = kb_pente::kDefaultBoardSize;
    std::size_t repetitions = kDefaultRepetitions;
    std::size_t warmups = kDefaultWarmups;
    std::vector<std::size_t> workers{
        kDefaultWorkers.begin(), kDefaultWorkers.end()};
};

struct Measurement final {
    std::size_t worker_count = 0U;
    std::size_t repetition = 0U;
    double seconds = 0.0;
    std::uint64_t aggregate_simulations = 0U;
};

struct Summary final {
    std::size_t worker_count = 0U;
    double median_seconds = 0.0;
    double throughput = 0.0;
    double speedup = 0.0;
};

bool parse_unsigned(
    std::string_view value,
    std::uint64_t maximum,
    std::uint64_t& result) {
    if (value.empty()) {
        return false;
    }
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result);
    return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
           result <= maximum;
}

bool assign_numeric_option(
    std::string_view option,
    std::string_view value,
    BenchmarkConfig& config,
    std::string& error) {
    std::uint64_t parsed = 0U;
    std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    if (option == "--trees") {
        maximum = kMaxTrees;
    } else if (option == "--simulations") {
        maximum = std::numeric_limits<std::uint32_t>::max();
    } else if (option == "--board-size") {
        maximum = kb_pente::kMaxBoardSize;
    } else if (option == "--repetitions") {
        maximum = kMaxRepetitions;
    } else if (option == "--warmups") {
        maximum = kMaxWarmups;
    } else {
        error = "unknown numeric option " + std::string(option);
        return false;
    }

    if (!parse_unsigned(value, maximum, parsed)) {
        error = "invalid value for " + std::string(option);
        return false;
    }
    if (option == "--trees") {
        config.trees = static_cast<std::size_t>(parsed);
    } else if (option == "--simulations") {
        config.simulations = static_cast<std::uint32_t>(parsed);
    } else if (option == "--board-size") {
        config.board_size = static_cast<std::uint8_t>(parsed);
    } else if (option == "--repetitions") {
        config.repetitions = static_cast<std::size_t>(parsed);
    } else {
        config.warmups = static_cast<std::size_t>(parsed);
    }
    return true;
}

bool parse_workers(
    std::string_view value,
    std::vector<std::size_t>& workers,
    std::string& error) {
    workers.clear();
    std::size_t begin = 0U;
    while (begin <= value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::size_t end = comma == std::string_view::npos
                                    ? value.size()
                                    : comma;
        const auto token = value.substr(begin, end - begin);
        std::uint64_t parsed = 0U;
        if (!parse_unsigned(token, kMaxWorkers, parsed) || parsed == 0U) {
            error = "invalid worker count list";
            return false;
        }
        const auto worker_count = static_cast<std::size_t>(parsed);
        if (std::find(workers.begin(), workers.end(), worker_count) !=
            workers.end()) {
            error = "worker count list contains a duplicate";
            return false;
        }
        workers.push_back(worker_count);
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1U;
    }

    if (workers.empty() ||
        std::find(workers.begin(), workers.end(), 1U) == workers.end()) {
        error = "worker count list must include one worker for speedup";
        return false;
    }
    return true;
}

bool parse_arguments(
    int argc,
    char** argv,
    BenchmarkConfig& config,
    bool& show_help,
    std::string& error) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help") {
            show_help = true;
            continue;
        }
        if (argument.rfind("--", 0U) != 0U) {
            error = "unexpected positional argument";
            return false;
        }

        const std::size_t equals = argument.find('=');
        const std::string_view option = argument.substr(0U, equals);
        std::string_view value;
        if (equals != std::string_view::npos) {
            value = argument.substr(equals + 1U);
        } else {
            if (index + 1 >= argc) {
                error = "missing value for " + std::string(option);
                return false;
            }
            value = argv[++index];
        }

        if (option == "--workers" || option == "--worker-list") {
            if (!parse_workers(value, config.workers, error)) {
                return false;
            }
        } else if (!assign_numeric_option(option, value, config, error)) {
            return false;
        }
    }

    if (config.trees == 0U || config.simulations == 0U ||
        config.repetitions == 0U) {
        error = "trees, simulations, and repetitions must be positive";
        return false;
    }
    if (!kb_pente::is_supported_board_size(config.board_size)) {
        error = "board size must be between 5 and 19";
        return false;
    }
    if (config.workers.empty()) {
        error = "worker count list must not be empty";
        return false;
    }
    return true;
}

void print_usage(std::ostream& output) {
    output << "Usage: kb_pente_native_worker_benchmark [options]\n"
           << "  --trees N             independent trees (default 128)\n"
           << "  --simulations N       simulations per tree (default 400)\n"
           << "  --board-size N        Freestyle board size 5..19 (default 19)\n"
           << "  --repetitions N       timed repetitions (default 3)\n"
           << "  --warmups N            untimed warmups (default 1)\n"
           << "  --workers LIST        comma-separated counts including 1\n"
           << "  --help                show this message\n";
}

struct SearchState final {
    SearchState(
        std::uint8_t board_size,
        std::uint32_t simulations,
        std::uint64_t seed)
        : tree(
              kb_pente::Position::initial(board_size),
              kb_pente::Ruleset::Freestyle,
              kb_pente::SearchConfig(
                  1.5F, simulations, 0.0F, 0.03F, seed)),
          session(tree, kb_pente::SearchSessionConfig(1.0F, false)) {
        policy.fill(1.0F);
    }

    void run() {
        for (;;) {
            const auto leaf = session.select_evaluation_leaf();
            if (!leaf.has_value()) {
                return;
            }
            session.accept_evaluation(
                *leaf,
                policy.data(),
                tree.leaf_position(*leaf).action_count(),
                0.0F);
        }
    }

    kb_pente::Tree tree;
    kb_pente::SearchSession session;
    std::array<float, kb_pente::kMaxActions> policy{};
};

std::vector<std::unique_ptr<SearchState>> make_states(
    const BenchmarkConfig& config) {
    std::vector<std::unique_ptr<SearchState>> states;
    states.reserve(config.trees);
    for (std::size_t index = 0U; index < config.trees; ++index) {
        states.push_back(std::make_unique<SearchState>(
            config.board_size,
            config.simulations,
            static_cast<std::uint64_t>(index)));
    }
    return states;
}

std::uint64_t expected_simulations(const BenchmarkConfig& config) {
    const auto trees = static_cast<std::uint64_t>(config.trees);
    const auto simulations = static_cast<std::uint64_t>(config.simulations);
    if (trees > std::numeric_limits<std::uint64_t>::max() / simulations) {
        throw std::overflow_error("aggregate simulation count overflows uint64");
    }
    return trees * simulations;
}

std::uint64_t validate_states(
    const std::vector<std::unique_ptr<SearchState>>& states,
    const BenchmarkConfig& config) {
    const std::uint64_t expected = expected_simulations(config);
    std::uint64_t aggregate = 0U;
    for (const auto& state : states) {
        if (!state->session.complete() ||
            state->session.completed_simulations() != config.simulations ||
            state->session.has_pending_evaluation()) {
            throw std::runtime_error("a tree did not complete its exact budget");
        }
        const auto completed = static_cast<std::uint64_t>(
            state->session.completed_simulations());
        if (aggregate > std::numeric_limits<std::uint64_t>::max() - completed) {
            throw std::overflow_error("aggregate simulation count overflows uint64");
        }
        aggregate += completed;
    }
    if (aggregate != expected) {
        throw std::runtime_error("aggregate simulation count is incomplete");
    }
    return aggregate;
}

std::vector<Measurement> run_worker_count(
    const BenchmarkConfig& config,
    std::size_t worker_count) {
    kb_pente::WorkerPool pool(worker_count);
    for (std::size_t warmup = 0U; warmup < config.warmups; ++warmup) {
        auto states = make_states(config);
        pool.parallel_for(states.size(), [&states](std::size_t index) {
            states[index]->run();
        });
        (void)validate_states(states, config);
    }

    std::vector<Measurement> measurements;
    measurements.reserve(config.repetitions);
    for (std::size_t repetition = 0U;
         repetition < config.repetitions;
         ++repetition) {
        auto states = make_states(config);
        const auto start = std::chrono::steady_clock::now();
        pool.parallel_for(states.size(), [&states](std::size_t index) {
            states[index]->run();
        });
        const auto finish = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(finish - start).count();
        const std::uint64_t aggregate = validate_states(states, config);
        if (!std::isfinite(seconds) || seconds <= 0.0) {
            throw std::runtime_error("timed wave returned an invalid duration");
        }
        measurements.push_back(
            Measurement{worker_count, repetition + 1U, seconds, aggregate});
    }
    return measurements;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if (values.size() % 2U == 0U) {
        return (values[middle - 1U] + values[middle]) / 2.0;
    }
    return values[middle];
}

std::vector<Summary> summarize(
    const BenchmarkConfig& config,
    const std::vector<Measurement>& measurements) {
    const auto expected = static_cast<double>(expected_simulations(config));
    std::vector<Summary> summaries;
    summaries.reserve(config.workers.size());
    for (const std::size_t worker_count : config.workers) {
        std::vector<double> durations;
        for (const auto& measurement : measurements) {
            if (measurement.worker_count == worker_count) {
                durations.push_back(measurement.seconds);
            }
        }
        if (durations.size() != config.repetitions) {
            throw std::runtime_error("missing benchmark repetition");
        }
        const double median_seconds = median(std::move(durations));
        const double throughput = expected / median_seconds;
        if (!std::isfinite(throughput) || throughput <= 0.0) {
            throw std::runtime_error("benchmark throughput is not finite");
        }
        summaries.push_back(
            Summary{worker_count, median_seconds, throughput, 0.0});
    }

    const auto baseline = std::find_if(
        summaries.begin(), summaries.end(), [](const Summary& summary) {
            return summary.worker_count == 1U;
        });
    if (baseline == summaries.end()) {
        throw std::runtime_error("one-worker benchmark baseline is missing");
    }
    for (auto& summary : summaries) {
        summary.speedup = summary.throughput / baseline->throughput;
    }
    return summaries;
}

void print_worker_list(
    std::ostream& output,
    const std::vector<std::size_t>& workers) {
    for (std::size_t index = 0U; index < workers.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << workers[index];
    }
}

void print_results(
    const BenchmarkConfig& config,
    const std::vector<Measurement>& measurements,
    const std::vector<Summary>& summaries) {
    const auto expected = expected_simulations(config);
    std::cout << "format=kb_pente_native_mcts_benchmark_v1\n"
              << "config.trees=" << config.trees << '\n'
              << "config.simulations_per_tree=" << config.simulations << '\n'
              << "config.board_size=" << static_cast<unsigned>(config.board_size)
              << '\n'
              << "config.ruleset=freestyle\n"
              << "config.evaluator=uniform_zero\n"
              << "config.root_noise=false\n"
              << "config.repetitions=" << config.repetitions << '\n'
              << "config.warmups=" << config.warmups << '\n'
              << "config.hardware_concurrency="
              << std::thread::hardware_concurrency() << '\n'
              << "config.expected_aggregate_simulations=" << expected << '\n'
              << "config.worker_counts=";
    print_worker_list(std::cout, config.workers);
    std::cout << '\n';

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    for (const auto& measurement : measurements) {
        std::cout << "record=repetition"
                  << " worker_count=" << measurement.worker_count
                  << " repetition=" << measurement.repetition
                  << " seconds=" << measurement.seconds
                  << " aggregate_simulations="
                  << measurement.aggregate_simulations << '\n';
    }
    for (const auto& summary : summaries) {
        std::cout << "record=summary"
                  << " worker_count=" << summary.worker_count
                  << " median_seconds=" << summary.median_seconds
                  << " throughput_simulations_per_second="
                  << summary.throughput << " speedup_vs_one="
                  << summary.speedup << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    BenchmarkConfig config;
    bool show_help = false;
    std::string error;
    if (!parse_arguments(argc, argv, config, show_help, error)) {
        std::cerr << "error=" << error << '\n';
        print_usage(std::cerr);
        return 2;
    }
    if (show_help) {
        print_usage(std::cout);
        return 0;
    }

    try {
        std::vector<Measurement> measurements;
        measurements.reserve(config.workers.size() * config.repetitions);
        for (const std::size_t worker_count : config.workers) {
            auto worker_measurements = run_worker_count(config, worker_count);
            measurements.insert(
                measurements.end(),
                worker_measurements.begin(),
                worker_measurements.end());
        }
        const auto summaries = summarize(config, measurements);
        print_results(config, measurements, summaries);
    } catch (const std::exception& failure) {
        std::cerr << "error=" << failure.what() << '\n';
        return 2;
    }
    return 0;
}
