#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <torch/extension.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kb_pente/game.h"
#include "kb_pente/mcts/search_batch.h"
#include "kb_pente/mcts/search_config.h"
#include "kb_pente/mcts/search_session.h"
#include "kb_pente/rules.h"

namespace py = pybind11;

namespace {

using kb_pente::Action;
using kb_pente::BatchToken;
using kb_pente::DeduplicationStats;
using kb_pente::DeduplicationTelemetry;
using kb_pente::GameStatus;
using kb_pente::Position;
using kb_pente::RootAdvanceStats;
using kb_pente::Ruleset;
using kb_pente::SearchBatch;
using kb_pente::SearchConfig;
using kb_pente::SearchBatchGenerationTelemetry;
using kb_pente::SearchBatchStageTelemetry;
using kb_pente::SearchBatchTimingTelemetry;
using kb_pente::SearchSessionConfig;
using kb_pente::SearchTelemetry;
using kb_pente::SlotId;
using kb_pente::TerminalResult;
using kb_pente::WinReason;
using kb_pente::WorkerPoolWaveTelemetry;

constexpr std::size_t kConstructorOptionCount = 10U;

[[nodiscard]] std::uint64_t parse_unsigned(
    const py::handle& value,
    const char* name) {
    if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be an integer");
    }

    const unsigned long long parsed = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw py::value_error(std::string(name) + " is out of range");
    }
    if (parsed > std::numeric_limits<std::uint64_t>::max()) {
        throw py::value_error(std::string(name) + " is out of range");
    }
    return static_cast<std::uint64_t>(parsed);
}

[[nodiscard]] std::int64_t parse_signed(
    const py::handle& value,
    const char* name) {
    if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be an integer");
    }

    const long long parsed = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw py::value_error(std::string(name) + " is out of range");
    }
    if (parsed < std::numeric_limits<std::int64_t>::min() ||
        parsed > std::numeric_limits<std::int64_t>::max()) {
        throw py::value_error(std::string(name) + " is out of range");
    }
    return static_cast<std::int64_t>(parsed);
}

[[nodiscard]] std::size_t parse_size(
    const py::handle& value,
    const char* name) {
    const std::uint64_t parsed = parse_unsigned(value, name);
    if (parsed > std::numeric_limits<std::size_t>::max()) {
        throw py::value_error(std::string(name) + " is out of range");
    }
    return static_cast<std::size_t>(parsed);
}

[[nodiscard]] std::uint32_t parse_uint32(
    const py::handle& value,
    const char* name) {
    const std::uint64_t parsed = parse_unsigned(value, name);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw py::value_error(std::string(name) + " is out of range");
    }
    return static_cast<std::uint32_t>(parsed);
}

[[nodiscard]] Action parse_action(
    const py::handle& value,
    const char* name) {
    const std::uint64_t parsed = parse_unsigned(value, name);
    if (parsed > std::numeric_limits<Action>::max()) {
        throw py::value_error(std::string(name) + " is out of range");
    }
    return static_cast<Action>(parsed);
}

[[nodiscard]] float parse_float(
    const py::handle& value,
    const char* name) {
    if ((!PyFloat_Check(value.ptr()) && !PyLong_Check(value.ptr())) ||
        PyBool_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be a real scalar");
    }
    const double parsed = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw py::value_error(std::string(name) + " is out of range");
    }
    if (parsed > std::numeric_limits<float>::max() ||
        parsed < -std::numeric_limits<float>::max()) {
        throw py::value_error(std::string(name) + " is out of range");
    }
    return static_cast<float>(parsed);
}

[[nodiscard]] bool parse_bool(
    const py::handle& value,
    const char* name) {
    if (!PyBool_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be a bool");
    }
    return value.cast<bool>();
}

[[nodiscard]] std::string parse_string(
    const py::handle& value,
    const char* name) {
    if (!PyUnicode_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be a string");
    }
    return py::cast<std::string>(value);
}

[[nodiscard]] Ruleset parse_ruleset(const py::handle& value) {
    const std::string name = parse_string(value, "ruleset");
    if (name == "standard") {
        return Ruleset::Standard;
    }
    if (name == "tournament") {
        return Ruleset::Tournament;
    }
    if (name == "freestyle") {
        return Ruleset::Freestyle;
    }
    throw py::value_error(
        "ruleset must be one of 'standard', 'tournament', or 'freestyle'");
}

struct ConstructorOptions final {
    std::size_t board_size = kb_pente::kDefaultBoardSize;
    Ruleset ruleset = kb_pente::kDefaultRuleset;
    std::uint32_t simulations = 400U;
    std::size_t active_games = 128U;
    std::size_t threads = 1U;
    std::uint64_t seed = 0U;
    float c_puct = 1.5F;
    float root_noise_epsilon = 0.25F;
    float root_dirichlet_alpha = 0.03F;
    bool pin_memory = true;
};

[[nodiscard]] int constructor_option_index(const std::string& name) {
    if (name == "board_size") {
        return 0;
    }
    if (name == "ruleset") {
        return 1;
    }
    if (name == "simulations" || name == "simulation_budget") {
        return 2;
    }
    if (name == "active_games" || name == "max_active_games" ||
        name == "max_batch") {
        return 3;
    }
    if (name == "threads" || name == "worker_threads") {
        return 4;
    }
    if (name == "seed") {
        return 5;
    }
    if (name == "c_puct") {
        return 6;
    }
    if (name == "root_noise_epsilon") {
        return 7;
    }
    if (name == "root_dirichlet_alpha") {
        return 8;
    }
    if (name == "pin_memory") {
        return 9;
    }
    return -1;
}

[[nodiscard]] ConstructorOptions parse_constructor(
    const py::args& args,
    const py::kwargs& kwargs) {
    if (args.size() > kConstructorOptionCount) {
        throw py::type_error("SearchBatch received too many positional arguments");
    }

    std::vector<py::object> values(kConstructorOptionCount, py::none());
    std::array<bool, kConstructorOptionCount> supplied{};
    for (std::size_t index = 0U; index < args.size(); ++index) {
        values[index] = args[index];
        supplied[index] = true;
    }

    for (const auto item : kwargs) {
        const std::string name = py::cast<std::string>(item.first);
        const int index = constructor_option_index(name);
        if (index < 0) {
            throw py::type_error(
                "SearchBatch received an unexpected constructor argument: " +
                name);
        }
        if (supplied[static_cast<std::size_t>(index)]) {
            throw py::type_error(
                "SearchBatch received multiple values for constructor argument: " +
                name);
        }
        values[static_cast<std::size_t>(index)] =
            py::reinterpret_borrow<py::object>(item.second);
        supplied[static_cast<std::size_t>(index)] = true;
    }

    ConstructorOptions result{};
    if (supplied[0U]) {
        result.board_size = parse_size(values[0U], "board_size");
    }
    if (supplied[1U]) {
        result.ruleset = parse_ruleset(values[1U]);
    }
    if (supplied[2U]) {
        result.simulations = parse_uint32(values[2U], "simulations");
    }
    if (supplied[3U]) {
        result.active_games = parse_size(values[3U], "active_games");
    }
    if (supplied[4U]) {
        result.threads = parse_size(values[4U], "threads");
    }
    if (supplied[5U]) {
        result.seed = parse_unsigned(values[5U], "seed");
    }
    if (supplied[6U]) {
        result.c_puct = parse_float(values[6U], "c_puct");
    }
    if (supplied[7U]) {
        result.root_noise_epsilon =
            parse_float(values[7U], "root_noise_epsilon");
    }
    if (supplied[8U]) {
        result.root_dirichlet_alpha =
            parse_float(values[8U], "root_dirichlet_alpha");
    }
    if (supplied[9U]) {
        result.pin_memory = parse_bool(values[9U], "pin_memory");
    }
    return result;
}

void validate_tensor(
    const torch::Tensor& tensor,
    torch::ScalarType dtype,
    std::initializer_list<std::int64_t> shape,
    const char* name) {
    if (!tensor.defined()) {
        throw py::value_error(std::string(name) + " must be defined");
    }
    if (!tensor.device().is_cpu()) {
        throw py::value_error(std::string(name) + " must be a CPU tensor");
    }
    if (tensor.scalar_type() != dtype) {
        throw py::value_error(std::string(name) + " has the wrong dtype");
    }
    if (!tensor.is_contiguous()) {
        throw py::value_error(
            std::string(name) + " must be contiguous in row-major order");
    }
    if (tensor.dim() != static_cast<std::int64_t>(shape.size())) {
        throw py::value_error(std::string(name) + " has the wrong rank");
    }
    std::size_t dimension = 0U;
    for (const std::int64_t expected : shape) {
        if (tensor.size(static_cast<std::int64_t>(dimension)) !=
            expected) {
            throw py::value_error(std::string(name) + " has the wrong shape");
        }
        ++dimension;
    }
}

[[nodiscard]] Position import_position(
    const torch::Tensor& stones,
    const torch::Tensor& captures,
    std::int64_t current_player,
    std::int64_t ply,
    const py::object& last_action,
    std::uint8_t board_size,
    Ruleset ruleset) {
    const std::size_t area = kb_pente::board_area(board_size);
    validate_tensor(
        stones,
        torch::kInt8,
        {static_cast<std::int64_t>(board_size),
         static_cast<std::int64_t>(board_size)},
        "stones");
    validate_tensor(captures, torch::kInt16, {2}, "captures");

    if (current_player != static_cast<std::int64_t>(kb_pente::Player::One) &&
        current_player != static_cast<std::int64_t>(kb_pente::Player::Two)) {
        throw py::value_error("current_player must be 1 or -1");
    }
    if (ply < 0 ||
        ply > static_cast<std::int64_t>(std::numeric_limits<std::uint16_t>::max())) {
        throw py::value_error("ply must fit an unsigned 16-bit integer");
    }
    if (!kb_pente::is_valid_ruleset_configuration(board_size, ruleset)) {
        throw py::value_error("board size is invalid for the configured ruleset");
    }

    Action native_last_action = kb_pente::kInvalidAction;
    if (!last_action.is_none()) {
        const std::int64_t parsed_last_action =
            parse_signed(last_action, "last_action");
        if (parsed_last_action == -1 ||
            parsed_last_action == static_cast<std::int64_t>(
                                      kb_pente::kInvalidAction)) {
            native_last_action = kb_pente::kInvalidAction;
        } else if (parsed_last_action < 0 ||
                   parsed_last_action >= static_cast<std::int64_t>(area)) {
            throw py::value_error("last_action is outside the active board");
        } else {
            native_last_action = static_cast<Action>(parsed_last_action);
        }
    }

    Position position{};
    position.board_size = board_size;
    position.current_player = static_cast<kb_pente::Player>(current_player);
    position.ply = static_cast<std::uint16_t>(ply);
    position.last_action = native_last_action;

    const auto* stone_data = stones.data_ptr<std::int8_t>();
    for (std::size_t action = 0U; action < area; ++action) {
        position.stones[action] = stone_data[action];
    }

    const auto* capture_data = captures.data_ptr<std::int16_t>();
    for (std::size_t player = 0U; player < kb_pente::kPlayerCount; ++player) {
        if (capture_data[player] < 0 || capture_data[player] > 255) {
            throw py::value_error("captures must fit unsigned 8-bit counts");
        }
        position.captures[player] =
            static_cast<std::uint8_t>(capture_data[player]);
    }

    position.refresh_hash();
    position.validate();
    const auto terminal = kb_pente::check_terminal(position);
    if (terminal.is_terminal()) {
        throw py::value_error("SearchBatch roots must be nonterminal");
    }
    return position;
}

void validate_staging_tensor(
    const torch::Tensor& tensor,
    std::initializer_list<std::int64_t> shape,
    bool pinned,
    const void* expected_data,
    const char* name) {
    validate_tensor(tensor, torch::kFloat32, shape, name);
    if (tensor.is_pinned() != pinned) {
        throw py::value_error(std::string(name) + " has unexpected pinning");
    }
    if (static_cast<const void*>(tensor.data_ptr<float>()) != expected_data) {
        throw py::value_error(std::string(name) + " storage was replaced");
    }
}

[[nodiscard]] const char* terminal_status_name(GameStatus status) {
    switch (status) {
        case GameStatus::InProgress:
            return "in_progress";
        case GameStatus::Draw:
            return "draw";
        case GameStatus::Win:
            return "win";
    }
    throw std::logic_error("TerminalResult has an unknown status");
}

[[nodiscard]] const char* terminal_reason_name(WinReason reason) {
    switch (reason) {
        case WinReason::None:
            return "none";
        case WinReason::Line:
            return "line";
        case WinReason::Capture:
            return "capture";
    }
    throw std::logic_error("TerminalResult has an unknown reason");
}

[[nodiscard]] py::dict terminal_result_to_dict(
    const TerminalResult& terminal) {
    py::dict result;
    result["status"] = terminal_status_name(terminal.status);
    result["reason"] = terminal_reason_name(terminal.reason);
    if (terminal.winner.has_value()) {
        result["winner"] = static_cast<int>(*terminal.winner);
    } else {
        result["winner"] = py::none();
    }
    return result;
}

[[nodiscard]] py::dict root_advance_stats_to_dict(
    const RootAdvanceStats& stats) {
    py::dict result;
    result["reused_subtree"] = stats.reused_subtree;
    result["previous_node_count"] = stats.previous_node_count;
    result["retained_node_count"] = stats.retained_node_count;
    result["discarded_node_count"] = stats.discarded_node_count;
    result["previous_owned_bytes"] = stats.previous_owned_bytes;
    result["new_owned_bytes"] = stats.new_owned_bytes;
    return result;
}

[[nodiscard]] py::dict search_telemetry_to_dict(
    const SearchTelemetry& telemetry) {
    py::dict result;
    result["completed_simulations"] = telemetry.completed_simulations;
    result["evaluator_completions"] = telemetry.evaluator_completions;
    result["terminal_simulations"] = telemetry.terminal_simulations;
    result["selected_leaves"] = telemetry.selected_leaves;
    result["max_selected_path_depth"] = telemetry.max_selected_path_depth;
    result["root_legal_actions"] = telemetry.root_legal_actions;
    result["root_edge_visits"] = telemetry.root_edge_visits;
    result["root_children_visited"] = telemetry.root_children_visited;
    result["root_visit_entropy"] = telemetry.root_visit_entropy;
    result["root_max_visit_share"] = telemetry.root_max_visit_share;
    result["root_collapse_eligible"] = telemetry.root_collapse_eligible;
    result["root_search_collapsed"] = telemetry.root_search_collapsed;
    result["invalid_policy_fallbacks"] = telemetry.invalid_policy_fallbacks;
    result["zero_visit_fallbacks"] = telemetry.zero_visit_fallbacks;
    return result;
}

[[nodiscard]] py::dict deduplication_stats_to_dict(
    const DeduplicationStats& stats) {
    py::dict result;
    result["selection_waves"] = stats.selection_waves;
    result["raw_evaluation_requests"] = stats.raw_evaluation_requests;
    result["unique_evaluations"] = stats.unique_evaluations;
    result["eliminated_duplicate_evaluations"] =
        stats.eliminated_duplicate_evaluations;
    result["duplicate_leaf_rate"] = stats.duplicate_leaf_rate;
    return result;
}

[[nodiscard]] py::dict deduplication_telemetry_to_dict(
    const DeduplicationTelemetry& telemetry) {
    py::dict result;
    result["cumulative"] = deduplication_stats_to_dict(telemetry.cumulative);
    result["last_wave"] = deduplication_stats_to_dict(telemetry.last_wave);
    return result;
}

[[nodiscard]] py::dict worker_telemetry_to_dict(
    const WorkerPoolWaveTelemetry& telemetry) {
    py::dict result;
    result["items"] = telemetry.items;
    result["workers"] = telemetry.workers;
    result["wall_seconds"] = telemetry.wall_seconds;
    result["callback_busy_seconds"] = telemetry.callback_busy_seconds;
    result["busy_fraction"] = telemetry.busy_fraction;
    return result;
}

[[nodiscard]] py::dict stage_telemetry_to_dict(
    const SearchBatchStageTelemetry& telemetry) {
    py::dict result;
    result["successful_operations"] = telemetry.successful_operations;
    result["token"] = telemetry.token;
    result["wall_seconds"] = telemetry.wall_seconds;
    result["worker"] = worker_telemetry_to_dict(telemetry.worker);
    return result;
}

[[nodiscard]] py::dict generation_telemetry_to_dict(
    const SearchBatchGenerationTelemetry& telemetry) {
    py::dict result;
    result["token"] = telemetry.token;
    result["select"] = stage_telemetry_to_dict(telemetry.select);
    result["dedup"] = stage_telemetry_to_dict(telemetry.dedup);
    result["features"] = stage_telemetry_to_dict(telemetry.features);
    result["backup"] = stage_telemetry_to_dict(telemetry.backup);
    return result;
}

[[nodiscard]] py::dict timing_telemetry_to_dict(
    const SearchBatchTimingTelemetry& telemetry) {
    py::dict result;
    result["cumulative"] =
        generation_telemetry_to_dict(telemetry.cumulative);
    result["latest_generation"] =
        generation_telemetry_to_dict(telemetry.latest_generation);
    return result;
}

class SelectionBatch final {
public:
    SelectionBatch(
        torch::Tensor features,
        BatchToken token,
        std::size_t size,
        std::size_t raw_size)
        : features_(std::move(features)),
          token_(token),
          size_(size),
          raw_size_(raw_size) {}

    [[nodiscard]] const torch::Tensor& features() const noexcept {
        return features_;
    }
    [[nodiscard]] BatchToken token() const noexcept { return token_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t raw_size() const noexcept { return raw_size_; }
    [[nodiscard]] std::size_t unique_size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0U; }

private:
    torch::Tensor features_;
    BatchToken token_ = kb_pente::kInvalidBatchToken;
    std::size_t size_ = 0U;
    std::size_t raw_size_ = 0U;
};

class SearchBatchBinding final {
public:
    static std::unique_ptr<SearchBatchBinding> construct(
        const py::args& args,
        const py::kwargs& kwargs) {
        const ConstructorOptions options = parse_constructor(args, kwargs);
        if (options.board_size < kb_pente::kMinBoardSize ||
            options.board_size > kb_pente::kMaxBoardSize) {
            throw py::value_error("board_size must be between 5 and 19");
        }
        const auto board_size = static_cast<std::uint8_t>(options.board_size);
        if (!kb_pente::is_valid_ruleset_configuration(
                board_size,
                options.ruleset)) {
            throw py::value_error(
                "board_size is invalid for the configured ruleset");
        }
        if (options.active_games == 0U) {
            throw py::value_error("active_games must be positive");
        }
        if (options.active_games >
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw py::value_error("active_games is too large for tensor shapes");
        }
        if (options.threads == 0U) {
            throw py::value_error("threads must be positive");
        }

        return std::unique_ptr<SearchBatchBinding>(new SearchBatchBinding(
            board_size,
            options.ruleset,
            SearchConfig(
                options.c_puct,
                options.simulations,
                options.root_noise_epsilon,
                options.root_dirichlet_alpha,
                options.seed),
            options.active_games,
            options.threads,
            options.pin_memory));
    }

    SearchBatchBinding(
        std::uint8_t board_size,
        Ruleset ruleset,
        SearchConfig config,
        std::size_t active_games,
        std::size_t threads,
        bool pin_memory)
        : board_size_(board_size),
          ruleset_(ruleset),
          max_active_games_(active_games),
          pin_memory_requested_(pin_memory),
          batch_(std::move(config), active_games, threads) {
        const auto options = torch::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(torch::kCPU)
                                 .pinned_memory(pin_memory_requested_);
        const auto board = static_cast<std::int64_t>(board_size_);
        const auto rows = static_cast<std::int64_t>(max_active_games_);
        features_ = torch::empty({rows, 4, board, board}, options);
        policies_ = torch::empty({rows, kb_pente::kMaxActions}, options);
        values_ = torch::empty({rows}, options);

        pin_memory_realized_ = features_.is_pinned() && policies_.is_pinned() &&
                               values_.is_pinned();
        if (pin_memory_realized_ != pin_memory_requested_) {
            throw std::runtime_error(
                "requested staging pinning was not realized; refusing fallback");
        }
        features_data_ = features_.data_ptr<float>();
        policies_data_ = policies_.data_ptr<float>();
        values_data_ = values_.data_ptr<float>();
    }

    [[nodiscard]] std::size_t add(
        const torch::Tensor& stones,
        const torch::Tensor& captures,
        const py::object& current_player,
        const py::object& ply,
        const py::object& last_action,
        const py::object& temperature,
        const py::object& add_root_noise) {
        const std::int64_t parsed_current_player =
            parse_signed(current_player, "current_player");
        const std::int64_t parsed_ply = parse_signed(ply, "ply");
        const float parsed_temperature =
            parse_float(temperature, "temperature");
        const bool parsed_add_root_noise =
            parse_bool(add_root_noise, "add_root_noise");
        const Position position = import_position(
            stones,
            captures,
            parsed_current_player,
            parsed_ply,
            last_action,
            board_size_,
            ruleset_);
        const SearchSessionConfig session_config(
            parsed_temperature,
            parsed_add_root_noise);

        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.add(position, ruleset_, session_config);
    }

    [[nodiscard]] SelectionBatch select() {
        validate_staging_buffers();

        float* feature_data = features_.data_ptr<float>();
        BatchToken token = kb_pente::kInvalidBatchToken;
        std::size_t unique_size = 0U;
        std::size_t raw_size = 0U;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            const kb_pente::Selection selection = batch_.select();
            token = selection.token;
            unique_size = selection.size();
            raw_size = selection.raw_size();
            if (unique_size != 0U) {
                batch_.write_features(
                    token,
                    feature_data,
                    unique_size,
                    4U,
                    board_size_,
                    board_size_);
            }
        }

        const torch::Tensor feature_view = features_.narrow(
            0,
            0,
            static_cast<std::int64_t>(unique_size));
        return SelectionBatch(
            feature_view,
            token,
            unique_size,
            raw_size);
    }

    void backup(const py::object& token, const py::object& rows) {
        const BatchToken parsed_token = parse_unsigned(token, "token");
        const std::size_t parsed_rows = parse_size(rows, "rows");
        validate_staging_buffers();
        if (parsed_rows > max_active_games_) {
            throw py::value_error("rows exceeds SearchBatch capacity");
        }

        const auto* policy_data = policies_.data_ptr<float>();
        const auto* value_data = values_.data_ptr<float>();
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            batch_.backup(
                parsed_token,
                policy_data,
                parsed_rows,
                value_data,
                parsed_rows);
        }
    }

    [[nodiscard]] torch::Tensor root_policy(const py::object& slot) {
        const SlotId parsed_slot = parse_size(slot, "slot");
        std::array<float, kb_pente::kMaxActions> policy{};
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            policy = batch_.root_policy(parsed_slot);
        }

        const auto options = torch::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(torch::kCPU);
        torch::Tensor result = torch::empty(
            {static_cast<std::int64_t>(kb_pente::board_area(board_size_))},
            options);
        std::copy_n(
            policy.data(),
            kb_pente::board_area(board_size_),
            result.data_ptr<float>());
        return result;
    }

    [[nodiscard]] py::dict advance_root(
        const py::object& slot,
        const py::object& action,
        const py::object& temperature,
        const py::object& add_root_noise) {
        const SlotId parsed_slot = parse_size(slot, "slot");
        const Action parsed_action = parse_action(action, "action");
        const SearchSessionConfig session_config(
            parse_float(temperature, "temperature"),
            parse_bool(add_root_noise, "add_root_noise"));

        RootAdvanceStats stats{};
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            stats = batch_.advance_root(
                parsed_slot,
                parsed_action,
                session_config);
        }
        return root_advance_stats_to_dict(stats);
    }

    void remove(const py::object& slot) {
        const SlotId parsed_slot = parse_size(slot, "slot");
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        batch_.remove(parsed_slot);
    }

    void replace_root(
        const py::object& slot,
        const torch::Tensor& stones,
        const torch::Tensor& captures,
        const py::object& current_player,
        const py::object& ply,
        const py::object& last_action,
        const py::object& temperature,
        const py::object& add_root_noise) {
        const SlotId parsed_slot = parse_size(slot, "slot");
        const std::int64_t parsed_current_player =
            parse_signed(current_player, "current_player");
        const std::int64_t parsed_ply = parse_signed(ply, "ply");
        const SearchSessionConfig session_config(
            parse_float(temperature, "temperature"),
            parse_bool(add_root_noise, "add_root_noise"));
        Position position = import_position(
            stones,
            captures,
            parsed_current_player,
            parsed_ply,
            last_action,
            board_size_,
            ruleset_);

        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        batch_.replace_root(
            parsed_slot,
            std::move(position),
            ruleset_,
            session_config);
    }

    [[nodiscard]] py::dict root_terminal(const py::object& slot) const {
        const SlotId parsed_slot = parse_size(slot, "slot");
        TerminalResult terminal{};
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            terminal = batch_.root_terminal(parsed_slot);
        }
        return terminal_result_to_dict(terminal);
    }

    [[nodiscard]] py::dict slot_telemetry(const py::object& slot) const {
        const SlotId parsed_slot = parse_size(slot, "slot");
        SearchTelemetry telemetry{};
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            telemetry = batch_.slot_telemetry(parsed_slot);
        }
        return search_telemetry_to_dict(telemetry);
    }

    [[nodiscard]] py::dict deduplication_telemetry() const {
        DeduplicationTelemetry telemetry{};
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            telemetry = batch_.deduplication_telemetry();
        }
        return deduplication_telemetry_to_dict(telemetry);
    }

    [[nodiscard]] py::dict timing_telemetry() const {
        SearchBatchTimingTelemetry telemetry{};
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            telemetry = batch_.timing_telemetry();
        }
        return timing_telemetry_to_dict(telemetry);
    }

    [[nodiscard]] bool complete() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.complete();
    }

    [[nodiscard]] bool slot_active(const py::object& slot) const {
        const SlotId parsed_slot = parse_size(slot, "slot");
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.slot_active(parsed_slot);
    }

    [[nodiscard]] bool slot_complete(const py::object& slot) const {
        const SlotId parsed_slot = parse_size(slot, "slot");
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.slot_complete(parsed_slot);
    }

    [[nodiscard]] py::dict status() const {
        bool complete_value = false;
        std::size_t active_count_value = 0U;
        std::size_t capacity_value = 0U;
        bool pending_value = false;
        bool poisoned_value = false;
        BatchToken pending_token_value = kb_pente::kInvalidBatchToken;
        std::size_t pending_rows_value = 0U;
        std::size_t pending_raw_rows_value = 0U;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            complete_value = batch_.complete();
            poisoned_value = batch_.poisoned();
            active_count_value = batch_.active_count();
            capacity_value = batch_.capacity();
            pending_value = batch_.has_pending();
            pending_token_value = batch_.pending_token();
            pending_rows_value = batch_.pending_request_count();
            pending_raw_rows_value = batch_.pending_selected_count();
        }
        py::dict result;
        result["complete"] = complete_value;
        result["poisoned"] = poisoned_value;
        result["active_count"] = active_count_value;
        result["capacity"] = capacity_value;
        result["pending"] = pending_value;
        result["pending_token"] = pending_token_value;
        result["pending_rows"] = pending_rows_value;
        result["pending_raw_rows"] = pending_raw_rows_value;
        return result;
    }

    [[nodiscard]] const torch::Tensor& features() const noexcept {
        return features_;
    }
    [[nodiscard]] const torch::Tensor& policies() const noexcept {
        return policies_;
    }
    [[nodiscard]] const torch::Tensor& values() const noexcept {
        return values_;
    }
    [[nodiscard]] bool pin_memory_requested() const noexcept {
        return pin_memory_requested_;
    }
    [[nodiscard]] bool pin_memory_realized() const noexcept {
        return pin_memory_realized_;
    }
    [[nodiscard]] std::size_t board_size() const noexcept {
        return board_size_;
    }
    [[nodiscard]] std::size_t active_count() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.active_count();
    }
    [[nodiscard]] std::size_t capacity() const noexcept {
        return max_active_games_;
    }
    [[nodiscard]] std::size_t pending_request_count() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.pending_request_count();
    }
    [[nodiscard]] std::size_t pending_selected_count() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.pending_selected_count();
    }
    [[nodiscard]] bool has_pending() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.has_pending();
    }
    [[nodiscard]] BatchToken pending_token() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.pending_token();
    }
    [[nodiscard]] BatchToken last_token() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_.last_token();
    }
    [[nodiscard]] std::size_t thread_count() const noexcept {
        return batch_.thread_count();
    }

private:
    void validate_staging_buffers() const {
        const auto board = static_cast<std::int64_t>(board_size_);
        const auto rows = static_cast<std::int64_t>(max_active_games_);
        validate_staging_tensor(
            features_,
            {rows, 4, board, board},
            pin_memory_realized_,
            features_data_,
            "features");
        validate_staging_tensor(
            policies_,
            {rows, kb_pente::kMaxActions},
            pin_memory_realized_,
            policies_data_,
            "policies");
        validate_staging_tensor(
            values_,
            {rows},
            pin_memory_realized_,
            values_data_,
            "values");
    }

    std::uint8_t board_size_ = kb_pente::kDefaultBoardSize;
    Ruleset ruleset_ = kb_pente::kDefaultRuleset;
    std::size_t max_active_games_ = 0U;
    bool pin_memory_requested_ = true;
    bool pin_memory_realized_ = false;
    torch::Tensor features_;
    torch::Tensor policies_;
    torch::Tensor values_;
    void* features_data_ = nullptr;
    void* policies_data_ = nullptr;
    void* values_data_ = nullptr;
    mutable std::mutex mutex_;
    SearchBatch batch_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<SelectionBatch>(module, "SelectionBatch")
        .def_property_readonly("features", &SelectionBatch::features)
        .def_property_readonly("token", &SelectionBatch::token)
        .def_property_readonly("size", &SelectionBatch::size)
        .def_property_readonly("raw_size", &SelectionBatch::raw_size)
        .def_property_readonly("unique_size", &SelectionBatch::unique_size)
        .def_property_readonly("empty", &SelectionBatch::empty)
        .def("__len__", &SelectionBatch::size);

    py::class_<SearchBatchBinding>(module, "SearchBatch")
        .def(py::init([](py::args args, py::kwargs kwargs) {
            return SearchBatchBinding::construct(args, kwargs);
        }))
        .def(
            "add",
            &SearchBatchBinding::add,
            py::arg("stones"),
            py::arg("captures"),
            py::arg("current_player"),
            py::arg("ply"),
            py::arg("last_action") = py::none(),
            py::arg("temperature") = 1.0F,
            py::arg("add_root_noise") = false)
        .def("select", &SearchBatchBinding::select)
        .def(
            "backup",
            &SearchBatchBinding::backup,
            py::arg("token"),
            py::arg("rows"))
        .def("root_policy", &SearchBatchBinding::root_policy, py::arg("slot"))
        .def(
            "advance_root",
            &SearchBatchBinding::advance_root,
            py::arg("slot"),
            py::arg("action"),
            py::arg("temperature") = 1.0F,
            py::arg("add_root_noise") = false)
        .def(
            "remove",
            &SearchBatchBinding::remove,
            py::arg("slot"))
        .def(
            "replace_root",
            &SearchBatchBinding::replace_root,
            py::arg("slot"),
            py::arg("stones"),
            py::arg("captures"),
            py::arg("current_player"),
            py::arg("ply"),
            py::arg("last_action") = py::none(),
            py::arg("temperature") = 1.0F,
            py::arg("add_root_noise") = false)
        .def(
            "root_terminal",
            &SearchBatchBinding::root_terminal,
            py::arg("slot"))
        .def(
            "slot_telemetry",
            &SearchBatchBinding::slot_telemetry,
            py::arg("slot"))
        .def(
            "deduplication_telemetry",
            &SearchBatchBinding::deduplication_telemetry)
        .def("timing_telemetry", &SearchBatchBinding::timing_telemetry)
        .def("complete", &SearchBatchBinding::complete)
        .def("slot_active", &SearchBatchBinding::slot_active, py::arg("slot"))
        .def(
            "slot_complete",
            &SearchBatchBinding::slot_complete,
            py::arg("slot"))
        .def("status", &SearchBatchBinding::status)
        .def_property_readonly("features", &SearchBatchBinding::features)
        .def_property_readonly("policies", &SearchBatchBinding::policies)
        .def_property_readonly("values", &SearchBatchBinding::values)
        .def_property_readonly(
            "pin_memory_requested",
            &SearchBatchBinding::pin_memory_requested)
        .def_property_readonly(
            "pin_memory_realized",
            &SearchBatchBinding::pin_memory_realized)
        .def_property_readonly("pin_memory", &SearchBatchBinding::pin_memory_requested)
        .def_property_readonly("board_size", &SearchBatchBinding::board_size)
        .def_property_readonly("active_count", &SearchBatchBinding::active_count)
        .def_property_readonly("capacity", &SearchBatchBinding::capacity)
        .def_property_readonly(
            "pending_request_count",
            &SearchBatchBinding::pending_request_count)
        .def_property_readonly(
            "pending_selected_count",
            &SearchBatchBinding::pending_selected_count)
        .def_property_readonly("has_pending", &SearchBatchBinding::has_pending)
        .def_property_readonly("pending_token", &SearchBatchBinding::pending_token)
        .def_property_readonly("last_token", &SearchBatchBinding::last_token)
        .def_property_readonly("thread_count", &SearchBatchBinding::thread_count);
}
