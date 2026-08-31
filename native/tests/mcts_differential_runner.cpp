#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>

#include "kb_pente/mcts/search_session.h"

namespace {

constexpr std::size_t kHeaderMagicSize = 8U;
constexpr std::uint16_t kProtocolVersion = 1U;
constexpr std::uint32_t kMaximumCases = 100'000U;
constexpr std::array<char, kHeaderMagicSize> kRequestMagic{{
    'K', 'P', 'M', 'C', 'T', 'S', '6', 'B',
}};
constexpr std::array<char, kHeaderMagicSize> kResponseMagic{{
    'K', 'P', 'M', 'R', 'E', 'S', '6', 'B',
}};

enum class EvaluatorMode : std::uint8_t {
    UniformZero = 0,
    FixedNonuniform = 1,
    ConstantPositive = 2,
    ConstantNegative = 3,
    AllZero = 4,
    PositionDependent = 5,
};

[[nodiscard]] bool read_bytes(void* destination, std::size_t size) {
    std::cin.read(
        static_cast<char*>(destination), static_cast<std::streamsize>(size));
    return std::cin.good() &&
           static_cast<std::size_t>(std::cin.gcount()) == size;
}

[[nodiscard]] bool read_u8(std::uint8_t& value) {
    return read_bytes(&value, sizeof(value));
}

[[nodiscard]] bool read_i8(std::int8_t& value) {
    std::uint8_t raw = 0U;
    if (!read_u8(raw)) {
        return false;
    }
    value = static_cast<std::int8_t>(raw);
    return true;
}

[[nodiscard]] bool read_u16(std::uint16_t& value) {
    std::array<std::uint8_t, 2> bytes{};
    if (!read_bytes(bytes.data(), bytes.size())) {
        return false;
    }
    value = static_cast<std::uint16_t>(bytes[0]) |
            (static_cast<std::uint16_t>(bytes[1]) << 8U);
    return true;
}

[[nodiscard]] bool read_u32(std::uint32_t& value) {
    std::array<std::uint8_t, 4> bytes{};
    if (!read_bytes(bytes.data(), bytes.size())) {
        return false;
    }
    value = static_cast<std::uint32_t>(bytes[0]) |
            (static_cast<std::uint32_t>(bytes[1]) << 8U) |
            (static_cast<std::uint32_t>(bytes[2]) << 16U) |
            (static_cast<std::uint32_t>(bytes[3]) << 24U);
    return true;
}

[[nodiscard]] bool read_float(float& value) {
    std::uint32_t bits = 0U;
    if (!read_u32(bits)) {
        return false;
    }
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

void write_bytes(const void* source, std::size_t size) {
    std::cout.write(
        static_cast<const char*>(source), static_cast<std::streamsize>(size));
}

void write_u8(std::uint8_t value) {
    write_bytes(&value, sizeof(value));
}

void write_u16(std::uint16_t value) {
    const std::array<std::uint8_t, 2> bytes{{
        static_cast<std::uint8_t>(value & 0xffU),
        static_cast<std::uint8_t>((value >> 8U) & 0xffU),
    }};
    write_bytes(bytes.data(), bytes.size());
}

void write_u32(std::uint32_t value) {
    const std::array<std::uint8_t, 4> bytes{{
        static_cast<std::uint8_t>(value & 0xffU),
        static_cast<std::uint8_t>((value >> 8U) & 0xffU),
        static_cast<std::uint8_t>((value >> 16U) & 0xffU),
        static_cast<std::uint8_t>((value >> 24U) & 0xffU),
    }};
    write_bytes(bytes.data(), bytes.size());
}

void write_u64(std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t index = 0U; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(
            (value >> (8U * index)) & static_cast<std::uint64_t>(0xffU));
    }
    write_bytes(bytes.data(), bytes.size());
}

void write_float(float value) {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(value));
    write_u32(bits);
}

struct WirePosition final {
    std::array<std::int8_t, kb_pente::kMaxActions> stones{};
    std::array<std::uint8_t, kb_pente::kPlayerCount> captures{};
    std::uint16_t ply = 0U;
    kb_pente::Action last_action = kb_pente::kInvalidAction;
    std::uint8_t board_size = 0U;
    std::int8_t current_player = 0;
};

[[nodiscard]] bool read_wire_position(WirePosition& wire) {
    if (!read_bytes(wire.stones.data(), wire.stones.size())) {
        return false;
    }
    for (auto& capture : wire.captures) {
        if (!read_u8(capture)) {
            return false;
        }
    }
    return read_u16(wire.ply) && read_u16(wire.last_action) &&
           read_u8(wire.board_size) && read_i8(wire.current_player);
}

[[nodiscard]] kb_pente::Position decode_position(const WirePosition& wire) {
    kb_pente::Position position{};
    position.stones = wire.stones;
    position.captures = wire.captures;
    position.ply = wire.ply;
    position.last_action = wire.last_action;
    position.board_size = wire.board_size;
    position.current_player =
        static_cast<kb_pente::Player>(wire.current_player);
    return position;
}

struct Evaluation final {
    std::array<float, kb_pente::kMaxActions> policy{};
    float value = 0.0F;
};

[[nodiscard]] std::int64_t position_checksum(
    const kb_pente::Position& position) {
    std::int64_t checksum = static_cast<std::int64_t>(position.ply);
    checksum += 3 * static_cast<std::int64_t>(position.captures[0]);
    checksum += 5 * static_cast<std::int64_t>(position.captures[1]);
    checksum += position.current_player == kb_pente::Player::One ? 11 : 17;
    for (std::size_t index = 0U; index < position.action_count(); ++index) {
        checksum += static_cast<std::int64_t>(index + 1U) *
                    (static_cast<std::int64_t>(position.stones[index]) + 1);
    }
    return checksum;
}

[[nodiscard]] Evaluation evaluate(
    const kb_pente::Position& position,
    EvaluatorMode mode) {
    Evaluation result{};
    const std::size_t active_actions = position.action_count();
    const std::int64_t checksum = position_checksum(position);
    for (std::size_t index = 0U; index < active_actions; ++index) {
        float probability = 0.0F;
        switch (mode) {
            case EvaluatorMode::UniformZero:
            case EvaluatorMode::ConstantPositive:
            case EvaluatorMode::ConstantNegative:
                probability = 1.0F;
                break;
            case EvaluatorMode::FixedNonuniform:
                probability = std::array<float, 4>{{
                    1.0F, 0.5F, 0.25F, 0.125F,
                }}[index % 4U];
                break;
            case EvaluatorMode::AllZero:
                probability = 0.0F;
                break;
            case EvaluatorMode::PositionDependent:
                probability = std::array<float, 4>{{
                    0.0F, 0.25F, 0.5F, 1.0F,
                }}[static_cast<std::size_t>((checksum +
                                               7 * static_cast<std::int64_t>(index)) %
                                              4)];
                break;
        }
        result.policy[index] = probability;
    }

    switch (mode) {
        case EvaluatorMode::ConstantPositive:
            result.value = 0.25F;
            break;
        case EvaluatorMode::ConstantNegative:
            result.value = -0.25F;
            break;
        case EvaluatorMode::PositionDependent:
            result.value = std::array<float, 3>{{
                -0.25F, 0.0F, 0.25F,
            }}[static_cast<std::size_t>(checksum % 3)];
            break;
        case EvaluatorMode::UniformZero:
        case EvaluatorMode::FixedNonuniform:
        case EvaluatorMode::AllZero:
            result.value = 0.0F;
            break;
    }
    return result;
}

int protocol_error(const char* message) {
    std::cerr << "native MCTS differential protocol error: " << message
              << '\n';
    return 2;
}

void write_result(
    const kb_pente::SearchSession& session,
    const std::array<float, kb_pente::kMaxActions>& root_policy) {
    const kb_pente::Tree& tree = session.tree();
    const kb_pente::NodeMeta& root = tree.arena().node(tree.root_id());
    const kb_pente::ConstEdgeRowView row =
        tree.arena().edge_row(tree.root_id());
    const std::size_t active_actions = root.position.action_count();
    write_u16(static_cast<std::uint16_t>(active_actions));
    for (std::size_t index = 0U; index < active_actions; ++index) {
        write_u32(row.visit_count(static_cast<kb_pente::Action>(index)));
    }
    for (std::size_t index = 0U; index < active_actions; ++index) {
        write_float(row.value_sum(static_cast<kb_pente::Action>(index)));
    }
    for (std::size_t index = 0U; index < active_actions; ++index) {
        write_float(root_policy[index]);
    }

    const kb_pente::SearchTelemetry telemetry = session.telemetry();
    write_u64(telemetry.completed_simulations);
    write_u64(telemetry.evaluator_completions);
    write_u64(telemetry.terminal_simulations);
    write_u64(telemetry.selected_leaves);
    write_u64(telemetry.max_selected_path_depth);
    write_u64(telemetry.root_legal_actions);
    write_u64(telemetry.root_edge_visits);
    write_u64(telemetry.root_children_visited);
    write_float(telemetry.root_visit_entropy);
    write_float(telemetry.root_max_visit_share);
    write_u64(telemetry.invalid_policy_fallbacks);
    write_u64(telemetry.zero_visit_fallbacks);
    write_u8(telemetry.root_collapse_eligible ? 1U : 0U);
    write_u8(telemetry.root_search_collapsed ? 1U : 0U);
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::array<char, kHeaderMagicSize> request_magic{};
    if (!read_bytes(request_magic.data(), request_magic.size())) {
        return protocol_error("truncated request header");
    }
    if (request_magic != kRequestMagic) {
        return protocol_error("unexpected request magic");
    }

    std::uint16_t version = 0U;
    if (!read_u16(version)) {
        return protocol_error("truncated protocol version");
    }
    if (version != kProtocolVersion) {
        return protocol_error("unsupported protocol version");
    }

    std::uint32_t request_count = 0U;
    if (!read_u32(request_count)) {
        return protocol_error("truncated request count");
    }
    if (request_count > kMaximumCases) {
        return protocol_error("request count is too large");
    }

    write_bytes(kResponseMagic.data(), kResponseMagic.size());
    write_u16(kProtocolVersion);
    write_u32(request_count);

    for (std::uint32_t request_index = 0U; request_index < request_count;
         ++request_index) {
        std::uint8_t ruleset_code = 0U;
        std::uint8_t mode_code = 0U;
        std::uint32_t simulation_budget = 0U;
        float c_puct = 0.0F;
        float temperature = 0.0F;
        WirePosition wire{};
        if (!read_u8(ruleset_code) || !read_u8(mode_code) ||
            !read_u32(simulation_budget) || !read_float(c_puct) ||
            !read_float(temperature) || !read_wire_position(wire)) {
            return protocol_error("truncated request record");
        }
        if (ruleset_code > 2U) {
            return protocol_error("unknown ruleset code");
        }
        if (mode_code > static_cast<std::uint8_t>(EvaluatorMode::PositionDependent)) {
            return protocol_error("unknown evaluator mode");
        }
        if (simulation_budget == 0U) {
            return protocol_error("simulation budget must be positive");
        }
        if (!std::isfinite(c_puct) || c_puct <= 0.0F) {
            return protocol_error("c_puct must be finite and positive");
        }
        if (!std::isfinite(temperature) || temperature < 0.0F) {
            return protocol_error(
                "temperature must be finite and non-negative");
        }

        try {
            const auto ruleset = static_cast<kb_pente::Ruleset>(ruleset_code);
            const auto mode = static_cast<EvaluatorMode>(mode_code);
            const kb_pente::Position position = decode_position(wire);
            position.validate();
            const kb_pente::SearchConfig config(
                c_puct, simulation_budget, 0.0F, 0.03F, 0U);
            kb_pente::Tree tree(position, ruleset, config);
            kb_pente::SearchSession session(
                tree, kb_pente::SearchSessionConfig(temperature, false));
            while (true) {
                const auto leaf = session.select_evaluation_leaf();
                if (!leaf.has_value()) {
                    break;
                }
                const kb_pente::Position& leaf_position =
                    session.tree().leaf_position(*leaf);
                const Evaluation evaluation = evaluate(leaf_position, mode);
                session.accept_evaluation(
                    *leaf,
                    evaluation.policy.data(),
                    leaf_position.action_count(),
                    evaluation.value);
            }
            const auto root_policy = session.root_policy();
            write_result(session, root_policy);
        } catch (const std::exception& error) {
            std::cerr << "native MCTS differential request " << request_index
                      << " failed: " << error.what() << '\n';
            return 3;
        }
    }

    std::cout.flush();
    return std::cout.good() ? 0 : protocol_error("response write failed");
}
