#include "kb_pente/game.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

#include "kb_pente/position_hash.h"

namespace kb_pente {

namespace {

struct Direction final {
    int row;
    int column;
};

constexpr std::array<Direction, 4> kAxes{{
    {0, 1},
    {1, 0},
    {1, 1},
    {1, -1},
}};

[[nodiscard]] bool in_bounds(
    int row,
    int column,
    std::uint8_t board_size) noexcept {
    return row >= 0 && column >= 0 && row < board_size && column < board_size;
}

[[nodiscard]] Action action_at(
    int row,
    int column,
    std::uint8_t board_size) noexcept {
    return static_cast<Action>(row * board_size + column);
}

[[nodiscard]] bool has_five_from_action(
    const Position& position,
    Player player,
    Action action) noexcept {
    if (!position.is_active_action(action) ||
        position.stones[action] != static_cast<std::int8_t>(player)) {
        return false;
    }

    const int row = action / position.board_size;
    const int column = action % position.board_size;
    for (const Direction axis : kAxes) {
        int count = 1;
        for (const int direction : {-1, 1}) {
            int next_row = row + direction * axis.row;
            int next_column = column + direction * axis.column;
            while (in_bounds(next_row, next_column, position.board_size) &&
                   position.stones[action_at(
                       next_row, next_column, position.board_size)] ==
                       static_cast<std::int8_t>(player)) {
                ++count;
                next_row += direction * axis.row;
                next_column += direction * axis.column;
            }
        }
        if (count >= 5) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool has_any_five(
    const Position& position,
    Player player) noexcept {
    for (Action action = 0; action < position.action_count(); ++action) {
        if (has_five_from_action(position, player, action)) {
            return true;
        }
    }
    return false;
}

}  // namespace

TerminalResult check_terminal(const Position& position) {
    position.validate();

    const auto player_one_captures =
        position.captures[player_index(Player::One)];
    const auto player_two_captures =
        position.captures[player_index(Player::Two)];
    if (player_one_captures >= kCapturesToWin &&
        player_two_captures >= kCapturesToWin) {
        throw std::invalid_argument("Position has multiple capture winners");
    }
    if (player_one_captures >= kCapturesToWin) {
        return TerminalResult::win(Player::One, WinReason::Capture);
    }
    if (player_two_captures >= kCapturesToWin) {
        return TerminalResult::win(Player::Two, WinReason::Capture);
    }

    if (position.last_action != kInvalidAction) {
        const auto previous_player = opponent(position.current_player);
        if (has_five_from_action(position, previous_player,
                                 position.last_action)) {
            return TerminalResult::win(previous_player, WinReason::Line);
        }
    } else {
        const bool player_one_line = has_any_five(position, Player::One);
        const bool player_two_line = has_any_five(position, Player::Two);
        if (player_one_line && player_two_line) {
            throw std::invalid_argument("Position has multiple line winners");
        }
        if (player_one_line) {
            return TerminalResult::win(Player::One, WinReason::Line);
        }
        if (player_two_line) {
            return TerminalResult::win(Player::Two, WinReason::Line);
        }
    }

    const auto stones_on_board =
        static_cast<std::size_t>(position.ply) -
        2U * (static_cast<std::size_t>(player_one_captures) +
              player_two_captures);
    if (stones_on_board == position.action_count()) {
        return TerminalResult::draw();
    }
    return TerminalResult::in_progress();
}

Transition apply_action(
    const Position& parent,
    Action action,
    Ruleset ruleset) {
    parent.validate();
    if (!parent.has_consistent_hash()) {
        throw std::invalid_argument("Pente position has a stale hash cache");
    }
    if (!is_valid_ruleset_configuration(parent.board_size, ruleset)) {
        throw std::invalid_argument("Invalid board size for Pente ruleset");
    }
    if (!is_legal_action(parent, ruleset, action)) {
        throw std::invalid_argument("Invalid Pente action");
    }

    Position child = parent;
    PositionHash child_hash = parent.hash();
    const Player moving_player = parent.current_player;
    const Player opposing_player = opponent(moving_player);
    child.stones[action] = static_cast<std::int8_t>(moving_player);
    position_hash_detail::toggle_stone(
        child_hash,
        action,
        static_cast<std::int8_t>(moving_player));

    const int row = action / parent.board_size;
    const int column = action % parent.board_size;
    std::uint8_t captured_pairs = 0;
    for (const Direction axis : kAxes) {
        for (const int direction : {-1, 1}) {
            const int first_row = row + direction * axis.row;
            const int first_column = column + direction * axis.column;
            const int second_row = row + 2 * direction * axis.row;
            const int second_column = column + 2 * direction * axis.column;
            const int third_row = row + 3 * direction * axis.row;
            const int third_column = column + 3 * direction * axis.column;
            if (!in_bounds(third_row, third_column, parent.board_size)) {
                continue;
            }

            const Action first =
                action_at(first_row, first_column, parent.board_size);
            const Action second =
                action_at(second_row, second_column, parent.board_size);
            const Action third =
                action_at(third_row, third_column, parent.board_size);
            if (child.stones[first] ==
                    static_cast<std::int8_t>(opposing_player) &&
                child.stones[second] ==
                    static_cast<std::int8_t>(opposing_player) &&
                child.stones[third] ==
                    static_cast<std::int8_t>(moving_player)) {
                child.stones[first] = 0;
                child.stones[second] = 0;
                position_hash_detail::toggle_stone(
                    child_hash,
                    first,
                    static_cast<std::int8_t>(opposing_player));
                position_hash_detail::toggle_stone(
                    child_hash,
                    second,
                    static_cast<std::int8_t>(opposing_player));
                ++captured_pairs;
            }
        }
    }

    const std::size_t moving_player_index = player_index(moving_player);
    const std::uint8_t previous_captures =
        parent.captures[moving_player_index];
    auto& captures = child.captures[moving_player_index];
    if (captured_pairs >
        std::numeric_limits<std::uint8_t>::max() - captures) {
        throw std::overflow_error("Pente capture count overflow");
    }
    captures = static_cast<std::uint8_t>(captures + captured_pairs);
    if (captures != previous_captures) {
        position_hash_detail::toggle_capture_count(
            child_hash,
            moving_player,
            previous_captures);
        position_hash_detail::toggle_capture_count(
            child_hash,
            moving_player,
            captures);
    }

    child.ply = static_cast<std::uint16_t>(parent.ply + 1U);
    position_hash_detail::toggle_ply(child_hash, parent.ply);
    position_hash_detail::toggle_ply(child_hash, child.ply);
    position_hash_detail::toggle_opening(child_hash, parent.ply == 0U);
    position_hash_detail::toggle_opening(child_hash, child.ply == 0U);

    child.last_action = action;
    position_hash_detail::toggle_last_action(child_hash, parent.last_action);
    position_hash_detail::toggle_last_action(child_hash, child.last_action);

    child.current_player = opposing_player;
    position_hash_detail::toggle_current_player(
        child_hash,
        parent.current_player);
    position_hash_detail::toggle_current_player(
        child_hash,
        child.current_player);

    child.hash_lo = child_hash.lo;
    child.hash_hi = child_hash.hi;

    return Transition{child, check_terminal(child)};
}

void write_features(const Position& position, float* output) {
    position.validate();
    if (output == nullptr) {
        throw std::invalid_argument("Feature output cannot be null");
    }

    const std::size_t area = position.action_count();
    const auto current = static_cast<std::int8_t>(position.current_player);
    const auto opposing =
        static_cast<std::int8_t>(opponent(position.current_player));
    for (std::size_t action = 0; action < area; ++action) {
        output[action] = position.stones[action] == current ? 1.0F : 0.0F;
        output[area + action] =
            position.stones[action] == opposing ? 1.0F : 0.0F;
    }

    const float current_captures = static_cast<float>(
        position.capture_count(position.current_player)) /
        static_cast<float>(kCapturesToWin);
    const float opposing_captures = static_cast<float>(
        position.capture_count(opponent(position.current_player))) /
        static_cast<float>(kCapturesToWin);
    std::fill_n(output + 2U * area, area, current_captures);
    std::fill_n(output + 3U * area, area, opposing_captures);
}

}  // namespace kb_pente
