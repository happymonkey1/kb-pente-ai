#include "kb_pente/rules.h"

#include <stdexcept>

namespace kb_pente {

namespace {

[[nodiscard]] bool is_opening_action(
    const Position& position,
    Action action) noexcept {
    const auto center = static_cast<std::uint8_t>(position.board_size / 2U);
    const auto center_action = static_cast<Action>(
        static_cast<std::uint16_t>(center) * position.board_size + center);
    return position.ply == 0 && action == center_action;
}

[[nodiscard]] bool is_outside_tournament_exclusion(
    const Position& position,
    Action action) noexcept {
    const auto center = static_cast<int>(position.board_size / 2U);
    const auto row = static_cast<int>(action / position.board_size);
    const auto column = static_cast<int>(action % position.board_size);
    const auto row_distance = row > center ? row - center : center - row;
    const auto column_distance =
        column > center ? column - center : center - column;
    return row_distance >= 3 || column_distance >= 3;
}

}  // namespace

bool is_legal_action(
    const Position& position,
    Ruleset ruleset,
    Action action) noexcept {
    if (!position.is_valid() ||
        !is_valid_ruleset_configuration(position.board_size, ruleset) ||
        !position.is_active_action(action) || position.stones[action] != 0) {
        return false;
    }

    if (ruleset == Ruleset::Freestyle) {
        return true;
    }

    if (position.ply == 0) {
        return is_opening_action(position, action);
    }

    return ruleset != Ruleset::Tournament || position.ply != 2 ||
           position.current_player != Player::One ||
           is_outside_tournament_exclusion(position, action);
}

ActionMask legal_action_mask(const Position& position, Ruleset ruleset) {
    position.validate();
    if (!is_valid_ruleset_configuration(position.board_size, ruleset)) {
        throw std::invalid_argument("Invalid board size for Pente ruleset");
    }

    ActionMask mask;
    for (Action action = 0; action < position.action_count(); ++action) {
        if (is_legal_action(position, ruleset, action)) {
            mask.set(action);
        }
    }
    return mask;
}

}  // namespace kb_pente
