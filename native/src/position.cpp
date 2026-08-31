#include "kb_pente/position.h"

#include <stdexcept>

namespace kb_pente {

Position Position::initial(std::uint8_t requested_board_size) {
    if (!is_supported_board_size(requested_board_size)) {
        throw std::invalid_argument("Pente board size must be between 5 and 19");
    }

    Position position{};
    position.board_size = requested_board_size;
    return position;
}

bool Position::is_valid() const noexcept {
    if (!is_supported_board_size(board_size) ||
        !is_valid_player(current_player)) {
        return false;
    }

    const std::size_t active_actions = action_count();
    std::size_t occupied_stones = 0;
    for (std::size_t action = 0; action < kMaxActions; ++action) {
        const std::int8_t stone = stones[action];
        if (stone != 0 && stone != static_cast<std::int8_t>(Player::One) &&
            stone != static_cast<std::int8_t>(Player::Two)) {
            return false;
        }

        if (action >= active_actions) {
            if (stone != 0) {
                return false;
            }
        } else if (stone != 0) {
            ++occupied_stones;
        }
    }

    const std::size_t captured_stones =
        2U * (static_cast<std::size_t>(captures[0]) + captures[1]);
    if (static_cast<std::size_t>(ply) != occupied_stones + captured_stones) {
        return false;
    }

    const Player expected_player =
        ply % 2 == 0 ? Player::One : Player::Two;
    if (current_player != expected_player) {
        return false;
    }

    if (last_action == kInvalidAction) {
        return true;
    }

    if (ply == 0 || last_action >= active_actions) {
        return false;
    }

    const auto previous_player = opponent(current_player);
    return stones[last_action] == static_cast<std::int8_t>(previous_player);
}

void Position::validate() const {
    if (!is_valid()) {
        throw std::invalid_argument("Invalid Pente position");
    }
}

}  // namespace kb_pente
