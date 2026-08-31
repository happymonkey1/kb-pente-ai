#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "kb_pente/constants.h"
#include "kb_pente/player.h"

namespace kb_pente {

// A board uses the first board_size * board_size entries of stones in
// row-major order. The remaining entries stay zero so every position has a
// deterministic fixed-size representation.
struct Position final {
    std::array<std::int8_t, kMaxActions> stones{};
    std::array<std::uint8_t, kPlayerCount> captures{};
    std::uint16_t ply = 0;
    Action last_action = kInvalidAction;
    std::uint8_t board_size = kDefaultBoardSize;
    Player current_player = Player::One;

    [[nodiscard]] static Position initial(
        std::uint8_t board_size = kDefaultBoardSize);

    [[nodiscard]] constexpr std::size_t action_count() const noexcept {
        return board_area(board_size);
    }

    [[nodiscard]] constexpr bool is_active_action(Action action) const noexcept {
        return is_supported_board_size(board_size) &&
               action < action_count();
    }

    [[nodiscard]] std::uint8_t capture_count(Player player) const noexcept {
        return captures[player_index(player)];
    }

    // Returns false without allocating or throwing. validate() is available
    // for callers that want an exception when importing an invalid position.
    [[nodiscard]] bool is_valid() const noexcept;
    void validate() const;
};

}  // namespace kb_pente
