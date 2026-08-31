#pragma once

#include <cstddef>
#include <cstdint>

namespace kb_pente {

enum class Player : std::int8_t {
    Two = -1,
    One = 1,
};

[[nodiscard]] constexpr bool is_valid_player(Player player) noexcept {
    return player == Player::One || player == Player::Two;
}

// Player values are intentionally not used as array indexes. The mapping is
// part of the core data contract: Player One is slot zero and Player Two is
// slot one.
[[nodiscard]] constexpr std::size_t player_index(Player player) noexcept {
    return player == Player::One ? 0U : 1U;
}

[[nodiscard]] constexpr Player opponent(Player player) noexcept {
    return player == Player::One ? Player::Two : Player::One;
}

}  // namespace kb_pente
