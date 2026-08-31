#pragma once

#include <cstdint>

#include "kb_pente/constants.h"
#include "kb_pente/player.h"

namespace kb_pente {

struct Position;

// PositionHash is the cached two-lane identity used by later native
// inference coordination. It intentionally excludes ruleset identity.
struct PositionHash final {
    std::uint64_t lo = 0U;
    std::uint64_t hi = 0U;
};

[[nodiscard]] constexpr bool operator==(
    const PositionHash& left,
    const PositionHash& right) noexcept {
    return left.lo == right.lo && left.hi == right.hi;
}

[[nodiscard]] constexpr bool operator!=(
    const PositionHash& left,
    const PositionHash& right) noexcept {
    return !(left == right);
}

namespace position_hash_detail {

// These fixed-seed token operations are shared by full recomputation and the
// game transition hot path so their representations cannot drift apart.
[[nodiscard]] PositionHash recompute(const Position& position) noexcept;
void toggle_stone(
    PositionHash& hash,
    Action action,
    std::int8_t stone) noexcept;
void toggle_capture_count(
    PositionHash& hash,
    Player player,
    std::uint8_t count) noexcept;
void toggle_ply(PositionHash& hash, std::uint16_t ply) noexcept;
void toggle_opening(PositionHash& hash, bool opening) noexcept;
void toggle_last_action(PositionHash& hash, Action action) noexcept;
void toggle_current_player(PositionHash& hash, Player player) noexcept;
void toggle_board_size(PositionHash& hash, std::uint8_t board_size) noexcept;

}  // namespace position_hash_detail

}  // namespace kb_pente
