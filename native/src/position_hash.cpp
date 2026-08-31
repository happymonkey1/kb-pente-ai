#include "kb_pente/position_hash.h"

#include <cstddef>

#include "kb_pente/position.h"

namespace kb_pente {
namespace position_hash_detail {
namespace {

constexpr std::uint64_t kSeedLo = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t kSeedHi = 0x13198a2e03707344ULL;
constexpr std::uint64_t kMixConstant = 0x9e3779b97f4a7c15ULL;

[[nodiscard]] constexpr std::uint64_t mix(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] constexpr std::uint64_t token(
    std::uint64_t seed,
    std::uint64_t domain,
    std::uint64_t value) noexcept {
    return mix(seed ^ (domain * kMixConstant) ^ (value + kMixConstant));
}

void xor_token(
    PositionHash& hash,
    std::uint64_t domain,
    std::uint64_t value) noexcept {
    hash.lo ^= token(kSeedLo, domain, value);
    hash.hi ^= token(kSeedHi, domain, value);
}

constexpr std::uint64_t kBoardSizeDomain = 1U;
constexpr std::uint64_t kStoneDomain = 2U;
constexpr std::uint64_t kCaptureDomain = 3U;
constexpr std::uint64_t kPlyDomain = 4U;
constexpr std::uint64_t kOpeningDomain = 5U;
constexpr std::uint64_t kLastActionDomain = 6U;
constexpr std::uint64_t kCurrentPlayerDomain = 7U;

}  // namespace

PositionHash recompute(const Position& position) noexcept {
    PositionHash hash{};
    toggle_board_size(hash, position.board_size);
    toggle_capture_count(hash, Player::One, position.captures[0]);
    toggle_capture_count(hash, Player::Two, position.captures[1]);
    toggle_ply(hash, position.ply);
    toggle_opening(hash, position.ply == 0U);
    toggle_last_action(hash, position.last_action);
    toggle_current_player(hash, position.current_player);

    for (std::size_t index = 0U; index < kMaxActions; ++index) {
        const auto stone = position.stones[index];
        if (stone != 0) {
            toggle_stone(hash, static_cast<Action>(index), stone);
        }
    }
    return hash;
}

void toggle_stone(
    PositionHash& hash,
    Action action,
    std::int8_t stone) noexcept {
    if (stone == 0) {
        return;
    }
    const std::uint64_t value =
        (static_cast<std::uint64_t>(action) << 8U) |
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(stone));
    xor_token(hash, kStoneDomain, value);
}

void toggle_capture_count(
    PositionHash& hash,
    Player player,
    std::uint8_t count) noexcept {
    const std::uint64_t value =
        (static_cast<std::uint64_t>(player_index(player)) << 8U) | count;
    xor_token(hash, kCaptureDomain, value);
}

void toggle_ply(PositionHash& hash, std::uint16_t ply) noexcept {
    xor_token(hash, kPlyDomain, ply);
}

void toggle_opening(PositionHash& hash, bool opening) noexcept {
    xor_token(hash, kOpeningDomain, opening ? 1U : 0U);
}

void toggle_last_action(PositionHash& hash, Action action) noexcept {
    xor_token(hash, kLastActionDomain, action);
}

void toggle_current_player(PositionHash& hash, Player player) noexcept {
    xor_token(
        hash,
        kCurrentPlayerDomain,
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(static_cast<std::int8_t>(player))));
}

void toggle_board_size(PositionHash& hash, std::uint8_t board_size) noexcept {
    xor_token(hash, kBoardSizeDomain, board_size);
}

}  // namespace position_hash_detail
}  // namespace kb_pente
