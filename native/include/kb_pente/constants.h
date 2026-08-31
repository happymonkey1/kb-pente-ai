#pragma once

#include <cstddef>
#include <cstdint>

namespace kb_pente {

// An action is a row-major index into the fixed-capacity board storage.
using Action = std::uint16_t;

inline constexpr std::uint8_t kMinBoardSize = 5;
inline constexpr std::uint8_t kMaxBoardSize = 19;
inline constexpr std::uint8_t kDefaultBoardSize = kMaxBoardSize;
inline constexpr std::uint8_t kPlayerCount = 2;
inline constexpr std::uint8_t kCapturesToWin = 5;
inline constexpr std::uint16_t kMaxActions =
    static_cast<std::uint16_t>(kMaxBoardSize) * kMaxBoardSize;
inline constexpr Action kInvalidAction = static_cast<Action>(0xffffU);

[[nodiscard]] constexpr bool is_supported_board_size(
    std::uint8_t board_size) noexcept {
    return board_size >= kMinBoardSize && board_size <= kMaxBoardSize;
}

[[nodiscard]] constexpr std::size_t board_area(
    std::uint8_t board_size) noexcept {
    return static_cast<std::size_t>(board_size) * board_size;
}

}  // namespace kb_pente
