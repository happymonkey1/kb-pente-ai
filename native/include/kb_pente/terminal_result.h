#pragma once

#include <cstdint>
#include <optional>

#include "kb_pente/player.h"

namespace kb_pente {

enum class GameStatus : std::uint8_t {
    InProgress,
    Draw,
    Win,
};

enum class WinReason : std::uint8_t {
    None,
    Line,
    Capture,
};

// These aliases keep status and reason terminology explicit at call sites that
// describe a terminal result.
using TerminalStatus = GameStatus;
using TerminalReason = WinReason;

// TerminalResult is the small value object passed between game and search
// layers. A winner is present only for a win; a win reason is optional because
// callers may know the winner before they classify the cause.
struct TerminalResult final {
    GameStatus status = GameStatus::InProgress;
    std::optional<Player> winner;
    WinReason reason = WinReason::None;

    [[nodiscard]] static constexpr TerminalResult in_progress() noexcept {
        return TerminalResult{GameStatus::InProgress, std::nullopt,
                              WinReason::None};
    }

    [[nodiscard]] static constexpr TerminalResult draw() noexcept {
        return TerminalResult{GameStatus::Draw, std::nullopt, WinReason::None};
    }

    [[nodiscard]] static constexpr TerminalResult win(
        Player player,
        WinReason win_reason = WinReason::None) noexcept {
        return TerminalResult{GameStatus::Win, player, win_reason};
    }

    [[nodiscard]] constexpr bool is_terminal() const noexcept {
        return status != GameStatus::InProgress;
    }

    [[nodiscard]] constexpr bool is_valid() const noexcept {
        switch (status) {
            case GameStatus::InProgress:
            case GameStatus::Draw:
                return !winner.has_value() && reason == WinReason::None;
            case GameStatus::Win:
                return winner.has_value() && is_valid_player(*winner);
        }

        return false;
    }

    [[nodiscard]] constexpr float value_for(Player player) const noexcept {
        if (status != GameStatus::Win || !winner.has_value()) {
            return 0.0F;
        }

        return *winner == player ? 1.0F : -1.0F;
    }
};

}  // namespace kb_pente
