#pragma once

#include <cstdint>

#include "kb_pente/action_mask.h"
#include "kb_pente/position.h"

namespace kb_pente {

// Ruleset identifies the rule variant used by later game-transition code.
// The enum intentionally carries no transition behavior in the core skeleton.
enum class Ruleset : std::uint8_t {
    Standard,
    Tournament,
    Freestyle,
};

inline constexpr Ruleset kDefaultRuleset = Ruleset::Standard;

[[nodiscard]] constexpr bool is_valid_ruleset(Ruleset ruleset) noexcept {
    switch (ruleset) {
        case Ruleset::Standard:
        case Ruleset::Tournament:
        case Ruleset::Freestyle:
            return true;
    }

    return false;
}

// Standard and Tournament rules require an odd board so that the mandatory
// opening center is a single intersection. Freestyle also supports even
// boards within the fixed native capacity.
[[nodiscard]] constexpr bool is_valid_ruleset_configuration(
    std::uint8_t board_size,
    Ruleset ruleset) noexcept {
    if (!is_supported_board_size(board_size) || !is_valid_ruleset(ruleset)) {
        return false;
    }

    return ruleset == Ruleset::Freestyle || board_size % 2U == 1U;
}

[[nodiscard]] bool is_legal_action(
    const Position& position,
    Ruleset ruleset,
    Action action) noexcept;

// Invalid positions or ruleset configurations are reported as
// std::invalid_argument. A valid result contains only actions in the active
// board area.
[[nodiscard]] ActionMask legal_action_mask(
    const Position& position,
    Ruleset ruleset);

}  // namespace kb_pente
