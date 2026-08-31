#pragma once

#include <cstdint>

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

}  // namespace kb_pente
