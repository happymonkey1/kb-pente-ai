#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "kb_pente/constants.h"

namespace kb_pente {

// A fixed-capacity bit mask for the 361 possible board actions. Bits above
// kMaxActions are kept clear and are never exposed as legal actions.
struct ActionMask final {
    std::array<std::uint64_t, 6> words{};

    void clear() noexcept { words.fill(0); }

    void set(Action action) noexcept {
        if (action >= kMaxActions) {
            return;
        }

        words[action / 64U] |= std::uint64_t{1} << (action % 64U);
    }

    void clear(Action action) noexcept {
        if (action >= kMaxActions) {
            return;
        }

        words[action / 64U] &=
            ~(std::uint64_t{1} << (action % 64U));
    }

    [[nodiscard]] bool contains(Action action) const noexcept {
        if (action >= kMaxActions) {
            return false;
        }

        return (words[action / 64U] &
                (std::uint64_t{1} << (action % 64U))) != 0;
    }

    [[nodiscard]] std::size_t count() const noexcept {
        std::size_t result = 0;
        for (std::uint64_t word : words) {
            while (word != 0) {
                word &= word - 1;
                ++result;
            }
        }
        return result;
    }
};

[[nodiscard]] inline bool operator==(
    const ActionMask& left,
    const ActionMask& right) noexcept {
    return left.words == right.words;
}

[[nodiscard]] inline bool operator!=(
    const ActionMask& left,
    const ActionMask& right) noexcept {
    return !(left == right);
}

}  // namespace kb_pente
