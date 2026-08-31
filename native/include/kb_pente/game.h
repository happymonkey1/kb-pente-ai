#pragma once

#include "kb_pente/features.h"
#include "kb_pente/rules.h"
#include "kb_pente/terminal_result.h"

namespace kb_pente {

struct Transition final {
    Position position;
    TerminalResult terminal;
};

[[nodiscard]] inline bool operator==(
    const Transition& left,
    const Transition& right) noexcept {
    return left.position == right.position && left.terminal == right.terminal;
}

[[nodiscard]] inline bool operator!=(
    const Transition& left,
    const Transition& right) noexcept {
    return !(left == right);
}

// Apply one legal action, including captures and terminal detection. Invalid
// positions, configurations, or actions throw std::invalid_argument.
[[nodiscard]] Transition apply_action(
    const Position& parent,
    Action action,
    Ruleset rules);

// Check an imported position. When last_action is set, only the previous
// player's lines through that action are examined, matching the fast path
// used after transitions. Without a last action, both players are scanned.
[[nodiscard]] TerminalResult check_terminal(const Position& position);

}  // namespace kb_pente
