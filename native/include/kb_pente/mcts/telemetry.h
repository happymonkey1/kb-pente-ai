#pragma once

#include <cstddef>
#include <cstdint>

namespace kb_pente {

// A root needs this many edge visits before a one-child search is considered
// a meaningful collapse rather than an under-sampled opening.
inline constexpr std::uint64_t kSearchCollapseMinRootVisits = 8U;

// SearchTelemetry is a detached value report of one completed or in-progress
// SearchSession. It copies counters and root edge statistics so callers never
// observe or mutate the live search state through the report.
struct SearchTelemetry final {
    std::uint64_t completed_simulations = 0U;
    std::uint64_t evaluator_completions = 0U;
    std::uint64_t terminal_simulations = 0U;
    std::uint64_t selected_leaves = 0U;
    std::size_t max_selected_path_depth = 0U;

    std::size_t root_legal_actions = 0U;
    std::uint64_t root_edge_visits = 0U;
    std::size_t root_children_visited = 0U;
    float root_visit_entropy = 0.0F;
    float root_max_visit_share = 0.0F;
    bool root_collapse_eligible = false;
    bool root_search_collapsed = false;

    std::uint64_t invalid_policy_fallbacks = 0U;
    std::uint64_t zero_visit_fallbacks = 0U;
};

[[nodiscard]] inline bool operator==(
    const SearchTelemetry& left,
    const SearchTelemetry& right) noexcept {
    return left.completed_simulations == right.completed_simulations &&
           left.evaluator_completions == right.evaluator_completions &&
           left.terminal_simulations == right.terminal_simulations &&
           left.selected_leaves == right.selected_leaves &&
           left.max_selected_path_depth == right.max_selected_path_depth &&
           left.root_legal_actions == right.root_legal_actions &&
           left.root_edge_visits == right.root_edge_visits &&
           left.root_children_visited == right.root_children_visited &&
           left.root_visit_entropy == right.root_visit_entropy &&
           left.root_max_visit_share == right.root_max_visit_share &&
           left.root_collapse_eligible == right.root_collapse_eligible &&
           left.root_search_collapsed == right.root_search_collapsed &&
           left.invalid_policy_fallbacks == right.invalid_policy_fallbacks &&
           left.zero_visit_fallbacks == right.zero_visit_fallbacks;
}

[[nodiscard]] inline bool operator!=(
    const SearchTelemetry& left,
    const SearchTelemetry& right) noexcept {
    return !(left == right);
}

}  // namespace kb_pente
