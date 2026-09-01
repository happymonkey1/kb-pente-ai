#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "kb_pente/mcts/telemetry.h"
#include "kb_pente/mcts/tree.h"

namespace kb_pente {

// SearchSessionConfig contains the evaluator-facing settings for one bounded
// search. The simulation budget and noise distribution belong to Tree's
// SearchConfig so successive sessions share one search RNG stream.
struct SearchSessionConfig final {
    explicit SearchSessionConfig(
        float temperature_value = 1.0F,
        bool add_root_noise_value = false);

    void validate() const;

    float temperature = 1.0F;
    bool add_root_noise = false;
};

// SearchSession owns evaluator-facing lifecycle state over one Tree. It
// resolves terminal leaves internally and keeps root noise outside the arena.
class SearchSession final {
public:
    explicit SearchSession(
        Tree& tree,
        SearchSessionConfig config = SearchSessionConfig{});

    ~SearchSession() noexcept;

    SearchSession(const SearchSession&) = delete;
    SearchSession& operator=(const SearchSession&) = delete;
    SearchSession(SearchSession&&) = delete;
    SearchSession& operator=(SearchSession&&) = delete;

    // Return the next nonterminal leaf needing evaluation, or nullopt after
    // the exact simulation budget is complete.
    [[nodiscard]] std::optional<NodeId> select_evaluation_leaf();

    // Complete the pending nonterminal leaf with policy probabilities and a
    // value from that leaf's side-to-move perspective.
    void accept_evaluation(
        NodeId leaf,
        const float* policy,
        std::size_t policy_length,
        float value);

    template <std::size_t PolicySize>
    void accept_evaluation(
        NodeId leaf,
        const std::array<float, PolicySize>& policy,
        float value) {
        accept_evaluation(leaf, policy.data(), PolicySize, value);
    }

    // Return legal root probabilities derived from completed edge visits.
    // Calling this before completion is a lifecycle error.
    [[nodiscard]] std::array<float, kMaxActions> root_policy();

    [[nodiscard]] bool complete() const noexcept {
        return completed_simulations_ >= tree_.config().simulation_budget;
    }

    [[nodiscard]] std::uint32_t completed_simulations() const noexcept {
        return completed_simulations_;
    }

    [[nodiscard]] std::uint64_t evaluator_completions() const noexcept {
        return evaluator_completions_;
    }

    [[nodiscard]] std::uint32_t simulation_budget() const noexcept {
        return tree_.config().simulation_budget;
    }

    [[nodiscard]] bool has_pending_evaluation() const noexcept {
        return tree_.has_pending_evaluation();
    }

    // A pristine session has not selected or completed any simulation.
    [[nodiscard]] bool pristine() const noexcept {
        return completed_simulations_ == 0U && selected_leaves_ == 0U;
    }

    [[nodiscard]] std::uint64_t zero_visit_fallbacks() const noexcept {
        return zero_visit_fallbacks_;
    }

    // Return a value snapshot of session counters and root visit quality.
    // The snapshot is valid while a session is pending, which makes rejected
    // evaluator requests observable without changing the search state.
    [[nodiscard]] SearchTelemetry telemetry() const;

    [[nodiscard]] bool root_priors_initialized() const noexcept {
        return root_priors_initialized_;
    }

    [[nodiscard]] bool root_noise_initialized() const noexcept {
        return root_priors_initialized_ && config_.add_root_noise;
    }

    // Root priors are unavailable until the root has been expanded.
    [[nodiscard]] const std::array<float, kMaxActions>&
    root_search_priors() const;

    [[nodiscard]] float root_search_prior(Action action) const;

    [[nodiscard]] const SearchSessionConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] const Tree& tree() const noexcept { return tree_; }

private:
    void initialize_root_priors();
    void fill_zero_visit_fallback(
        std::array<float, kMaxActions>& policy);

    Tree& tree_;
    SearchSessionConfig config_;
    std::array<float, kMaxActions> root_search_priors_{};
    std::uint32_t completed_simulations_ = 0U;
    std::uint64_t evaluator_completions_ = 0U;
    std::uint64_t terminal_simulations_ = 0U;
    std::uint64_t selected_leaves_ = 0U;
    std::size_t max_selected_path_depth_ = 0U;
    std::uint64_t invalid_policy_fallback_baseline_ = 0U;
    std::uint64_t zero_visit_fallbacks_ = 0U;
    bool root_priors_initialized_ = false;
};

}  // namespace kb_pente
