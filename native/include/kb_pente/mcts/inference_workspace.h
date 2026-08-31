#pragma once

#include <cstddef>
#include <vector>

#include "kb_pente/mcts/tree_arena.h"
#include "kb_pente/rules.h"

namespace kb_pente {

// InferenceCandidate is the identity and routing metadata for one selected
// leaf. The position pointer is borrowed from its owning Tree and must remain
// valid until the workspace is reset or regrouped.
struct InferenceCandidate final {
    std::size_t slot = 0U;
    NodeId leaf = kInvalidNode;
    const Position* position = nullptr;
    Ruleset ruleset = kDefaultRuleset;
};

// InferenceWorkspace groups selected leaves without allocating during a
// selection wave. Hashes narrow candidate comparisons; semantic Position
// equality is still required before two candidates share an evaluation row.
class InferenceWorkspace final {
public:
    explicit InferenceWorkspace(std::size_t capacity);

    InferenceWorkspace(const InferenceWorkspace&) = delete;
    InferenceWorkspace& operator=(const InferenceWorkspace&) = delete;
    InferenceWorkspace(InferenceWorkspace&&) = delete;
    InferenceWorkspace& operator=(InferenceWorkspace&&) = delete;

    // Discard the current wave while retaining every reserved allocation.
    void clear() noexcept;

    // Add one candidate in deterministic raw-selection order.
    void add(InferenceCandidate candidate);

    // Sort and group all candidates added since clear(). Repeated calls are
    // harmless and do not change the resulting representative order.
    void finalize();

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t raw_count() const noexcept {
        return candidates_.size();
    }
    [[nodiscard]] std::size_t unique_count() const noexcept {
        return representatives_.size();
    }

    [[nodiscard]] const InferenceCandidate& raw_candidate(
        std::size_t index) const;
    [[nodiscard]] const InferenceCandidate& representative(
        std::size_t index) const;
    [[nodiscard]] std::size_t evaluation_index_for_raw(
        std::size_t index) const;

    // These capacities are exposed so callers can assert that waves never
    // grow the fixed-capacity grouping workspace.
    [[nodiscard]] std::size_t candidate_capacity() const noexcept {
        return candidates_.capacity();
    }
    [[nodiscard]] std::size_t sorted_index_capacity() const noexcept {
        return sorted_indices_.capacity();
    }
    [[nodiscard]] std::size_t representative_capacity() const noexcept {
        return representatives_.capacity();
    }
    [[nodiscard]] std::size_t selected_to_evaluation_capacity() const noexcept {
        return selected_to_evaluation_.capacity();
    }
    [[nodiscard]] std::size_t raw_selected_request_capacity() const noexcept {
        return candidates_.capacity();
    }

private:
    [[nodiscard]] bool same_key(
        std::size_t left_index,
        std::size_t right_index) const noexcept;

    std::size_t capacity_ = 0U;
    std::vector<InferenceCandidate> candidates_;
    std::vector<std::size_t> sorted_indices_;
    std::vector<std::size_t> representatives_;
    std::vector<std::size_t> selected_to_representative_;
    std::vector<std::size_t> representative_to_evaluation_;
    std::vector<std::size_t> selected_to_evaluation_;
    bool finalized_ = false;
};

}  // namespace kb_pente
