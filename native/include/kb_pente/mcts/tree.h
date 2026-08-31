#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "kb_pente/game.h"
#include "kb_pente/mcts/search_config.h"
#include "kb_pente/mcts/tree_arena.h"

namespace kb_pente {

class SearchSession;

// PathEdge identifies the parent edge traversed while selecting a leaf.
// Keeping IDs instead of positions makes the pending path reusable and cheap.
struct PathEdge final {
    NodeId node = kInvalidNode;
    Action action = kInvalidAction;
};

[[nodiscard]] inline bool operator==(
    const PathEdge& left,
    const PathEdge& right) noexcept {
    return left.node == right.node && left.action == right.action;
}

[[nodiscard]] inline bool operator!=(
    const PathEdge& left,
    const PathEdge& right) noexcept {
    return !(left == right);
}

// Tree owns one Pente root and the deterministic single-tree search state.
// Evaluation is supplied by callers through accept_evaluation or resolved
// internally for terminal leaves.
class Tree final {
public:
    static constexpr std::size_t kInitialPathCapacity = 64;

    explicit Tree(
        Position root_position,
        Ruleset ruleset = kDefaultRuleset,
        SearchConfig config = SearchConfig{});

    ~Tree() = default;

    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;
    Tree(Tree&& other);
    Tree& operator=(Tree&& other);

    // Select one unexpanded or terminal leaf and retain its path as pending.
    [[nodiscard]] NodeId select_leaf();

    [[nodiscard]] const Position& leaf_position(NodeId leaf) const;
    [[nodiscard]] const TerminalResult& leaf_terminal(NodeId leaf) const;

    // Complete a nonterminal pending leaf with policy probabilities and a
    // leaf-side value. The policy length must equal the active board area.
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

    // Resolve a pending terminal leaf without invoking an evaluator.
    void resolve_terminal(NodeId leaf);

    [[nodiscard]] bool has_pending_evaluation() const noexcept {
        return pending_active_;
    }

    [[nodiscard]] NodeId pending_leaf() const noexcept {
        return pending_leaf_;
    }

    [[nodiscard]] std::size_t pending_path_size() const noexcept {
        return pending_path_.size();
    }

    [[nodiscard]] std::size_t pending_path_capacity() const noexcept {
        return pending_path_.capacity();
    }

    [[nodiscard]] std::uint64_t invalid_policy_fallbacks() const noexcept {
        return invalid_policy_fallbacks_;
    }

    [[nodiscard]] NodeId root_id() const noexcept { return root_; }

    [[nodiscard]] const Position& root_position() const;

    [[nodiscard]] Ruleset ruleset() const noexcept { return ruleset_; }

    [[nodiscard]] const SearchConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] const TreeArena& arena() const noexcept { return arena_; }

private:
    friend class SearchSession;

    static constexpr float kPuctEpsilon = 1.0e-8F;
    static const Tree& validate_move_source(const Tree& tree);

    // SearchSession uses this narrow internal boundary for temporary root
    // priors. Direct Tree callers retain only the deterministic base-prior API.
    [[nodiscard]] NodeId select_leaf(
        const float* root_priors,
        std::size_t root_policy_length);

    void accept_evaluation_for_session(
        NodeId leaf,
        const float* policy,
        std::size_t policy_length,
        float value);

    void resolve_terminal_for_session(NodeId leaf);

    void accept_evaluation_impl(
        NodeId leaf,
        const float* policy,
        std::size_t policy_length,
        float value);

    void resolve_terminal_impl(NodeId leaf);

    [[nodiscard]] std::size_t legal_action_count(
        const NodeMeta& node) const noexcept;

    [[nodiscard]] Action select_action(
        NodeId node_id,
        const NodeMeta& node,
        const float* root_priors) const;

    [[nodiscard]] NodeId create_child(NodeId parent, Action action);

    void validate_pending_leaf(NodeId leaf) const;
    void validate_backup(float leaf_value) const;
    void backup(float leaf_value);
    void finish_pending() noexcept;

    Ruleset ruleset_;
    SearchConfig config_;
    std::mt19937_64 rng_;
    TreeArena arena_;
    NodeId root_ = kInvalidNode;
    std::vector<PathEdge> pending_path_;
    NodeId pending_leaf_ = kInvalidNode;
    bool pending_active_ = false;
    std::uint64_t invalid_policy_fallbacks_ = 0;
    SearchSession* session_owner_ = nullptr;
};

}  // namespace kb_pente
