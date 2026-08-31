#include "kb_pente/mcts/tree.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kb_pente {

namespace {

[[nodiscard]] std::size_t action_count(const NodeMeta& node) noexcept {
    return node.position.action_count();
}

[[nodiscard]] bool path_contains_node(
    const std::vector<PathEdge>& path,
    NodeId node) noexcept {
    for (const PathEdge& edge : path) {
        if (edge.node == node) {
            return true;
        }
    }
    return false;
}

}  // namespace

Tree::Tree(
    Position root_position,
    Ruleset ruleset,
    SearchConfig config)
    : ruleset_(ruleset), config_(config), rng_(config_.seed) {
    config_.validate();
    root_position.validate();
    if (!is_valid_ruleset_configuration(root_position.board_size, ruleset_)) {
        throw std::invalid_argument("Invalid board size for Pente ruleset");
    }

    const TerminalResult terminal = check_terminal(root_position);
    if (!terminal.is_valid()) {
        throw std::invalid_argument("Invalid root terminal result");
    }
    if (terminal.is_terminal()) {
        throw std::invalid_argument("Cannot search a terminal root");
    }

    NodeMeta root_meta{};
    root_meta.position = root_position;
    root_meta.terminal = terminal;
    root_meta.legal = legal_action_mask(root_position, ruleset_);
    if (legal_action_count(root_meta) == 0U) {
        throw std::logic_error("Non-terminal root has no legal action");
    }

    pending_path_.reserve(kInitialPathCapacity);
    root_ = arena_.allocate(std::move(root_meta));
}

const Tree& Tree::validate_move_source(const Tree& tree) {
    if (tree.session_owner_ != nullptr) {
        throw std::logic_error("Cannot move a Tree with a live search session");
    }
    return tree;
}

Tree::Tree(Tree&& other)
    : ruleset_(validate_move_source(other).ruleset_),
      config_(validate_move_source(other).config_),
      rng_(std::move(other.rng_)),
      arena_(std::move(other.arena_)),
      root_(other.root_),
      pending_path_(std::move(other.pending_path_)),
      pending_leaf_(other.pending_leaf_),
      pending_active_(other.pending_active_),
      invalid_policy_fallbacks_(other.invalid_policy_fallbacks_),
      session_owner_(nullptr) {
    other.root_ = kInvalidNode;
    other.pending_leaf_ = kInvalidNode;
    other.pending_active_ = false;
    other.invalid_policy_fallbacks_ = 0U;
}

Tree& Tree::operator=(Tree&& other) {
    if (this == &other) {
        return *this;
    }
    if (session_owner_ != nullptr || other.session_owner_ != nullptr) {
        throw std::logic_error(
            "Cannot move a Tree with a live search session");
    }

    ruleset_ = other.ruleset_;
    config_ = other.config_;
    rng_ = std::move(other.rng_);
    arena_ = std::move(other.arena_);
    root_ = other.root_;
    pending_path_ = std::move(other.pending_path_);
    pending_leaf_ = other.pending_leaf_;
    pending_active_ = other.pending_active_;
    invalid_policy_fallbacks_ = other.invalid_policy_fallbacks_;

    other.root_ = kInvalidNode;
    other.pending_leaf_ = kInvalidNode;
    other.pending_active_ = false;
    other.invalid_policy_fallbacks_ = 0U;
    return *this;
}

NodeId Tree::select_leaf() {
    if (session_owner_ != nullptr) {
        throw std::logic_error(
            "Tree direct selection is unavailable while a session is live");
    }
    return select_leaf(nullptr, 0U);
}

NodeId Tree::select_leaf(
    const float* root_priors,
    std::size_t root_policy_length) {
    if (pending_active_) {
        throw std::logic_error(
            "Cannot select a leaf while an evaluation is pending");
    }
    if ((root_priors == nullptr) != (root_policy_length == 0U)) {
        throw std::invalid_argument(
            "Root prior override must be null or cover the active action area");
    }
    if (root_priors != nullptr &&
        root_policy_length != root_position().action_count()) {
        throw std::invalid_argument(
            "Root prior override length must equal the active action count");
    }

    pending_path_.clear();
    pending_leaf_ = kInvalidNode;
    NodeId current = root_;
    try {
        while (true) {
            if (path_contains_node(pending_path_, current)) {
                throw std::logic_error("Tree traversal encountered a cycle");
            }

            NodeMeta& node = arena_.node(current);
            if (!node.terminal.is_valid()) {
                throw std::logic_error("Tree node has an invalid terminal result");
            }
            if (node.terminal.is_terminal()) {
                pending_leaf_ = current;
                pending_active_ = true;
                return current;
            }

            const std::size_t legal_count = legal_action_count(node);
            if (!node.expanded) {
                if (legal_count == 0U) {
                    throw std::logic_error(
                        "Non-terminal node has no legal action");
                }
                pending_leaf_ = current;
                pending_active_ = true;
                return current;
            }
            if (legal_count == 0U) {
                throw std::logic_error(
                    "Non-terminal node has no legal action");
            }

            const float* node_root_priors =
                current == root_ ? root_priors : nullptr;
            const Action action =
                select_action(current, node, node_root_priors);
            pending_path_.push_back(PathEdge{current, action});

            NodeId child = arena_.edge_row(current).child(action);
            if (child == kInvalidNode) {
                child = create_child(current, action);
                // Allocation may reallocate all edge vectors, so reacquire
                // the row before recording the new child ID.
                arena_.edge_row(current).child(action) = child;
            }
            current = child;
        }
    } catch (...) {
        pending_path_.clear();
        pending_leaf_ = kInvalidNode;
        pending_active_ = false;
        throw;
    }
}

const Position& Tree::leaf_position(NodeId leaf) const {
    return arena_.node(leaf).position;
}

const TerminalResult& Tree::leaf_terminal(NodeId leaf) const {
    return arena_.node(leaf).terminal;
}

void Tree::accept_evaluation(
    NodeId leaf,
    const float* policy,
    std::size_t policy_length,  // NOLINT(bugprone-easily-swappable-parameters)
    float value) {
    if (session_owner_ != nullptr) {
        throw std::logic_error(
            "Tree direct evaluation is unavailable while a session is live");
    }
    accept_evaluation_impl(leaf, policy, policy_length, value);
}

void Tree::accept_evaluation_for_session(
    NodeId leaf,
    const float* policy,
    std::size_t policy_length,  // NOLINT(bugprone-easily-swappable-parameters)
    float value) {
    if (session_owner_ == nullptr) {
        throw std::logic_error("Tree session evaluation requires an owner");
    }
    accept_evaluation_impl(leaf, policy, policy_length, value);
}

void Tree::accept_evaluation_impl(
    NodeId leaf,
    const float* policy,
    std::size_t policy_length,  // NOLINT(bugprone-easily-swappable-parameters)
    float value) {
    validate_pending_leaf(leaf);
    NodeMeta& node = arena_.node(leaf);
    if (!node.terminal.is_valid()) {
        throw std::logic_error("Leaf has an invalid terminal result");
    }
    if (node.terminal.is_terminal()) {
        throw std::logic_error(
            "Cannot evaluate a terminal leaf; resolve it instead");
    }
    if (node.expanded) {
        throw std::logic_error("Leaf has already been expanded");
    }

    const std::size_t active_actions = action_count(node);
    if (policy_length != active_actions) {
        throw std::invalid_argument(
            "Evaluator policy length must equal the active action count");
    }
    if (policy == nullptr) {
        throw std::invalid_argument("Evaluator policy cannot be null");
    }
    if (!std::isfinite(value) || value < -1.0F || value > 1.0F) {
        throw std::invalid_argument(
            "Evaluator value must be finite and in [-1, 1]");
    }

    const std::size_t legal_actions = legal_action_count(node);
    if (legal_actions == 0U) {
        throw std::logic_error("Non-terminal node has no legal action");
    }

    double legal_mass = 0.0;
    for (std::size_t index = 0; index < active_actions; ++index) {
        const float probability = policy[index];
        if (!std::isfinite(probability) || probability < 0.0F) {
            throw std::invalid_argument(
                "Evaluator policy must contain finite non-negative values");
        }
        if (node.legal.contains(static_cast<Action>(index))) {
            legal_mass += static_cast<double>(probability);
        }
    }
    if (!std::isfinite(legal_mass)) {
        throw std::invalid_argument("Evaluator legal policy mass is not finite");
    }

    const bool use_fallback = legal_mass == 0.0;
    if (use_fallback &&
        invalid_policy_fallbacks_ ==
            std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Invalid policy fallback counter overflow");
    }

    const float uniform_prior =
        1.0F / static_cast<float>(legal_actions);
    for (std::size_t index = 0; index < active_actions; ++index) {
        if (!node.legal.contains(static_cast<Action>(index))) {
            continue;
        }
        const float prior = use_fallback
                                ? uniform_prior
                                : static_cast<float>(
                                      static_cast<double>(policy[index]) /
                                      legal_mass);
        if (!std::isfinite(prior) || prior < 0.0F) {
            throw std::invalid_argument(
                "Evaluator policy normalization is not finite");
        }
    }

    // Validate the complete backup before changing priors or visit counts.
    validate_backup(value);

    EdgeRowView row = arena_.edge_row(leaf);
    for (std::size_t index = 0; index < TreeArena::edge_stride(); ++index) {
        const Action action = static_cast<Action>(index);
        float prior = 0.0F;
        if (index < active_actions && node.legal.contains(action)) {
            prior = use_fallback
                        ? uniform_prior
                        : static_cast<float>(
                              static_cast<double>(policy[index]) /
                              legal_mass);
        }
        row.prior(action) = prior;
    }
    node.expanded = true;
    if (use_fallback) {
        ++invalid_policy_fallbacks_;
    }
    backup(value);
    finish_pending();
}

void Tree::resolve_terminal(NodeId leaf) {
    if (session_owner_ != nullptr) {
        throw std::logic_error(
            "Tree direct terminal resolution is unavailable while a session is live");
    }
    resolve_terminal_impl(leaf);
}

void Tree::resolve_terminal_for_session(NodeId leaf) {
    if (session_owner_ == nullptr) {
        throw std::logic_error("Tree session resolution requires an owner");
    }
    resolve_terminal_impl(leaf);
}

void Tree::resolve_terminal_impl(NodeId leaf) {
    validate_pending_leaf(leaf);
    const NodeMeta& node = arena_.node(leaf);
    if (!node.terminal.is_valid()) {
        throw std::logic_error("Leaf has an invalid terminal result");
    }
    if (!node.terminal.is_terminal()) {
        throw std::logic_error(
            "Cannot resolve a non-terminal leaf as terminal");
    }

    const float value =
        node.terminal.value_for(node.position.current_player);
    validate_backup(value);
    backup(value);
    finish_pending();
}

std::size_t Tree::legal_action_count(const NodeMeta& node) const noexcept {
    return node.legal.count();
}

Action Tree::select_action(
    NodeId node_id,  // NOLINT(bugprone-easily-swappable-parameters)
    const NodeMeta& node,
    const float* root_priors) const {
    const std::size_t active_actions = action_count(node);
    const float sqrt_parent = std::sqrt(
        static_cast<float>(node.total_visits) + kPuctEpsilon);
    if (!std::isfinite(sqrt_parent)) {
        throw std::overflow_error("PUCT parent visit scale is not finite");
    }

    const ConstEdgeRowView row = arena_.edge_row(node_id);
    float best_score = -std::numeric_limits<float>::infinity();
    Action best_action = kInvalidAction;
    for (std::size_t index = 0; index < active_actions; ++index) {
        const Action action = static_cast<Action>(index);
        if (!node.legal.contains(action)) {
            continue;
        }

        const float prior =
            root_priors == nullptr ? row.prior(action) : root_priors[index];
        if (!std::isfinite(prior) || prior < 0.0F) {
            throw std::logic_error("PUCT prior is invalid");
        }
        const std::uint32_t visits = row.visit_count(action);
        const float value_sum = row.value_sum(action);
        if (!std::isfinite(value_sum)) {
            throw std::logic_error("PUCT value sum is not finite");
        }
        const float q = visits == 0U
                            ? 0.0F
                            : value_sum / static_cast<float>(visits);
        if (!std::isfinite(q)) {
            throw std::overflow_error("PUCT Q value is not finite");
        }

        const float exploration =
            prior == 0.0F
                ? 0.0F
                : config_.c_puct * prior * sqrt_parent /
                      (1.0F + static_cast<float>(visits));
        const float score = q + exploration;
        if (!std::isfinite(score)) {
            throw std::overflow_error("PUCT score is not finite");
        }
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }

    if (best_action == kInvalidAction) {
        throw std::logic_error("Non-terminal node has no legal action");
    }
    return best_action;
}

NodeId Tree::create_child(
    NodeId parent,  // NOLINT(bugprone-easily-swappable-parameters)
    Action action) {
    const Position parent_position = arena_.node(parent).position;
    const Transition transition =
        apply_action(parent_position, action, ruleset_);
    if (!transition.terminal.is_valid()) {
        throw std::logic_error("Native transition returned invalid terminal state");
    }

    NodeMeta child{};
    child.position = transition.position;
    child.terminal = transition.terminal;
    if (!child.terminal.is_terminal()) {
        child.legal = legal_action_mask(child.position, ruleset_);
        if (legal_action_count(child) == 0U) {
            throw std::logic_error("Non-terminal node has no legal action");
        }
    }
    return arena_.allocate(std::move(child));
}

void Tree::validate_pending_leaf(NodeId leaf) const {
    if (!pending_active_) {
        throw std::logic_error("No pending leaf evaluation");
    }
    if (leaf != pending_leaf_) {
        throw std::logic_error("Leaf does not match pending evaluation");
    }
}

void Tree::validate_backup(float leaf_value) const {
    if (!std::isfinite(leaf_value)) {
        throw std::overflow_error("Backup value is not finite");
    }

    float value = leaf_value;
    for (auto edge = pending_path_.rbegin();
         edge != pending_path_.rend(); ++edge) {
        value = -value;
        if (!std::isfinite(value)) {
            throw std::overflow_error("Backup value is not finite");
        }

        const NodeMeta& parent = arena_.node(edge->node);
        if (!parent.position.is_active_action(edge->action) ||
            !parent.legal.contains(edge->action)) {
            throw std::logic_error("Backup path contains an invalid action");
        }
        if (parent.total_visits == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Tree node visit count overflow");
        }

        const ConstEdgeRowView row = arena_.edge_row(edge->node);
        const std::uint32_t visits = row.visit_count(edge->action);
        if (visits == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Tree edge visit count overflow");
        }
        const float value_sum = row.value_sum(edge->action);
        if (!std::isfinite(value_sum) ||
            !std::isfinite(value_sum + value)) {
            throw std::overflow_error("Tree edge value sum overflow");
        }
    }
}

void Tree::backup(float leaf_value) {
    float value = leaf_value;
    for (auto edge = pending_path_.rbegin();
         edge != pending_path_.rend(); ++edge) {
        value = -value;
        EdgeRowView row = arena_.edge_row(edge->node);
        ++row.visit_count(edge->action);
        row.value_sum(edge->action) += value;
        ++arena_.node(edge->node).total_visits;
    }
}

void Tree::finish_pending() noexcept {
    pending_path_.clear();
    pending_leaf_ = kInvalidNode;
    pending_active_ = false;
}

const Position& Tree::root_position() const {
    return arena_.node(root_).position;
}

}  // namespace kb_pente
