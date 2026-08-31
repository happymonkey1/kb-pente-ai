#include "kb_pente/mcts/search_session.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace kb_pente {

SearchSessionConfig::SearchSessionConfig(
    float temperature_value,
    bool add_root_noise_value)
    : temperature(temperature_value), add_root_noise(add_root_noise_value) {
    validate();
}

void SearchSessionConfig::validate() const {
    if (!std::isfinite(temperature) || temperature < 0.0F) {
        throw std::invalid_argument(
            "Search temperature must be finite and non-negative");
    }
}

SearchSession::SearchSession(Tree& tree, SearchSessionConfig config)
    : tree_(tree), config_(config) {
    config_.validate();
    if (tree_.session_owner_ != nullptr) {
        throw std::logic_error("Tree already has a live search session");
    }
    if (tree_.has_pending_evaluation()) {
        throw std::logic_error(
            "Cannot create a search session with a pending tree evaluation");
    }
    invalid_policy_fallback_baseline_ = tree_.invalid_policy_fallbacks();

    const NodeMeta& root = tree_.arena().node(tree_.root_id());
    if (!root.terminal.is_valid()) {
        throw std::logic_error("Tree root has an invalid terminal result");
    }
    if (root.terminal.is_terminal()) {
        throw std::invalid_argument(
            "Cannot create a search session for a terminal root");
    }
    const bool root_expanded = root.expanded;
    tree_.reserve_for_search();
    if (root_expanded) {
        initialize_root_priors();
    }
    tree_.session_owner_ = this;
}

SearchSession::~SearchSession() noexcept {
    if (tree_.session_owner_ == this) {
        tree_.session_owner_ = nullptr;
    }
}

std::optional<NodeId> SearchSession::select_evaluation_leaf() {
    if (tree_.has_pending_evaluation()) {
        throw std::logic_error(
            "Cannot select while a session evaluation is pending");
    }

    const NodeMeta& root = tree_.arena().node(tree_.root_id());
    if (root.expanded && !root_priors_initialized_) {
        initialize_root_priors();
    }

    while (!complete()) {
        const std::size_t root_action_count =
            tree_.root_position().action_count();
        const NodeId leaf = root_priors_initialized_
                                ? tree_.select_leaf(
                                      root_search_priors_.data(),
                                      root_action_count)
                                : tree_.select_leaf(nullptr, 0U);
        ++selected_leaves_;
        const std::size_t selected_depth = tree_.pending_path_size();
        if (selected_depth > max_selected_path_depth_) {
            max_selected_path_depth_ = selected_depth;
        }
        if (tree_.leaf_terminal(leaf).is_terminal()) {
            tree_.resolve_terminal_for_session(leaf);
            ++completed_simulations_;
            ++terminal_simulations_;
            continue;
        }
        return leaf;
    }

    return std::nullopt;
}

void SearchSession::accept_evaluation(
    NodeId leaf,
    const float* policy,
    std::size_t policy_length,  // NOLINT(bugprone-easily-swappable-parameters)
    float value) {
    if (complete()) {
        throw std::logic_error(
            "Cannot accept an evaluation after the simulation budget completes");
    }
    if (!tree_.has_pending_evaluation()) {
        throw std::logic_error("No pending session evaluation");
    }

    const bool is_root = leaf == tree_.root_id();
    tree_.accept_evaluation_for_session(leaf, policy, policy_length, value);
    ++completed_simulations_;
    ++evaluator_completions_;
    if (is_root && !root_priors_initialized_) {
        initialize_root_priors();
    }
}

SearchTelemetry SearchSession::telemetry() const {
    SearchTelemetry result{};
    result.completed_simulations = completed_simulations_;
    result.evaluator_completions = evaluator_completions_;
    result.terminal_simulations = terminal_simulations_;
    result.selected_leaves = selected_leaves_;
    result.max_selected_path_depth = max_selected_path_depth_;
    result.zero_visit_fallbacks = zero_visit_fallbacks_;

    const std::uint64_t cumulative_fallbacks =
        tree_.invalid_policy_fallbacks();
    if (cumulative_fallbacks < invalid_policy_fallback_baseline_) {
        throw std::logic_error(
            "Tree invalid-policy fallback counter moved backwards");
    }
    result.invalid_policy_fallbacks =
        cumulative_fallbacks - invalid_policy_fallback_baseline_;

    const NodeMeta& root = tree_.arena().node(tree_.root_id());
    const ConstEdgeRowView row = tree_.arena().edge_row(tree_.root_id());
    const std::size_t active_actions = root.position.action_count();
    result.root_legal_actions = root.legal.count();

    std::uint64_t total_visit_count = 0U;
    std::uint32_t maximum_visits = 0U;
    for (std::size_t index = 0U; index < active_actions; ++index) {
        const Action action = static_cast<Action>(index);
        if (!root.legal.contains(action)) {
            continue;
        }
        const std::uint32_t visits = row.visit_count(action);
        total_visit_count += static_cast<std::uint64_t>(visits);
        if (visits > 0U) {
            ++result.root_children_visited;
            if (visits > maximum_visits) {
                maximum_visits = visits;
            }
        }
    }
    const double total_visits = static_cast<double>(total_visit_count);
    if (!std::isfinite(total_visits)) {
        throw std::overflow_error("Root visit count is not finite");
    }
    result.root_edge_visits = total_visit_count;

    if (total_visits > 0.0) {
        double entropy = 0.0;
        for (std::size_t index = 0U; index < active_actions; ++index) {
            const Action action = static_cast<Action>(index);
            if (!root.legal.contains(action)) {
                continue;
            }
            const std::uint32_t visits = row.visit_count(action);
            if (visits == 0U) {
                continue;
            }
            const double probability =
                static_cast<double>(visits) / total_visits;
            entropy -= probability * std::log(probability);
        }
        const double maximum_share =
            static_cast<double>(maximum_visits) / total_visits;
        if (!std::isfinite(entropy) || !std::isfinite(maximum_share)) {
            throw std::overflow_error("Root visit statistics are not finite");
        }
        result.root_visit_entropy = static_cast<float>(entropy);
        result.root_max_visit_share = static_cast<float>(maximum_share);
    }

    result.root_collapse_eligible =
        result.root_legal_actions > 1U &&
        result.root_edge_visits >= kSearchCollapseMinRootVisits;
    result.root_search_collapsed =
        result.root_collapse_eligible && result.root_children_visited <= 1U;
    return result;
}

std::array<float, kMaxActions> SearchSession::root_policy() {
    if (!complete()) {
        throw std::logic_error(
            "Root policy is unavailable before the simulation budget completes");
    }
    if (tree_.has_pending_evaluation()) {
        throw std::logic_error(
            "Root policy is unavailable while an evaluation is pending");
    }
    if (!root_priors_initialized_) {
        initialize_root_priors();
    }
    std::array<float, kMaxActions> policy{};
    const NodeMeta& root = tree_.arena().node(tree_.root_id());
    const ConstEdgeRowView row = tree_.arena().edge_row(tree_.root_id());
    const std::size_t active_actions = root.position.action_count();

    bool has_visit = false;
    std::uint32_t maximum_visits = 0U;
    Action best_action = kInvalidAction;
    for (std::size_t index = 0; index < active_actions; ++index) {
        const Action action = static_cast<Action>(index);
        if (!root.legal.contains(action)) {
            continue;
        }
        const std::uint32_t visits = row.visit_count(action);
        if (visits == 0U) {
            continue;
        }
        if (!has_visit || visits > maximum_visits) {
            has_visit = true;
            maximum_visits = visits;
            best_action = action;
        }
    }

    if (!has_visit) {
        fill_zero_visit_fallback(policy);
        if (config_.temperature == 0.0F) {
            Action best_fallback_action = kInvalidAction;
            float best_fallback_probability = 0.0F;
            for (std::size_t index = 0U; index < active_actions; ++index) {
                const Action action = static_cast<Action>(index);
                if (!root.legal.contains(action) ||
                    (best_fallback_action != kInvalidAction &&
                     policy[index] <= best_fallback_probability)) {
                    continue;
                }
                best_fallback_action = action;
                best_fallback_probability = policy[index];
            }
            if (best_fallback_action == kInvalidAction) {
                throw std::logic_error(
                    "Root fallback has no legal action for temperature zero");
            }
            policy.fill(0.0F);
            policy[best_fallback_action] = 1.0F;
        }
        return policy;
    }

    if (config_.temperature == 0.0F) {
        policy[best_action] = 1.0F;
        return policy;
    }

    const double inverse_temperature =
        1.0 / static_cast<double>(config_.temperature);
    if (!std::isfinite(inverse_temperature)) {
        policy[best_action] = 1.0F;
        return policy;
    }

    double total_weight = 0.0;
    for (std::size_t index = 0; index < active_actions; ++index) {
        const Action action = static_cast<Action>(index);
        if (!root.legal.contains(action)) {
            continue;
        }
        const std::uint32_t visits = row.visit_count(action);
        if (visits == 0U) {
            continue;
        }
        const double relative_visits =
            static_cast<double>(visits) /
            static_cast<double>(maximum_visits);
        const double log_weight =
            std::log(relative_visits) * inverse_temperature;
        if (!std::isfinite(log_weight)) {
            throw std::overflow_error("Root policy weight is not finite");
        }
        const double weight = std::exp(log_weight);
        if (!std::isfinite(weight)) {
            throw std::overflow_error("Root policy weight is not finite");
        }
        total_weight += weight;
        if (!std::isfinite(total_weight)) {
            throw std::overflow_error("Root policy normalization is not finite");
        }
        policy[index] = static_cast<float>(weight);
    }

    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        throw std::overflow_error("Root policy has no finite visit mass");
    }
    for (std::size_t index = 0; index < active_actions; ++index) {
        if (policy[index] == 0.0F) {
            continue;
        }
        policy[index] = static_cast<float>(
            static_cast<double>(policy[index]) / total_weight);
        if (!std::isfinite(policy[index])) {
            throw std::overflow_error("Root policy normalization is not finite");
        }
    }
    return policy;
}

const std::array<float, kMaxActions>& SearchSession::root_search_priors() const {
    if (!root_priors_initialized_) {
        throw std::logic_error(
            "Root search priors are unavailable before root expansion");
    }
    return root_search_priors_;
}

float SearchSession::root_search_prior(Action action) const {
    if (action >= kMaxActions) {
        throw std::out_of_range("Root search action is out of range");
    }
    return root_search_priors()[static_cast<std::size_t>(action)];
}

void SearchSession::initialize_root_priors() {
    if (root_priors_initialized_) {
        return;
    }

    const NodeMeta& root = tree_.arena().node(tree_.root_id());
    if (!root.expanded) {
        throw std::logic_error("Root search priors require an expanded root");
    }
    const std::size_t active_actions = root.position.action_count();
    const std::size_t legal_actions = root.legal.count();
    if (legal_actions == 0U) {
        throw std::logic_error("Expanded root has no legal action");
    }

    const ConstEdgeRowView row = tree_.arena().edge_row(tree_.root_id());
    std::array<double, kMaxActions> noise{};
    double noise_mass = 0.0;
    const bool use_noise = config_.add_root_noise &&
                           tree_.config().root_noise_epsilon > 0.0F;
    std::gamma_distribution<double> dirichlet_component(
        static_cast<double>(tree_.config().root_dirichlet_alpha), 1.0);
    for (std::size_t index = 0; index < active_actions; ++index) {
        const Action action = static_cast<Action>(index);
        if (!root.legal.contains(action)) {
            continue;
        }

        const float base_prior = row.prior(action);
        if (!std::isfinite(base_prior) || base_prior < 0.0F) {
            throw std::logic_error("Root base prior is invalid");
        }
        if (use_noise) {
            const double component = dirichlet_component(tree_.rng_);
            if (!std::isfinite(component) || component < 0.0) {
                throw std::overflow_error("Root Dirichlet component is invalid");
            }
            noise[index] = component;
            noise_mass += component;
            if (!std::isfinite(noise_mass)) {
                throw std::overflow_error("Root Dirichlet mass is not finite");
            }
        }
    }
    if (use_noise && !(noise_mass > 0.0)) {
        throw std::runtime_error("Root Dirichlet noise has zero mass");
    }

    root_search_priors_.fill(0.0F);
    const double epsilon =
        static_cast<double>(tree_.config().root_noise_epsilon);
    for (std::size_t index = 0; index < active_actions; ++index) {
        const Action action = static_cast<Action>(index);
        if (!root.legal.contains(action)) {
            continue;
        }
        const double base_prior = static_cast<double>(row.prior(action));
        const double noise_prior =
            use_noise ? noise[index] / noise_mass : 0.0;
        const double mixed_prior = use_noise
                                       ? (1.0 - epsilon) * base_prior +
                                             epsilon * noise_prior
                                       : base_prior;
        const float result = static_cast<float>(mixed_prior);
        if (!std::isfinite(result) || result < 0.0F) {
            throw std::overflow_error("Root search prior is not finite");
        }
        root_search_priors_[index] = result;
    }

    root_priors_initialized_ = true;
}

void SearchSession::fill_zero_visit_fallback(
    std::array<float, kMaxActions>& policy) {
    const NodeMeta& root = tree_.arena().node(tree_.root_id());
    const ConstEdgeRowView row = tree_.arena().edge_row(tree_.root_id());
    const std::size_t active_actions = root.position.action_count();
    const std::size_t legal_actions = root.legal.count();
    if (legal_actions == 0U) {
        throw std::logic_error("Root has no legal action for fallback");
    }

    double legal_mass = 0.0;
    for (std::size_t index = 0; index < active_actions; ++index) {
        const Action action = static_cast<Action>(index);
        if (!root.legal.contains(action)) {
            continue;
        }
        const float prior = row.prior(action);
        if (!std::isfinite(prior) || prior < 0.0F) {
            throw std::logic_error("Root base prior is invalid");
        }
        legal_mass += static_cast<double>(prior);
    }

    if (legal_mass > 0.0 && std::isfinite(legal_mass)) {
        for (std::size_t index = 0; index < active_actions; ++index) {
            const Action action = static_cast<Action>(index);
            if (!root.legal.contains(action)) {
                continue;
            }
            const float normalized = static_cast<float>(
                static_cast<double>(row.prior(action)) / legal_mass);
            if (!std::isfinite(normalized) || normalized < 0.0F) {
                throw std::overflow_error(
                    "Root fallback normalization is not finite");
            }
            policy[index] = normalized;
        }
    } else {
        const float uniform = 1.0F / static_cast<float>(legal_actions);
        if (!std::isfinite(uniform)) {
            throw std::overflow_error("Root fallback uniform is not finite");
        }
        for (std::size_t index = 0; index < active_actions; ++index) {
            if (root.legal.contains(static_cast<Action>(index))) {
                policy[index] = uniform;
            }
        }
    }

    if (zero_visit_fallbacks_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Zero-visit fallback counter overflow");
    }
    ++zero_visit_fallbacks_;
}

}  // namespace kb_pente
