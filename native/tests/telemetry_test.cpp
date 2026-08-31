#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "kb_pente/mcts/search_session.h"

namespace {

class TestFailure final : public std::runtime_error {
public:
    explicit TestFailure(const std::string& message)
        : std::runtime_error(message) {}
};

void expect(bool condition, const char* message) {
    if (!condition) {
        throw TestFailure(message);
    }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (...) {
        throw TestFailure(message);
    }

    throw TestFailure(message);
}

void expect_close(
    float left,
    float right,
    float tolerance,
    const char* message) {
    if (std::fabs(left - right) > tolerance) {
        throw TestFailure(message);
    }
}

kb_pente::SearchConfig tree_config(std::uint32_t simulation_budget) {
    return kb_pente::SearchConfig(
        1.5F, simulation_budget, 0.25F, 0.03F, 17U);
}

std::array<float, kb_pente::kMaxActions> uniform_policy() {
    std::array<float, kb_pente::kMaxActions> policy{};
    policy.fill(1.0F);
    return policy;
}

kb_pente::Position make_draw_root() {
    constexpr std::array<std::array<std::int8_t, 5>, 5> pattern{{
        {{1, 1, -1, -1, 1}},
        {{-1, -1, 1, 1, -1}},
        {{1, 1, -1, -1, 1}},
        {{-1, -1, 1, 1, -1}},
        {{1, -1, 1, -1, 1}},
    }};

    kb_pente::Position position = kb_pente::Position::initial(5);
    for (std::uint8_t row = 0; row < 5; ++row) {
        for (std::uint8_t column = 0; column < 5; ++column) {
            position.stones[row * 5U + column] = pattern[row][column];
        }
    }
    position.stones[24] = 0;
    position.ply = 24;
    position.current_player = kb_pente::Player::One;
    position.last_action = kb_pente::kInvalidAction;
    position.validate();
    expect(!kb_pente::check_terminal(position).is_terminal(),
           "draw fixture starts in progress");
    return position;
}

std::size_t run_with_policy(
    kb_pente::SearchSession& session,
    const std::array<float, kb_pente::kMaxActions>& policy,
    float value = 0.0F) {
    std::size_t evaluator_requests = 0U;
    while (true) {
        const auto leaf = session.select_evaluation_leaf();
        if (!leaf.has_value()) {
            break;
        }
        ++evaluator_requests;
        const std::size_t action_count =
            session.tree().leaf_position(*leaf).action_count();
        session.accept_evaluation(
            *leaf, policy.data(), action_count, value);
    }
    return evaluator_requests;
}

void test_value_snapshot_and_collapse_regression() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(16U));
    kb_pente::SearchSession session(tree);
    const auto policy = uniform_policy();

    expect(run_with_policy(session, policy) == 16U,
           "uniform root completes the configured simulations");
    const kb_pente::SearchTelemetry telemetry = session.telemetry();
    expect(telemetry.completed_simulations == 16U,
           "telemetry reports completed simulations");
    expect(telemetry.evaluator_completions == 16U,
           "telemetry reports successful evaluator completions");
    expect(telemetry.terminal_simulations == 0U,
           "uniform opening has no terminal simulations");
    expect(telemetry.selected_leaves == 16U,
           "telemetry reports every selected leaf");
    expect(telemetry.max_selected_path_depth > 0U,
           "search reaches a child leaf");
    expect(telemetry.root_legal_actions == 25U,
           "telemetry reports active legal root actions");
    expect(telemetry.root_edge_visits == 15U,
           "root edge visits exclude the root evaluator expansion");
    expect(telemetry.root_children_visited > 1U,
           "uniform zero-value search visits multiple root children");
    expect(telemetry.root_visit_entropy > 0.0F,
           "multiple root visits have positive entropy");
    expect(telemetry.root_max_visit_share < 1.0F,
           "multiple root visits do not collapse to one action");
    expect(telemetry.root_collapse_eligible,
           "root with eight or more visits is collapse eligible");
    expect(!telemetry.root_search_collapsed,
           "uniform root does not report a collapsed search");
    expect(telemetry.invalid_policy_fallbacks == 0U,
           "valid evaluator policies do not use fallback");
    expect(telemetry.zero_visit_fallbacks == 0U,
           "root policy has visited edges");

    const kb_pente::SearchTelemetry copy = telemetry;
    expect(copy == telemetry, "telemetry snapshots compare by value");
}

void test_collapsed_search_is_reported() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(
            static_cast<std::uint32_t>(
                kb_pente::kSearchCollapseMinRootVisits + 1U)));
    kb_pente::SearchSession session(tree);
    std::array<float, kb_pente::kMaxActions> policy{};
    policy[0] = 1.0F;

    run_with_policy(session, policy);
    const auto telemetry = session.telemetry();
    expect(telemetry.root_legal_actions > 1U,
           "collapsed fixture has multiple legal root actions");
    expect(telemetry.root_edge_visits == kb_pente::kSearchCollapseMinRootVisits,
           "collapsed fixture reaches the collapse threshold");
    expect(telemetry.root_children_visited == 1U,
           "single-prior fixture visits one root child");
    expect(telemetry.root_collapse_eligible,
           "single-prior fixture is collapse eligible");
    expect(telemetry.root_search_collapsed,
           "single-prior fixture reports collapsed search");
}

void test_terminal_composition_and_depth() {
    kb_pente::Tree tree(
        make_draw_root(), kb_pente::Ruleset::Freestyle, tree_config(4U));
    kb_pente::SearchSession session(tree);
    const auto policy = uniform_policy();

    expect(run_with_policy(session, policy) == 1U,
           "full-board draw fixture needs one evaluator request");
    const auto telemetry = session.telemetry();
    expect(telemetry.completed_simulations == 4U,
           "terminal resolution contributes to completion");
    expect(telemetry.evaluator_completions == 1U,
           "terminal leaves do not count as evaluator completions");
    expect(telemetry.terminal_simulations == 3U,
           "terminal leaves are counted after successful resolution");
    expect(telemetry.selected_leaves == 4U,
           "terminal and evaluator leaves are both selected");
    expect(telemetry.max_selected_path_depth == 1U,
           "terminal child selection reports one edge of depth");
    expect(telemetry.root_legal_actions == 1U,
           "draw fixture has one legal root action");
    expect(telemetry.root_edge_visits == 3U,
           "terminal backups visit the only root edge");
    expect(telemetry.root_children_visited == 1U,
           "terminal child is the one visited root child");
    expect(!telemetry.root_collapse_eligible,
           "forced root is not collapse eligible");
    expect(!telemetry.root_search_collapsed,
           "forced root does not report collapse");
}

void test_pending_and_retry_safe_accounting() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(2U));
    kb_pente::SearchSession session(tree);
    const auto policy = uniform_policy();
    const auto root = tree.root_id();

    const auto leaf = session.select_evaluation_leaf();
    expect(leaf.has_value() && *leaf == root,
           "pending fixture selects the unexpanded root");
    const auto pending_snapshot = session.telemetry();
    expect(pending_snapshot.completed_simulations == 0U,
           "pending selection does not complete a simulation");
    expect(pending_snapshot.evaluator_completions == 0U,
           "pending selection has no evaluator completion");
    expect(pending_snapshot.terminal_simulations == 0U,
           "pending selection has no terminal completion");
    expect(pending_snapshot.selected_leaves == 1U,
           "pending selection is counted once");
    expect(pending_snapshot.max_selected_path_depth == 0U,
           "root selection has zero path depth");

    expect_throws<std::logic_error>(
        [&session, &policy] {
            session.accept_evaluation(
                kb_pente::kInvalidNode, policy.data(), 25U, 0.0F);
        },
        "wrong leaf does not complete a pending evaluation");
    auto invalid_policy = policy;
    invalid_policy[0] = std::numeric_limits<float>::quiet_NaN();
    expect_throws<std::invalid_argument>(
        [&session, leaf, &invalid_policy] {
            session.accept_evaluation(
                *leaf, invalid_policy.data(), 25U, 0.0F);
        },
        "invalid policy does not complete a pending evaluation");
    expect_throws<std::invalid_argument>(
        [&session, leaf, &policy] {
            session.accept_evaluation(
                *leaf, policy.data(), 25U,
                std::numeric_limits<float>::infinity());
        },
        "invalid value does not complete a pending evaluation");
    expect(session.telemetry() == pending_snapshot,
           "rejected evaluation attempts preserve telemetry exactly");

    session.accept_evaluation(*leaf, policy.data(), 25U, 0.0F);
    const auto after_root = session.telemetry();
    expect(after_root.completed_simulations == 1U,
           "successful evaluator completion increments simulations");
    expect(after_root.evaluator_completions == 1U,
           "successful evaluator completion increments its counter");
    expect(after_root.selected_leaves == 1U,
           "accepting a leaf does not double-count selection");
    expect(after_root.invalid_policy_fallbacks == 0U,
           "valid retry leaves fallback delta unchanged");

    const auto child = session.select_evaluation_leaf();
    expect(child.has_value(), "second simulation selects a child leaf");
    const auto pending_child = session.telemetry();
    expect(pending_child.selected_leaves == 2U,
           "second selection increments selected-leaf count");
    expect(pending_child.max_selected_path_depth == 1U,
           "child selection records maximum path depth");
    session.accept_evaluation(
        *child,
        policy.data(),
        session.tree().leaf_position(*child).action_count(),
        0.0F);
    expect(session.telemetry().completed_simulations == 2U,
           "second successful evaluation completes the session");
}

void test_fallback_deltas_and_zero_visit_policy() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    const auto root = tree.select_leaf();
    std::array<float, kb_pente::kMaxActions> zero_policy{};
    tree.accept_evaluation(root, zero_policy.data(), 25U, 0.0F);
    expect(tree.invalid_policy_fallbacks() == 1U,
           "direct fallback establishes the cumulative baseline");

    {
        kb_pente::SearchSession session(tree);
        const auto leaf = session.select_evaluation_leaf();
        expect(leaf.has_value(), "expanded root can start a fresh session");
        const auto policy = uniform_policy();
        session.accept_evaluation(
            *leaf,
            policy.data(),
            session.tree().leaf_position(*leaf).action_count(),
            0.0F);
        const auto telemetry = session.telemetry();
        expect(telemetry.invalid_policy_fallbacks == 0U,
               "session fallback telemetry is a delta from its baseline");
    }

    kb_pente::Tree fallback_tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    kb_pente::SearchSession fallback_session(fallback_tree);
    const auto fallback_leaf = fallback_session.select_evaluation_leaf();
    fallback_session.accept_evaluation(
        *fallback_leaf, zero_policy.data(), 25U, 0.0F);
    expect(fallback_session.telemetry().invalid_policy_fallbacks == 1U,
           "successful zero policy increments session fallback delta");
    expect(fallback_session.telemetry().zero_visit_fallbacks == 0U,
           "fallback counter starts before root policy extraction");

    const auto root_policy = fallback_session.root_policy();
    for (std::size_t index = 0U; index < 25U; ++index) {
        expect_close(
            root_policy[index], 1.0F / 25.0F, 1.0e-7F,
            "zero policy uses legal uniform root policy fallback");
    }
    const auto telemetry = fallback_session.telemetry();
    expect(telemetry.zero_visit_fallbacks == 1U,
           "one-simulation root policy increments zero-visit fallback");
    expect(telemetry.root_edge_visits == 0U,
           "one root expansion has no root edge visits");
}

}  // namespace

int main() {
    try {
        test_value_snapshot_and_collapse_regression();
        test_collapsed_search_is_reported();
        test_terminal_composition_and_depth();
        test_pending_and_retry_safe_accounting();
        test_fallback_deltas_and_zero_visit_policy();
    } catch (const std::exception& error) {
        std::cerr << "telemetry test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "telemetry tests passed\n";
    return EXIT_SUCCESS;
}
