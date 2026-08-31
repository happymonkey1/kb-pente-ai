#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "kb_pente/game.h"
#include "kb_pente/mcts/tree.h"

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

kb_pente::Action action_at(
    std::uint8_t board_size,
    std::uint8_t row,
    std::uint8_t column) {
    return static_cast<kb_pente::Action>(row * board_size + column);
}

std::array<float, kb_pente::kMaxActions> policy_with(
    kb_pente::Action action,
    float probability) {
    std::array<float, kb_pente::kMaxActions> policy{};
    policy[action] = probability;
    return policy;
}

std::array<float, kb_pente::kMaxActions> uniform_policy() {
    std::array<float, kb_pente::kMaxActions> policy{};
    policy.fill(1.0F);
    return policy;
}

kb_pente::NodeId evaluate_leaf(
    kb_pente::Tree& tree,
    kb_pente::NodeId leaf,
    const std::array<float, kb_pente::kMaxActions>& policy,
    float value = 0.0F) {
    const std::size_t action_count =
        tree.leaf_position(leaf).action_count();
    tree.accept_evaluation(leaf, policy.data(), action_count, value);
    return leaf;
}

std::size_t root_child_count(const kb_pente::Tree& tree) {
    const auto root = tree.root_id();
    const auto& position = tree.root_position();
    const auto& arena = tree.arena();
    std::size_t count = 0;
    for (std::size_t index = 0; index < position.action_count(); ++index) {
        if (arena.child(root, static_cast<kb_pente::Action>(index)) !=
            kb_pente::kInvalidNode) {
            ++count;
        }
    }
    return count;
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
            position.stones[action_at(5, row, column)] = pattern[row][column];
        }
    }
    position.stones[24] = 0;
    position.ply = 24;
    position.current_player = kb_pente::Player::One;
    position.last_action = kb_pente::kInvalidAction;
    position.validate();
    return position;
}

kb_pente::Position make_full_draw_position() {
    kb_pente::Position position = make_draw_root();
    position.stones[24] = static_cast<std::int8_t>(kb_pente::Player::One);
    position.ply = 25;
    position.current_player = kb_pente::Player::Two;
    position.validate();
    expect(kb_pente::check_terminal(position).status ==
               kb_pente::GameStatus::Draw,
           "full-board fixture is a draw");
    return position;
}

kb_pente::Position make_two_ply_line_root() {
    kb_pente::Position position = kb_pente::Position::initial(7);
    for (std::uint8_t column = 0; column < 4; ++column) {
        position.stones[action_at(7, 0, column)] =
            static_cast<std::int8_t>(kb_pente::Player::Two);
    }
    position.stones[action_at(7, 6, 5)] =
        static_cast<std::int8_t>(kb_pente::Player::One);
    position.stones[action_at(7, 6, 6)] =
        static_cast<std::int8_t>(kb_pente::Player::One);
    position.ply = 6;
    position.current_player = kb_pente::Player::One;
    position.last_action = kb_pente::kInvalidAction;
    position.validate();
    expect(!kb_pente::check_terminal(position).is_terminal(),
           "line fixture starts in progress");
    return position;
}

void test_search_config() {
    const kb_pente::SearchConfig config;
    expect(config.c_puct > 0.0F, "default PUCT constant is positive");

    expect_throws<std::invalid_argument>(
        [] { (void)kb_pente::SearchConfig(0.0F); },
        "zero PUCT is rejected");
    expect_throws<std::invalid_argument>(
        [] { (void)kb_pente::SearchConfig(-1.0F); },
        "negative PUCT is rejected");
    expect_throws<std::invalid_argument>(
        [] {
            (void)kb_pente::SearchConfig(
                std::numeric_limits<float>::infinity());
        },
        "infinite PUCT is rejected");
    expect_throws<std::invalid_argument>(
        [] {
            (void)kb_pente::SearchConfig(
                std::numeric_limits<float>::quiet_NaN());
        },
        "NaN PUCT is rejected");

    kb_pente::SearchConfig mutated;
    mutated.c_puct = 0.0F;
    expect_throws<std::invalid_argument>(
        [&mutated] {
            kb_pente::Tree tree(
                kb_pente::Position::initial(5),
                kb_pente::Ruleset::Freestyle,
                mutated);
        },
        "Tree validates a mutated search config");
}

void test_uniform_exploration_and_unvisited_q() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(9), kb_pente::Ruleset::Freestyle);
    const auto root = tree.root_id();
    const auto policy = uniform_policy();

    const auto root_leaf = tree.select_leaf();
    expect(root_leaf == root, "initial selection returns the root");
    evaluate_leaf(tree, root_leaf, policy);

    const auto first_child_leaf = tree.select_leaf();
    expect(first_child_leaf != root, "first PUCT selection descends to a child");
    const auto first_child_count = root_child_count(tree);
    expect(first_child_count == 1U, "one child is created on first traversal");
    evaluate_leaf(tree, first_child_leaf, policy);

    kb_pente::Action first_action = kb_pente::kInvalidAction;
    for (std::size_t index = 0; index < tree.root_position().action_count();
         ++index) {
        const auto action = static_cast<kb_pente::Action>(index);
        if (tree.arena().child(root, action) != kb_pente::kInvalidNode) {
            first_action = action;
            break;
        }
    }
    expect(first_action != kb_pente::kInvalidAction,
           "the visited root action is discoverable");
    expect(tree.arena().visit_count(root, first_action) == 1U,
           "first root edge has one visit");
    expect(tree.arena().value_sum(root, first_action) == 0.0F,
           "zero leaf value leaves the first Q numerator at zero");

    const auto second_child_leaf = tree.select_leaf();
    expect(root_child_count(tree) == 2U,
           "an unvisited legal edge remains selectable after one visit");
    evaluate_leaf(tree, second_child_leaf, policy);

    for (int simulation = 0; simulation < 14; ++simulation) {
        const auto leaf = tree.select_leaf();
        evaluate_leaf(tree, leaf, policy);
    }
    expect(root_child_count(tree) >= 16U,
           "uniform zero-value search visits multiple root children");

    const auto& root_row = tree.arena().edge_row(root);
    for (std::size_t index = tree.root_position().action_count();
         index < kb_pente::TreeArena::edge_stride(); ++index) {
        const auto action = static_cast<kb_pente::Action>(index);
        expect(root_row.prior(action) == 0.0F,
               "padding actions keep zero priors");
        expect(root_row.visit_count(action) == 0U,
               "padding actions keep zero visits");
    }
}

void test_policy_masking_and_fallback() {
    kb_pente::Position occupied_root =
        kb_pente::apply_action(
            kb_pente::Position::initial(5), 0,
            kb_pente::Ruleset::Freestyle)
            .position;
    kb_pente::Tree tree(occupied_root, kb_pente::Ruleset::Freestyle);
    const auto leaf = tree.select_leaf();
    auto policy = uniform_policy();
    policy[0] = 100.0F;
    evaluate_leaf(tree, leaf, policy);

    const auto root = tree.root_id();
    const auto& row = tree.arena().edge_row(root);
    expect(row.prior(0) == 0.0F, "occupied action receives no prior");
    float legal_sum = 0.0F;
    for (std::size_t index = 1; index < occupied_root.action_count(); ++index) {
        legal_sum += row.prior(static_cast<kb_pente::Action>(index));
    }
    expect(std::fabs(legal_sum - 1.0F) < 1.0e-5F,
           "legal priors are normalized after masking");
    expect(row.prior(1) > 0.0F, "legal action retains normalized prior");
    for (std::size_t index = occupied_root.action_count();
         index < kb_pente::TreeArena::edge_stride(); ++index) {
        expect(row.prior(static_cast<kb_pente::Action>(index)) == 0.0F,
               "padding prior remains zero after expansion");
    }

    kb_pente::Tree fallback(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto fallback_leaf = fallback.select_leaf();
    std::array<float, kb_pente::kMaxActions> zero_policy{};
    evaluate_leaf(fallback, fallback_leaf, zero_policy);
    expect(fallback.invalid_policy_fallbacks() == 1U,
           "zero legal policy increments the fallback counter");
    const auto& fallback_row = fallback.arena().edge_row(fallback.root_id());
    const float expected = 1.0F / 25.0F;
    for (std::size_t index = 0; index < 25; ++index) {
        expect(std::fabs(fallback_row.prior(
                             static_cast<kb_pente::Action>(index)) - expected) <
                   1.0e-6F,
               "zero legal policy uses uniform legal priors");
    }
    expect(fallback_row.prior(25) == 0.0F,
           "fallback does not write outside the active board");
}

void test_policy_validation_is_retryable() {
    auto expect_retryable = [](const std::array<float, kb_pente::kMaxActions>&
                                   policy,
                               std::size_t policy_length,
                               float value,
                               const char* message) {
        kb_pente::Tree tree(
            kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
        const auto leaf = tree.select_leaf();
        const auto before = tree.arena().node(tree.root_id());
        const auto node_count = tree.arena().node_count();
        const auto edge_count = tree.arena().edge_count();
        const auto fallbacks = tree.invalid_policy_fallbacks();
        expect_throws<std::invalid_argument>(
            [&tree, leaf, &policy, policy_length, value] {
                tree.accept_evaluation(
                    leaf, policy.data(), policy_length, value);
            },
            message);
        expect(tree.has_pending_evaluation(),
               "invalid evaluation keeps the leaf pending");
        expect(tree.pending_leaf() == leaf,
               "invalid evaluation preserves the pending leaf ID");
        expect(tree.arena().node(tree.root_id()) == before,
               "invalid evaluation preserves node metadata");
        expect(tree.arena().node_count() == node_count,
               "invalid evaluation does not allocate a node");
        expect(tree.arena().edge_count() == edge_count,
               "invalid evaluation does not change edge storage");
        expect(tree.invalid_policy_fallbacks() == fallbacks,
               "invalid evaluation does not increment fallback telemetry");

        const auto valid = uniform_policy();
        evaluate_leaf(tree, leaf, valid);
        expect(!tree.has_pending_evaluation(),
               "a corrected evaluation clears pending state");
    };

    const auto valid = uniform_policy();
    expect_retryable(valid, 24U, 0.0F,
                     "policy length mismatch is rejected");

    auto negative = valid;
    negative[0] = -1.0F;
    expect_retryable(negative, 25U, 0.0F,
                     "negative policy entry is rejected");

    auto nan = valid;
    nan[0] = std::numeric_limits<float>::quiet_NaN();
    expect_retryable(nan, 25U, 0.0F,
                     "NaN policy entry is rejected");

    auto infinity = valid;
    infinity[0] = std::numeric_limits<float>::infinity();
    expect_retryable(infinity, 25U, 0.0F,
                     "infinite policy entry is rejected");

    expect_retryable(valid, 25U, 1.1F, "out-of-range value is rejected");
    expect_retryable(
        valid, 25U, std::numeric_limits<float>::quiet_NaN(),
        "NaN value is rejected");
    expect_retryable(
        valid, 25U, std::numeric_limits<float>::infinity(),
        "infinite value is rejected");
}

void test_alternating_terminal_backup() {
    const auto root_position = make_two_ply_line_root();
    kb_pente::Tree tree(root_position, kb_pente::Ruleset::Freestyle);
    const auto root = tree.root_id();
    const auto first_action = action_at(7, 6, 4);
    const auto second_action = action_at(7, 0, 4);

    const auto root_leaf = tree.select_leaf();
    evaluate_leaf(tree, root_leaf, policy_with(first_action, 1.0F));

    const auto child_leaf = tree.select_leaf();
    expect(tree.leaf_position(child_leaf).current_player ==
               kb_pente::Player::Two,
           "first child changes the side to move");
    evaluate_leaf(tree, child_leaf, policy_with(second_action, 1.0F));
    const auto child = tree.arena().child(root, first_action);

    const auto terminal_leaf = tree.select_leaf();
    const auto terminal = tree.leaf_terminal(terminal_leaf);
    expect(terminal.is_terminal(), "second action reaches a terminal leaf");
    expect(terminal.winner == kb_pente::Player::Two,
           "terminal fixture is a Player Two win");
    expect(terminal.reason == kb_pente::WinReason::Line,
           "terminal fixture reports a line win");
    expect(tree.pending_path_size() == 2U,
           "terminal selection retains both path edges");
    tree.resolve_terminal(terminal_leaf);

    expect(tree.arena().value_sum(root, first_action) == -1.0F,
           "root edge receives the negative terminal value");
    expect(tree.arena().visit_count(root, first_action) == 2U,
           "root edge includes both traversals");
    expect(tree.arena().value_sum(child, second_action) == 1.0F,
           "child edge receives the positive terminal value");
    expect(tree.arena().visit_count(child, second_action) == 1U,
           "child edge receives one terminal visit");
    expect(tree.arena().node(root).total_visits == 2U,
           "root visit total follows reverse backup");
    expect(tree.arena().node(child).total_visits == 1U,
           "child visit total follows reverse backup");
    expect(!tree.has_pending_evaluation(),
           "terminal resolution clears pending state");
}

void test_terminal_draw_resolution() {
    kb_pente::Tree tree(
        make_draw_root(), kb_pente::Ruleset::Freestyle);
    const auto root = tree.root_id();
    const auto root_leaf = tree.select_leaf();
    evaluate_leaf(tree, root_leaf, policy_with(24, 1.0F));

    const auto terminal_leaf = tree.select_leaf();
    expect(tree.leaf_terminal(terminal_leaf).status ==
               kb_pente::GameStatus::Draw,
           "full-board child is a draw");
    tree.resolve_terminal(terminal_leaf);
    expect(tree.arena().visit_count(root, 24) == 1U,
           "draw terminal contributes one edge visit");
    expect(tree.arena().value_sum(root, 24) == 0.0F,
           "draw terminal contributes zero value");
    expect(tree.arena().node_count() == 2U,
           "draw resolution does not allocate an evaluator node");
}

void test_transition_reuse_and_arena_inspection() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(9), kb_pente::Ruleset::Freestyle);
    const auto root = tree.root_id();
    const auto first_action = static_cast<kb_pente::Action>(0);
    const auto root_leaf = tree.select_leaf();
    evaluate_leaf(tree, root_leaf, policy_with(first_action, 1.0F));

    const auto first_child_leaf = tree.select_leaf();
    const auto first_child = tree.arena().child(root, first_action);
    expect(first_child_leaf == first_child,
           "first traversal returns the allocated integer child ID");
    evaluate_leaf(tree, first_child_leaf, uniform_policy());
    const auto count_after_first = tree.arena().node_count();

    const auto second_leaf = tree.select_leaf();
    expect(tree.arena().child(root, first_action) == first_child,
           "second traversal reuses the existing child ID");
    expect(tree.arena().node_count() == count_after_first + 1U,
           "second traversal allocates only its next descendant");
    evaluate_leaf(tree, second_leaf, uniform_policy());

    const kb_pente::Tree& read_only = tree;
    expect(read_only.root_id() == root, "const tree exposes its root ID");
    expect(read_only.arena().node(root) == tree.arena().node(root),
           "const arena inspection matches mutable storage");
}

void test_pending_state_and_path_reuse() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(9), kb_pente::Ruleset::Freestyle);
    expect(tree.pending_path_capacity() >= kb_pente::Tree::kInitialPathCapacity,
           "tree reserves the documented pending path depth");

    const auto root_leaf = tree.select_leaf();
    expect(tree.has_pending_evaluation(), "selection marks evaluation pending");
    expect_throws<std::logic_error>(
        [&tree] { (void)tree.select_leaf(); },
        "selecting twice is rejected");
    expect_throws<std::logic_error>(
        [&tree] { tree.resolve_terminal(kb_pente::kInvalidNode); },
        "resolving a wrong leaf is rejected");
    expect_throws<std::logic_error>(
        [&tree] {
            const auto policy = uniform_policy();
            tree.accept_evaluation(
                kb_pente::kInvalidNode, policy.data(),
                static_cast<std::size_t>(9U) * 9U, 0.0F);
        },
        "evaluating a wrong pending leaf is rejected");
    expect(tree.has_pending_evaluation(),
           "pending state survives wrong-leaf errors");
    evaluate_leaf(tree, root_leaf, uniform_policy());
    expect(tree.pending_path_size() == 0U,
           "successful evaluation clears the reusable path");
    const auto retained_capacity = tree.pending_path_capacity();

    const auto child_leaf = tree.select_leaf();
    expect(tree.pending_path_size() == 1U,
           "descent records one parent edge");
    evaluate_leaf(tree, child_leaf, uniform_policy());
    expect(tree.pending_path_size() == 0U,
           "second evaluation clears the same path storage");
    expect(tree.pending_path_capacity() == retained_capacity,
           "path capacity is retained between selections");

    kb_pente::Tree nonterminal(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto nonterminal_leaf = nonterminal.select_leaf();
    expect_throws<std::logic_error>(
        [&nonterminal, nonterminal_leaf] {
            nonterminal.resolve_terminal(nonterminal_leaf);
        },
        "nonterminal leaf cannot be terminal-resolved");
    expect(nonterminal.has_pending_evaluation(),
           "nonterminal resolve error keeps evaluation pending");
    evaluate_leaf(nonterminal, nonterminal_leaf, uniform_policy());

    kb_pente::Tree terminal_tree(
        make_draw_root(), kb_pente::Ruleset::Freestyle);
    const auto terminal_root = terminal_tree.select_leaf();
    evaluate_leaf(terminal_tree, terminal_root, policy_with(24, 1.0F));
    const auto terminal_leaf = terminal_tree.select_leaf();
    const auto policy = uniform_policy();
    expect_throws<std::logic_error>(
        [&terminal_tree, terminal_leaf, &policy] {
            terminal_tree.accept_evaluation(
                terminal_leaf, policy.data(), 25U, 0.0F);
        },
        "terminal leaf cannot be evaluator-resolved");
    expect(terminal_tree.has_pending_evaluation(),
           "terminal evaluator error keeps resolution pending");
    terminal_tree.resolve_terminal(terminal_leaf);
    expect_throws<std::logic_error>(
        [&terminal_tree, terminal_leaf] {
            terminal_tree.resolve_terminal(terminal_leaf);
        },
        "terminal leaf cannot be resolved twice");
}

void test_no_legal_action() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Tournament);
    const auto center = action_at(5, 2, 2);
    const auto root_leaf = tree.select_leaf();
    evaluate_leaf(tree, root_leaf, policy_with(center, 1.0F));
    const auto second_leaf = tree.select_leaf();
    evaluate_leaf(tree, second_leaf, policy_with(0, 1.0F));
    expect_throws<std::logic_error>(
        [&tree] { (void)tree.select_leaf(); },
        "nonterminal node without legal actions is rejected");
    expect(!tree.has_pending_evaluation(),
           "failed no-legal selection does not create pending state");

}

void test_terminal_root_rejection() {
    expect_throws<std::invalid_argument>(
        [] {
            kb_pente::Tree tree(
                make_full_draw_position(), kb_pente::Ruleset::Freestyle);
        },
        "terminal roots are rejected");
    expect_throws<std::invalid_argument>(
        [] {
            kb_pente::Tree tree(
                kb_pente::Position::initial(6), kb_pente::Ruleset::Standard);
        },
        "invalid ruleset board configurations are rejected");
}

}  // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    try {
        test_search_config();
        test_uniform_exploration_and_unvisited_q();
        test_policy_masking_and_fallback();
        test_policy_validation_is_retryable();
        test_alternating_terminal_backup();
        test_terminal_draw_resolution();
        test_transition_reuse_and_arena_inspection();
        test_pending_state_and_path_reuse();
        test_no_legal_action();
        test_terminal_root_rejection();
    } catch (const TestFailure& failure) {
        std::cerr << "FAIL: " << failure.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "native tree search tests passed\n";
    return EXIT_SUCCESS;
}
