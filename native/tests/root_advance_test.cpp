#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "kb_pente/game.h"
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

kb_pente::Action action_at(
    std::uint8_t board_size,
    std::uint8_t row,
    std::uint8_t column) {
    return static_cast<kb_pente::Action>(row * board_size + column);
}

kb_pente::SearchConfig tree_config(
    std::uint32_t simulation_budget,
    std::uint64_t seed = 0U) {
    return kb_pente::SearchConfig(
        1.5F, simulation_budget, 0.25F, 0.03F, seed);
}

std::array<float, kb_pente::kMaxActions> uniform_policy() {
    std::array<float, kb_pente::kMaxActions> policy{};
    policy.fill(1.0F);
    return policy;
}

std::array<float, kb_pente::kMaxActions> policy_with(
    kb_pente::Action action,
    float probability) {
    std::array<float, kb_pente::kMaxActions> policy{};
    policy[action] = probability;
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
            position.stones[action_at(5, row, column)] = pattern[row][column];
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

kb_pente::Action find_child_action(
    const kb_pente::Tree& tree,
    kb_pente::NodeId parent,
    kb_pente::NodeId child) {
    const auto& row = tree.arena().edge_row(parent);
    for (std::size_t index = 0; index < tree.arena().edge_stride(); ++index) {
        const auto action = static_cast<kb_pente::Action>(index);
        if (row.child(action) == child) {
            return action;
        }
    }
    return kb_pente::kInvalidAction;
}

struct EdgeSnapshot final {
    std::array<float, kb_pente::kMaxActions> priors{};
    std::array<float, kb_pente::kMaxActions> value_sums{};
    std::array<std::uint32_t, kb_pente::kMaxActions> visits{};
    std::array<kb_pente::NodeId, kb_pente::kMaxActions> children{};
};

EdgeSnapshot snapshot_edges(
    const kb_pente::Tree& tree,
    kb_pente::NodeId node) {
    EdgeSnapshot snapshot{};
    const auto row = tree.arena().edge_row(node);
    for (std::size_t index = 0; index < kb_pente::kMaxActions; ++index) {
        const auto action = static_cast<kb_pente::Action>(index);
        snapshot.priors[index] = row.prior(action);
        snapshot.value_sums[index] = row.value_sum(action);
        snapshot.visits[index] = row.visit_count(action);
        snapshot.children[index] = row.child(action);
    }
    return snapshot;
}

void expect_same_edges(
    const kb_pente::Tree& tree,
    kb_pente::NodeId node,
    const EdgeSnapshot& expected,
    kb_pente::NodeId expected_descendant,
    kb_pente::NodeId actual_descendant,
    const char* message) {
    const EdgeSnapshot actual = snapshot_edges(tree, node);
    expect(actual.priors == expected.priors, message);
    expect(actual.value_sums == expected.value_sums, message);
    expect(actual.visits == expected.visits, message);
    for (std::size_t index = 0; index < kb_pente::kMaxActions; ++index) {
        if (expected.children[index] == kb_pente::kInvalidNode) {
            expect(actual.children[index] == kb_pente::kInvalidNode, message);
        } else if (expected.children[index] == expected_descendant) {
            expect(actual.children[index] == actual_descendant, message);
        } else {
            expect(actual.children[index] == expected.children[index], message);
        }
    }
}

void test_allocated_subtree_compaction() {
    const kb_pente::Position original_root =
        kb_pente::Position::initial(5);
    kb_pente::Tree tree(
        original_root, kb_pente::Ruleset::Freestyle, tree_config(4U));

    const auto root = tree.root_id();
    std::array<float, kb_pente::kMaxActions> root_policy{};
    root_policy[0] = 0.5F;
    root_policy[1] = 0.5F;
    const auto root_leaf = tree.select_leaf();
    tree.accept_evaluation(
        root_leaf, root_policy.data(), 25U, 0.0F);

    const auto selected_child = tree.select_leaf();
    const auto child_zero = tree.arena().child(root, 0U);
    expect(child_zero == selected_child,
           "first deterministic root child is action zero");
    const auto child_zero_policy = policy_with(2U, 1.0F);
    tree.accept_evaluation(
        selected_child, child_zero_policy.data(), 25U, 0.0F);

    const auto sibling_leaf = tree.select_leaf();
    const auto child_one = tree.arena().child(root, 1U);
    expect(child_one == sibling_leaf,
           "second configured root child is action one");
    const auto sibling_policy = uniform_policy();
    tree.accept_evaluation(
        sibling_leaf, sibling_policy.data(), 25U, 0.0F);

    const auto descendant_leaf = tree.select_leaf();
    const auto descendant = tree.arena().child(child_zero, 2U);
    expect(descendant == descendant_leaf,
           "selected child creates its configured descendant");
    tree.accept_evaluation(
        descendant_leaf, sibling_policy.data(), 25U, 0.0F);

    const auto old_child_zero_meta = tree.arena().node(child_zero);
    const auto old_descendant_meta = tree.arena().node(descendant);
    const auto old_child_zero_edges = snapshot_edges(tree, child_zero);
    const auto old_descendant_edges = snapshot_edges(tree, descendant);
    const std::size_t old_node_count = tree.arena().node_count();
    const std::size_t old_owned_bytes = tree.arena().owned_bytes();
    const auto expected_transition = kb_pente::apply_action(
        original_root, 0U, kb_pente::Ruleset::Freestyle);

    const kb_pente::RootAdvanceStats stats = tree.advance_root(0U);
    expect(stats.reused_subtree, "allocated child subtree is reused");
    expect(stats.previous_node_count == old_node_count,
           "advance stats report the previous node count");
    expect(stats.retained_node_count == 2U,
           "advance stats count the selected subtree exactly");
    expect(stats.discarded_node_count == old_node_count - 2U,
           "advance stats count unreachable siblings");
    expect(stats.previous_owned_bytes == old_owned_bytes,
           "advance stats report pre-compaction owned bytes");
    expect(stats.new_owned_bytes == tree.arena().owned_bytes(),
           "advance stats report post-compaction owned bytes");
    expect(stats.new_owned_bytes < stats.previous_owned_bytes,
           "compaction releases unreachable arena storage");
    expect(tree.root_id() == 0U, "compaction makes the new root ID zero");
    expect(tree.arena().node_count() == 2U,
           "compaction discards the unreachable sibling subtree");
    expect(tree.root_position() == expected_transition.position,
           "advanced root position equals the exact game transition");
    expect(tree.arena().node(0U) == old_child_zero_meta,
           "compaction preserves retained root metadata");
    expect(tree.arena().node(1U) == old_descendant_meta,
           "compaction preserves retained descendant metadata");

    const auto remapped_descendant = find_child_action(tree, 0U, 1U);
    expect(remapped_descendant == 2U,
           "compaction remaps the retained child ID on its original action");
    expect_same_edges(
        tree, 0U, old_child_zero_edges, descendant, 1U,
        "compaction preserves retained root edge data and remaps children");
    expect(snapshot_edges(tree, 1U).priors == old_descendant_edges.priors,
           "compaction preserves descendant priors");
    expect(snapshot_edges(tree, 1U).value_sums == old_descendant_edges.value_sums,
           "compaction preserves descendant value sums");
    expect(snapshot_edges(tree, 1U).visits == old_descendant_edges.visits,
           "compaction preserves descendant visits");
    expect(snapshot_edges(tree, 1U).children == old_descendant_edges.children,
           "compaction preserves descendant child sentinels");
    expect(tree.arena().node_capacity() == tree.arena().node_count(),
           "compaction reserves exactly the retained node count");
    expect(tree.arena().node_capacity() <
               tree.arena().node_count() + tree_config(4U).simulation_budget +
                   kb_pente::Tree::kSearchReserveMargin,
           "compaction does not retain a future-search reserve");
}

void test_unallocated_and_terminal_advancement() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(3U));
    const auto original = tree.root_position();
    const auto stats = tree.advance_root(0U);
    const auto expected = kb_pente::apply_action(
        original, 0U, kb_pente::Ruleset::Freestyle);
    expect(!stats.reused_subtree,
           "unallocated action reports no subtree reuse");
    expect(stats.previous_node_count == 1U &&
               stats.retained_node_count == 1U &&
               stats.discarded_node_count == 1U,
           "unallocated advancement reports one fresh retained root");
    expect(tree.root_id() == 0U && tree.arena().node_count() == 1U,
           "unallocated advancement creates one zero root");
    expect(tree.root_position() == expected.position,
           "unallocated advancement applies the requested transition");
    expect(tree.arena().node(0U).terminal == expected.terminal,
           "unallocated advancement preserves terminal state");
    expect(tree.arena().node_capacity() == 1U,
           "unallocated compaction uses exact fresh capacity");

    kb_pente::SearchSession session(tree);
    expect(tree.arena().node_capacity() >=
               tree.arena().node_count() + tree.config().simulation_budget +
                   kb_pente::Tree::kSearchReserveMargin,
           "a new search reserves retained nodes, budget, and margin");

    kb_pente::Tree terminal_tree(
        make_draw_root(), kb_pente::Ruleset::Freestyle, tree_config(2U));
    const auto terminal_stats = terminal_tree.advance_root(24U);
    expect(!terminal_stats.reused_subtree &&
               terminal_stats.retained_node_count == 1U,
           "terminal unallocated advancement creates one retained root");
    expect(terminal_tree.arena().node(0U).terminal.status ==
               kb_pente::GameStatus::Draw,
           "terminal advancement creates a draw root");
    expect_throws<std::logic_error>(
        [&terminal_tree] { (void)terminal_tree.select_leaf(); },
        "direct selection from a terminal root is rejected");
    expect_throws<std::invalid_argument>(
        [&terminal_tree] { kb_pente::SearchSession session(terminal_tree); },
        "a SearchSession rejects a terminal root");
    expect_throws<std::logic_error>(
        [&terminal_tree] { (void)terminal_tree.advance_root(24U); },
        "advancement from a terminal root is rejected");
}

void test_rejection_and_session_ownership() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(2U));
    const auto original_position = tree.root_position();
    const std::size_t original_nodes = tree.arena().node_count();
    expect_throws<std::invalid_argument>(
        [&tree] { (void)tree.advance_root(kb_pente::kInvalidAction); },
        "out-of-range advancement action is rejected");
    expect(tree.root_position() == original_position &&
               tree.arena().node_count() == original_nodes,
           "out-of-range rejection leaves the tree unchanged");

    kb_pente::Position occupied_position = kb_pente::Position::initial(5);
    occupied_position.stones[0] =
        static_cast<std::int8_t>(kb_pente::Player::One);
    occupied_position.ply = 1U;
    occupied_position.current_player = kb_pente::Player::Two;
    occupied_position.last_action = 0U;
    occupied_position.validate();
    kb_pente::Tree occupied_tree(
        occupied_position, kb_pente::Ruleset::Freestyle, tree_config(2U));
    const auto occupied_before = occupied_tree.root_position();
    expect_throws<std::invalid_argument>(
        [&occupied_tree] { (void)occupied_tree.advance_root(0U); },
        "occupied advancement action is rejected");
    expect(occupied_tree.root_position() == occupied_before,
           "occupied rejection leaves the tree unchanged");

    {
        kb_pente::SearchSession session(tree);
        expect_throws<std::logic_error>(
            [&tree] { (void)tree.advance_root(0U); },
            "live SearchSession blocks root advancement");
        expect(tree.root_position() == original_position &&
                   tree.arena().node_count() == original_nodes,
               "live-session rejection leaves tree content unchanged");
    }

    const auto pending_leaf = tree.select_leaf();
    expect_throws<std::logic_error>(
        [&tree] { (void)tree.advance_root(0U); },
        "pending evaluation blocks root advancement");
    expect(tree.has_pending_evaluation() &&
               tree.pending_leaf() == pending_leaf,
           "pending rejection preserves evaluator state");
    const auto policy = uniform_policy();
    tree.accept_evaluation(pending_leaf, policy.data(), 25U, 0.0F);
    (void)tree.advance_root(0U);
}

void test_rng_continuity_after_advancement() {
    auto make_first_noise = [](kb_pente::Tree& tree) {
        kb_pente::SearchSession session(
            tree, kb_pente::SearchSessionConfig(1.0F, true));
        const auto leaf = session.select_evaluation_leaf();
        const auto policy = uniform_policy();
        session.accept_evaluation(
            *leaf, policy.data(), tree.root_position().action_count(), 0.0F);
        return session.root_search_priors();
    };

    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U, 8128U));
    const auto first_noise = make_first_noise(tree);
    (void)tree.advance_root(0U);
    const auto second_noise = make_first_noise(tree);
    bool shared_legal_difference = false;
    const auto& legal = tree.arena().node(tree.root_id()).legal;
    for (std::size_t index = 0; index < tree.root_position().action_count();
         ++index) {
        if (legal.contains(static_cast<kb_pente::Action>(index)) &&
            first_noise[index] != second_noise[index]) {
            shared_legal_difference = true;
            break;
        }
    }
    expect(shared_legal_difference,
           "advancement preserves the Tree RNG stream for fresh noise");
}

void test_cumulative_telemetry_and_move_safety() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(2U, 42U));
    const auto root = tree.select_leaf();
    std::array<float, kb_pente::kMaxActions> zero_policy{};
    tree.accept_evaluation(root, zero_policy.data(), 25U, 0.0F);
    expect(tree.invalid_policy_fallbacks() == 1U,
           "tree records a policy fallback before advancement");
    const auto selected_child = tree.select_leaf();
    expect(selected_child != root, "telemetry fixture allocates a root child");
    const auto policy = uniform_policy();
    tree.accept_evaluation(selected_child, policy.data(), 25U, 0.0F);
    (void)tree.advance_root(0U);
    expect(tree.invalid_policy_fallbacks() == 1U,
           "advancement preserves cumulative tree telemetry");

    kb_pente::Tree movable(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    kb_pente::Tree moved(std::move(movable));
    expect(moved.root_id() == 0U,
           "session-free root advancement Trees retain move semantics");
}

}  // namespace

int main() {
    try {
        test_allocated_subtree_compaction();
        test_unallocated_and_terminal_advancement();
        test_rejection_and_session_ownership();
        test_rng_continuity_after_advancement();
        test_cumulative_telemetry_and_move_safety();
    } catch (const std::exception& error) {
        std::cerr << "root advance test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "root advance tests passed\n";
    return EXIT_SUCCESS;
}
