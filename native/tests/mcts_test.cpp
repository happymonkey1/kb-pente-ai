#include <array>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "kb_pente/mcts/tree_arena.h"

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

// Fixture arguments mirror the metadata fields that each test varies.
kb_pente::NodeMeta make_meta(
    std::uint8_t board_size,  // NOLINT(bugprone-easily-swappable-parameters)
    std::uint32_t total_visits,
    bool expanded) {
    kb_pente::NodeMeta meta{};
    meta.position = kb_pente::Position::initial(board_size);
    meta.terminal = kb_pente::TerminalResult::in_progress();
    meta.total_visits = total_visits;
    meta.legal.set(static_cast<kb_pente::Action>(board_size));
    meta.expanded = expanded;
    return meta;
}

void expect_neutral_edges(
    const kb_pente::TreeArena& arena,
    kb_pente::NodeId node) {
    for (std::size_t action_index = 0;
         action_index < kb_pente::TreeArena::edge_stride(); ++action_index) {
        const auto action = static_cast<kb_pente::Action>(action_index);
        expect(arena.prior(node, action) == 0.0F,
               "new edge prior is neutral");
        expect(arena.value_sum(node, action) == 0.0F,
               "new edge value sum is neutral");
        expect(arena.visit_count(node, action) == 0U,
               "new edge visit count is neutral");
        expect(arena.child(node, action) == kb_pente::kInvalidNode,
               "new edge child is invalid");
    }
}

void test_allocate_and_metadata() {
    kb_pente::TreeArena arena;
    auto first = make_meta(5, 7, true);
    first.terminal = kb_pente::TerminalResult::win(
        kb_pente::Player::One, kb_pente::WinReason::Capture);
    first.legal.set(360);
    auto second = make_meta(19, 11, false);
    second.terminal = kb_pente::TerminalResult::draw();
    second.legal.set(0);

    const auto first_id = arena.allocate(first);
    const auto second_id = arena.allocate(second);
    expect(first_id == 0U, "first node receives ID zero");
    expect(second_id == 1U, "second node receives the next ID");
    expect(arena.node_count() == 2U, "multiple nodes are retained");
    expect(arena.edge_count() == 2U * kb_pente::TreeArena::edge_stride(),
           "edge count uses one full stride per node");
    expect(arena.node(first_id) == first, "first node metadata is retained");
    expect(arena.node(second_id) == second,
           "second node metadata is retained");
    expect_neutral_edges(arena, first_id);
    expect_neutral_edges(arena, second_id);
}

void test_stride_and_isolation() {
    kb_pente::TreeArena arena;
    const auto small_id = arena.allocate(make_meta(5, 1, false));
    const auto large_id = arena.allocate(make_meta(19, 2, true));
    const auto last_action = static_cast<kb_pente::Action>(
        kb_pente::TreeArena::edge_stride() - 1U);

    expect(arena.edge_index(small_id, 0) == 0U,
           "first node starts at edge zero");
    expect(arena.edge_index(small_id, last_action) == last_action,
           "first node has the complete action stride");
    expect(arena.edge_index(large_id, 0) ==
               kb_pente::TreeArena::edge_stride(),
           "second node starts after one full action stride");
    expect(arena.edge_index(large_id, last_action) ==
               2U * kb_pente::TreeArena::edge_stride() - 1U,
           "second node ends at its stride boundary");

    arena.prior(small_id, last_action) = 1.25F;
    arena.value_sum(small_id, last_action) = -2.5F;
    arena.visit_count(small_id, last_action) = 3U;
    arena.child(small_id, last_action) = large_id;
    arena.prior(large_id, 0) = 4.0F;
    arena.value_sum(large_id, 0) = 5.0F;
    arena.visit_count(large_id, 0) = 6U;
    arena.child(large_id, 0) = small_id;

    expect(arena.prior(small_id, last_action) == 1.25F,
           "small-node prior mutation persists");
    expect(arena.value_sum(small_id, last_action) == -2.5F,
           "small-node value mutation persists");
    expect(arena.visit_count(small_id, last_action) == 3U,
           "small-node visit mutation persists");
    expect(arena.child(small_id, last_action) == large_id,
           "small-node child mutation persists");
    expect(arena.prior(large_id, 0) == 4.0F,
           "large-node prior mutation persists");
    expect(arena.value_sum(large_id, 0) == 5.0F,
           "large-node value mutation persists");
    expect(arena.visit_count(large_id, 0) == 6U,
           "large-node visit mutation persists");
    expect(arena.child(large_id, 0) == small_id,
           "large-node child mutation persists");
    expect(arena.prior(small_id, 0) == 0.0F,
           "small-node action zero is isolated");
    expect(arena.prior(large_id, last_action) == 0.0F,
           "large-node final action is isolated");
}

void test_edge_row_views() {
    kb_pente::TreeArena arena;
    const auto first_id = arena.allocate(make_meta(5, 0, false));
    const auto second_id = arena.allocate(make_meta(19, 0, false));
    auto first_row = arena.edge_row(first_id);
    auto second_row = arena.edge_row(second_id);
    const auto last_action = static_cast<kb_pente::Action>(
        kb_pente::TreeArena::edge_stride() - 1U);

    expect(kb_pente::EdgeRowView::size() ==
               kb_pente::TreeArena::edge_stride(),
           "mutable edge row has the fixed action extent");
    expect(kb_pente::ConstEdgeRowView::size() ==
               kb_pente::TreeArena::edge_stride(),
           "const edge row has the fixed action extent");
    expect(first_row.child(0) == kb_pente::kInvalidNode,
           "mutable row starts with a neutral child");
    expect(second_row.prior(last_action) == 0.0F,
           "second row starts with a neutral prior");

    first_row.prior(last_action) = 1.5F;
    first_row.value_sum(last_action) = -3.0F;
    first_row.visit_count(last_action) = 8U;
    first_row.child(last_action) = second_id;
    second_row.prior(0) = 2.5F;
    second_row.value_sum(0) = 4.0F;
    second_row.visit_count(0) = 9U;
    second_row.child(0) = first_id;

    expect(first_row.prior(last_action) == 1.5F,
           "mutable row prior mutation persists");
    expect(first_row.value_sum(last_action) == -3.0F,
           "mutable row value mutation persists");
    expect(first_row.visit_count(last_action) == 8U,
           "mutable row visit mutation persists");
    expect(first_row.child(last_action) == second_id,
           "mutable row child mutation persists");
    expect(second_row.prior(0) == 2.5F,
           "second row prior mutation persists");
    expect(first_row.prior(0) == 0.0F,
           "mutable rows do not alias at action zero");
    expect(second_row.prior(last_action) == 0.0F,
           "mutable rows do not alias at the final action");

    const kb_pente::TreeArena& read_only = arena;
    const auto first_const_row = read_only.edge_row(first_id);
    expect(first_const_row.prior(last_action) == 1.5F,
           "const row reads mutable row values");
    expect(first_const_row.child(last_action) == second_id,
           "const row reads child IDs");

    expect_throws<std::out_of_range>(
        [&first_row] { (void)first_row.prior(kb_pente::kInvalidAction); },
        "mutable row rejects an invalid action");
    expect_throws<std::out_of_range>(
        [&first_const_row] {
            (void)first_const_row.visit_count(kb_pente::kInvalidAction);
        },
        "const row rejects an invalid action");
    expect_throws<std::out_of_range>(
        [&arena] { (void)arena.edge_row(kb_pente::kInvalidNode); },
        "edge row rejects an invalid node");
}

void test_checked_boundaries() {
    kb_pente::TreeArena arena;
    const auto node = arena.allocate(make_meta(5, 0, false));
    const auto invalid_action = kb_pente::kInvalidAction;
    const auto after_stride = static_cast<kb_pente::Action>(
        kb_pente::TreeArena::edge_stride());

    expect_throws<std::out_of_range>(
        [&arena] { (void)arena.node(kb_pente::kInvalidNode); },
        "invalid node sentinel is rejected");
    expect_throws<std::out_of_range>(
        [&arena] { (void)arena.node(1U); },
        "unallocated node ID is rejected");
    expect_throws<std::out_of_range>(
        [&arena] { (void)arena.edge_index(kb_pente::kInvalidNode, 0); },
        "invalid node is rejected by edge indexing");
    expect_throws<std::out_of_range>(
        [&arena, node] { (void)arena.edge_index(node, kb_pente::kInvalidAction); },
        "invalid action sentinel is rejected by edge indexing");
    expect_throws<std::out_of_range>(
        [&arena, node, after_stride] { (void)arena.prior(node, after_stride); },
        "action after the fixed stride is rejected");
    expect_throws<std::out_of_range>(
        [&arena, node, invalid_action] { (void)arena.value_sum(node, invalid_action); },
        "invalid action is rejected by value access");
    expect_throws<std::out_of_range>(
        [&arena, node, invalid_action] { (void)arena.visit_count(node, invalid_action); },
        "invalid action is rejected by visit access");
    expect_throws<std::out_of_range>(
        [&arena, node, invalid_action] { (void)arena.child(node, invalid_action); },
        "invalid action is rejected by child access");
}

void test_reserve_and_clear() {
    kb_pente::TreeArena arena;
    auto expected = make_meta(9, 17, true);
    const auto node = arena.allocate(expected);
    arena.prior(node, 17) = 0.75F;
    arena.value_sum(node, 17) = 2.5F;
    arena.visit_count(node, 17) = 4U;
    arena.child(node, 17) = node;

    arena.reserve(8);
    expect(arena.node_count() == 1U, "reserve preserves node count");
    expect(arena.node(node) == expected,
           "reserve preserves complete node metadata");
    expect(arena.prior(node, 17) == 0.75F,
           "reserve preserves priors");
    expect(arena.value_sum(node, 17) == 2.5F,
           "reserve preserves value sums");
    expect(arena.visit_count(node, 17) == 4U,
           "reserve preserves visit counts");
    expect(arena.child(node, 17) == node,
           "reserve preserves child IDs");
    expect(arena.node_capacity() >= 8U,
           "reserve grows node capacity");
    expect(arena.edge_capacity() >= 8U * kb_pente::TreeArena::edge_stride(),
           "reserve grows every edge array");

    const auto retained_node_capacity = arena.node_capacity();
    const auto retained_edge_capacity = arena.edge_capacity();
    arena.clear();
    expect(arena.node_count() == 0U, "clear removes all nodes");
    expect(arena.edge_count() == 0U, "clear removes all edges");
    expect(arena.node_capacity() >= retained_node_capacity,
           "clear retains reusable node capacity");
    expect(arena.edge_capacity() >= retained_edge_capacity,
           "clear retains reusable edge capacity");

    const auto reused_node = arena.allocate(expected);
    expect(reused_node == 0U, "cleared arena reuses node ID zero");
    expect(arena.node(reused_node) == expected,
           "cleared arena accepts new metadata");
    expect_neutral_edges(arena, reused_node);

    arena.reset();
    expect(arena.node_count() == 0U, "reset restores empty state");
    expect(arena.edge_count() == 0U, "reset clears edge count");
}

void test_accounting_and_capacity_guards() {
    kb_pente::TreeArena arena(3);
    expect(arena.node_count() == 0U, "capacity constructor starts empty");
    expect(arena.edge_count() == 0U, "capacity constructor has no edges");
    expect(arena.node_capacity() >= 3U,
           "capacity constructor reserves requested nodes");
    expect(arena.edge_capacity() >= 3U * kb_pente::TreeArena::edge_stride(),
           "capacity constructor reserves requested edges");

    const auto estimated = kb_pente::TreeArena::estimated_bytes_for_nodes(3);
    expect(arena.owned_bytes() >= estimated,
           "owned bytes cover the logical requested storage");
    expect(kb_pente::TreeArena::estimated_bytes_for_nodes(0) == 0U,
           "zero-node estimate is empty");
    expect(
        kb_pente::TreeArena::edge_bytes_per_node() ==
            kb_pente::TreeArena::edge_stride() *
                (sizeof(float) * 2U + sizeof(std::uint32_t) +
                 sizeof(kb_pente::NodeId)),
        "edge byte estimate includes all four arrays");

    const auto before = arena.owned_bytes();
    arena.reserve(12);
    expect(arena.owned_bytes() >= before,
           "growing reserve does not reduce owned bytes");
    expect(arena.owned_bytes() >=
               kb_pente::TreeArena::estimated_bytes_for_nodes(12),
           "grown owned bytes cover requested storage");

    expect_throws<std::length_error>(
        [] {
            kb_pente::TreeArena too_large;
            too_large.reserve(
                std::numeric_limits<std::size_t>::max());
        },
        "reserve rejects an impossible node count");
    expect_throws<std::length_error>(
        [] {
            (void)kb_pente::TreeArena::estimated_bytes_for_nodes(
                std::numeric_limits<std::size_t>::max());
        },
        "byte estimate rejects an impossible node count");
}

}  // namespace

// Boundary tests intentionally execute APIs that report invalid input by
// throwing; the surrounding test harness catches those exceptions.
// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    static_assert(std::is_same_v<kb_pente::NodeId, std::uint32_t>);
    static_assert(sizeof(kb_pente::NodeId) == sizeof(std::uint32_t));
    static_assert(sizeof(float) == 4);
    static_assert(std::is_standard_layout_v<kb_pente::NodeMeta>);

    try {
        test_allocate_and_metadata();
        test_stride_and_isolation();
        test_edge_row_views();
        test_checked_boundaries();
        test_reserve_and_clear();
        test_accounting_and_capacity_guards();
    } catch (const TestFailure& failure) {
        std::cerr << "FAIL: " << failure.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "native tree arena tests passed\n";
    return EXIT_SUCCESS;
}
