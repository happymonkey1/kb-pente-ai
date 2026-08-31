#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kb_pente/action_mask.h"
#include "kb_pente/position.h"
#include "kb_pente/terminal_result.h"

namespace kb_pente {

// Integer indices keep child references compact and relocation-safe.
using NodeId = std::uint32_t;

inline constexpr NodeId kInvalidNode =
    std::numeric_limits<NodeId>::max();

// Metadata is kept together because it is read once for each selected node.
struct NodeMeta final {
    Position position{};
    TerminalResult terminal{};
    std::uint32_t total_visits = 0;
    ActionMask legal{};
    bool expanded = false;
};

[[nodiscard]] inline bool operator==(
    const NodeMeta& left,
    const NodeMeta& right) noexcept {
    return left.position == right.position && left.terminal == right.terminal &&
           left.total_visits == right.total_visits &&
           left.legal == right.legal && left.expanded == right.expanded;
}

[[nodiscard]] inline bool operator!=(
    const NodeMeta& left,
    const NodeMeta& right) noexcept {
    return !(left == right);
}

// Provides checked access to one node's fixed-width edge row after the node
// boundary has been validated by TreeArena. The view is invalidated by clear,
// reset, and operations that can reallocate edge storage.
class EdgeRowView final {
public:
    [[nodiscard]] static constexpr std::size_t size() noexcept {
        return static_cast<std::size_t>(kMaxActions);
    }

    [[nodiscard]] float& prior(Action action) const {
        return priors_[checked_action(action)];
    }

    [[nodiscard]] float& value_sum(Action action) const {
        return value_sums_[checked_action(action)];
    }

    [[nodiscard]] std::uint32_t& visit_count(Action action) const {
        return visits_[checked_action(action)];
    }

    [[nodiscard]] NodeId& child(Action action) const {
        return children_[checked_action(action)];
    }

private:
    friend class TreeArena;

    EdgeRowView(
        float* priors,
        float* value_sums,
        std::uint32_t* visits,
        NodeId* children)
        : priors_(priors),
          value_sums_(value_sums),
          visits_(visits),
          children_(children) {}

    [[nodiscard]] static std::size_t checked_action(Action action) {
        if (action >= kMaxActions) {
            throw std::out_of_range("TreeArena action index is out of range");
        }
        return static_cast<std::size_t>(action);
    }

    float* priors_;
    float* value_sums_;
    std::uint32_t* visits_;
    NodeId* children_;
};

// Read-only counterpart to EdgeRowView for selection and inspection paths.
class ConstEdgeRowView final {
public:
    [[nodiscard]] static constexpr std::size_t size() noexcept {
        return static_cast<std::size_t>(kMaxActions);
    }

    [[nodiscard]] const float& prior(Action action) const {
        return priors_[checked_action(action)];
    }

    [[nodiscard]] const float& value_sum(Action action) const {
        return value_sums_[checked_action(action)];
    }

    [[nodiscard]] const std::uint32_t& visit_count(Action action) const {
        return visits_[checked_action(action)];
    }

    [[nodiscard]] const NodeId& child(Action action) const {
        return children_[checked_action(action)];
    }

private:
    friend class TreeArena;

    ConstEdgeRowView(
        const float* priors,
        const float* value_sums,
        const std::uint32_t* visits,
        const NodeId* children)
        : priors_(priors),
          value_sums_(value_sums),
          visits_(visits),
          children_(children) {}

    [[nodiscard]] static std::size_t checked_action(Action action) {
        if (action >= kMaxActions) {
            throw std::out_of_range("TreeArena action index is out of range");
        }
        return static_cast<std::size_t>(action);
    }

    const float* priors_;
    const float* value_sums_;
    const std::uint32_t* visits_;
    const NodeId* children_;
};

// Owns integer-indexed tree nodes and dense per-action edge arrays. The arena
// has no game or search policy; later MCTS code supplies those behaviors.
class TreeArena final {
public:
    TreeArena() = default;
    explicit TreeArena(std::size_t requested_node_capacity);

    [[nodiscard]] NodeId allocate(NodeMeta meta);
    [[nodiscard]] NodeId allocate(
        const Position& position,
        const TerminalResult& terminal);

    // Reserve capacity for at least this many nodes without changing data.
    void reserve(std::size_t requested_node_capacity);
    void clear() noexcept;
    void reset() noexcept { clear(); }

    [[nodiscard]] std::size_t node_count() const noexcept {
        return nodes_.size();
    }

    [[nodiscard]] std::size_t node_capacity() const noexcept {
        return nodes_.capacity();
    }

    [[nodiscard]] std::size_t edge_count() const noexcept {
        return priors_.size();
    }

    // This is the complete edge capacity common to all four edge arrays.
    [[nodiscard]] std::size_t edge_capacity() const noexcept;

    // Bytes held by vector allocations, including unused reserved capacity.
    [[nodiscard]] std::size_t owned_bytes() const;

    // Logical bytes required for a requested node count.
    [[nodiscard]] static std::size_t estimated_bytes_for_nodes(
        std::size_t node_count);

    [[nodiscard]] static constexpr std::size_t edge_stride() noexcept {
        return static_cast<std::size_t>(kMaxActions);
    }

    [[nodiscard]] static constexpr std::size_t edge_bytes_per_node() noexcept {
        return edge_stride() *
               (sizeof(float) * 2U + sizeof(std::uint32_t) + sizeof(NodeId));
    }

    // Return the dense edge index and reject both public dimensions.
    [[nodiscard]] std::size_t edge_index(
        NodeId node,  // NOLINT(bugprone-easily-swappable-parameters)
        Action action) const;

    [[nodiscard]] NodeMeta& node(NodeId node);
    [[nodiscard]] const NodeMeta& node(NodeId node) const;

    [[nodiscard]] EdgeRowView edge_row(NodeId node);
    [[nodiscard]] ConstEdgeRowView edge_row(NodeId node) const;

    [[nodiscard]] float& prior(NodeId node, Action action);
    [[nodiscard]] const float& prior(NodeId node, Action action) const;

    [[nodiscard]] float& value_sum(NodeId node, Action action);
    [[nodiscard]] const float& value_sum(NodeId node, Action action) const;

    [[nodiscard]] std::uint32_t& visit_count(NodeId node, Action action);
    [[nodiscard]] const std::uint32_t& visit_count(
        NodeId node,
        Action action) const;

    [[nodiscard]] NodeId& child(NodeId node, Action action);
    [[nodiscard]] const NodeId& child(NodeId node, Action action) const;

private:
    [[nodiscard]] std::size_t checked_node_index(NodeId node) const;
    [[nodiscard]] static std::size_t checked_edge_count(
        std::size_t node_count);

    std::vector<NodeMeta> nodes_;
    std::vector<float> priors_;
    std::vector<float> value_sums_;
    std::vector<std::uint32_t> visits_;
    std::vector<NodeId> children_;
};

}  // namespace kb_pente
