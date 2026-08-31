#include "kb_pente/mcts/tree_arena.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kb_pente {

namespace {

[[nodiscard]] std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* description) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(description);
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* description) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(description);
    }
    return left * right;
}

void validate_node_capacity(std::size_t node_count) {
    if (node_count > static_cast<std::size_t>(kInvalidNode)) {
        throw std::length_error("TreeArena node count exceeds NodeId range");
    }
}

[[nodiscard]] std::size_t checked_storage_bytes(
    std::size_t element_count,
    std::size_t element_size,
    const char* description) {
    return checked_multiply(element_count, element_size, description);
}

}  // namespace

TreeArena::TreeArena(std::size_t requested_node_capacity) {
    reserve(requested_node_capacity);
}

NodeId TreeArena::allocate(NodeMeta meta) {
    if (nodes_.size() >= static_cast<std::size_t>(kInvalidNode)) {
        throw std::length_error("TreeArena has exhausted the NodeId range");
    }

    const std::size_t next_node_count = nodes_.size() + 1U;
    reserve(next_node_count);

    const auto node = static_cast<NodeId>(nodes_.size());
    const std::size_t old_edge_count = priors_.size();
    try {
        nodes_.push_back(std::move(meta));
        priors_.insert(priors_.end(), edge_stride(), 0.0F);
        value_sums_.insert(value_sums_.end(), edge_stride(), 0.0F);
        visits_.insert(visits_.end(), edge_stride(), 0U);
        children_.insert(children_.end(), edge_stride(), kInvalidNode);
    } catch (...) {
        nodes_.resize(node);
        priors_.resize(old_edge_count);
        value_sums_.resize(old_edge_count);
        visits_.resize(old_edge_count);
        children_.resize(old_edge_count);
        throw;
    }
    return node;
}

NodeId TreeArena::allocate(
    const Position& position,
    const TerminalResult& terminal) {
    NodeMeta meta{};
    meta.position = position;
    meta.terminal = terminal;
    return allocate(std::move(meta));
}

void TreeArena::reserve(std::size_t requested_node_capacity) {
    validate_node_capacity(requested_node_capacity);
    const std::size_t requested_edge_capacity =
        checked_edge_count(requested_node_capacity);

    if (requested_node_capacity > nodes_.max_size() ||
        requested_edge_capacity > priors_.max_size() ||
        requested_edge_capacity > value_sums_.max_size() ||
        requested_edge_capacity > visits_.max_size() ||
        requested_edge_capacity > children_.max_size()) {
        throw std::length_error("TreeArena capacity exceeds vector limits");
    }

    nodes_.reserve(requested_node_capacity);
    priors_.reserve(requested_edge_capacity);
    value_sums_.reserve(requested_edge_capacity);
    visits_.reserve(requested_edge_capacity);
    children_.reserve(requested_edge_capacity);
}

void TreeArena::clear() noexcept {
    nodes_.clear();
    priors_.clear();
    value_sums_.clear();
    visits_.clear();
    children_.clear();
}

std::size_t TreeArena::edge_capacity() const noexcept {
    return std::min({
        priors_.capacity(),
        value_sums_.capacity(),
        visits_.capacity(),
        children_.capacity(),
    });
}

std::size_t TreeArena::owned_bytes() const {
    std::size_t total = 0;
    total = checked_add(
        total,
        checked_storage_bytes(
            nodes_.capacity(), sizeof(NodeMeta),
            "TreeArena node byte accounting overflow"),
        "TreeArena byte accounting overflow");
    total = checked_add(
        total,
        checked_storage_bytes(
            priors_.capacity(), sizeof(float),
            "TreeArena prior byte accounting overflow"),
        "TreeArena byte accounting overflow");
    total = checked_add(
        total,
        checked_storage_bytes(
            value_sums_.capacity(), sizeof(float),
            "TreeArena value byte accounting overflow"),
        "TreeArena byte accounting overflow");
    total = checked_add(
        total,
        checked_storage_bytes(
            visits_.capacity(), sizeof(std::uint32_t),
            "TreeArena visit byte accounting overflow"),
        "TreeArena byte accounting overflow");
    return checked_add(
        total,
        checked_storage_bytes(
            children_.capacity(), sizeof(NodeId),
            "TreeArena child byte accounting overflow"),
        "TreeArena byte accounting overflow");
}

std::size_t TreeArena::estimated_bytes_for_nodes(std::size_t node_count) {
    validate_node_capacity(node_count);
    const std::size_t edge_count = checked_edge_count(node_count);
    const std::size_t node_bytes = checked_storage_bytes(
        node_count, sizeof(NodeMeta), "TreeArena node byte estimate overflow");
    const std::size_t edge_bytes = checked_storage_bytes(
        edge_count,
        sizeof(float) * 2U + sizeof(std::uint32_t) + sizeof(NodeId),
        "TreeArena edge byte estimate overflow");
    return checked_add(
        node_bytes, edge_bytes, "TreeArena byte estimate overflow");
}

std::size_t TreeArena::edge_index(
    NodeId node,  // NOLINT(bugprone-easily-swappable-parameters)
    Action action) const {
    const std::size_t node_index = checked_node_index(node);
    if (action >= kMaxActions) {
        throw std::out_of_range("TreeArena action index is out of range");
    }

    const std::size_t base = checked_multiply(
        node_index, edge_stride(), "TreeArena edge index overflow");
    return checked_add(
        base, static_cast<std::size_t>(action),
        "TreeArena edge index overflow");
}

NodeMeta& TreeArena::node(NodeId node_id) {
    return nodes_.at(checked_node_index(node_id));
}

const NodeMeta& TreeArena::node(NodeId node_id) const {
    return nodes_.at(checked_node_index(node_id));
}

EdgeRowView TreeArena::edge_row(NodeId node_id) {
    const std::size_t node_index = checked_node_index(node_id);
    const std::size_t base = checked_multiply(
        node_index, edge_stride(), "TreeArena edge index overflow");
    return EdgeRowView(
        priors_.data() + base,
        value_sums_.data() + base,
        visits_.data() + base,
        children_.data() + base);
}

ConstEdgeRowView TreeArena::edge_row(NodeId node_id) const {
    const std::size_t node_index = checked_node_index(node_id);
    const std::size_t base = checked_multiply(
        node_index, edge_stride(), "TreeArena edge index overflow");
    return ConstEdgeRowView(
        priors_.data() + base,
        value_sums_.data() + base,
        visits_.data() + base,
        children_.data() + base);
}

float& TreeArena::prior(NodeId node_id, Action action) {
    return priors_.at(edge_index(node_id, action));
}

const float& TreeArena::prior(NodeId node_id, Action action) const {
    return priors_.at(edge_index(node_id, action));
}

float& TreeArena::value_sum(NodeId node_id, Action action) {
    return value_sums_.at(edge_index(node_id, action));
}

const float& TreeArena::value_sum(NodeId node_id, Action action) const {
    return value_sums_.at(edge_index(node_id, action));
}

std::uint32_t& TreeArena::visit_count(NodeId node_id, Action action) {
    return visits_.at(edge_index(node_id, action));
}

const std::uint32_t& TreeArena::visit_count(
    NodeId node_id,
    Action action) const {
    return visits_.at(edge_index(node_id, action));
}

NodeId& TreeArena::child(NodeId node_id, Action action) {
    return children_.at(edge_index(node_id, action));
}

const NodeId& TreeArena::child(NodeId node_id, Action action) const {
    return children_.at(edge_index(node_id, action));
}

std::size_t TreeArena::checked_node_index(NodeId node) const {
    if (node == kInvalidNode) {
        throw std::out_of_range("TreeArena node ID is invalid");
    }
    const std::size_t index = static_cast<std::size_t>(node);
    if (index >= nodes_.size()) {
        throw std::out_of_range("TreeArena node ID is out of range");
    }
    return index;
}

std::size_t TreeArena::checked_edge_count(std::size_t node_count) {
    validate_node_capacity(node_count);
    return checked_multiply(
        node_count, edge_stride(), "TreeArena edge count overflow");
}

}  // namespace kb_pente
