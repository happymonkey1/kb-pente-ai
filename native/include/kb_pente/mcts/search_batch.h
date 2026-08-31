#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "kb_pente/mcts/search_session.h"
#include "kb_pente/parallel/worker_pool.h"

namespace kb_pente {

// A generation identifies exactly one selection wave and prevents a result
// from being applied to a later pending wave.
using BatchToken = std::uint64_t;

inline constexpr BatchToken kInvalidBatchToken = 0U;
using SlotId = std::size_t;
inline constexpr SlotId kInvalidSlot = static_cast<SlotId>(-1);

// LeafRequest is an immutable view of one evaluator request. Its position
// pointer remains valid until the next mutating SearchBatch operation.
struct LeafRequest final {
    SlotId slot = kInvalidSlot;
    NodeId leaf = kInvalidNode;
    const Position* position = nullptr;

    [[nodiscard]] SlotId slot_id() const noexcept { return slot; }
    [[nodiscard]] NodeId leaf_id() const noexcept { return leaf; }

    [[nodiscard]] const Position& leaf_position() const {
        if (position == nullptr) {
            throw std::logic_error("SearchBatch leaf request has no position");
        }
        return *position;
    }
};

// Selection is a non-owning, slot-ordered view returned by SearchBatch::select.
// It is invalidated by the next mutating SearchBatch operation.
struct Selection final {
    BatchToken token = kInvalidBatchToken;
    const LeafRequest* data = nullptr;
    std::size_t count = 0U;

    [[nodiscard]] const LeafRequest* begin() const noexcept {
        return count == 0U ? nullptr : data;
    }
    [[nodiscard]] const LeafRequest* end() const noexcept {
        return count == 0U || data == nullptr ? nullptr : data + count;
    }
    [[nodiscard]] const LeafRequest& operator[](std::size_t index) const {
        if (index >= count || data == nullptr) {
            throw std::out_of_range("SearchBatch selection index is out of range");
        }
        return data[index];
    }
    [[nodiscard]] std::size_t size() const noexcept { return count; }
    [[nodiscard]] bool empty() const noexcept { return count == 0U; }
};

// SearchBatch coordinates independent single-tree sessions through
// synchronous evaluator-selection waves. Slot admission is stable, while
// later slices may add root replacement and removal.
class SearchBatch final {
public:
    SearchBatch(
        SearchConfig config,
        std::size_t max_active_games,
        std::size_t worker_threads);

    ~SearchBatch() = default;

    SearchBatch(const SearchBatch&) = delete;
    SearchBatch& operator=(const SearchBatch&) = delete;
    SearchBatch(SearchBatch&&) = delete;
    SearchBatch& operator=(SearchBatch&&) = delete;

    // Admit a nonterminal root into the lowest available slot.
    [[nodiscard]] SlotId add(
        Position root,
        Ruleset ruleset = kDefaultRuleset,
        SearchSessionConfig session_config = SearchSessionConfig{});

    // Select at most one nonterminal evaluator leaf per incomplete slot.
    // Terminal leaves are resolved internally by each SearchSession.
    [[nodiscard]] Selection select();

    // Apply a complete contiguous [request_count, kMaxActions] policy matrix
    // and [request_count] value vector for the latest selection token.
    void backup(
        BatchToken token,
        const float* policies,
        std::size_t policy_rows,
        std::size_t policy_stride,
        const float* values,
        std::size_t value_count);

    // Convenience overload for the fixed native action stride.
    void backup(
        BatchToken token,
        const float* policies,
        std::size_t policy_rows,
        const float* values,
        std::size_t value_count) {
        backup(
            token,
            policies,
            policy_rows,
            kMaxActions,
            values,
            value_count);
    }

    [[nodiscard]] bool complete() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return slots_.size();
    }
    [[nodiscard]] std::size_t active_count() const noexcept {
        return active_count_;
    }
    [[nodiscard]] std::size_t pending_request_count() const noexcept {
        return requests_.size();
    }
    [[nodiscard]] bool has_pending() const noexcept {
        return pending_token_ != kInvalidBatchToken;
    }
    [[nodiscard]] BatchToken pending_token() const noexcept {
        return pending_token_;
    }
    [[nodiscard]] BatchToken last_token() const noexcept {
        return last_token_;
    }
    [[nodiscard]] std::size_t thread_count() const noexcept {
        return worker_pool_.thread_count();
    }
    [[nodiscard]] const SearchConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] bool slot_active(SlotId slot) const;
    [[nodiscard]] bool slot_complete(SlotId slot) const;
    // Return the deterministic per-admission seed assigned to one slot.
    [[nodiscard]] std::uint64_t slot_seed(SlotId slot) const;
    [[nodiscard]] const Position& root_position(SlotId slot) const;
    // Return the current root status, including terminal roots after advance.
    [[nodiscard]] TerminalResult root_terminal(SlotId slot) const;
    // Return the completed session's legal root policy by value.
    [[nodiscard]] std::array<float, kMaxActions> root_policy(SlotId slot);
    [[nodiscard]] SearchTelemetry slot_telemetry(SlotId slot) const;

    // Advance a completed session onto one legal root action. The retained
    // subtree is compacted by Tree, and a nonterminal root receives a fresh
    // session with the supplied configuration.
    [[nodiscard]] RootAdvanceStats advance_root(
        SlotId slot,
        Action action,
        SearchSessionConfig session_config = SearchSessionConfig{});

    // Remove a completed active slot. Slot IDs become available to add().
    void remove(SlotId slot);

    // Replace a completed active slot without changing its slot ID. The new
    // tree and session are constructed before the existing slot is touched.
    void replace_root(
        SlotId slot,
        Position root,
        Ruleset ruleset = kDefaultRuleset,
        SearchSessionConfig session_config = SearchSessionConfig{});

private:
    struct Slot final {
        std::unique_ptr<Tree> tree;
        std::unique_ptr<SearchSession> session;
    };

    struct SelectionScratch final {
        bool has_request = false;
        NodeId leaf = kInvalidNode;
    };

    static SearchConfig validate_config(SearchConfig config);
    static std::size_t validate_capacity(std::size_t capacity);
    static std::size_t validate_workers(std::size_t workers);

    void ensure_usable() const;
    void ensure_no_pending() const;
    [[nodiscard]] const Slot& checked_slot(SlotId slot) const;
    [[nodiscard]] Slot& checked_slot(SlotId slot);
    void poison() noexcept;

    SearchConfig config_;
    std::vector<Slot> slots_;
    std::vector<SelectionScratch> selection_scratch_;
    std::vector<LeafRequest> requests_;
    std::size_t active_count_ = 0U;
    std::uint64_t admission_count_ = 0U;
    BatchToken last_token_ = kInvalidBatchToken;
    BatchToken pending_token_ = kInvalidBatchToken;
    bool poisoned_ = false;
    WorkerPool worker_pool_;
};

}  // namespace kb_pente
