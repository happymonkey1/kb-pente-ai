#include "kb_pente/mcts/search_batch.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kb_pente {

SearchConfig SearchBatch::validate_config(SearchConfig config) {
    config.validate();
    return config;
}

std::size_t SearchBatch::validate_capacity(std::size_t capacity) {
    if (capacity == 0U) {
        throw std::invalid_argument(
            "SearchBatch maximum active games must be positive");
    }
    return capacity;
}

std::size_t SearchBatch::validate_workers(std::size_t workers) {
    if (workers == 0U) {
        throw std::invalid_argument(
            "SearchBatch worker thread count must be positive");
    }
    return workers;
}

SearchBatch::SearchBatch(
    SearchConfig config,
    std::size_t max_active_games,
    std::size_t worker_threads)
    : config_(validate_config(config)),
      slots_(validate_capacity(max_active_games)),
      selection_scratch_(max_active_games),
      worker_pool_(validate_workers(worker_threads)) {
    requests_.reserve(max_active_games);
}

SlotId SearchBatch::add(
    Position root,
    Ruleset ruleset,
    SearchSessionConfig session_config) {
    ensure_usable();
    ensure_no_pending();
    session_config.validate();

    SlotId free_slot = kInvalidSlot;
    for (SlotId slot = 0U; slot < slots_.size(); ++slot) {
        if (!slots_[slot].tree) {
            free_slot = slot;
            break;
        }
    }
    if (free_slot == kInvalidSlot) {
        throw std::length_error("SearchBatch has no available slot");
    }

    if (admission_count_ == std::numeric_limits<std::uint64_t>::max() ||
        admission_count_ >
            std::numeric_limits<std::uint64_t>::max() - config_.seed) {
        throw std::overflow_error("SearchBatch admission seed overflow");
    }
    SearchConfig tree_config = config_;
    tree_config.seed = config_.seed + admission_count_;
    auto tree = std::make_unique<Tree>(root, ruleset, tree_config);
    auto session = std::make_unique<SearchSession>(*tree, session_config);
    slots_[free_slot].tree = std::move(tree);
    slots_[free_slot].session = std::move(session);
    ++active_count_;
    ++admission_count_;
    return free_slot;
}

Selection SearchBatch::select() {
    ensure_usable();
    ensure_no_pending();
    if (complete()) {
        throw std::logic_error("Cannot select from a complete SearchBatch");
    }
    if (last_token_ == std::numeric_limits<BatchToken>::max()) {
        throw std::overflow_error("SearchBatch token counter overflow");
    }

    requests_.clear();
    for (SelectionScratch& scratch : selection_scratch_) {
        scratch.has_request = false;
        scratch.leaf = kInvalidNode;
    }

    try {
        worker_pool_.parallel_for(
            slots_.size(), [this](std::size_t slot_index) {
                Slot& slot = slots_[slot_index];
                if (!slot.tree || slot.session->complete()) {
                    return;
                }
                const auto leaf = slot.session->select_evaluation_leaf();
                if (leaf.has_value()) {
                    selection_scratch_[slot_index].has_request = true;
                    selection_scratch_[slot_index].leaf = *leaf;
                }
            });

        for (SlotId slot = 0U; slot < slots_.size(); ++slot) {
            const SelectionScratch& scratch = selection_scratch_[slot];
            if (!scratch.has_request) {
                continue;
            }
            const Slot& selected_slot = slots_[slot];
            if (!selected_slot.tree || !selected_slot.session ||
                !selected_slot.tree->has_pending_evaluation() ||
                selected_slot.tree->pending_leaf() != scratch.leaf) {
                throw std::logic_error(
                    "SearchBatch selected leaf is not pending");
            }
            requests_.push_back(LeafRequest{
                slot,
                scratch.leaf,
                &selected_slot.session->tree().leaf_position(scratch.leaf),
            });
        }
    } catch (...) {
        poison();
        throw;
    }

    ++last_token_;
    if (!requests_.empty()) {
        pending_token_ = last_token_;
    }
    return Selection{last_token_, requests_.data(), requests_.size()};
}

void SearchBatch::backup(
    BatchToken token,
    const float* policies,
    std::size_t policy_rows,
    std::size_t policy_stride,
    const float* values,
    std::size_t value_count) {
    ensure_usable();
    if (pending_token_ == kInvalidBatchToken) {
        throw std::logic_error("SearchBatch has no pending evaluations");
    }
    if (token != pending_token_) {
        throw std::invalid_argument("SearchBatch backup token is stale");
    }
    if (policy_rows != requests_.size() || value_count != requests_.size()) {
        throw std::invalid_argument(
            "SearchBatch backup shape does not match pending requests");
    }
    if (policy_stride != kMaxActions) {
        throw std::invalid_argument(
            "SearchBatch policy stride must equal kMaxActions");
    }
    if (policies == nullptr || values == nullptr) {
        throw std::invalid_argument(
            "SearchBatch backup policy and value buffers cannot be null");
    }

    for (std::size_t row = 0U; row < policy_rows; ++row) {
        const float* policy = policies + row * policy_stride;
        for (std::size_t action = 0U; action < policy_stride; ++action) {
            if (!std::isfinite(policy[action]) || policy[action] < 0.0F) {
                throw std::invalid_argument(
                    "SearchBatch policy must contain finite non-negative values");
            }
        }
        const float value = values[row];
        if (!std::isfinite(value) || value < -1.0F || value > 1.0F) {
            throw std::invalid_argument(
                "SearchBatch value must be finite and in [-1, 1]");
        }
    }

    try {
        worker_pool_.parallel_for(
            requests_.size(), [this, policies, policy_stride, values](
                                  std::size_t request_index) {
                const LeafRequest& request = requests_[request_index];
                Slot& slot = slots_[request.slot];
                if (!slot.tree || !slot.session ||
                    !slot.tree->has_pending_evaluation() ||
                    slot.tree->pending_leaf() != request.leaf) {
                    throw std::logic_error(
                        "SearchBatch backup leaf is no longer pending");
                }
                const std::size_t active_actions =
                    request.position->action_count();
                slot.session->accept_evaluation(
                    request.leaf,
                    policies + request_index * policy_stride,
                    active_actions,
                    values[request_index]);
            });
    } catch (...) {
        poison();
        throw;
    }

    pending_token_ = kInvalidBatchToken;
    requests_.clear();
}

bool SearchBatch::complete() const noexcept {
    if (poisoned_) {
        return false;
    }
    for (const Slot& slot : slots_) {
        if (slot.session && !slot.session->complete()) {
            return false;
        }
    }
    return true;
}

bool SearchBatch::slot_active(SlotId slot) const {
    return checked_slot(slot).tree != nullptr;
}

bool SearchBatch::slot_complete(SlotId slot) const {
    const Slot& checked = checked_slot(slot);
    if (!checked.session) {
        throw std::out_of_range("SearchBatch slot is not active");
    }
    return checked.session->complete();
}

std::uint64_t SearchBatch::slot_seed(SlotId slot) const {
    const Slot& checked = checked_slot(slot);
    if (!checked.tree) {
        throw std::out_of_range("SearchBatch slot is not active");
    }
    return checked.tree->config().seed;
}

const Position& SearchBatch::root_position(SlotId slot) const {
    const Slot& checked = checked_slot(slot);
    if (!checked.tree) {
        throw std::out_of_range("SearchBatch slot is not active");
    }
    return checked.tree->root_position();
}

SearchTelemetry SearchBatch::slot_telemetry(SlotId slot) const {
    const Slot& checked = checked_slot(slot);
    if (!checked.session) {
        throw std::out_of_range("SearchBatch slot is not active");
    }
    return checked.session->telemetry();
}

void SearchBatch::ensure_usable() const {
    if (poisoned_) {
        throw std::logic_error("SearchBatch is poisoned after an internal failure");
    }
}

void SearchBatch::ensure_no_pending() const {
    if (pending_token_ != kInvalidBatchToken) {
        throw std::logic_error(
            "SearchBatch cannot mutate while evaluations are pending");
    }
}

const SearchBatch::Slot& SearchBatch::checked_slot(SlotId slot) const {
    if (slot >= slots_.size()) {
        throw std::out_of_range("SearchBatch slot is out of range");
    }
    return slots_[slot];
}

SearchBatch::Slot& SearchBatch::checked_slot(SlotId slot) {
    if (slot >= slots_.size()) {
        throw std::out_of_range("SearchBatch slot is out of range");
    }
    return slots_[slot];
}

void SearchBatch::poison() noexcept {
    poisoned_ = true;
}

}  // namespace kb_pente
