#include "kb_pente/mcts/inference_workspace.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace kb_pente {

InferenceWorkspace::InferenceWorkspace(std::size_t capacity)
    : capacity_(capacity) {
    if (capacity_ == 0U) {
        throw std::invalid_argument(
            "Inference workspace capacity must be positive");
    }

    candidates_.reserve(capacity_);
    sorted_indices_.reserve(capacity_);
    representatives_.reserve(capacity_);
    selected_to_representative_.reserve(capacity_);
    representative_to_evaluation_.reserve(capacity_);
    selected_to_evaluation_.reserve(capacity_);
}

void InferenceWorkspace::clear() noexcept {
    candidates_.clear();
    sorted_indices_.clear();
    representatives_.clear();
    selected_to_representative_.clear();
    representative_to_evaluation_.clear();
    selected_to_evaluation_.clear();
    finalized_ = false;
}

void InferenceWorkspace::add(InferenceCandidate candidate) {
    if (finalized_) {
        throw std::logic_error(
            "Cannot add to a finalized inference workspace");
    }
    if (candidate.position == nullptr) {
        throw std::invalid_argument(
            "Inference candidate position cannot be null");
    }
    if (!is_valid_ruleset(candidate.ruleset)) {
        throw std::invalid_argument("Inference candidate has an invalid ruleset");
    }
    if (candidates_.size() == capacity_) {
        throw std::length_error("Inference workspace capacity exceeded");
    }

    candidates_.push_back(candidate);
}

bool InferenceWorkspace::same_key(
    std::size_t left_index,
    std::size_t right_index) const noexcept {
    const InferenceCandidate& left = candidates_[left_index];
    const InferenceCandidate& right = candidates_[right_index];
    return left.ruleset == right.ruleset &&
           left.position->hash() == right.position->hash();
}

void InferenceWorkspace::finalize() {
    if (finalized_) {
        return;
    }

    sorted_indices_.clear();
    selected_to_representative_.assign(candidates_.size(), 0U);
    selected_to_evaluation_.assign(candidates_.size(), 0U);
    representatives_.clear();
    representative_to_evaluation_.clear();

    for (std::size_t index = 0U; index < candidates_.size(); ++index) {
        sorted_indices_.push_back(index);
    }

    std::sort(
        sorted_indices_.begin(),
        sorted_indices_.end(),
        [this](std::size_t left_index, std::size_t right_index) noexcept {
            const InferenceCandidate& left = candidates_[left_index];
            const InferenceCandidate& right = candidates_[right_index];
            const PositionHash left_hash = left.position->hash();
            const PositionHash right_hash = right.position->hash();
            if (left_hash.lo != right_hash.lo) {
                return left_hash.lo < right_hash.lo;
            }
            if (left_hash.hi != right_hash.hi) {
                return left_hash.hi < right_hash.hi;
            }
            const auto left_ruleset =
                static_cast<std::uint8_t>(left.ruleset);
            const auto right_ruleset =
                static_cast<std::uint8_t>(right.ruleset);
            if (left_ruleset != right_ruleset) {
                return left_ruleset < right_ruleset;
            }
            if (left.slot != right.slot) {
                return left.slot < right.slot;
            }
            return left_index < right_index;
        });

    std::size_t run_start = 0U;
    while (run_start < sorted_indices_.size()) {
        std::size_t run_end = run_start + 1U;
        while (run_end < sorted_indices_.size() &&
               same_key(sorted_indices_[run_start], sorted_indices_[run_end])) {
            ++run_end;
        }

        const std::size_t first_representative = representatives_.size();
        for (std::size_t sorted_index = run_start; sorted_index < run_end;
             ++sorted_index) {
            const std::size_t candidate_index = sorted_indices_[sorted_index];
            const InferenceCandidate& candidate = candidates_[candidate_index];
            std::size_t representative_index = first_representative;
            for (; representative_index < representatives_.size();
                 ++representative_index) {
                const InferenceCandidate& representative = candidates_[
                    representatives_[representative_index]];
                if (*candidate.position == *representative.position) {
                    break;
                }
            }
            if (representative_index == representatives_.size()) {
                representatives_.push_back(candidate_index);
            }
            selected_to_representative_[candidate_index] =
                representative_index;
        }

        run_start = run_end;
    }

    std::sort(
        representatives_.begin(),
        representatives_.end(),
        [this](std::size_t left_index, std::size_t right_index) noexcept {
            const InferenceCandidate& left = candidates_[left_index];
            const InferenceCandidate& right = candidates_[right_index];
            if (left.slot != right.slot) {
                return left.slot < right.slot;
            }
            return left_index < right_index;
        });

    representative_to_evaluation_.resize(representatives_.size());
    for (std::size_t evaluation_index = 0U;
         evaluation_index < representatives_.size();
         ++evaluation_index) {
        const std::size_t candidate_index =
            representatives_[evaluation_index];
        const std::size_t representative_index =
            selected_to_representative_[candidate_index];
        representative_to_evaluation_[representative_index] =
            evaluation_index;
    }

    for (std::size_t candidate_index = 0U;
         candidate_index < candidates_.size();
         ++candidate_index) {
        selected_to_evaluation_[candidate_index] =
            representative_to_evaluation_[
                selected_to_representative_[candidate_index]];
    }

    finalized_ = true;
}

const InferenceCandidate& InferenceWorkspace::raw_candidate(
    std::size_t index) const {
    if (index >= candidates_.size()) {
        throw std::out_of_range("Inference raw candidate index is out of range");
    }
    return candidates_[index];
}

const InferenceCandidate& InferenceWorkspace::representative(
    std::size_t index) const {
    if (!finalized_) {
        throw std::logic_error(
            "Inference workspace representatives are not finalized");
    }
    if (index >= representatives_.size()) {
        throw std::out_of_range(
            "Inference representative index is out of range");
    }
    return candidates_[representatives_[index]];
}

std::size_t InferenceWorkspace::evaluation_index_for_raw(
    std::size_t index) const {
    if (!finalized_) {
        throw std::logic_error(
            "Inference workspace mappings are not finalized");
    }
    if (index >= selected_to_evaluation_.size()) {
        throw std::out_of_range(
            "Inference raw candidate mapping index is out of range");
    }
    return selected_to_evaluation_[index];
}

}  // namespace kb_pente
