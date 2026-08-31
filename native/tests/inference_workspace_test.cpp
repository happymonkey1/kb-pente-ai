#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "kb_pente/mcts/inference_workspace.h"

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

void test_grouping_and_representative_order() {
    kb_pente::InferenceWorkspace workspace(8U);
    kb_pente::Position base = kb_pente::Position::initial(5);
    kb_pente::Position same = base;
    kb_pente::Position other = kb_pente::Position::initial(9);
    kb_pente::Position collision_equal = base;
    kb_pente::Position collision_unequal = base;
    collision_unequal.stones[0] =
        static_cast<std::int8_t>(kb_pente::Player::One);
    collision_unequal.ply = 1U;
    collision_unequal.last_action = 0U;
    collision_unequal.current_player = kb_pente::Player::Two;
    collision_unequal.refresh_hash();
    collision_unequal.hash_lo = base.hash_lo;
    collision_unequal.hash_hi = base.hash_hi;

    // The insertion order deliberately differs from slot order. The
    // representative for the identical base group must still be slot two.
    workspace.add({
        7U,
        1U,
        &base,
        kb_pente::Ruleset::Freestyle,
    });
    workspace.add({
        2U,
        2U,
        &same,
        kb_pente::Ruleset::Freestyle,
    });
    workspace.add({
        8U,
        3U,
        &other,
        kb_pente::Ruleset::Freestyle,
    });
    workspace.add({
        4U,
        4U,
        &base,
        kb_pente::Ruleset::Standard,
    });
    workspace.add({
        6U,
        5U,
        &collision_equal,
        kb_pente::Ruleset::Freestyle,
    });
    workspace.add({
        1U,
        6U,
        &collision_unequal,
        kb_pente::Ruleset::Freestyle,
    });

    expect(workspace.raw_count() == 6U, "all raw candidates are retained");
    workspace.finalize();
    expect(workspace.unique_count() == 4U,
           "equal positions merge while collisions remain separate");
    expect(workspace.representative(0U).slot == 1U,
           "lowest collision representative is first");
    expect(workspace.representative(1U).slot == 2U,
           "lowest identical-position representative is retained");
    expect(workspace.representative(2U).slot == 4U,
           "mixed ruleset remains a separate representative");
    expect(workspace.representative(3U).slot == 8U,
           "distinct position representative follows slot order");

    expect(workspace.evaluation_index_for_raw(0U) == 1U,
           "base candidate maps to its lowest representative");
    expect(workspace.evaluation_index_for_raw(1U) == 1U,
           "identical candidate maps to the same evaluation");
    expect(workspace.evaluation_index_for_raw(2U) == 3U,
           "distinct position maps to its own evaluation");
    expect(workspace.evaluation_index_for_raw(3U) == 2U,
           "mixed ruleset maps to its own evaluation");
    expect(workspace.evaluation_index_for_raw(4U) == 1U,
           "equal-hash equal-position candidate merges safely");
    expect(workspace.evaluation_index_for_raw(5U) == 0U,
           "equal-hash unequal-position candidate is not merged");
    workspace.finalize();
    expect(workspace.representative(1U).slot == 2U,
           "repeated finalization preserves deterministic grouping");
}

void test_fixed_capacity_and_invalid_candidates() {
    kb_pente::InferenceWorkspace workspace(1U);
    const kb_pente::Position position = kb_pente::Position::initial(5);
    const auto candidate = kb_pente::InferenceCandidate{
        0U,
        1U,
        &position,
        kb_pente::Ruleset::Freestyle,
    };
    workspace.add(candidate);
    expect(workspace.candidate_capacity() == 1U,
           "candidate storage is preallocated to capacity");
    expect(workspace.sorted_index_capacity() == 1U,
           "sorted-index storage is preallocated to capacity");
    expect(workspace.representative_capacity() == 1U,
           "representative storage is preallocated to capacity");
    expect(workspace.selected_to_evaluation_capacity() == 1U,
           "selected mapping storage is preallocated to capacity");
    expect(workspace.raw_selected_request_capacity() == 1U,
           "raw request storage is preallocated to capacity");
    workspace.finalize();
    workspace.clear();
    expect(workspace.candidate_capacity() == 1U,
           "clear retains candidate capacity");
    workspace.add(candidate);

    expect_throws<std::length_error>(
        [&workspace, &candidate] { workspace.add(candidate); },
        "adding beyond fixed capacity is rejected");
}

}  // namespace

int main() {
    try {
        test_grouping_and_representative_order();
        test_fixed_capacity_and_invalid_candidates();
    } catch (const std::exception& failure) {
        std::cerr << "inference workspace test failed: " << failure.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "inference workspace tests passed\n";
    return EXIT_SUCCESS;
}
