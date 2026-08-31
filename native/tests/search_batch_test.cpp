#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "kb_pente/mcts/search_batch.h"

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

kb_pente::SearchConfig batch_config(
    std::uint32_t simulation_budget = 4U,
    std::uint64_t seed = 7U) {
    return kb_pente::SearchConfig(
        1.5F, simulation_budget, 0.0F, 0.03F, seed);
}

kb_pente::Action action_at(
    std::uint8_t board_size,
    std::uint8_t row,
    std::uint8_t column) {
    return static_cast<kb_pente::Action>(row * board_size + column);
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
    position.refresh_hash();
    expect(!kb_pente::check_terminal(position).is_terminal(),
           "draw root is nonterminal before its final move");
    return position;
}

kb_pente::Position make_terminal_draw_root() {
    kb_pente::Position position = make_draw_root();
    position.stones[24] = static_cast<std::int8_t>(kb_pente::Player::One);
    position.ply = 25;
    position.current_player = kb_pente::Player::Two;
    position.validate();
    expect(kb_pente::check_terminal(position).status ==
               kb_pente::GameStatus::Draw,
           "terminal draw fixture is terminal");
    return position;
}

std::vector<float> uniform_policies(std::size_t request_count) {
    return std::vector<float>(
        request_count * kb_pente::kMaxActions, 1.0F);
}

void backup_uniform(
    kb_pente::SearchBatch& batch,
    const kb_pente::Selection& selection) {
    if (selection.empty()) {
        return;
    }
    std::vector<float> policies = uniform_policies(selection.size());
    std::vector<float> values(selection.size(), 0.0F);
    batch.backup(
        selection.token,
        policies.data(),
        selection.size(),
        kb_pente::kMaxActions,
        values.data(),
        values.size());
}

void backup_first_one_hot(
    kb_pente::SearchBatch& batch,
    const kb_pente::Selection& selection,
    kb_pente::Action action) {
    expect(!selection.empty(), "one-hot helper expects a request");
    std::vector<float> policies = uniform_policies(selection.size());
    for (std::size_t index = 0; index < kb_pente::kMaxActions; ++index) {
        policies[index] = 0.0F;
    }
    policies[action] = 1.0F;
    std::vector<float> values(selection.size(), 0.0F);
    batch.backup(
        selection.token,
        policies.data(),
        selection.size(),
        kb_pente::kMaxActions,
        values.data(),
        values.size());
}

void run_to_completion(kb_pente::SearchBatch& batch) {
    std::size_t waves = 0U;
    while (!batch.complete()) {
        const kb_pente::Selection selection = batch.select();
        backup_uniform(batch, selection);
        ++waves;
        if (waves > 128U) {
            throw TestFailure("batch did not complete within the test bound");
        }
    }
    expect(!batch.has_pending(), "completed batch has no pending requests");
}

void expect_normalized_policy(
    const std::array<float, kb_pente::kMaxActions>& policy,
    std::size_t active_actions,
    const char* message) {
    double total = 0.0;
    for (std::size_t index = 0U; index < active_actions; ++index) {
        expect(std::isfinite(policy[index]) && policy[index] >= 0.0F, message);
        total += static_cast<double>(policy[index]);
    }
    expect(std::abs(total - 1.0) < 1.0e-6, message);
    for (std::size_t index = active_actions; index < kb_pente::kMaxActions;
         ++index) {
        expect(policy[index] == 0.0F, message);
    }
}

void test_root_policy_and_advancement() {
    const kb_pente::Position original_root =
        kb_pente::Position::initial(5);
    kb_pente::SearchBatch batch(batch_config(1U, 41U), 1U, 1U);
    const auto slot = batch.add(
        original_root,
        kb_pente::Ruleset::Freestyle,
        kb_pente::SearchSessionConfig(1.0F, false));

    expect(batch.root_terminal(slot) == kb_pente::TerminalResult::in_progress(),
           "new root is reported as in progress");
    expect_throws<std::logic_error>(
        [&batch, slot] { (void)batch.root_policy(slot); },
        "incomplete slots do not expose a root policy");

    const auto selection = batch.select();
    expect_throws<std::logic_error>(
        [&batch, slot] { (void)batch.root_policy(slot); },
        "pending slots do not expose a root policy");
    backup_uniform(batch, selection);

    expect(batch.complete() && batch.slot_complete(slot),
           "root policy fixture completes its session");
    const auto policy = batch.root_policy(slot);
    expect_normalized_policy(policy, 25U, "root policy is normalized");
    for (std::size_t index = 0U; index < 25U; ++index) {
        expect(std::abs(policy[index] - 1.0F / 25.0F) < 1.0e-6F,
               "positive-temperature fallback uses the base prior");
    }

    const auto expected = kb_pente::apply_action(
        original_root, 0U, kb_pente::Ruleset::Freestyle);
    const auto stats = batch.advance_root(
        slot, 0U, kb_pente::SearchSessionConfig(0.0F, false));
    expect(!stats.reused_subtree && stats.retained_node_count == 1U,
           "unallocated root advancement reports a fresh root");
    expect(batch.root_position(slot) == expected.position,
           "unallocated advancement applies the exact transition");
    expect(batch.root_terminal(slot) == expected.terminal,
           "advancement reports the new root terminal state");
    expect(!batch.slot_complete(slot),
           "nonterminal advancement creates a fresh session");
    run_to_completion(batch);
    const auto continued_policy = batch.root_policy(slot);
    std::size_t first_legal_action = kb_pente::kMaxActions;
    for (std::size_t index = 0U; index < expected.position.action_count();
         ++index) {
        if (expected.position.stones[index] == 0) {
            first_legal_action = index;
            break;
        }
    }
    expect(first_legal_action < expected.position.action_count(),
           "continued root has a legal action");
    expect(continued_policy[first_legal_action] == 1.0F,
           "advancement applies the supplied zero-temperature session config");
    for (std::size_t index = 0U;
         index < batch.root_position(slot).action_count();
         ++index) {
        expect(continued_policy[index] ==
                   (index == first_legal_action ? 1.0F : 0.0F),
               "new zero-temperature session clears other actions");
    }

    kb_pente::SearchBatch pending_other(batch_config(1U, 42U), 2U, 1U);
    const auto completed_slot = pending_other.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto completed_selection = pending_other.select();
    backup_uniform(pending_other, completed_selection);
    const auto pending_peer = pending_other.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto pending_selection = pending_other.select();
    expect(pending_selection.size() == 1U &&
               pending_selection[0].slot_id() == pending_peer,
           "one completed peer leaves one pending request");
    expect_throws<std::logic_error>(
        [&pending_other, completed_slot] {
            (void)pending_other.root_policy(completed_slot);
        },
        "root policy rejects a pending wave in another slot");
    backup_uniform(pending_other, pending_selection);
}

void test_allocated_root_advancement_and_continuation() {
    const kb_pente::Position original_root =
        kb_pente::Position::initial(5);
    kb_pente::SearchBatch batch(batch_config(3U, 51U), 1U, 1U);
    const auto slot = batch.add(
        original_root,
        kb_pente::Ruleset::Freestyle,
        kb_pente::SearchSessionConfig(0.0F, false));

    const auto first = batch.select();
    backup_first_one_hot(batch, first, 0U);
    const auto second = batch.select();
    backup_first_one_hot(batch, second, 2U);
    const auto third = batch.select();
    backup_first_one_hot(batch, third, 3U);
    expect(batch.complete(), "allocated advancement fixture completes");

    const auto expected = kb_pente::apply_action(
        original_root, 0U, kb_pente::Ruleset::Freestyle);
    const auto stats = batch.advance_root(slot, 0U);
    expect(stats.reused_subtree && stats.retained_node_count >= 2U,
           "allocated root advancement retains the selected subtree");
    expect(stats.discarded_node_count > 0U,
           "allocated advancement reports discarded storage");
    expect(stats.new_owned_bytes < stats.previous_owned_bytes,
           "allocated advancement releases unreachable arena storage");
    expect(batch.root_position(slot) == expected.position,
           "allocated advancement preserves the selected child position");
    expect(batch.root_terminal(slot) == expected.terminal,
           "allocated advancement preserves the child terminal state");
    expect(!batch.slot_complete(slot),
           "allocated advancement starts a new nonterminal session");

    run_to_completion(batch);
    const auto continued_policy = batch.root_policy(slot);
    expect_normalized_policy(
        continued_policy,
        batch.root_position(slot).action_count(),
        "continued search exposes a normalized root policy");
}

void test_terminal_advancement_and_slot_removal() {
    kb_pente::SearchBatch batch(batch_config(1U, 61U), 2U, 2U);
    const auto terminal_slot = batch.add(
        make_draw_root(), kb_pente::Ruleset::Freestyle);
    const auto terminal_selection = batch.select();
    backup_first_one_hot(batch, terminal_selection, 24U);
    expect(batch.slot_complete(terminal_slot),
           "terminal fixture completes before advancement");

    const auto expected_terminal = kb_pente::apply_action(
        make_draw_root(), 24U, kb_pente::Ruleset::Freestyle);
    const auto stats = batch.advance_root(terminal_slot, 24U);
    expect(!stats.reused_subtree && stats.retained_node_count == 1U,
           "terminal advancement creates one fresh root");
    expect(batch.root_position(terminal_slot) == expected_terminal.position &&
               batch.root_terminal(terminal_slot) == expected_terminal.terminal,
           "terminal advancement preserves the exact terminal transition");
    expect(batch.slot_complete(terminal_slot),
           "terminal root remains an active complete slot");
    expect_throws<std::logic_error>(
        [&batch, terminal_slot] { (void)batch.root_policy(terminal_slot); },
        "terminal slots do not expose a root policy");
    expect_throws<std::logic_error>(
        [&batch, terminal_slot] { (void)batch.slot_telemetry(terminal_slot); },
        "sessionless terminal slots do not expose session telemetry");

    const auto peer = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto peer_selection = batch.select();
    expect(peer_selection.size() == 1U && peer_selection[0].slot_id() == peer,
           "selection skips an active terminal slot");
    backup_uniform(batch, peer_selection);
    expect(batch.complete(), "terminal and peer slots complete");

    batch.replace_root(
        terminal_slot,
        kb_pente::Position::initial(9),
        kb_pente::Ruleset::Freestyle);
    expect(batch.slot_active(terminal_slot) &&
               batch.root_position(terminal_slot) ==
                   kb_pente::Position::initial(9) &&
               !batch.slot_complete(terminal_slot),
           "sessionless terminal slots can be replaced");
    run_to_completion(batch);

    batch.remove(terminal_slot);
    expect(!batch.slot_active(terminal_slot) && batch.active_count() == 1U,
           "removal clears the terminal slot and active count");
    expect_throws<std::out_of_range>(
        [&batch, terminal_slot] { batch.remove(terminal_slot); },
        "removing an inactive slot is rejected");
    batch.remove(peer);
    expect(batch.active_count() == 0U && batch.complete(),
           "removing all slots restores an empty complete batch");
}

void test_lifecycle_rejection_and_lowest_free_reuse() {
    kb_pente::SearchBatch batch(batch_config(2U, 71U), 2U, 2U);
    const auto first = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto second = batch.add(
        kb_pente::Position::initial(9), kb_pente::Ruleset::Freestyle);
    const auto original_first = batch.root_position(first);

    expect_throws<std::logic_error>(
        [&batch, first] { batch.remove(first); },
        "incomplete slot removal is rejected");
    const auto selection = batch.select();
    expect_throws<std::logic_error>(
        [&batch, first] { (void)batch.advance_root(first, 0U); },
        "pending selection blocks root advancement");
    expect_throws<std::logic_error>(
        [&batch, first] { batch.remove(first); },
        "pending selection blocks removal");
    expect_throws<std::logic_error>(
        [&batch, first] {
            batch.replace_root(
                first,
                kb_pente::Position::initial(5),
                kb_pente::Ruleset::Freestyle);
        },
        "pending selection blocks replacement");
    backup_uniform(batch, selection);
    const auto second_selection = batch.select();
    backup_uniform(batch, second_selection);
    expect(batch.complete(), "both lifecycle fixture sessions complete");

    kb_pente::SearchSessionConfig invalid_session_config;
    invalid_session_config.temperature = -1.0F;
    expect_throws<std::invalid_argument>(
        [&batch, first, &invalid_session_config] {
            (void)batch.advance_root(
                first,
                0U,
                invalid_session_config);
        },
        "invalid next session config is rejected before advancement");
    expect(batch.root_position(first) == original_first &&
               batch.slot_complete(first),
           "invalid next session config preserves the old completed session");

    expect_throws<std::invalid_argument>(
        [&batch, first] {
            (void)batch.advance_root(first, kb_pente::kInvalidAction);
        },
        "invalid root action is rejected after completion");
    expect(batch.root_position(first) == original_first,
           "invalid root action preserves the completed slot");

    batch.remove(first);
    expect(!batch.slot_active(first) && batch.active_count() == 1U,
           "completed slot removal updates active count");
    const auto reused = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    expect(reused == first && batch.slot_seed(reused) == 73U,
           "new admission reuses the lowest free slot and advances seed");
    run_to_completion(batch);
    batch.remove(second);
    batch.remove(reused);
    expect(batch.active_count() == 0U, "all reusable lifecycle slots remove");
}

void test_strong_replacement_and_seed_progression() {
    kb_pente::SearchBatch batch(batch_config(1U, 100U), 2U, 1U);
    const auto slot = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto selection = batch.select();
    backup_uniform(batch, selection);
    expect(batch.complete(), "replacement fixture completes its session");

    const auto original_root = batch.root_position(slot);
    const auto original_seed = batch.slot_seed(slot);
    expect_throws<std::invalid_argument>(
        [&batch, slot] {
            batch.replace_root(
                slot,
                make_terminal_draw_root(),
                kb_pente::Ruleset::Freestyle);
        },
        "terminal replacement roots are rejected");
    expect(batch.root_position(slot) == original_root &&
               batch.slot_seed(slot) == original_seed && batch.active_count() == 1U,
           "rejected replacement leaves the old slot untouched");

    kb_pente::SearchSessionConfig invalid_session_config;
    invalid_session_config.temperature = -1.0F;
    expect_throws<std::invalid_argument>(
        [&batch, slot, &invalid_session_config] {
            batch.replace_root(
                slot,
                kb_pente::Position::initial(9),
                kb_pente::Ruleset::Freestyle,
                invalid_session_config);
        },
        "invalid replacement session config is rejected");
    expect(batch.root_position(slot) == original_root &&
               batch.slot_seed(slot) == original_seed,
           "invalid replacement config preserves the old slot");

    batch.replace_root(
        slot,
        kb_pente::Position::initial(9),
        kb_pente::Ruleset::Freestyle,
        kb_pente::SearchSessionConfig(0.0F, false));
    expect(batch.slot_seed(slot) == 101U &&
               batch.root_position(slot) == kb_pente::Position::initial(9) &&
               !batch.slot_complete(slot),
           "successful replacement keeps the slot and consumes one seed");
    run_to_completion(batch);

    const auto next_slot = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    expect(next_slot == 1U && batch.slot_seed(next_slot) == 102U,
           "later admission follows successful replacement seed order");
    run_to_completion(batch);
    batch.remove(slot);
    batch.remove(next_slot);
}

void test_worker_equivalence_after_lifecycle_changes() {
    kb_pente::SearchBatch one_worker(batch_config(3U, 81U), 2U, 1U);
    kb_pente::SearchBatch many_workers(batch_config(3U, 81U), 2U, 4U);
    for (const auto board_size : {5U, 9U}) {
        const auto root = kb_pente::Position::initial(
            static_cast<std::uint8_t>(board_size));
        (void)one_worker.add(
            root,
            kb_pente::Ruleset::Freestyle,
            kb_pente::SearchSessionConfig(0.0F, false));
        (void)many_workers.add(
            root,
            kb_pente::Ruleset::Freestyle,
            kb_pente::SearchSessionConfig(0.0F, false));
    }
    run_to_completion(one_worker);
    run_to_completion(many_workers);
    for (std::size_t slot = 0U; slot < 2U; ++slot) {
        const auto one_policy = one_worker.root_policy(slot);
        const auto many_policy = many_workers.root_policy(slot);
        expect(one_policy == many_policy,
               "worker counts agree before lifecycle continuation");
        const auto one_stats = one_worker.advance_root(slot, 0U);
        const auto many_stats = many_workers.advance_root(slot, 0U);
        expect(one_stats == many_stats,
               "worker counts agree on root advancement stats");
    }
    run_to_completion(one_worker);
    run_to_completion(many_workers);
    for (std::size_t slot = 0U; slot < 2U; ++slot) {
        expect(one_worker.root_position(slot) == many_workers.root_position(slot),
               "worker counts agree on continued roots");
        expect(one_worker.slot_telemetry(slot) == many_workers.slot_telemetry(slot),
               "worker counts agree on continued telemetry");
    }
}

void test_construction_and_admission() {
    expect_throws<std::invalid_argument>(
        [] { kb_pente::SearchBatch(batch_config(), 0U, 1U); },
        "zero batch capacity is rejected");
    expect_throws<std::invalid_argument>(
        [] { kb_pente::SearchBatch(batch_config(), 1U, 0U); },
        "zero worker count is rejected");

    kb_pente::SearchConfig invalid_config = batch_config();
    invalid_config.simulation_budget = 0U;
    expect_throws<std::invalid_argument>(
        [&invalid_config] {
            kb_pente::SearchBatch batch(invalid_config, 1U, 1U);
        },
        "mutated invalid search config is rejected");

    kb_pente::SearchBatch batch(batch_config(), 2U, 2U);
    expect(batch.capacity() == 2U, "batch capacity is retained");
    expect(batch.active_count() == 0U, "batch starts with no active slots");
    expect(batch.thread_count() == 2U, "batch worker count is retained");
    expect(batch.complete(), "empty batch is complete");
    expect(batch.last_token() == kb_pente::kInvalidBatchToken,
           "empty batch has no selection token");
    expect_throws<std::logic_error>(
        [&batch] { (void)batch.select(); },
        "selection from an empty complete batch is rejected");
    expect_throws<std::out_of_range>(
        [&batch] { (void)batch.slot_complete(0U); },
        "completion inspection rejects an inactive slot");

    const auto first = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto second = batch.add(
        kb_pente::Position::initial(9), kb_pente::Ruleset::Freestyle);
    expect(first == 0U && second == 1U,
           "admission uses stable lowest available slots");
    expect(batch.active_count() == 2U, "admission increments active count");
    expect(batch.slot_active(first) && batch.slot_active(second),
           "admitted slots are active");
    expect(batch.root_position(first) == kb_pente::Position::initial(5),
           "first root is inspectable");
    expect(batch.root_position(second) == kb_pente::Position::initial(9),
           "second root is inspectable");
    expect(!batch.slot_complete(first) && !batch.slot_complete(second),
           "new slots are incomplete");

    kb_pente::SearchBatch seeded(batch_config(1U, 1234U), 2U, 1U);
    const auto seeded_first = seeded.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto seeded_second = seeded.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    expect(seeded.slot_seed(seeded_first) == 1234U &&
               seeded.slot_seed(seeded_second) == 1235U,
           "admissions receive distinct deterministic seeds");
    kb_pente::SearchBatch repeated_seed(batch_config(1U, 1234U), 1U, 1U);
    const auto repeated_slot = repeated_seed.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    expect(repeated_seed.slot_seed(repeated_slot) ==
               seeded.slot_seed(seeded_first),
           "run-level seeds reproduce admission seeds");

    expect_throws<std::length_error>(
        [&batch] {
            (void)batch.add(
                kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
        },
        "admission beyond fixed capacity is rejected");
    expect_throws<std::out_of_range>(
        [&batch] { (void)batch.slot_active(2U); },
        "out-of-range slot inspection is rejected");
    expect_throws<std::out_of_range>(
        [&batch] { (void)batch.root_position(2U); },
        "out-of-range root inspection is rejected");

    kb_pente::SearchBatch root_validation(batch_config(), 1U, 1U);
    expect_throws<std::invalid_argument>(
        [&root_validation] {
            (void)root_validation.add(
                make_terminal_draw_root(), kb_pente::Ruleset::Freestyle);
        },
        "terminal roots are rejected at admission");
    expect_throws<std::invalid_argument>(
        [&root_validation] {
            (void)root_validation.add(
                kb_pente::Position::initial(6), kb_pente::Ruleset::Standard);
        },
        "invalid ruleset board combinations are rejected");
}

void test_selection_order_and_request_views() {
    kb_pente::SearchBatch batch(batch_config(3U), 3U, 2U);
    const auto slot0 = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto slot1 = batch.add(
        kb_pente::Position::initial(9), kb_pente::Ruleset::Freestyle);
    const auto slot2 = batch.add(
        kb_pente::Position::initial(19), kb_pente::Ruleset::Freestyle);
    const kb_pente::Selection selection = batch.select();
    expect(selection.token == 1U, "first selection receives token one");
    expect(selection.size() == 3U, "one request is returned per active tree");
    expect(selection.raw_size() == 3U,
           "raw selected count matches unique requests without duplicates");
    expect(batch.pending_request_count() == 3U,
           "pending count matches returned requests");
    expect(batch.has_pending() && batch.pending_token() == selection.token,
           "selection token is pending");
    expect(selection[0].slot_id() == slot0 && selection[1].slot_id() == slot1 &&
               selection[2].slot_id() == slot2,
           "requests are gathered in slot order");
    expect(selection[0].leaf_position() == kb_pente::Position::initial(5),
           "request exposes the first leaf position");
    expect(selection[1].leaf_position() == kb_pente::Position::initial(9),
           "request exposes the second leaf position");
    expect(selection[2].leaf_position() == kb_pente::Position::initial(19),
           "request exposes the third leaf position");
    expect(selection.begin() + selection.size() == selection.end(),
           "selection view has a contiguous end");

    expect_throws<std::logic_error>(
        [&batch] { (void)batch.select(); },
        "selection while requests are pending is rejected");
    expect_throws<std::logic_error>(
        [&batch] {
            (void)batch.add(
                kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
        },
        "admission while requests are pending is rejected");

    backup_uniform(batch, selection);
    expect(!batch.has_pending() && batch.pending_request_count() == 0U,
           "successful backup clears pending requests");
    expect(batch.last_token() == selection.token,
           "backup does not change the completed selection token");

    const kb_pente::Selection next = batch.select();
    expect(next.token > selection.token, "selection tokens are monotonic");
    expect(next.size() == 3U, "each incomplete slot requests again");
    expect(next.data == selection.data,
           "reserved request storage is reused across waves");
    backup_uniform(batch, next);
}

void test_deduplication_and_fanout() {
    kb_pente::SearchBatch batch(batch_config(2U, 29U), 2U, 2U);
    const auto first_slot = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto second_slot = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto raw_request_capacity = batch.raw_request_capacity();
    const auto unique_request_capacity = batch.unique_request_capacity();
    expect(raw_request_capacity == 2U && unique_request_capacity == 2U,
           "raw and unique request storage are preallocated to capacity");

    const kb_pente::Selection first = batch.select();
    expect(first.size() == 1U && first.raw_size() == 2U,
           "identical roots collapse to one evaluator row");
    expect(first[0].slot_id() == first_slot,
           "lowest slot is the duplicate representative");
    expect(batch.pending_request_count() == 1U &&
               batch.pending_selected_count() == 2U,
           "pending counts expose unique rows and raw leaves");
    expect(batch.raw_request_capacity() == raw_request_capacity &&
               batch.unique_request_capacity() == unique_request_capacity,
           "selection does not grow dedup workspace capacity");

    std::vector<float> raw_policies = uniform_policies(first.raw_size());
    std::vector<float> raw_values(first.raw_size(), 0.5F);
    expect_throws<std::invalid_argument>(
        [&batch, &first, &raw_policies, &raw_values] {
            batch.backup(
                first.token,
                raw_policies.data(),
                first.raw_size(),
                kb_pente::kMaxActions,
                raw_values.data(),
                raw_values.size());
        },
        "deduplicated backup rejects raw-row policy shape");
    expect(batch.has_pending() && batch.pending_request_count() == 1U &&
               batch.pending_selected_count() == 2U,
           "raw-row rejection preserves deduplicated pending state");

    std::vector<float> policies(kb_pente::kMaxActions, 0.0F);
    policies[0] = 1.0F;
    const std::vector<float> values{0.5F};
    batch.backup(
        first.token,
        policies.data(),
        first.size(),
        kb_pente::kMaxActions,
        values.data(),
        values.size());
    expect(batch.pending_request_count() == 0U &&
               batch.pending_selected_count() == 0U,
           "successful fan-out clears all pending request storage");
    expect(batch.raw_request_capacity() == raw_request_capacity &&
               batch.unique_request_capacity() == unique_request_capacity,
           "backup does not grow or release dedup workspace capacity");
    const auto first_telemetry = batch.slot_telemetry(first_slot);
    const auto second_telemetry = batch.slot_telemetry(second_slot);
    expect(first_telemetry.evaluator_completions == 1U &&
               second_telemetry.evaluator_completions == 1U,
           "one evaluator result fans out to every duplicate leaf");

    const auto telemetry_after_first = batch.deduplication_telemetry();
    const kb_pente::DeduplicationStats expected_first{
        1U,
        2U,
        1U,
        1U,
        0.5,
    };
    expect(telemetry_after_first.cumulative == expected_first &&
               telemetry_after_first.last_wave == expected_first,
           "first-wave duplicate telemetry is exact");

    const kb_pente::Selection second = batch.select();
    expect(second.size() == 1U && second.raw_size() == 2U,
           "identical descendants collapse on the next wave");
    backup_uniform(batch, second);
    expect(batch.complete(), "deduplicated batch completes after fan-out");
    const auto telemetry_after_second = batch.deduplication_telemetry();
    const kb_pente::DeduplicationStats expected_cumulative{
        2U,
        4U,
        2U,
        2U,
        0.5,
    };
    expect(telemetry_after_second.cumulative == expected_cumulative &&
               telemetry_after_second.last_wave == expected_first,
           "multiwave duplicate telemetry accumulates deterministically");
}

void test_deduplication_respects_ruleset_and_backup_shape() {
    kb_pente::SearchBatch batch(batch_config(1U, 31U), 2U, 2U);
    (void)batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Standard);
    (void)batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const kb_pente::Selection selection = batch.select();
    expect(selection.size() == 2U && selection.raw_size() == 2U,
           "ruleset participates in the deduplication key");
    expect(batch.deduplication_telemetry().last_wave.eliminated_duplicate_evaluations ==
               0U,
           "mixed rulesets do not eliminate identical semantic roots");

    std::vector<float> policies = uniform_policies(selection.size());
    std::vector<float> values(selection.size(), 0.0F);
    expect_throws<std::invalid_argument>(
        [&batch, &selection, &policies, &values] {
            batch.backup(
                selection.token,
                policies.data(),
                1U,
                kb_pente::kMaxActions,
                values.data(),
                1U);
        },
        "backup shape is validated against unique evaluator rows");
    expect(batch.has_pending() && batch.pending_request_count() == 2U &&
               batch.pending_selected_count() == 2U,
           "rejected unique-row shape preserves all pending leaves");

    batch.backup(
        selection.token,
        policies.data(),
        selection.size(),
        kb_pente::kMaxActions,
        values.data(),
        values.size());
    expect(batch.complete(), "mixed-ruleset deduplication fixture completes");
    const auto telemetry = batch.deduplication_telemetry();
    const kb_pente::DeduplicationStats expected{
        1U,
        2U,
        2U,
        0U,
        0.0,
    };
    expect(telemetry.cumulative == expected && telemetry.last_wave == expected,
           "mixed-ruleset telemetry reports no eliminated evaluations");
}

void test_deduplication_worker_equivalence() {
    kb_pente::SearchBatch one_worker(batch_config(4U, 37U), 2U, 1U);
    kb_pente::SearchBatch many_workers(batch_config(4U, 37U), 2U, 4U);
    for (kb_pente::SearchBatch* batch : {&one_worker, &many_workers}) {
        (void)batch->add(
            kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
        (void)batch->add(
            kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    }

    std::size_t waves = 0U;
    while (!one_worker.complete()) {
        const kb_pente::Selection one = one_worker.select();
        const kb_pente::Selection many = many_workers.select();
        expect(one.size() == many.size() && one.raw_size() == many.raw_size(),
               "worker counts agree on deduplicated request shapes");
        backup_uniform(one_worker, one);
        backup_uniform(many_workers, many);
        ++waves;
        if (waves > 128U) {
            throw TestFailure("deduplicated worker fixture did not complete");
        }
    }

    expect(many_workers.complete(), "both deduplicated worker batches complete");
    expect(one_worker.deduplication_telemetry() ==
               many_workers.deduplication_telemetry(),
           "worker counts agree on duplicate telemetry");
    for (std::size_t slot = 0U; slot < 2U; ++slot) {
        expect(one_worker.root_position(slot) == many_workers.root_position(slot),
               "worker counts agree on deduplicated root positions");
        expect(one_worker.root_policy(slot) == many_workers.root_policy(slot),
               "worker counts agree on deduplicated root policies");
    }
    expect(one_worker.deduplication_telemetry()
                   .cumulative.eliminated_duplicate_evaluations > 0U,
           "duplicate worker fixture records eliminated evaluations");
}

void test_deduplication_multiwave_split_and_merge() {
    kb_pente::SearchBatch batch(batch_config(1U, 41U), 2U, 2U);
    const auto first_slot = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto second_slot = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);

    const auto first = batch.select();
    expect(first.size() == 1U && first.raw_size() == 2U,
           "split-and-merge fixture starts with a collapsed wave");
    backup_uniform(batch, first);

    (void)batch.advance_root(first_slot, 0U);
    const auto split = batch.select();
    expect(split.size() == 1U && split.raw_size() == 1U &&
               split[0].slot_id() == first_slot,
           "advancing one completed slot produces a distinct wave");
    backup_uniform(batch, split);

    batch.replace_root(
        first_slot,
        kb_pente::Position::initial(5),
        kb_pente::Ruleset::Freestyle);
    batch.replace_root(
        second_slot,
        kb_pente::Position::initial(5),
        kb_pente::Ruleset::Freestyle);
    const auto merged = batch.select();
    expect(merged.size() == 1U && merged.raw_size() == 2U,
           "replaced identical roots merge again on a later wave");
    backup_uniform(batch, merged);
    expect(batch.complete(), "split-and-merge fixture completes");

    const auto telemetry = batch.deduplication_telemetry();
    const kb_pente::DeduplicationStats cumulative{
        3U,
        5U,
        3U,
        2U,
        0.4,
    };
    const kb_pente::DeduplicationStats last_wave{
        1U,
        2U,
        1U,
        1U,
        0.5,
    };
    expect(telemetry.cumulative == cumulative && telemetry.last_wave == last_wave,
           "split-and-merge telemetry tracks every wave");
}

void test_backup_validation_and_retry() {
    kb_pente::SearchBatch batch(batch_config(2U), 1U, 1U);
    (void)batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const kb_pente::Selection selection = batch.select();
    std::vector<float> valid_policy = uniform_policies(selection.size());
    std::vector<float> valid_values(selection.size(), 0.0F);
    const auto before = batch.slot_telemetry(0U);
    const auto token = selection.token;

    auto expect_retryable = [&](auto&& backup, const char* message) {
        expect_throws<std::invalid_argument>(backup, message);
        expect(batch.has_pending() && batch.pending_token() == token,
               "rejected backup preserves pending token");
        expect(batch.slot_telemetry(0U) == before,
               "rejected backup preserves session state");
    };

    expect_retryable(
        [&] {
            batch.backup(
                token, valid_policy.data(), 0U, kb_pente::kMaxActions,
                valid_values.data(), 1U);
        },
        "policy row mismatch is rejected");
    expect_retryable(
        [&] {
            batch.backup(
                token, valid_policy.data(), 1U, kb_pente::kMaxActions - 1U,
                valid_values.data(), 1U);
        },
        "policy stride mismatch is rejected");
    expect_retryable(
        [&] {
            batch.backup(
                token, nullptr, 1U, kb_pente::kMaxActions,
                valid_values.data(), 1U);
        },
        "null policy is rejected");
    expect_retryable(
        [&] {
            batch.backup(
                token, valid_policy.data(), 1U, kb_pente::kMaxActions,
                nullptr, 1U);
        },
        "null values are rejected");
    expect_retryable(
        [&] {
            batch.backup(
                token - 1U, valid_policy.data(), 1U, kb_pente::kMaxActions,
                valid_values.data(), 1U);
        },
        "stale token is rejected");

    auto negative_policy = valid_policy;
    negative_policy[0] = -1.0F;
    expect_retryable(
        [&] {
            batch.backup(
                token, negative_policy.data(), 1U, kb_pente::kMaxActions,
                valid_values.data(), 1U);
        },
        "negative policy values are rejected");
    auto nan_policy = valid_policy;
    nan_policy[0] = std::numeric_limits<float>::quiet_NaN();
    expect_retryable(
        [&] {
            batch.backup(
                token, nan_policy.data(), 1U, kb_pente::kMaxActions,
                valid_values.data(), 1U);
        },
        "NaN policy values are rejected");
    auto infinite_policy = valid_policy;
    infinite_policy[0] = std::numeric_limits<float>::infinity();
    expect_retryable(
        [&] {
            batch.backup(
                token, infinite_policy.data(), 1U, kb_pente::kMaxActions,
                valid_values.data(), 1U);
        },
        "infinite policy values are rejected");

    auto out_of_range_values = valid_values;
    out_of_range_values[0] = 1.01F;
    expect_retryable(
        [&] {
            batch.backup(
                token, valid_policy.data(), 1U, kb_pente::kMaxActions,
                out_of_range_values.data(), 1U);
        },
        "out-of-range values are rejected");
    auto nan_values = valid_values;
    nan_values[0] = std::numeric_limits<float>::quiet_NaN();
    expect_retryable(
        [&] {
            batch.backup(
                token, valid_policy.data(), 1U, kb_pente::kMaxActions,
                nan_values.data(), 1U);
        },
        "NaN values are rejected");
    auto infinite_values = valid_values;
    infinite_values[0] = std::numeric_limits<float>::infinity();
    expect_retryable(
        [&] {
            batch.backup(
                token, valid_policy.data(), 1U, kb_pente::kMaxActions,
                infinite_values.data(), 1U);
        },
        "infinite values are rejected");

    batch.backup(
        token, valid_policy.data(), 1U, kb_pente::kMaxActions,
        valid_values.data(), 1U);
    expect_throws<std::logic_error>(
        [&batch, token, &valid_policy, &valid_values] {
            batch.backup(
                token, valid_policy.data(), 1U, kb_pente::kMaxActions,
                valid_values.data(), 1U);
        },
        "duplicate backup is rejected");
}

void test_terminal_progress_and_completed_slots() {
    kb_pente::SearchBatch batch(batch_config(4U), 2U, 2U);
    const auto terminal_slot = batch.add(
        make_draw_root(), kb_pente::Ruleset::Freestyle);
    const auto ordinary_slot = batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const kb_pente::Selection first = batch.select();
    expect(first.size() == 2U, "terminal and ordinary roots request together");
    expect(first[0].slot_id() == terminal_slot &&
               first[1].slot_id() == ordinary_slot,
           "terminal requests preserve slot order");
    backup_first_one_hot(batch, first, 24U);

    const kb_pente::Selection second = batch.select();
    expect(second.size() == 1U && second[0].slot_id() == ordinary_slot,
           "terminal progress skips its completed slot");
    const auto terminal_telemetry = batch.slot_telemetry(terminal_slot);
    expect(terminal_telemetry.completed_simulations == 4U,
           "terminal-only slot reaches its exact budget");
    expect(terminal_telemetry.evaluator_completions == 1U,
           "terminal-only slot uses one evaluator request");
    expect(terminal_telemetry.terminal_simulations == 3U,
           "terminal-only slot resolves remaining leaves internally");
    expect(batch.slot_complete(terminal_slot),
           "terminal-only slot is complete after internal progress");
    backup_uniform(batch, second);
    run_to_completion(batch);

    expect(batch.complete(), "all active slots eventually complete");
    const auto ordinary_telemetry = batch.slot_telemetry(ordinary_slot);
    expect(ordinary_telemetry.completed_simulations == 4U,
           "ordinary slot reaches its exact budget");

    expect_throws<std::logic_error>(
        [&batch] { (void)batch.select(); },
        "selection after all slots complete is rejected");
    expect(!batch.has_pending(), "completed batch has no pending state");

    kb_pente::SearchBatch terminal_only(batch_config(4U), 1U, 1U);
    (void)terminal_only.add(
        make_draw_root(), kb_pente::Ruleset::Freestyle);
    const kb_pente::Selection terminal_first = terminal_only.select();
    backup_first_one_hot(terminal_only, terminal_first, 24U);
    const kb_pente::Selection terminal_completion = terminal_only.select();
    expect(terminal_completion.empty() && terminal_only.complete(),
           "incomplete terminal progress may return an empty selection");
    expect(terminal_completion.begin() == nullptr &&
               terminal_completion.end() == nullptr,
           "empty selection exposes a null empty range");
    std::size_t empty_range_count = 0U;
    for (const auto& request : terminal_completion) {
        (void)request;
        ++empty_range_count;
    }
    expect(empty_range_count == 0U, "empty selection range has no requests");
    const auto empty_wave_telemetry = terminal_only.deduplication_telemetry();
    expect(empty_wave_telemetry.cumulative.selection_waves == 2U &&
               empty_wave_telemetry.cumulative.raw_evaluation_requests == 1U &&
               empty_wave_telemetry.cumulative.unique_evaluations == 1U &&
               empty_wave_telemetry.cumulative.eliminated_duplicate_evaluations ==
                   0U &&
               empty_wave_telemetry.cumulative.duplicate_leaf_rate == 0.0 &&
               empty_wave_telemetry.last_wave.selection_waves == 1U &&
               empty_wave_telemetry.last_wave.raw_evaluation_requests == 0U &&
               empty_wave_telemetry.last_wave.unique_evaluations == 0U &&
               empty_wave_telemetry.last_wave.eliminated_duplicate_evaluations ==
                   0U &&
               empty_wave_telemetry.last_wave.duplicate_leaf_rate == 0.0,
           "terminal-only empty waves update duplicate telemetry");
}

void test_worker_equivalence_and_telemetry() {
    kb_pente::SearchBatch one_worker(batch_config(8U, 19U), 3U, 1U);
    kb_pente::SearchBatch many_workers(batch_config(8U, 19U), 3U, 4U);
    for (const auto board_size : {5U, 9U, 19U}) {
        const auto root = kb_pente::Position::initial(
            static_cast<std::uint8_t>(board_size));
        (void)one_worker.add(root, kb_pente::Ruleset::Freestyle);
        (void)many_workers.add(root, kb_pente::Ruleset::Freestyle);
    }
    run_to_completion(one_worker);
    run_to_completion(many_workers);

    expect(one_worker.complete() && many_workers.complete(),
           "both worker configurations complete");
    for (std::size_t slot = 0U; slot < 3U; ++slot) {
        expect(one_worker.root_position(slot) == many_workers.root_position(slot),
               "worker count does not change root positions");
        expect(one_worker.slot_telemetry(slot) == many_workers.slot_telemetry(slot),
               "worker count does not change deterministic telemetry");
    }
}

void test_two_row_backup_validation_is_transactional() {
    kb_pente::SearchBatch batch(batch_config(2U), 2U, 2U);
    (void)batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    (void)batch.add(
        kb_pente::Position::initial(9), kb_pente::Ruleset::Freestyle);
    const kb_pente::Selection selection = batch.select();
    expect(selection.size() == 2U, "transactional fixture has two requests");

    std::vector<float> policies = uniform_policies(selection.size());
    std::vector<float> values(selection.size(), 0.0F);
    policies[kb_pente::kMaxActions] = -1.0F;
    const auto first_before = batch.slot_telemetry(0U);
    const auto second_before = batch.slot_telemetry(1U);
    expect_throws<std::invalid_argument>(
        [&] {
            batch.backup(
                selection.token,
                policies.data(),
                selection.size(),
                kb_pente::kMaxActions,
                values.data(),
                values.size());
        },
        "a later invalid policy row rejects the entire wave");
    expect(batch.slot_telemetry(0U) == first_before &&
               batch.slot_telemetry(1U) == second_before,
           "whole-wave validation mutates neither slot");
    expect(batch.has_pending(), "transactional rejection keeps pending state");

    policies[kb_pente::kMaxActions] = 1.0F;
    batch.backup(
        selection.token,
        policies.data(),
        selection.size(),
        kb_pente::kMaxActions,
        values.data(),
        values.size());
    expect(batch.slot_telemetry(0U).completed_simulations == 1U &&
               batch.slot_telemetry(1U).completed_simulations == 1U,
           "corrected transactional backup advances both slots");
}

void test_multiwave_token_lifecycle() {
    kb_pente::SearchBatch batch(batch_config(1U), 1U, 1U);
    (void)batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    const auto first = batch.select();
    expect(first.token == 1U && first.size() == 1U,
           "first one-simulation wave requests evaluation");
    backup_uniform(batch, first);
    expect(batch.complete(), "one-simulation batch completes after backup");
    expect_throws<std::logic_error>(
        [&batch] { (void)batch.select(); },
        "selection after completion is rejected");
    expect(batch.last_token() == first.token,
           "rejected completed selection does not advance the token");
    expect_throws<std::logic_error>(
        [&batch, &first] {
            batch.backup(
                first.token, nullptr, 0U, kb_pente::kMaxActions, nullptr, 0U);
        },
        "backup without a pending request is rejected");
}

void expect_finite_timing_stage(
    const kb_pente::SearchBatchStageTelemetry& stage,
    const char* message) {
    expect(std::isfinite(stage.wall_seconds) &&
               std::isfinite(stage.worker.wall_seconds) &&
               std::isfinite(stage.worker.callback_busy_seconds) &&
               std::isfinite(stage.worker.busy_fraction),
           message);
    expect(stage.wall_seconds >= 0.0 && stage.worker.wall_seconds >= 0.0 &&
               stage.worker.callback_busy_seconds >= 0.0 &&
               stage.worker.busy_fraction >= 0.0 &&
               stage.worker.busy_fraction <= 1.0,
           message);
}

void test_search_batch_timing_telemetry() {
    kb_pente::SearchBatch batch(batch_config(2U, 101U), 1U, 2U);
    (void)batch.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);

    const auto initial = batch.timing_telemetry();
    expect(initial.cumulative.token == kb_pente::kInvalidBatchToken &&
               initial.latest_generation.token == kb_pente::kInvalidBatchToken,
           "timing telemetry starts without a generation");
    expect(initial.cumulative.select.successful_operations == 0U &&
               initial.cumulative.dedup.successful_operations == 0U &&
               initial.cumulative.features.successful_operations == 0U &&
               initial.cumulative.backup.successful_operations == 0U,
           "timing telemetry starts with zero successful operations");

    const auto first = batch.select();
    const auto after_select = batch.timing_telemetry();
    expect(after_select.latest_generation.token == first.token &&
               after_select.cumulative.token == first.token,
           "selection publishes the generation token");
    expect(after_select.latest_generation.select.successful_operations == 1U &&
               after_select.latest_generation.dedup.successful_operations == 1U &&
               after_select.latest_generation.features.successful_operations ==
                   0U &&
               after_select.latest_generation.backup.successful_operations ==
                   0U,
           "selection publishes select and dedup timing only");
    expect(after_select.latest_generation.select.worker.items == 1U &&
               after_select.latest_generation.select.worker.workers == 2U &&
               after_select.latest_generation.dedup.worker.items == 0U &&
               after_select.latest_generation.dedup.worker.workers == 0U,
           "select and serial dedup worker metrics are separated");
    expect_finite_timing_stage(
        after_select.latest_generation.select,
        "select timing is finite and bounded");
    expect_finite_timing_stage(
        after_select.latest_generation.dedup,
        "dedup timing is finite and bounded");

    std::vector<float> features(4U * 25U, 0.0F);
    const auto before_rejected_feature = batch.timing_telemetry();
    expect_throws<std::invalid_argument>(
        [&batch, &first, &features] {
            batch.write_features(
                first.token,
                features.data(),
                2U,
                4U,
                5U,
                5U);
        },
        "rejected feature timing call throws");
    expect(batch.timing_telemetry() == before_rejected_feature,
           "rejected feature calls preserve timing telemetry");

    batch.write_features(
        first.token,
        features.data(),
        1U,
        4U,
        5U,
        5U);
    const auto after_first_feature = batch.timing_telemetry();
    batch.write_features(
        first.token,
        features.data(),
        1U,
        4U,
        5U,
        5U);
    const auto after_second_feature = batch.timing_telemetry();
    expect(after_second_feature.latest_generation.features.successful_operations ==
               2U &&
               after_second_feature.cumulative.features.successful_operations ==
                   2U &&
               after_second_feature.latest_generation.features.worker.items ==
                   2U,
           "repeated feature writes accumulate in the generation");
    expect(after_second_feature.latest_generation.select ==
               after_first_feature.latest_generation.select &&
               after_second_feature.latest_generation.dedup ==
                   after_first_feature.latest_generation.dedup,
           "feature timing does not rewrite select or dedup snapshots");
    expect_finite_timing_stage(
        after_second_feature.latest_generation.features,
        "feature timing is finite and bounded");

    std::vector<float> policies = uniform_policies(first.size());
    std::vector<float> values(first.size(), 0.0F);
    const auto before_rejected_backup = batch.timing_telemetry();
    expect_throws<std::invalid_argument>(
        [&batch, &first, &values] {
            batch.backup(
                first.token,
                nullptr,
                first.size(),
                kb_pente::kMaxActions,
                values.data(),
                values.size());
        },
        "rejected backup timing call throws");
    expect(batch.timing_telemetry() == before_rejected_backup,
           "rejected backup calls preserve timing telemetry");

    batch.backup(
        first.token,
        policies.data(),
        first.size(),
        kb_pente::kMaxActions,
        values.data(),
        values.size());
    const auto after_backup = batch.timing_telemetry();
    expect(after_backup.latest_generation.backup.successful_operations == 1U &&
               after_backup.latest_generation.backup.worker.items == 1U &&
               after_backup.cumulative.select.successful_operations == 1U &&
               after_backup.cumulative.dedup.successful_operations == 1U &&
               after_backup.cumulative.features.successful_operations == 2U &&
               after_backup.cumulative.backup.successful_operations == 1U,
           "successful backup publishes and accumulates timing");
    expect_finite_timing_stage(
        after_backup.latest_generation.backup,
        "backup timing is finite and bounded");

    const auto second = batch.select();
    const auto after_second_select = batch.timing_telemetry();
    expect(second.token > first.token &&
               after_second_select.latest_generation.token == second.token &&
               after_second_select.latest_generation.select.successful_operations ==
                   1U &&
               after_second_select.latest_generation.dedup.successful_operations ==
                   1U &&
               after_second_select.latest_generation.features.successful_operations ==
                   0U &&
               after_second_select.latest_generation.backup.successful_operations ==
                   0U,
           "a new generation resets latest stage counters");
    expect(after_second_select.cumulative.select.successful_operations == 2U &&
               after_second_select.cumulative.dedup.successful_operations == 2U &&
               after_second_select.cumulative.features.successful_operations ==
                   2U &&
               after_second_select.cumulative.backup.successful_operations == 1U,
           "a new generation preserves cumulative stage counters");
    expect_finite_timing_stage(
        after_second_select.latest_generation.select,
        "second select timing is finite and bounded");

    backup_uniform(batch, second);
    expect(batch.complete(), "timing fixture completes");

    kb_pente::SearchBatch terminal(batch_config(4U, 103U), 1U, 1U);
    (void)terminal.add(make_draw_root(), kb_pente::Ruleset::Freestyle);
    const auto terminal_first = terminal.select();
    backup_first_one_hot(terminal, terminal_first, 24U);
    const auto terminal_empty = terminal.select();
    const auto terminal_timing = terminal.timing_telemetry();
    expect(terminal_empty.empty() && terminal.complete() &&
               terminal_timing.latest_generation.token == terminal_empty.token &&
               terminal_timing.latest_generation.select.successful_operations ==
                   1U &&
               terminal_timing.latest_generation.dedup.successful_operations ==
                   1U &&
               terminal_timing.latest_generation.features.successful_operations ==
                   0U &&
               terminal_timing.latest_generation.backup.successful_operations ==
                   0U,
           "terminal-only generation publishes select and dedup timing");
    expect_finite_timing_stage(
        terminal_timing.latest_generation.select,
        "terminal-only select timing is finite and bounded");
    expect_finite_timing_stage(
        terminal_timing.latest_generation.dedup,
        "terminal-only dedup timing is finite and bounded");
}

}  // namespace

int main() {
    try {
        test_construction_and_admission();
        test_selection_order_and_request_views();
        test_deduplication_and_fanout();
        test_deduplication_respects_ruleset_and_backup_shape();
        test_deduplication_worker_equivalence();
        test_deduplication_multiwave_split_and_merge();
        test_backup_validation_and_retry();
        test_terminal_progress_and_completed_slots();
        test_worker_equivalence_and_telemetry();
        test_two_row_backup_validation_is_transactional();
        test_multiwave_token_lifecycle();
        test_root_policy_and_advancement();
        test_allocated_root_advancement_and_continuation();
        test_terminal_advancement_and_slot_removal();
        test_lifecycle_rejection_and_lowest_free_reuse();
        test_strong_replacement_and_seed_progression();
        test_worker_equivalence_after_lifecycle_changes();
        test_search_batch_timing_telemetry();
    } catch (const std::exception& failure) {
        std::cerr << "FAIL: " << failure.what() << '\n';
        return 1;
    }
    std::cout << "SearchBatch tests passed\n";
    return 0;
}
