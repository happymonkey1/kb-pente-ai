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

}  // namespace

int main() {
    try {
        test_construction_and_admission();
        test_selection_order_and_request_views();
        test_backup_validation_and_retry();
        test_terminal_progress_and_completed_slots();
        test_worker_equivalence_and_telemetry();
        test_two_row_backup_validation_is_transactional();
        test_multiwave_token_lifecycle();
    } catch (const std::exception& failure) {
        std::cerr << "FAIL: " << failure.what() << '\n';
        return 1;
    }
    std::cout << "SearchBatch tests passed\n";
    return 0;
}
