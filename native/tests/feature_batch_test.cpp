#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "kb_pente/features.h"
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
    std::uint32_t simulation_budget = 1U,
    std::uint64_t seed = 73U) {
    return kb_pente::SearchConfig(
        1.5F, simulation_budget, 0.0F, 0.03F, seed);
}

kb_pente::Position make_capture_position() {
    kb_pente::Position position = kb_pente::Position::initial(5);
    position.stones[0] = static_cast<std::int8_t>(kb_pente::Player::One);
    position.stones[1] = static_cast<std::int8_t>(kb_pente::Player::Two);
    position.stones[6] = static_cast<std::int8_t>(kb_pente::Player::One);
    position.captures[kb_pente::player_index(kb_pente::Player::One)] = 2U;
    position.captures[kb_pente::player_index(kb_pente::Player::Two)] = 4U;
    position.ply = 15U;
    position.last_action = 6U;
    position.current_player = kb_pente::Player::Two;
    position.validate();
    position.refresh_hash();
    return position;
}

kb_pente::Position make_single_stone(kb_pente::Action action) {
    kb_pente::Position position = kb_pente::Position::initial(5);
    position.stones[action] =
        static_cast<std::int8_t>(kb_pente::Player::One);
    position.ply = 1U;
    position.last_action = action;
    position.current_player = kb_pente::Player::Two;
    position.validate();
    position.refresh_hash();
    return position;
}

kb_pente::Position make_near_draw_position() {
    constexpr std::array<std::array<std::int8_t, 5>, 5> pattern{{
        {{1, 1, -1, -1, 1}},
        {{-1, -1, 1, 1, -1}},
        {{1, 1, -1, -1, 1}},
        {{-1, -1, 1, 1, -1}},
        {{1, -1, 1, -1, 1}},
    }};

    kb_pente::Position position = kb_pente::Position::initial(5);
    for (std::uint8_t row = 0U; row < 5U; ++row) {
        for (std::uint8_t column = 0U; column < 5U; ++column) {
            const auto action = static_cast<kb_pente::Action>(
                row * 5U + column);
            position.stones[action] = pattern[row][column];
        }
    }
    position.stones[24] = 0;
    position.ply = 24U;
    position.last_action = kb_pente::kInvalidAction;
    position.current_player = kb_pente::Player::One;
    position.validate();
    position.refresh_hash();
    expect(!kb_pente::check_terminal(position).is_terminal(),
           "near-draw feature fixture is nonterminal");
    return position;
}

std::size_t feature_area(std::size_t board_size) {
    return board_size * board_size;
}

std::size_t feature_row_size(std::size_t board_size) {
    return 4U * feature_area(board_size);
}

std::vector<float> guarded_storage(
    std::size_t rows,
    std::size_t board_size,
    float canary) {
    return std::vector<float>(
        rows * feature_row_size(board_size) + 2U,
        canary);
}

void expect_guarded_storage(
    const std::vector<float>& storage,
    float canary,
    const char* message) {
    expect(storage.front() == canary, message);
    expect(storage.back() == canary, message);
}

std::vector<float> guarded_body(const std::vector<float>& storage) {
    return std::vector<float>(storage.begin() + 1U, storage.end() - 1U);
}

std::vector<float> uniform_policies(std::size_t rows) {
    return std::vector<float>(rows * kb_pente::kMaxActions, 1.0F);
}

void backup_uniform(
    kb_pente::SearchBatch& batch,
    const kb_pente::Selection& selection) {
    expect(!selection.empty(), "uniform backup requires a pending row");
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

void backup_one_hot(
    kb_pente::SearchBatch& batch,
    const kb_pente::Selection& selection,
    kb_pente::Action action) {
    expect(!selection.empty(), "one-hot backup requires a pending row");
    std::vector<float> policies(
        selection.size() * kb_pente::kMaxActions,
        0.0F);
    for (std::size_t row = 0U; row < selection.size(); ++row) {
        policies[row * kb_pente::kMaxActions + action] = 1.0F;
    }
    std::vector<float> values(selection.size(), 0.0F);
    batch.backup(
        selection.token,
        policies.data(),
        selection.size(),
        kb_pente::kMaxActions,
        values.data(),
        values.size());
}

void test_exact_layout_repeatability_and_capacity() {
    kb_pente::SearchBatch batch(batch_config(), 1U, 2U);
    const auto slot = batch.add(
        make_capture_position(), kb_pente::Ruleset::Freestyle);
    const auto selection = batch.select();
    expect(selection.size() == 1U && selection.raw_size() == 1U,
           "one capture fixture produces one unique feature row");

    const auto raw_capacity = batch.raw_request_capacity();
    const auto unique_capacity = batch.unique_request_capacity();
    constexpr float canary = -91.25F;
    constexpr std::size_t board_size = 5U;
    std::vector<float> storage = guarded_storage(1U, board_size, canary);
    batch.write_features(
        selection.token,
        storage.data() + 1U,
        1U,
        4U,
        board_size,
        board_size);

    constexpr std::size_t area = 25U;
    const auto& position = batch.root_position(slot);
    const auto current = static_cast<std::int8_t>(position.current_player);
    const auto opposing = static_cast<std::int8_t>(
        kb_pente::opponent(position.current_player));
    for (std::size_t action = 0U; action < area; ++action) {
        expect(
            storage[1U + action] ==
                (position.stones[action] == current ? 1.0F : 0.0F),
            "current-player plane has exact row-major layout");
        expect(
            storage[1U + area + action] ==
                (position.stones[action] == opposing ? 1.0F : 0.0F),
            "opponent plane has exact row-major layout");
        expect(storage[1U + 2U * area + action] == 0.8F,
               "current capture plane uses the relative perspective");
        expect(storage[1U + 3U * area + action] == 0.4F,
               "opponent capture plane uses the relative perspective");
    }
    expect_guarded_storage(storage, canary, "feature writer respects canaries");
    const auto expected = guarded_body(storage);

    std::fill(storage.begin() + 1U, storage.end() - 1U, 7.0F);
    batch.write_features(
        selection.token,
        storage.data() + 1U,
        1U,
        4U,
        board_size,
        board_size);
    expect(guarded_body(storage) == expected,
           "successful feature writes are repeatable before backup");
    expect_guarded_storage(storage, canary, "repeated write preserves canaries");
    expect(batch.pending_request_count() == 1U &&
               batch.pending_selected_count() == 1U &&
               batch.raw_request_capacity() == raw_capacity &&
               batch.unique_request_capacity() == unique_capacity,
           "feature writes preserve pending state and capacities");

    backup_one_hot(batch, selection, 2U);
    expect(batch.complete(), "layout fixture completes after backup");
}

void test_deduplicated_representative_order() {
    kb_pente::SearchBatch batch(batch_config(), 3U, 2U);
    const auto first_slot = batch.add(
        make_single_stone(0U), kb_pente::Ruleset::Freestyle);
    const auto second_slot = batch.add(
        make_single_stone(1U), kb_pente::Ruleset::Freestyle);
    const auto duplicate_slot = batch.add(
        make_single_stone(0U), kb_pente::Ruleset::Freestyle);
    const auto selection = batch.select();
    expect(selection.size() == 2U && selection.raw_size() == 3U,
           "duplicate roots produce two rows for three raw leaves");
    expect(selection[0].slot_id() == first_slot &&
               selection[1].slot_id() == second_slot,
           "feature rows follow lowest representative slot order");
    expect(duplicate_slot != first_slot && duplicate_slot != second_slot,
           "duplicate fixture uses a distinct raw slot");

    constexpr float canary = -37.5F;
    constexpr std::size_t board_size = 5U;
    std::vector<float> storage = guarded_storage(2U, board_size, canary);
    batch.write_features(
        selection.token,
        storage.data() + 1U,
        2U,
        4U,
        board_size,
        board_size);
    std::vector<float> expected_first(feature_row_size(board_size), 0.0F);
    std::vector<float> expected_second(feature_row_size(board_size), 0.0F);
    kb_pente::write_features(
        make_single_stone(0U), expected_first.data());
    kb_pente::write_features(
        make_single_stone(1U), expected_second.data());
    expect(std::equal(
               expected_first.begin(),
               expected_first.end(),
               storage.begin() + 1U),
           "first feature row uses the lowest-slot representative");
    expect(std::equal(
               expected_second.begin(),
               expected_second.end(),
               storage.begin() + 1U + feature_row_size(board_size)),
           "second feature row remains in representative order");
    expect_guarded_storage(storage, canary,
                           "deduplicated rows respect canaries");

    backup_uniform(batch, selection);
    expect(batch.complete(), "deduplication feature fixture completes");
}

void test_validation_retry_and_bounds() {
    kb_pente::SearchBatch batch(batch_config(), 1U, 2U);
    (void)batch.add(
        make_single_stone(0U), kb_pente::Ruleset::Freestyle);
    const auto selection = batch.select();
    constexpr float canary = -12.75F;
    constexpr std::size_t board_size = 5U;
    std::vector<float> storage = guarded_storage(1U, board_size, canary);

    expect_throws<std::invalid_argument>(
        [&batch, &selection, &storage] {
            batch.write_features(
                selection.token + 1U,
                storage.data() + 1U,
                1U,
                4U,
                5U,
                5U);
        },
        "stale feature token is rejected");
    expect_guarded_storage(storage, canary,
                           "stale token does not write feature storage");

    expect_throws<std::invalid_argument>(
        [&batch, &selection] {
            batch.write_features(selection.token, nullptr, 1U, 4U, 5U, 5U);
        },
        "null feature storage is rejected");

    const auto reject_shape =
        [&batch, &selection, &storage](
            std::size_t rows,
            std::size_t planes,
            std::size_t height,
            std::size_t width,
            const char* message) {
            const auto before = storage;
            expect_throws<std::invalid_argument>(
                [&batch, &selection, &storage, rows, planes, height, width] {
                    batch.write_features(
                        selection.token,
                        storage.data() + 1U,
                        rows,
                        planes,
                        height,
                        width);
                },
                message);
            expect(storage == before,
                   "shape rejection leaves storage unchanged");
            expect(batch.has_pending() && batch.pending_request_count() == 1U,
                   "invalid feature calls preserve pending state");
        };

    reject_shape(2U, 4U, 5U, 5U, "wrong feature row count is rejected");
    reject_shape(1U, 3U, 5U, 5U, "wrong feature plane count is rejected");
    reject_shape(1U, 4U, 5U, 9U, "non-square feature shape is rejected");
    reject_shape(1U, 4U, 4U, 4U, "unsupported feature board is rejected");
    reject_shape(
        1U,
        4U,
        9U,
        9U,
        "representative board-size mismatch is rejected");
    reject_shape(
        std::numeric_limits<std::size_t>::max(),
        4U,
        5U,
        5U,
        "oversized feature row count is rejected safely");
    reject_shape(
        1U,
        4U,
        std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max(),
        "oversized feature dimensions are rejected safely");

    batch.write_features(
        selection.token,
        storage.data() + 1U,
        1U,
        4U,
        board_size,
        board_size);
    expect_guarded_storage(storage, canary,
                           "valid feature write remains available after errors");
    backup_uniform(batch, selection);
    expect(batch.complete(), "validation fixture completes after retry");

    kb_pente::SearchBatch heterogeneous(batch_config(1U, 77U), 2U, 2U);
    (void)heterogeneous.add(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle);
    (void)heterogeneous.add(
        kb_pente::Position::initial(9), kb_pente::Ruleset::Freestyle);
    const auto mixed_selection = heterogeneous.select();
    expect(mixed_selection.size() == 2U && mixed_selection.raw_size() == 2U,
           "mixed-size SearchBatch waves remain selectable");
    std::vector<float> mixed_storage = guarded_storage(2U, 5U, canary);
    const auto mixed_before = mixed_storage;
    expect_throws<std::invalid_argument>(
        [&heterogeneous, &mixed_selection, &mixed_storage] {
            heterogeneous.write_features(
                mixed_selection.token,
                mixed_storage.data() + 1U,
                2U,
                4U,
                5U,
                5U);
        },
        "heterogeneous pending boards are rejected before encoding");
    expect(mixed_storage == mixed_before,
           "mixed-size rejection leaves all destination storage unchanged");
    expect(heterogeneous.has_pending() &&
               heterogeneous.pending_request_count() == 2U &&
               heterogeneous.pending_selected_count() == 2U,
           "mixed-size rejection preserves the pending wave");
    backup_uniform(heterogeneous, mixed_selection);
    expect(heterogeneous.complete(),
           "mixed-size SearchBatch remains usable after rejection");
}

void test_terminal_wave_and_multiwave_reuse() {
    kb_pente::SearchBatch terminal_batch(batch_config(4U, 79U), 1U, 2U);
    (void)terminal_batch.add(
        make_near_draw_position(), kb_pente::Ruleset::Freestyle);
    const auto first = terminal_batch.select();
    constexpr float canary = -5.0F;
    std::vector<float> terminal_storage = guarded_storage(1U, 5U, canary);
    terminal_batch.write_features(
        first.token,
        terminal_storage.data() + 1U,
        1U,
        4U,
        5U,
        5U);
    backup_one_hot(terminal_batch, first, 24U);
    const auto empty = terminal_batch.select();
    expect(empty.empty() && terminal_batch.complete(),
           "terminal-only wave produces an empty completed selection");
    const auto before_empty_write = terminal_storage;
    expect_throws<std::logic_error>(
        [&terminal_batch, &empty, &terminal_storage] {
            terminal_batch.write_features(
                empty.token,
                terminal_storage.data() + 1U,
                0U,
                4U,
                5U,
                5U);
        },
        "feature writing without pending evaluator rows is rejected");
    expect(terminal_storage == before_empty_write,
           "empty terminal wave does not write storage");

    kb_pente::SearchBatch multiwave(batch_config(2U, 83U), 1U, 2U);
    (void)multiwave.add(
        make_single_stone(0U), kb_pente::Ruleset::Freestyle);
    const auto first_wave = multiwave.select();
    std::vector<float> first_storage = guarded_storage(1U, 5U, canary);
    multiwave.write_features(
        first_wave.token,
        first_storage.data() + 1U,
        1U,
        4U,
        5U,
        5U);
    const auto first_body = guarded_body(first_storage);
    backup_one_hot(multiwave, first_wave, 2U);
    const auto second_wave = multiwave.select();
    multiwave.write_features(
        second_wave.token,
        first_storage.data() + 1U,
        1U,
        4U,
        5U,
        5U);
    expect(guarded_body(first_storage) != first_body,
           "multiwave encoding follows the new representative position");
    expect_guarded_storage(first_storage, canary,
                           "multiwave reuse respects storage bounds");
    backup_uniform(multiwave, second_wave);
    expect(multiwave.complete(), "multiwave feature fixture completes");
}

void test_worker_equivalence() {
    kb_pente::SearchBatch one_worker(batch_config(), 3U, 1U);
    kb_pente::SearchBatch many_workers(batch_config(), 3U, 4U);
    for (const auto action : {0U, 1U, 2U}) {
        const auto root = make_single_stone(
            static_cast<kb_pente::Action>(action));
        (void)one_worker.add(root, kb_pente::Ruleset::Freestyle);
        (void)many_workers.add(root, kb_pente::Ruleset::Freestyle);
    }

    const auto one_selection = one_worker.select();
    const auto many_selection = many_workers.select();
    expect(one_selection.size() == 3U && many_selection.size() == 3U,
           "worker fixture exposes all unique rows");
    constexpr float canary = -44.0F;
    std::vector<float> one_storage = guarded_storage(3U, 5U, canary);
    std::vector<float> many_storage = guarded_storage(3U, 5U, canary);
    one_worker.write_features(
        one_selection.token,
        one_storage.data() + 1U,
        3U,
        4U,
        5U,
        5U);
    many_workers.write_features(
        many_selection.token,
        many_storage.data() + 1U,
        3U,
        4U,
        5U,
        5U);
    expect(guarded_body(one_storage) == guarded_body(many_storage),
           "one-worker and multiworker feature layouts match");
    expect_guarded_storage(one_storage, canary,
                           "one-worker feature encoding respects canaries");
    expect_guarded_storage(many_storage, canary,
                           "multiworker feature encoding respects canaries");
    backup_uniform(one_worker, one_selection);
    backup_uniform(many_workers, many_selection);
    expect(one_worker.complete() && many_workers.complete(),
           "worker-equivalent feature fixtures complete");
}

}  // namespace

int main() {
    try {
        test_exact_layout_repeatability_and_capacity();
        test_deduplicated_representative_order();
        test_validation_retry_and_bounds();
        test_terminal_wave_and_multiwave_reuse();
        test_worker_equivalence();
    } catch (const std::exception& failure) {
        std::cerr << "feature batch test failed: " << failure.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "feature batch tests passed\n";
    return EXIT_SUCCESS;
}
