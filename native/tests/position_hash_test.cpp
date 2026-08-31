#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "kb_pente/game.h"
#include "kb_pente/mcts/tree.h"

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

constexpr std::array<kb_pente::Ruleset, 3> kRulesets{{
    kb_pente::Ruleset::Standard,
    kb_pente::Ruleset::Tournament,
    kb_pente::Ruleset::Freestyle,
}};

void test_initial_and_copy_caches() {
    for (const std::uint8_t board_size : {5U, 6U, 9U, 19U}) {
        const kb_pente::Position position =
            kb_pente::Position::initial(board_size);
        expect(position.has_consistent_hash(),
               "initial position has a canonical hash");
        expect(position.hash() == position.recompute_hash(),
               "initial cache equals full recomputation");
        expect(position.hash().lo != position.hash().hi,
               "the two hash lanes are independently seeded");

        const kb_pente::Position copy = position;
        expect(copy == position, "position copies retain semantic equality");
        expect(copy.hash() == position.hash(),
               "position copies retain their cached hash");
    }
}

void test_semantic_sensitivity_and_cache_ignoring_equality() {
    const kb_pente::Position base = kb_pente::Position::initial(9);
    const kb_pente::PositionHash base_hash = base.recompute_hash();

    const auto expect_changed =
        [&base, base_hash](auto mutate, const char* message) {
            kb_pente::Position changed = base;
            mutate(changed);
            const kb_pente::PositionHash changed_hash =
                changed.recompute_hash();
            expect(changed_hash.lo != base_hash.lo, message);
            expect(changed_hash.hi != base_hash.hi, message);
        };

    expect_changed(
        [](kb_pente::Position& position) {
            position.stones[0] =
                static_cast<std::int8_t>(kb_pente::Player::One);
        },
        "Player One stones change both hash lanes");
    expect_changed(
        [](kb_pente::Position& position) {
            position.stones[1] =
                static_cast<std::int8_t>(kb_pente::Player::Two);
        },
        "Player Two stones change both hash lanes");
    expect_changed(
        [](kb_pente::Position& position) { position.captures[0] = 1U; },
        "Player One captures change both hash lanes");
    expect_changed(
        [](kb_pente::Position& position) { position.captures[1] = 1U; },
        "Player Two captures change both hash lanes");
    expect_changed(
        [](kb_pente::Position& position) { position.ply = 1U; },
        "ply and opening state change both hash lanes");
    expect_changed(
        [](kb_pente::Position& position) { position.last_action = 0U; },
        "last action changes both hash lanes");
    expect_changed(
        [](kb_pente::Position& position) {
            position.board_size = 10U;
        },
        "board size changes both hash lanes");
    expect_changed(
        [](kb_pente::Position& position) {
            position.current_player = kb_pente::Player::Two;
        },
        "current player changes both hash lanes");

    kb_pente::Position stale = base;
    stale.hash_lo ^= 1U;
    stale.hash_hi ^= 2U;
    expect(stale == base, "semantic equality ignores cached hash lanes");
    expect(!stale.has_consistent_hash(), "stale cache is detectable");
    stale.refresh_hash();
    expect(stale.has_consistent_hash(), "refresh restores cache consistency");
    expect(stale.hash() == base.hash(), "refresh restores the canonical value");
}

void test_invalid_recompute_is_bounded() {
    kb_pente::Position invalid{};
    invalid.board_size = 255U;
    invalid.stones[kb_pente::kMaxActions - 1U] =
        static_cast<std::int8_t>(kb_pente::Player::Two);
    const auto first = invalid.recompute_hash();
    invalid.refresh_hash();
    expect(invalid.hash() == first,
           "invalid board sizes still have bounded deterministic hashing");
    expect(invalid.has_consistent_hash(),
           "cache consistency remains inspectable for imports");
    expect(!invalid.is_valid(), "semantic validation still rejects the import");
}

void test_incremental_sequences() {
    for (std::uint8_t board_size = kb_pente::kMinBoardSize;
         board_size <= kb_pente::kMaxBoardSize; ++board_size) {
        for (const kb_pente::Ruleset ruleset : kRulesets) {
            if (!kb_pente::is_valid_ruleset_configuration(
                    board_size, ruleset)) {
                continue;
            }

            kb_pente::Position position =
                kb_pente::Position::initial(board_size);
            for (std::size_t step = 0U; step < 160U; ++step) {
                expect(position.has_consistent_hash(),
                       "sequence parent cache stays consistent");
                const auto legal =
                    kb_pente::legal_action_mask(position, ruleset);
                if (legal.count() == 0U) {
                    break;
                }

                const std::size_t offset =
                    (step * 17U + board_size * 3U) % position.action_count();
                kb_pente::Action selected = kb_pente::kInvalidAction;
                for (std::size_t probe = 0U; probe < position.action_count();
                     ++probe) {
                    const auto action = static_cast<kb_pente::Action>(
                        (offset + probe) % position.action_count());
                    if (legal.contains(action)) {
                        selected = action;
                        break;
                    }
                }
                expect(selected != kb_pente::kInvalidAction,
                       "legal mask yields a selected action");

                const kb_pente::Transition transition =
                    kb_pente::apply_action(position, selected, ruleset);
                expect(transition.position.has_consistent_hash(),
                       "incremental child cache is consistent");
                expect(transition.position.hash() ==
                           transition.position.recompute_hash(),
                       "incremental child equals full recomputation");
                position = transition.position;
                if (transition.terminal.is_terminal()) {
                    break;
                }
            }
        }
    }
}

void test_capture_incremental_update() {
    kb_pente::Position position = kb_pente::Position::initial(9);
    position.stones[0] = static_cast<std::int8_t>(kb_pente::Player::Two);
    position.stones[41] = static_cast<std::int8_t>(kb_pente::Player::Two);
    position.stones[42] = static_cast<std::int8_t>(kb_pente::Player::Two);
    position.stones[43] = static_cast<std::int8_t>(kb_pente::Player::One);
    position.captures[kb_pente::player_index(kb_pente::Player::One)] = 4U;
    position.ply = 12U;
    position.current_player = kb_pente::Player::One;
    expect(position.is_valid(), "capture hash fixture is valid");
    position.refresh_hash();

    const kb_pente::Transition transition =
        kb_pente::apply_action(position, 40U, kb_pente::Ruleset::Freestyle);
    expect(transition.position.capture_count(kb_pente::Player::One) == 5U,
           "incremental capture count includes the captured pair");
    expect(transition.position.stones[41] == 0 &&
               transition.position.stones[42] == 0,
           "incremental hash fixture removes both captured stones");
    expect(transition.terminal == kb_pente::TerminalResult::win(
                                      kb_pente::Player::One,
                                      kb_pente::WinReason::Capture),
           "capture fixture reaches the expected terminal result");
    expect(transition.position.has_consistent_hash(),
           "capture transition cache is consistent");
    expect(transition.position.hash() == transition.position.recompute_hash(),
           "capture transition equals full recomputation");
}

void test_stale_transition_rejection_and_imported_tree() {
    kb_pente::Position stale = kb_pente::Position::initial(5);
    stale.hash_lo ^= 0x100U;
    expect(!stale.has_consistent_hash(), "stale parent is detectable");
    expect_throws<std::invalid_argument>(
        [&stale] {
            (void)kb_pente::apply_action(
                stale, 0U, kb_pente::Ruleset::Freestyle);
        },
        "transition rejects a stale parent cache");

    const kb_pente::Position imported = stale;
    const kb_pente::Tree tree(
        imported, kb_pente::Ruleset::Freestyle, kb_pente::SearchConfig{});
    expect(tree.root_position() == imported,
           "Tree preserves imported semantic position fields");
    expect(tree.root_position().has_consistent_hash(),
           "Tree canonicalizes its by-value root cache");
    expect(imported.hash() != tree.root_position().hash(),
           "Tree canonicalization does not mutate the caller copy");
}

}  // namespace

int main() {
    static_assert(std::is_trivially_copyable_v<kb_pente::Position>);
    static_assert(std::is_standard_layout_v<kb_pente::Position>);
    static_assert(std::is_trivially_copyable_v<kb_pente::PositionHash>);

    try {
        test_initial_and_copy_caches();
        test_semantic_sensitivity_and_cache_ignoring_equality();
        test_invalid_recompute_is_bounded();
        test_incremental_sequences();
        test_capture_incremental_update();
        test_stale_transition_rejection_and_imported_tree();
    } catch (const TestFailure& failure) {
        std::cerr << "position hash test failed: " << failure.what() << '\n';
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "position hash test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "position hash tests passed\n";
    return EXIT_SUCCESS;
}
