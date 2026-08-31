#include <cstdlib>
#include <cstdint>
#include <array>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "kb_pente/constants.h"
#include "kb_pente/game.h"
#include "kb_pente/player.h"
#include "kb_pente/position.h"
#include "kb_pente/rules.h"
#include "kb_pente/terminal_result.h"

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

template <typename Function>
void expect_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }

    throw TestFailure(message);
}

kb_pente::Action action_at(
    std::uint8_t board_size,
    std::uint8_t row,
    std::uint8_t column) {
    return static_cast<kb_pente::Action>(row * board_size + column);
}

void test_player_helpers() {
    using kb_pente::Player;
    using kb_pente::opponent;
    using kb_pente::player_index;

    expect(kb_pente::is_valid_player(Player::One), "Player One is valid");
    expect(kb_pente::is_valid_player(Player::Two), "Player Two is valid");
    expect(player_index(Player::One) == 0, "Player One uses capture slot zero");
    expect(player_index(Player::Two) == 1, "Player Two uses capture slot one");
    expect(opponent(Player::One) == Player::Two, "Player One opponent");
    expect(opponent(Player::Two) == Player::One, "Player Two opponent");
}

void test_rulesets() {
    using kb_pente::Ruleset;

    expect(kb_pente::is_valid_ruleset(Ruleset::Standard),
           "standard ruleset is valid");
    expect(kb_pente::is_valid_ruleset(Ruleset::Tournament),
           "tournament ruleset is valid");
    expect(kb_pente::is_valid_ruleset(Ruleset::Freestyle),
           "freestyle ruleset is valid");
    expect(!kb_pente::is_valid_ruleset(static_cast<Ruleset>(255)),
           "unknown ruleset is rejected");

    expect(kb_pente::is_valid_ruleset_configuration(6, Ruleset::Freestyle),
           "freestyle supports an even board");
    expect(!kb_pente::is_valid_ruleset_configuration(6, Ruleset::Standard),
           "standard rejects an even board");
    expect(!kb_pente::is_valid_ruleset_configuration(6, Ruleset::Tournament),
           "tournament rejects an even board");
    expect(!kb_pente::is_valid_ruleset_configuration(4, Ruleset::Freestyle),
           "rulesets reject a board below the native minimum");
}

void test_action_mask_operations() {
    kb_pente::ActionMask mask;
    expect(mask.count() == 0, "new action mask is empty");

    mask.set(0);
    mask.set(63);
    mask.set(64);
    mask.set(kb_pente::kMaxActions - 1);
    mask.set(kb_pente::kInvalidAction);
    expect(mask.contains(0), "action mask stores first bit");
    expect(mask.contains(63), "action mask stores word boundary bit");
    expect(mask.contains(64), "action mask stores next word bit");
    expect(mask.contains(kb_pente::kMaxActions - 1),
           "action mask stores final action");
    expect(!mask.contains(kb_pente::kInvalidAction),
           "invalid action is never contained");
    expect(mask.count() == 4, "action mask counts set actions");

    mask.clear(63);
    expect(!mask.contains(63), "action mask clears a bit");
    mask.clear(kb_pente::kInvalidAction);
    expect(mask.count() == 3, "clearing invalid action has no effect");
    mask.clear();
    expect(mask.count() == 0, "action mask clears all bits");
}

void test_legal_actions_and_openings() {
    using kb_pente::Player;
    using kb_pente::Position;
    using kb_pente::Ruleset;

    Position standard = Position::initial(9);
    const auto standard_opening =
        kb_pente::legal_action_mask(standard, Ruleset::Standard);
    const auto center = action_at(9, 4, 4);
    expect(standard_opening.count() == 1,
           "standard opening has one legal action");
    expect(standard_opening.contains(center), "standard opening is center");
    expect(!kb_pente::is_legal_action(standard, Ruleset::Standard, 0),
           "standard rejects a non-center opening");
    expect(kb_pente::is_legal_action(standard, Ruleset::Freestyle, 0),
           "freestyle allows a corner opening");

    Position tournament = Position::initial(9);
    tournament = kb_pente::apply_action(
                     tournament, center, Ruleset::Tournament)
                     .position;
    tournament = kb_pente::apply_action(
                     tournament, 0, Ruleset::Tournament)
                     .position;
    expect(tournament.current_player == Player::One,
           "tournament returns to Player One on ply two");

    const auto tournament_mask =
        kb_pente::legal_action_mask(tournament, Ruleset::Tournament);
    expect(!tournament_mask.contains(action_at(9, 6, 4)),
           "tournament excludes Chebyshev radius two edge");
    expect(!tournament_mask.contains(action_at(9, 6, 6)),
           "tournament excludes Chebyshev radius two corner");
    expect(tournament_mask.contains(action_at(9, 7, 4)),
           "tournament allows outside-radius edge");
    expect(tournament_mask.contains(action_at(9, 7, 7)),
           "tournament allows outside-radius corner");

    const auto standard_mask_after_two =
        kb_pente::legal_action_mask(tournament, Ruleset::Standard);
    expect(standard_mask_after_two.contains(action_at(9, 5, 4)),
           "standard allows near-center second Player One move");
    expect(!standard_mask_after_two.contains(center),
           "occupied center is never legal");
}

void test_legal_action_rejection() {
    using kb_pente::Position;
    using kb_pente::Ruleset;

    Position position = Position::initial(9);
    expect(!kb_pente::is_legal_action(position, Ruleset::Freestyle,
                                      kb_pente::kInvalidAction),
           "invalid action sentinel is rejected");
    expect(!kb_pente::is_legal_action(position, Ruleset::Freestyle, 81),
           "action after board area is rejected");
    expect(kb_pente::is_legal_action(position, Ruleset::Freestyle, 0),
           "empty action is legal in freestyle");

    position = kb_pente::apply_action(position, 0, Ruleset::Freestyle).position;
    expect(!kb_pente::is_legal_action(position, Ruleset::Freestyle, 0),
           "occupied action is rejected");
    expect_invalid_argument(
        [&position] {
            (void)kb_pente::apply_action(position, 0,
                                         Ruleset::Freestyle);
        },
        "transition rejects an occupied action");
    expect_invalid_argument(
        [&position] {
            (void)kb_pente::apply_action(position, 81,
                                         Ruleset::Freestyle);
        },
        "transition rejects an out-of-range action");

    const Position even = Position::initial(6);
    expect(!kb_pente::is_legal_action(even, Ruleset::Standard, 0),
           "standard rejects actions on an even board");
    expect_invalid_argument(
        [&even] {
            (void)kb_pente::legal_action_mask(even, Ruleset::Standard);
        },
        "legal mask rejects an even standard board");
    expect_invalid_argument(
        [&even] {
            (void)kb_pente::apply_action(even, 0, Ruleset::Tournament);
        },
        "transition rejects an even tournament board");
}

void test_initial_positions_at_boundaries() {
    using kb_pente::Action;
    using kb_pente::Player;
    using kb_pente::Position;

    for (const std::uint8_t board_size : {kb_pente::kMinBoardSize,
                                           kb_pente::kMaxBoardSize}) {
        const Position position = Position::initial(board_size);
        expect(position.is_valid(), "initial position is valid");
        expect(position.board_size == board_size, "board size is preserved");
        expect(position.action_count() ==
                   static_cast<std::size_t>(board_size) * board_size,
               "action count matches board area");
        expect(position.current_player == Player::One,
               "Player One starts");
        expect(position.ply == 0, "initial ply is zero");
        expect(position.last_action == kb_pente::kInvalidAction,
               "initial position has no last action");
        expect(position.capture_count(Player::One) == 0,
               "Player One starts with no captures");
        expect(position.capture_count(Player::Two) == 0,
               "Player Two starts with no captures");
        expect(position.is_active_action(static_cast<Action>(
                   position.action_count() - 1)),
               "last board action is active");
        expect(!position.is_active_action(static_cast<Action>(
                   position.action_count())),
               "action after board area is inactive");
    }

    expect_invalid_argument([] { (void)Position::initial(4); },
                            "board smaller than five is rejected");
    expect_invalid_argument([] { (void)Position::initial(20); },
                            "board larger than nineteen is rejected");
}

void test_position_invariants() {
    using kb_pente::Player;
    using kb_pente::Position;

    Position position = Position::initial(5);
    position.stones[0] = static_cast<std::int8_t>(Player::One);
    position.stones[6] = static_cast<std::int8_t>(Player::Two);
    position.captures[kb_pente::player_index(Player::One)] = 1;
    position.ply = 4;
    position.current_player = Player::One;
    position.last_action = 6;
    expect(position.is_valid(), "populated position is valid");
    position.validate();

    position.current_player = Player::Two;
    expect(!position.is_valid(), "current player parity is enforced");
    position.current_player = Player::One;

    position.ply = 3;
    expect(!position.is_valid(), "ply must include captured stones");
    position.ply = 4;

    position.last_action = 0;
    expect(!position.is_valid(), "last action must belong to previous player");
    position.last_action = 6;

    position.stones[6] = 0;
    expect(!position.is_valid(), "last action must contain a stone");
    position.stones[6] = static_cast<std::int8_t>(Player::Two);

    position.stones[25] = static_cast<std::int8_t>(Player::One);
    expect(!position.is_valid(), "storage outside board must be empty");
    position.stones[25] = 0;

    position.stones[1] = 2;
    expect(!position.is_valid(), "stone values are restricted to players");
    expect_invalid_argument(
        [&position] { position.validate(); },
        "validate reports an invalid imported position");
}

void test_single_captures_for_both_players() {
    using kb_pente::Player;
    using kb_pente::Position;
    using kb_pente::Ruleset;

    Position player_one = Position::initial(9);
    for (const auto [row, column] : std::array{
             std::array<std::uint8_t, 2>{4, 0},
             std::array<std::uint8_t, 2>{4, 1},
             std::array<std::uint8_t, 2>{0, 0},
             std::array<std::uint8_t, 2>{4, 2},
             std::array<std::uint8_t, 2>{4, 3}}) {
        player_one = kb_pente::apply_action(
                         player_one, action_at(9, row, column),
                         Ruleset::Freestyle)
                         .position;
    }
    expect(player_one.capture_count(Player::One) == 1,
           "Player One receives a horizontal capture");
    expect(player_one.capture_count(Player::Two) == 0,
           "Player Two does not receive Player One capture");
    expect(player_one.stones[action_at(9, 4, 1)] == 0,
           "first captured stone is removed");
    expect(player_one.stones[action_at(9, 4, 2)] == 0,
           "second captured stone is removed");

    Position player_two = Position::initial(9);
    for (const auto [row, column] : std::array{
             std::array<std::uint8_t, 2>{0, 0},
             std::array<std::uint8_t, 2>{4, 0},
             std::array<std::uint8_t, 2>{4, 1},
             std::array<std::uint8_t, 2>{0, 1},
             std::array<std::uint8_t, 2>{4, 2},
             std::array<std::uint8_t, 2>{4, 3}}) {
        player_two = kb_pente::apply_action(
                         player_two, action_at(9, row, column),
                         Ruleset::Freestyle)
                         .position;
    }
    expect(player_two.capture_count(Player::One) == 0,
           "Player One does not receive Player Two capture");
    expect(player_two.capture_count(Player::Two) == 1,
           "Player Two receives a horizontal capture");
    expect(player_two.stones[action_at(9, 4, 1)] == 0,
           "Player Two removes first captured stone");
    expect(player_two.stones[action_at(9, 4, 2)] == 0,
           "Player Two removes second captured stone");
}

void test_captures_in_all_eight_directions() {
    using kb_pente::Player;
    using kb_pente::Position;
    using kb_pente::Ruleset;

    constexpr std::array<std::array<int, 2>, 8> directions{{
        {{0, 1}},
        {{0, -1}},
        {{1, 0}},
        {{-1, 0}},
        {{1, 1}},
        {{-1, -1}},
        {{1, -1}},
        {{-1, 1}},
    }};
    constexpr int center = 4;
    for (const auto [row_step, column_step] : directions) {
        Position position = Position::initial(9);
        const auto first = action_at(
            9, static_cast<std::uint8_t>(center + row_step),
            static_cast<std::uint8_t>(center + column_step));
        const auto second = action_at(
            9, static_cast<std::uint8_t>(center + 2 * row_step),
            static_cast<std::uint8_t>(center + 2 * column_step));
        const auto third = action_at(
            9, static_cast<std::uint8_t>(center + 3 * row_step),
            static_cast<std::uint8_t>(center + 3 * column_step));
        position.stones[0] = static_cast<std::int8_t>(Player::One);
        position.stones[first] = static_cast<std::int8_t>(Player::Two);
        position.stones[second] = static_cast<std::int8_t>(Player::Two);
        position.stones[third] = static_cast<std::int8_t>(Player::One);
        position.ply = 4;
        position.current_player = Player::One;
        expect(position.is_valid(), "directional capture fixture is valid");

        const auto moved = kb_pente::apply_action(
            position, action_at(9, center, center), Ruleset::Freestyle);
        expect(moved.position.capture_count(Player::One) == 1,
               "each direction captures one pair for Player One");
        expect(moved.position.stones[first] == 0,
               "directional capture removes first stone");
        expect(moved.position.stones[second] == 0,
               "directional capture removes second stone");
    }
}

void test_multiple_captures_and_capture_wins() {
    using kb_pente::Player;
    using kb_pente::Position;
    using kb_pente::Ruleset;
    using kb_pente::WinReason;

    Position multiple = Position::initial(9);
    multiple.stones[action_at(9, 4, 1)] = static_cast<std::int8_t>(Player::One);
    multiple.stones[action_at(9, 4, 2)] = static_cast<std::int8_t>(Player::Two);
    multiple.stones[action_at(9, 4, 3)] = static_cast<std::int8_t>(Player::Two);
    multiple.stones[action_at(9, 1, 4)] = static_cast<std::int8_t>(Player::One);
    multiple.stones[action_at(9, 2, 4)] = static_cast<std::int8_t>(Player::Two);
    multiple.stones[action_at(9, 3, 4)] = static_cast<std::int8_t>(Player::Two);
    multiple.captures[kb_pente::player_index(Player::One)] = 4;
    multiple.ply = 14;
    multiple.current_player = Player::One;
    expect(multiple.is_valid(), "multiple-capture fixture is valid");

    Position expected = multiple;
    expected.stones[action_at(9, 4, 2)] = 0;
    expected.stones[action_at(9, 4, 3)] = 0;
    expected.stones[action_at(9, 2, 4)] = 0;
    expected.stones[action_at(9, 3, 4)] = 0;
    expected.stones[action_at(9, 4, 4)] = static_cast<std::int8_t>(Player::One);
    expected.captures[kb_pente::player_index(Player::One)] = 6;
    expected.ply = 15;
    expected.last_action = action_at(9, 4, 4);
    expected.current_player = Player::Two;

    const auto transition = kb_pente::apply_action(
        multiple, action_at(9, 4, 4), Ruleset::Freestyle);
    expect(transition.position == expected,
           "simultaneous captures produce the complete expected position");
    expect(transition.terminal ==
               kb_pente::TerminalResult::win(Player::One, WinReason::Capture),
           "capture victory is reported after multiple captures");

    Position player_one_win = Position::initial(9);
    player_one_win.stones[action_at(9, 4, 4)] =
        static_cast<std::int8_t>(Player::One);
    player_one_win.stones[action_at(9, 4, 2)] =
        static_cast<std::int8_t>(Player::Two);
    player_one_win.stones[action_at(9, 4, 3)] =
        static_cast<std::int8_t>(Player::Two);
    player_one_win.stones[0] = static_cast<std::int8_t>(Player::One);
    player_one_win.captures[kb_pente::player_index(Player::One)] = 4;
    player_one_win.ply = 12;
    player_one_win.current_player = Player::One;
    const auto player_one_transition = kb_pente::apply_action(
        player_one_win, action_at(9, 4, 1), Ruleset::Freestyle);
    expect(player_one_transition.terminal ==
               kb_pente::TerminalResult::win(Player::One, WinReason::Capture),
           "Player One capture victory is reported");

    Position player_two_win = Position::initial(9);
    player_two_win.stones[action_at(9, 4, 8)] =
        static_cast<std::int8_t>(Player::Two);
    player_two_win.stones[action_at(9, 4, 6)] =
        static_cast<std::int8_t>(Player::One);
    player_two_win.stones[action_at(9, 4, 7)] =
        static_cast<std::int8_t>(Player::One);
    player_two_win.captures[kb_pente::player_index(Player::Two)] = 4;
    player_two_win.ply = 11;
    player_two_win.current_player = Player::Two;
    const auto player_two_transition = kb_pente::apply_action(
        player_two_win, action_at(9, 4, 5), Ruleset::Freestyle);
    expect(player_two_transition.terminal ==
               kb_pente::TerminalResult::win(Player::Two, WinReason::Capture),
           "Player Two capture victory is reported");

    Position capture_and_line = Position::initial(9);
    for (std::uint8_t column = 0; column < 4; ++column) {
        capture_and_line.stones[action_at(9, 4, column)] =
            static_cast<std::int8_t>(Player::One);
    }
    capture_and_line.stones[action_at(9, 2, 4)] =
        static_cast<std::int8_t>(Player::Two);
    capture_and_line.stones[action_at(9, 3, 4)] =
        static_cast<std::int8_t>(Player::Two);
    capture_and_line.stones[action_at(9, 1, 4)] =
        static_cast<std::int8_t>(Player::One);
    capture_and_line.stones[action_at(9, 0, 8)] =
        static_cast<std::int8_t>(Player::Two);
    capture_and_line.captures[kb_pente::player_index(Player::One)] = 4;
    capture_and_line.ply = 16;
    capture_and_line.current_player = Player::One;
    expect(capture_and_line.is_valid(),
           "capture-and-line precedence fixture is valid");
    const auto capture_and_line_transition = kb_pente::apply_action(
        capture_and_line, action_at(9, 4, 4), Ruleset::Freestyle);
    expect(capture_and_line_transition.terminal ==
               kb_pente::TerminalResult::win(Player::One,
                                              WinReason::Capture),
           "capture victory takes precedence over line victory");
}

void test_lines_for_both_players_and_overline() {
    using kb_pente::GameStatus;
    using kb_pente::Player;
    using kb_pente::Position;

    constexpr std::array<std::array<int, 2>, 4> axes{{
        {{0, 1}},
        {{1, 0}},
        {{1, 1}},
        {{1, -1}},
    }};
    for (const auto [row_step, column_step] : axes) {
        for (const Player player : {Player::One, Player::Two}) {
            Position position = Position::initial(9);
            const int start_column = column_step >= 0 ? 2 : 6;
            const auto opponent = kb_pente::opponent(player);
            for (int index = 0; index < 5; ++index) {
                const auto row = static_cast<std::uint8_t>(2 + index * row_step);
                const auto column = static_cast<std::uint8_t>(
                    start_column + index * column_step);
                position.stones[action_at(9, row, column)] =
                    static_cast<std::int8_t>(player);
            }

            constexpr std::array<std::array<std::uint8_t, 2>, 8> fillers{{
                {{0, 0}},
                {{0, 2}},
                {{0, 5}},
                {{1, 7}},
                {{3, 8}},
                {{5, 0}},
                {{7, 1}},
                {{8, 4}},
            }};
            std::size_t filler_count = player == Player::One ? 4 : 5;
            std::size_t placed_fillers = 0;
            for (const auto [row, column] : fillers) {
                const auto action = action_at(9, row, column);
                if (position.stones[action] == 0 &&
                    placed_fillers < filler_count) {
                    position.stones[action] =
                        static_cast<std::int8_t>(opponent);
                    ++placed_fillers;
                }
            }

            position.ply = static_cast<std::uint16_t>(5 + filler_count);
            position.current_player =
                position.ply % 2 == 0 ? Player::One : Player::Two;
            expect(position.is_valid(), "line fixture is valid");
            const auto result = kb_pente::check_terminal(position);
            expect(result.status == GameStatus::Win,
                   "five in a row is terminal");
            expect(result.winner == player, "line winner is identified");
            expect(result.reason == kb_pente::WinReason::Line,
                   "line win has line reason");
        }
    }

    Position overline = Position::initial(9);
    for (std::uint8_t column = 1; column <= 6; ++column) {
        overline.stones[action_at(9, 2, column)] =
            static_cast<std::int8_t>(Player::One);
    }
    overline.stones[0] = static_cast<std::int8_t>(Player::Two);
    overline.stones[action_at(9, 1, 8)] = static_cast<std::int8_t>(Player::Two);
    overline.stones[action_at(9, 3, 8)] = static_cast<std::int8_t>(Player::Two);
    overline.stones[action_at(9, 4, 8)] = static_cast<std::int8_t>(Player::Two);
    overline.ply = 10;
    overline.current_player = Player::One;
    expect(overline.is_valid(), "overline fixture is valid");
    expect(kb_pente::check_terminal(overline) ==
               kb_pente::TerminalResult::win(Player::One,
                                              kb_pente::WinReason::Line),
           "line longer than five is a win");
}

void test_terminal_precedence_and_imported_positions() {
    using kb_pente::GameStatus;
    using kb_pente::Player;
    using kb_pente::Position;
    using kb_pente::Ruleset;
    using kb_pente::TerminalResult;
    using kb_pente::WinReason;

    Position final_cell = Position::initial(5);
    constexpr std::array<std::int8_t, 25> final_cell_stones{{
        1, 1, 1, 1, 0,
        -1, -1, 1, -1, 1,
        1, -1, -1, 1, -1,
        -1, 1, -1, -1, 1,
        1, -1, 1, -1, -1,
    }};
    std::copy(final_cell_stones.begin(), final_cell_stones.end(),
              final_cell.stones.begin());
    final_cell.ply = 24;
    final_cell.current_player = Player::One;
    const auto final_transition =
        kb_pente::apply_action(final_cell, 4, Ruleset::Freestyle);
    expect(final_transition.terminal ==
               TerminalResult::win(Player::One, WinReason::Line),
           "line victory takes precedence over a full-board draw");
    expect(final_transition.terminal.status == GameStatus::Win,
           "final line transition is a win");

    Position imported = Position::initial(9);
    for (std::uint8_t column = 1; column <= 5; ++column) {
        imported.stones[action_at(9, 3, column)] =
            static_cast<std::int8_t>(Player::One);
    }
    imported.ply = 5;
    imported.current_player = Player::Two;
    imported.last_action = action_at(9, 3, 5);
    expect(imported.is_valid(), "imported last-action line is valid");
    expect(kb_pente::check_terminal(imported) ==
               TerminalResult::win(Player::One, WinReason::Line),
           "last-action fast path finds the previous player's line");

    imported.last_action = kb_pente::kInvalidAction;
    expect(kb_pente::check_terminal(imported) ==
               TerminalResult::win(Player::One, WinReason::Line),
           "no-last-action path scans imported lines");
}

void test_full_board_draw() {
    using kb_pente::GameStatus;
    using kb_pente::Player;
    using kb_pente::Position;

    Position position = Position::initial(5);
    constexpr std::array<std::int8_t, 25> stones{{
        1, 1, -1, -1, 1,
        -1, -1, 1, 1, -1,
        1, 1, -1, -1, 1,
        -1, -1, 1, 1, -1,
        1, -1, 1, -1, 1,
    }};
    std::copy(stones.begin(), stones.end(), position.stones.begin());
    position.ply = 25;
    position.current_player = Player::Two;
    expect(position.is_valid(), "full-board draw fixture is valid");
    const auto result = kb_pente::check_terminal(position);
    expect(result.status == GameStatus::Draw, "full board without line is draw");
    expect(result.is_terminal(), "draw is terminal");
    expect(result.value_for(Player::One) == 0.0F,
           "draw value is neutral for Player One");
    expect(result.value_for(Player::Two) == 0.0F,
           "draw value is neutral for Player Two");
}

void test_relative_features() {
    using kb_pente::Player;
    using kb_pente::Position;

    Position position = Position::initial(9);
    position.stones[0] = static_cast<std::int8_t>(Player::One);
    position.stones[1] = static_cast<std::int8_t>(Player::Two);
    position.stones[9] = static_cast<std::int8_t>(Player::One);
    position.captures[kb_pente::player_index(Player::One)] = 2;
    position.captures[kb_pente::player_index(Player::Two)] = 4;
    position.ply = 15;
    position.current_player = Player::Two;
    expect(position.is_valid(), "feature fixture is valid");

    std::array<float, 4 * kb_pente::kMaxActions> output;
    output.fill(-1.0F);
    kb_pente::write_features(position, output.data());
    constexpr std::size_t area = 81;
    expect(output[0] == 0.0F, "current plane ignores Player One stone");
    expect(output[1] == 1.0F, "current plane includes Player Two stone");
    expect(output[area] == 1.0F, "opponent plane includes Player One stone");
    expect(output[area + 1] == 0.0F,
           "opponent plane ignores current-player stone");
    expect(output[2 * area] == 0.8F, "current capture plane is relative");
    expect(output[3 * area] == 0.4F, "opponent capture plane is relative");
    expect(output[4 * area - 1] == 0.4F,
           "feature planes fill the complete active area");
    expect(output[4 * area] == -1.0F,
           "feature writer does not touch storage past active area");

    expect_invalid_argument(
        [&position] { kb_pente::write_features(position, nullptr); },
        "feature writer rejects null caller storage");
}

void test_terminal_results() {
    using kb_pente::GameStatus;
    using kb_pente::Player;
    using kb_pente::TerminalResult;
    using kb_pente::WinReason;

    const auto in_progress = TerminalResult::in_progress();
    expect(in_progress.is_valid(), "in-progress result is valid");
    expect(!in_progress.is_terminal(), "in-progress result is not terminal");
    expect(in_progress.value_for(Player::One) == 0.0F,
           "in-progress value is neutral");

    const auto draw = TerminalResult::draw();
    expect(draw.status == GameStatus::Draw, "draw status");
    expect(draw.is_valid(), "draw result is valid");
    expect(draw.is_terminal(), "draw result is terminal");

    const auto player_one_win =
        TerminalResult::win(Player::One, WinReason::Capture);
    expect(player_one_win.status == GameStatus::Win, "Player One win status");
    expect(player_one_win.winner == Player::One, "Player One winner");
    expect(player_one_win.reason == WinReason::Capture,
           "capture win reason");
    expect(player_one_win.value_for(Player::One) == 1.0F,
           "winner value from Player One perspective");
    expect(player_one_win.value_for(Player::Two) == -1.0F,
           "winner value from Player Two perspective");

    const auto player_two_win = TerminalResult::win(Player::Two, WinReason::Line);
    expect(player_two_win.is_valid(), "Player Two win is valid");
    expect(player_two_win.value_for(Player::One) == -1.0F,
           "Player One sees Player Two win as loss");
    expect(player_two_win.value_for(Player::Two) == 1.0F,
           "Player Two sees own win");

    const TerminalResult malformed{GameStatus::Draw, Player::One,
                                   WinReason::None};
    expect(!malformed.is_valid(), "draw cannot identify a winner");
}

}  // namespace

int main() {
    static_assert(std::is_trivially_copyable_v<kb_pente::Position>);
    static_assert(std::is_standard_layout_v<kb_pente::Position>);

    try {
        test_player_helpers();
        test_rulesets();
        test_action_mask_operations();
        test_initial_positions_at_boundaries();
        test_position_invariants();
        test_legal_actions_and_openings();
        test_legal_action_rejection();
        test_single_captures_for_both_players();
        test_captures_in_all_eight_directions();
        test_multiple_captures_and_capture_wins();
        test_lines_for_both_players_and_overline();
        test_terminal_precedence_and_imported_positions();
        test_full_board_draw();
        test_relative_features();
        test_terminal_results();
    } catch (const TestFailure& failure) {
        std::cerr << "FAIL: " << failure.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "native core tests passed\n";
    return EXIT_SUCCESS;
}
