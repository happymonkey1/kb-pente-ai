#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "kb_pente/constants.h"
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
        test_initial_positions_at_boundaries();
        test_position_invariants();
        test_terminal_results();
    } catch (const TestFailure& failure) {
        std::cerr << "FAIL: " << failure.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "native core tests passed\n";
    return EXIT_SUCCESS;
}
