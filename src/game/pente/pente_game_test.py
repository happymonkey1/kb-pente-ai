import unittest

import numpy as np

from src.game.game import Game, GameStatus
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset


class PenteGameTest(unittest.TestCase):
    BOARD_SIZE = 9

    def test_standard_opening_requires_center(self) -> None:
        game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.STANDARD)
        board = game.init_board()
        center_action = (self.BOARD_SIZE // 2) * self.BOARD_SIZE + self.BOARD_SIZE // 2

        legal = game.get_valid_moves(board, Game.PLAYER_ONE)

        self.assertEqual(1, int(legal.sum()))
        self.assertEqual(1, legal[center_action])

    def test_tournament_restricts_player_one_second_move(self) -> None:
        game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.TOURNAMENT)
        board = game.init_board()
        center = self.BOARD_SIZE // 2
        board, _ = game.apply_action(board, board.current_player, center * self.BOARD_SIZE + center)
        board, _ = game.apply_action(board, board.current_player, 0)

        legal = game.get_valid_moves(board, Game.PLAYER_ONE).reshape(self.BOARD_SIZE, self.BOARD_SIZE)

        self.assertEqual(0, legal[center + 2, center])
        self.assertEqual(0, legal[center + 2, center + 2])
        self.assertEqual(1, legal[center + 3, center])
        self.assertEqual(1, legal[center + 3, center + 3])

    def test_standard_allows_near_center_player_one_second_move(self) -> None:
        game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.STANDARD)
        board = game.init_board()
        center = self.BOARD_SIZE // 2
        board, _ = game.apply_action(board, board.current_player, center * self.BOARD_SIZE + center)
        board, _ = game.apply_action(board, board.current_player, 0)

        legal = game.get_valid_moves(board, Game.PLAYER_ONE).reshape(self.BOARD_SIZE, self.BOARD_SIZE)

        self.assertEqual(1, legal[center + 1, center])

    def test_five_in_a_row_for_every_direction_and_player(self) -> None:
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        filler_candidates = ((0, 0), (0, 2), (0, 5), (1, 7), (3, 8), (5, 0), (7, 1), (8, 4))

        for player in (Game.PLAYER_ONE, Game.PLAYER_TWO):
            for row_step, column_step in directions:
                with self.subTest(player=player, direction=(row_step, column_step)):
                    start = (2, 2) if column_step >= 0 else (2, 6)
                    targets = tuple(
                        (start[0] + index * row_step, start[1] + index * column_step)
                        for index in range(5)
                    )
                    filler_count = 4 if player == Game.PLAYER_ONE else 5
                    fillers = tuple(
                        coordinate for coordinate in filler_candidates if coordinate not in targets
                    )[:filler_count]
                    stones = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
                    for coordinate in targets:
                        stones[coordinate] = player
                    for coordinate in fillers:
                        stones[coordinate] = Game.opponent(player)
                    current_player = Game.opponent(player)
                    board = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=current_player)
                    game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.FREESTYLE)
                    result = game.check_game_end(board)

                    self.assertEqual(GameStatus.WIN, result.status)
                    self.assertEqual(player, result.winner)
                    self.assertEqual(1.0, result.value_for(player))
                    self.assertEqual(-1.0, result.value_for(Game.opponent(player)))

    def test_capture_win_for_both_players_through_transition(self) -> None:
        patterns = (
            (
                Game.PLAYER_ONE,
                np.array((4, 0), dtype=np.int16),
                ((4, 1), (4, 2), (4, 3)),
            ),
            (
                Game.PLAYER_TWO,
                np.array((0, 4), dtype=np.int16),
                ((4, 5), (4, 7), (4, 6)),
            ),
        )

        for player, captures, coordinates in patterns:
            with self.subTest(player=player):
                place, opponent_one, opponent_two = coordinates
                stones = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
                anchor_column = 4 if player == Game.PLAYER_ONE else 8
                stones[4, anchor_column] = player
                stones[opponent_one] = Game.opponent(player)
                stones[opponent_two] = Game.opponent(player)
                if player == Game.PLAYER_ONE:
                    stones[0, 0] = Game.PLAYER_ONE
                board = PenteBoard(stones, captures, current_player=player)
                game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.FREESTYLE)

                board, _ = game.apply_action(board, player, place[0] * self.BOARD_SIZE + place[1])
                result = game.check_game_end(board)

                self.assertEqual(GameStatus.WIN, result.status)
                self.assertEqual(player, result.winner)

    def test_full_board_without_winner_is_draw(self) -> None:
        stones = np.array(
            (
                (1, 1, -1, -1, 1),
                (-1, -1, 1, 1, -1),
                (1, 1, -1, -1, 1),
                (-1, -1, 1, 1, -1),
                (1, -1, 1, -1, 1),
            ),
            dtype=np.int8,
        )
        board = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=Game.PLAYER_TWO)
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)

        result = game.check_game_end(board)

        self.assertEqual(GameStatus.DRAW, result.status)
        self.assertTrue(result.is_terminal)
        self.assertEqual(0.0, result.value_for(Game.PLAYER_ONE))
        self.assertEqual(0.0, result.value_for(Game.PLAYER_TWO))

    def test_final_cell_line_win_precedes_draw(self) -> None:
        stones = np.array(
            (
                (1, 1, 1, 1, 0),
                (-1, -1, 1, -1, 1),
                (1, -1, -1, 1, -1),
                (-1, 1, -1, -1, 1),
                (1, -1, 1, -1, -1),
            ),
            dtype=np.int8,
        )
        board = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=Game.PLAYER_ONE)
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)

        board, _ = game.apply_action(board, Game.PLAYER_ONE, 4)
        result = game.check_game_end(board)

        self.assertEqual(GameStatus.WIN, result.status)
        self.assertEqual(Game.PLAYER_ONE, result.winner)

    def test_state_key_and_symmetry_preserve_complete_state(self) -> None:
        game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.FREESTYLE)
        board = self._play(game, ((4, 0), (4, 1), (0, 0), (4, 2), (4, 3)))
        policy = np.zeros(game.get_action_size(), dtype=np.float32)
        policy[8] = 1.0

        symmetries = game.get_symmetries(board, policy)

        self.assertEqual(8, len(symmetries))
        for transformed, transformed_policy in symmetries:
            np.testing.assert_array_equal(board.captures, transformed.captures)
            self.assertEqual(board.current_player, transformed.current_player)
            self.assertEqual(board.ply, transformed.ply)
            self.assertEqual(1.0, float(transformed_policy.sum()))
            self.assertEqual(
                Game.opponent(transformed.current_player),
                transformed.board.reshape(-1)[transformed.last_action],
            )

    def test_rejects_invalid_actions_and_players(self) -> None:
        game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.FREESTYLE)
        board = game.init_board()

        self.assertFalse(game.is_valid_move(board, Game.PLAYER_ONE, -1))
        self.assertFalse(game.is_valid_move(board, Game.PLAYER_ONE, game.get_action_size()))
        with self.assertRaisesRegex(ValueError, "Expected Player"):
            game.apply_action(board, Game.PLAYER_TWO, 0)
        with self.assertRaisesRegex(ValueError, "Invalid action"):
            game.apply_action(board, Game.PLAYER_ONE, game.get_action_size())

    @staticmethod
    def _play(game: PenteGame, moves: tuple[tuple[int, int], ...]) -> PenteBoard:
        board = game.init_board()
        for row, column in moves:
            action = row * game.get_board_size() + column
            board, _ = game.apply_action(board, board.current_player, action)
        return board


if __name__ == "__main__":
    unittest.main()
