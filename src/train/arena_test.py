import unittest

import numpy as np

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.train.arena import Arena
from src.train.player import Player


class FirstLegalPlayer(Player):
    def play(
        self,
        game: PenteGame,
        board: PenteBoard,
        player: int,
        debug: bool = False,
    ) -> int:
        return int(np.flatnonzero(game.get_valid_moves(board, player))[0])


class LastLegalPlayer(Player):
    def play(
        self,
        game: PenteGame,
        board: PenteBoard,
        player: int,
        debug: bool = False,
    ) -> int:
        return int(np.flatnonzero(game.get_valid_moves(board, player))[-1])


class ArenaTest(unittest.TestCase):
    def test_balances_colors_and_restores_player_order(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        first = FirstLegalPlayer()
        second = LastLegalPlayer()
        arena = Arena(first, second, game)

        stats = arena.play_games(5)

        self.assertEqual(5, stats.p1_wins + stats.p2_wins + stats.draws)
        self.assertEqual(
            stats.p1_wins + stats.p2_wins,
            stats.player_one_color_wins + stats.player_two_color_wins,
        )
        self.assertEqual(
            stats.p1_wins,
            stats.p1_as_player_one_wins + stats.p1_as_player_two_wins,
        )
        self.assertEqual(
            stats.p2_wins,
            stats.p2_as_player_one_wins + stats.p2_as_player_two_wins,
        )
        self.assertEqual(
            stats.paired_openings,
            stats.p1_pair_wins + stats.p1_pair_losses + stats.pair_ties,
        )
        self.assertGreater(stats.avg_moves, 0)
        self.assertIs(first, arena.player1)
        self.assertIs(second, arena.player2)

    def test_rejects_empty_evaluation(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        arena = Arena(FirstLegalPlayer(), LastLegalPlayer(), game)

        with self.assertRaisesRegex(ValueError, "at least one"):
            arena.play_games(0)

    def test_uses_seeded_paired_openings_for_color_balance(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        arena = Arena(
            FirstLegalPlayer(),
            LastLegalPlayer(),
            game,
            opening_plies=4,
            rng=np.random.default_rng(22),
        )

        stats = arena.play_games(6)

        self.assertEqual(4, stats.opening_plies)
        self.assertEqual(3, stats.unique_openings)
        self.assertEqual(3, stats.paired_openings)
        self.assertEqual(6, stats.p1_wins + stats.p2_wins + stats.draws)


if __name__ == "__main__":
    unittest.main()
