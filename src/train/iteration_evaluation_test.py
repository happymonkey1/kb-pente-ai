from __future__ import annotations

import unittest
from unittest.mock import patch

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.telemetry import InMemoryMetricSink
from src.train.arena import ArenaStats
from src.train.iteration_evaluation import evaluate_training_iteration


class _FakeNet:
    def eval(self) -> _FakeNet:
        return self


def _arena_stats() -> ArenaStats:
    return ArenaStats(
        p1_wins=1,
        p2_wins=1,
        draws=0,
        avg_moves=1.0,
        player_one_color_wins=1,
        player_two_color_wins=1,
        p1_as_player_one_wins=1,
        p1_as_player_two_wins=0,
        p2_as_player_one_wins=0,
        p2_as_player_two_wins=1,
        opening_plies=0,
        unique_openings=1,
        paired_openings=1,
        p1_pair_wins=0,
        p1_pair_losses=0,
        pair_ties=1,
    )


class _FakeArena:
    arenas: list[_FakeArena] = []

    def __init__(self, player1: object, player2: object, **kwargs: object) -> None:
        self.player1 = player1
        self.player2 = player2
        self.kwargs = kwargs
        self.arenas.append(self)

    def play_games(self, num_games: int) -> ArenaStats:
        self.num_games = num_games
        return _arena_stats()


class IterationEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeArena.arenas.clear()

    def test_all_mcts_arena_players_use_selected_backend_and_distinct_seeds(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        previous_net = _FakeNet()
        current_net = _FakeNet()
        mcts_args = MCTSArgs(num_simulations=2)
        created: list[dict[str, object]] = []

        def build(
            net: object,
            mcts: object,
            name: str,
            **kwargs: object,
        ) -> object:
            created.append({"net": net, "mcts": mcts, "name": name, **kwargs})
            return object()

        with patch(
            "src.train.iteration_evaluation.build_player",
            side_effect=build,
        ), patch(
            "src.train.iteration_evaluation.Arena",
            _FakeArena,
        ):
            evaluate_training_iteration(
                game,
                previous_net,  # type: ignore[arg-type]
                current_net,  # type: ignore[arg-type]
                mcts_args,
                InMemoryMetricSink(),
                iteration=3,
                num_games=2,
                opening_plies=0,
                debug=False,
                seed=11,
                search_backend="cpp",
                native_worker_threads=4,
            )

        self.assertEqual([11, 12, 13, 15], [call["seed"] for call in created])
        self.assertEqual(
            ["previous", "current", "current", "current"],
            [call["name"] for call in created],
        )
        self.assertTrue(
            all(
                call["search_backend"] == "cpp"
                and call["native_worker_threads"] == 4
                and call["game"] is game
                and call["mcts_args"] is mcts_args
                and call["mcts"] is None
                for call in created
            )
        )
        self.assertIs(previous_net, created[0]["net"])
        self.assertTrue(all(call["net"] is current_net for call in created[1:]))
        self.assertEqual(3, len(_FakeArena.arenas))

    def test_legacy_call_defaults_to_python_one_thread(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        created: list[dict[str, object]] = []

        def build(*args: object, **kwargs: object) -> object:
            created.append({"args": args, **kwargs})
            return object()

        with patch(
            "src.train.iteration_evaluation.build_player",
            side_effect=build,
        ), patch(
            "src.train.iteration_evaluation.Arena",
            _FakeArena,
        ):
            evaluate_training_iteration(
                game,
                _FakeNet(),  # type: ignore[arg-type]
                _FakeNet(),  # type: ignore[arg-type]
                MCTSArgs(num_simulations=1),
                InMemoryMetricSink(),
                1,
                2,
                0,
                False,
                7,
            )

        self.assertEqual(4, len(created))
        self.assertTrue(
            all(
                call["search_backend"] == "python"
                and call["native_worker_threads"] == 1
                for call in created
            )
        )


if __name__ == "__main__":
    unittest.main()
