from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
import unittest
from unittest.mock import patch

import numpy as np

from src.evaluation.tactical import TacticalSuiteStats
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.train.arena import ArenaStats
from src.train.player import Player
from src.train.random_player import RandomPlayer
from src.train.self_play_args import SearchBackend
from src.train.tactical_heuristic_player import TacticalHeuristicPlayer
from src.verification.search_quality import (
    BackendSearchQualityReport,
    evaluate_search_quality,
    summarize_statistical_parity,
)
from src.verification.search_quality_reports import (
    ColorResults,
    DirectTacticalMetrics,
    GameResults,
    MatchReport,
    PairResults,
    SearchQualityConfig,
    SearchTacticalReport,
    StatisticalParityCriteria,
    WilsonInterval,
)


class _FakeNet:
    def __init__(self, board_size: int = 9) -> None:
        self.eval_calls = 0
        self.board_size = board_size

    def eval(self) -> _FakeNet:
        self.eval_calls += 1
        return self


class _FakePlayer(Player):
    def __init__(self, game: PenteGame) -> None:
        self.game = game
        self.reset_calls = 0
        self.play_calls: list[int] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def play(
        self,
        game: PenteGame,
        board: object,
        player: int,
        debug: bool = False,
    ) -> int:
        del debug
        if game is not self.game:
            raise AssertionError("unexpected game")
        action = int(np.flatnonzero(game.get_valid_moves(board, player))[0])  # type: ignore[arg-type]
        self.play_calls.append(action)
        return action


def _arena_stats() -> ArenaStats:
    return ArenaStats(
        p1_wins=3,
        p2_wins=1,
        draws=0,
        avg_moves=12.0,
        player_one_color_wins=2,
        player_two_color_wins=2,
        p1_as_player_one_wins=2,
        p1_as_player_two_wins=1,
        p2_as_player_one_wins=1,
        p2_as_player_two_wins=0,
        opening_plies=4,
        unique_openings=2,
        paired_openings=2,
        p1_pair_wins=1,
        p1_pair_losses=0,
        pair_ties=1,
    )


class _FakeArena:
    arenas: list[_FakeArena] = []

    def __init__(
        self,
        player1: Player,
        player2: Player,
        game: PenteGame,
        **kwargs: object,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.kwargs = kwargs
        self.arenas.append(self)

    def play_games(self, games: int) -> ArenaStats:
        self.games = games
        return _arena_stats()


class SearchQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeArena.arenas.clear()
        self.game = PenteGame(9, ruleset=PenteRuleset.FREESTYLE)
        self.current = _FakeNet()
        self.previous = _FakeNet()

    def test_evaluates_three_opponents_and_all_search_tactical_cases(self) -> None:
        created: list[dict[str, object]] = []
        players: list[_FakePlayer] = []

        def build(
            net: object,
            mcts: object,
            name: str,
            **kwargs: object,
        ) -> _FakePlayer:
            del mcts
            player = _FakePlayer(self.game)
            players.append(player)
            created.append({"net": net, "name": name, **kwargs})
            return player

        tactical = TacticalSuiteStats(
            cases=6,
            correct=4,
            accuracy=2.0 / 3.0,
            mean_expected_policy_mass=0.4,
            category_accuracy={"capture_win": 1.0, "line_win": 0.5, "line_block": 0.5},
        )
        with (
            patch("src.verification.search_quality.build_player", side_effect=build),
            patch("src.verification.search_quality.Arena", _FakeArena),
            patch(
                "src.verification.search_quality.evaluate_tactical_suite",
                return_value=tactical,
            ),
        ):
            report = evaluate_search_quality(
                self.current,  # type: ignore[arg-type]
                self.previous,  # type: ignore[arg-type]
                self.game,
                MCTSArgs(num_simulations=2),
                games=4,
                opening_plies=4,
                seed=40,
                search_backend="cpp",
                native_worker_threads=3,
            )

        self.assertEqual("cpp", report.backend)
        self.assertEqual(3, len(_FakeArena.arenas))
        self.assertEqual("_FakePlayer", _FakeArena.arenas[0].player2.__class__.__name__)
        self.assertIsInstance(_FakeArena.arenas[1].player2, RandomPlayer)
        self.assertIsInstance(_FakeArena.arenas[2].player2, TacticalHeuristicPlayer)
        self.assertEqual(5, len(created))
        self.assertEqual(
            ["current", "previous", "current", "current", "current-tactical"],
            [call["name"] for call in created],
        )
        self.assertEqual(
            [self.current, self.previous, self.current, self.current, self.current],
            [call["net"] for call in created],
        )
        self.assertEqual([41, 42, 43, 45, 46], [call["seed"] for call in created])
        self.assertTrue(all(call["search_backend"] == "cpp" for call in created))
        self.assertTrue(all(call["native_worker_threads"] == 3 for call in created))
        self.assertEqual(6, len(report.search_tactical.cases))
        self.assertEqual(6, players[-1].reset_calls)
        self.assertEqual(2.0 / 3.0, report.current_direct_tactical.accuracy)
        self.assertEqual(4, report.current_vs_previous.game.games)
        self.assertEqual(1, report.current_vs_previous.pair.current_pair_wins)

    def test_python_and_cpp_reports_share_opening_seeds(self) -> None:
        arenas: list[_FakeArena] = []

        class ArenaWithRecord(_FakeArena):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                arenas.append(self)

        def build(*args: object, **kwargs: object) -> _FakePlayer:
            return _FakePlayer(self.game)

        tactical = TacticalSuiteStats(6, 6, 1.0, 0.5, {"all": 1.0})
        with (
            patch("src.verification.search_quality.build_player", side_effect=build),
            patch("src.verification.search_quality.Arena", ArenaWithRecord),
            patch("src.verification.search_quality.evaluate_tactical_suite", return_value=tactical),
        ):
            evaluate_search_quality(
                self.current,  # type: ignore[arg-type]
                self.previous,  # type: ignore[arg-type]
                self.game,
                MCTSArgs(num_simulations=1),
                games=2,
                seed=12,
                search_backend="python",
            )
            first_seeds = [
                cast(np.random.Generator, arena.kwargs["rng"]).bit_generator.state
                for arena in arenas
            ]
            arenas.clear()
            evaluate_search_quality(
                self.current,  # type: ignore[arg-type]
                self.previous,  # type: ignore[arg-type]
                self.game,
                MCTSArgs(num_simulations=1),
                games=2,
                seed=12,
                search_backend="cpp",
            )
            second_seeds = [
                cast(np.random.Generator, arena.kwargs["rng"]).bit_generator.state
                for arena in arenas
            ]

        # The API gives each backend the same deterministic offsets.  Compare
        # generated opening streams rather than RNG object identities.
        self.assertEqual(3, len(first_seeds))
        self.assertEqual(first_seeds, second_seeds)

    def test_validation_rejects_invalid_search_configuration(self) -> None:
        cases: tuple[tuple[dict[str, object], str], ...] = (
            ({"games": 0}, "games must be positive"),
            ({"games": 1}, "games must be at least two"),
            ({"games": 3}, "games must be even for paired openings"),
            ({"opening_plies": -1}, "opening_plies cannot be negative"),
            ({"seed": -1}, "seed cannot be negative"),
            ({"native_worker_threads": 0}, "native_worker_threads must be positive"),
            ({"search_backend": "bogus"}, "Search backend must be 'python' or 'cpp'"),
        )
        for overrides, message in cases:
            arguments: dict[str, object] = {
                "games": 2,
                "opening_plies": 4,
                "seed": 40,
                "search_backend": "python",
                "native_worker_threads": 1,
            }
            arguments.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                evaluate_search_quality(
                    self.current,  # type: ignore[arg-type]
                    self.previous,  # type: ignore[arg-type]
                    self.game,
                    MCTSArgs(num_simulations=1),
                    **arguments,  # type: ignore[arg-type]
                )

    def test_validation_rejects_nonpositive_simulations(self) -> None:
        invalid_args = object.__new__(MCTSArgs)
        object.__setattr__(invalid_args, "num_simulations", 0)
        with self.assertRaisesRegex(ValueError, "num_simulations must be positive"):
            evaluate_search_quality(
                self.current,  # type: ignore[arg-type]
                self.previous,  # type: ignore[arg-type]
                self.game,
                invalid_args,
                games=2,
            )

    def test_validation_rejects_incompatible_model_configuration(self) -> None:
        setattr(self.current, "config", SimpleNamespace(board_size=8, action_size=64))
        with self.assertRaisesRegex(ValueError, "current board size"):
            evaluate_search_quality(
                self.current,  # type: ignore[arg-type]
                self.previous,  # type: ignore[arg-type]
                self.game,
                MCTSArgs(num_simulations=1),
                games=2,
            )

        setattr(self.current, "config", SimpleNamespace(board_size=9, action_size=81, num_channels=8))
        setattr(self.previous, "config", SimpleNamespace(board_size=9, action_size=81, num_channels=4))
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            evaluate_search_quality(
                self.current,  # type: ignore[arg-type]
                self.previous,  # type: ignore[arg-type]
                self.game,
                MCTSArgs(num_simulations=1),
                games=2,
            )

    def test_illegal_tactical_selection_fails_closed(self) -> None:
        class IllegalPlayer(_FakePlayer):
            def play(
                self,
                game: PenteGame,
                board: object,
                player: int,
                debug: bool = False,
            ) -> int:
                del game, board, player, debug
                return 0

        created = 0

        def build(*args: object, **kwargs: object) -> _FakePlayer:
            nonlocal created
            created += 1
            return IllegalPlayer(self.game) if created == 5 else _FakePlayer(self.game)

        with (
            patch("src.verification.search_quality.build_player", side_effect=build),
            patch("src.verification.search_quality.Arena", _FakeArena),
            patch(
                "src.verification.search_quality.evaluate_tactical_suite",
                return_value=TacticalSuiteStats(6, 6, 1.0, 0.4, {"all": 1.0}),
            ),
            self.assertRaisesRegex(ValueError, "illegal tactical action"),
        ):
            evaluate_search_quality(
                self.current,  # type: ignore[arg-type]
                self.previous,  # type: ignore[arg-type]
                self.game,
                MCTSArgs(num_simulations=1),
                games=2,
            )

    def test_statistical_parity_uses_tolerances_and_not_raw_counts(self) -> None:
        config = SearchQualityConfig(
            games=2,
            opening_plies=4,
            seed=12,
            mcts_args=MCTSArgs(num_simulations=1),
            native_worker_threads=1,
            board_size=9,
            ruleset="freestyle",
        )
        with (
            patch(
                "src.verification.search_quality.build_player",
                side_effect=lambda *args, **kwargs: _FakePlayer(self.game),
            ),
            patch("src.verification.search_quality.Arena", _FakeArena),
            patch("src.verification.search_quality.evaluate_tactical_suite", return_value=TacticalSuiteStats(6, 6, 1.0, 0.4, {"all": 1.0})),
        ):
            python_report = evaluate_search_quality(
                self.current,  # type: ignore[arg-type]
                self.previous,  # type: ignore[arg-type]
                self.game,
                config.mcts_args,
                config.games,
                config.opening_plies,
                config.seed,
                "python",
            )
        cpp_report = replace(
            python_report,
            backend="cpp",
            search_tactical=replace(
                python_report.search_tactical,
                accuracy=python_report.search_tactical.accuracy + 0.1,
            ),
        )

        parity = summarize_statistical_parity(
            python_report,
            cpp_report,
            StatisticalParityCriteria(
                maximum_tactical_accuracy_difference=0.2,
                require_wilson_overlap=True,
            ),
        )
        self.assertTrue(parity.passed, parity.failures)
        self.assertEqual(3, len(parity.matches))

    def test_parity_rejects_statistically_separate_rates(self) -> None:
        config = SearchQualityConfig(
            games=2,
            opening_plies=0,
            seed=1,
            mcts_args=MCTSArgs(num_simulations=1),
            native_worker_threads=1,
            board_size=9,
            ruleset="freestyle",
        )
        report = _report_with_rate(config, "python", 1.0)
        other = _report_with_rate(config, "cpp", 0.0)
        parity = summarize_statistical_parity(report, other)
        self.assertFalse(parity.passed)
        self.assertTrue(any("decisive" in failure for failure in parity.failures))


def _report_with_rate(
    config: SearchQualityConfig,
    backend: str,
    rate: float,
) -> BackendSearchQualityReport:
    games = 100
    current_wins = int(games * rate)
    opponent_wins = games - current_wins
    game = GameResults(
        games=games,
        current_wins=current_wins,
        opponent_wins=opponent_wins,
        draws=0,
        decisive_games=games,
        current_decisive_win_rate=rate,
        current_decisive_win_rate_95pct=WilsonInterval(*(
            (0.8, 1.0) if rate else (0.0, 0.2)
        )),
        current_score=rate,
        current_elo=0.0,
    )
    color = ColorResults(current_wins // 2, current_wins - current_wins // 2, 50, 50)
    pair = PairResults(0, 50, 50, current_wins // 2, opponent_wins // 2, 0, rate, game.current_decisive_win_rate_95pct)
    match = MatchReport("previous", game, color, pair, 1.0, 1.0, 1.0)
    direct = DirectTacticalMetrics(6, 6, 1.0, 0.4, (("all", 1.0),))
    tactical = SearchTacticalReport((), 0, 0.0)
    return BackendSearchQualityReport(
        backend=cast_backend(backend),
        config=config,
        current_vs_previous=match,
        current_vs_random=replace(match, opponent="random"),
        current_vs_heuristic=replace(match, opponent="heuristic"),
        current_direct_tactical=direct,
        previous_direct_tactical=direct,
        search_tactical=tactical,
        elapsed_seconds=1.0,
    )


def cast_backend(value: str) -> SearchBackend:
    return cast(SearchBackend, value)


if __name__ == "__main__":
    unittest.main()
