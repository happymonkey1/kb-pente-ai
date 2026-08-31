from __future__ import annotations

import unittest
from unittest.mock import patch

from src.evaluation.tactical import TacticalSuiteStats
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.train.arena import ArenaStats
from src.verification.model_match import (
    ModelMatchCriteria,
    evaluate_model_match,
    summarize_model_match,
)


class _FakeNet:
    def eval(self) -> _FakeNet:
        return self


class _FakeArena:
    def __init__(
        self,
        player1: object,
        player2: object,
        game: object,
        **kwargs: object,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.kwargs = kwargs

    def play_games(self, num_games: int) -> ArenaStats:
        del num_games
        return ArenaStats(
            p1_wins=1,
            p2_wins=1,
            draws=0,
            avg_moves=4.0,
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


class ModelMatchVerificationTest(unittest.TestCase):
    def test_default_match_builds_direct_players_with_preserved_seed_offsets(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        candidate = _FakeNet()
        baseline = _FakeNet()
        created: list[dict[str, object]] = []

        def build(
            net: object,
            mcts: object,
            name: str,
            **kwargs: object,
        ) -> object:
            created.append({"net": net, "mcts": mcts, "name": name, **kwargs})
            return object()

        with (
            patch("src.verification.model_match.build_player", side_effect=build),
            patch("src.verification.model_match.Arena", _FakeArena),
            patch(
                "src.verification.model_match.evaluate_tactical_suite",
                side_effect=[self._tactical(1.0, 0.7), self._tactical(1.0, 0.7)],
            ),
        ):
            evaluate_model_match(
                candidate,  # type: ignore[arg-type]
                baseline,  # type: ignore[arg-type]
                game,
                games=2,
                seed=40,
                simulations=0,
            )

        self.assertEqual(["candidate", "baseline"], [call["name"] for call in created])
        self.assertEqual([41, 42], [call["seed"] for call in created])
        self.assertTrue(
            all(
                call["search_backend"] == "python"
                and call["native_worker_threads"] == 1
                and call["game"] is game
                and call["mcts_args"] is None
                and call["mcts"] is None
                for call in created
            )
        )
        self.assertIs(candidate, created[0]["net"])
        self.assertIs(baseline, created[1]["net"])

    def test_match_forwards_cpp_backend_threads_and_mcts_configuration(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        created: list[dict[str, object]] = []

        def build(
            net: object,
            mcts: object,
            name: str,
            **kwargs: object,
        ) -> object:
            created.append({"net": net, "mcts": mcts, "name": name, **kwargs})
            return object()

        with (
            patch("src.verification.model_match.build_player", side_effect=build),
            patch("src.verification.model_match.Arena", _FakeArena),
            patch(
                "src.verification.model_match.evaluate_tactical_suite",
                side_effect=[self._tactical(1.0, 0.7), self._tactical(1.0, 0.7)],
            ),
        ):
            evaluate_model_match(
                _FakeNet(),  # type: ignore[arg-type]
                _FakeNet(),  # type: ignore[arg-type]
                game,
                games=2,
                seed=7,
                simulations=3,
                criteria=ModelMatchCriteria(
                    minimum_games=2,
                    require_wins_as_both_colors=False,
                    require_tactical_non_regression=False,
                ),
                search_backend="cpp",
                native_worker_threads=4,
            )

        self.assertEqual([8, 9], [call["seed"] for call in created])
        self.assertTrue(
            all(
                call["search_backend"] == "cpp"
                and call["native_worker_threads"] == 4
                and call["game"] is game
                and call["mcts"] is None
                and isinstance(call["mcts_args"], MCTSArgs)
                and call["mcts_args"].num_simulations == 3  # type: ignore[union-attr]
                for call in created
            )
        )

    def test_passes_stronger_candidate_without_tactical_regression(self) -> None:
        stats = ArenaStats(
            p1_wins=75,
            p2_wins=20,
            draws=5,
            avg_moves=50.0,
            player_one_color_wins=50,
            player_two_color_wins=45,
            p1_as_player_one_wins=40,
            p1_as_player_two_wins=35,
            p2_as_player_one_wins=10,
            p2_as_player_two_wins=10,
            opening_plies=4,
            unique_openings=50,
            paired_openings=50,
            p1_pair_wins=35,
            p1_pair_losses=10,
            pair_ties=5,
        )
        candidate_tactical = self._tactical(accuracy=1.0, mass=0.7)
        baseline_tactical = self._tactical(accuracy=0.8, mass=0.6)

        report = summarize_model_match(
            stats,
            elapsed_seconds=20.0,
            candidate_tactical=candidate_tactical,
            baseline_tactical=baseline_tactical,
        )

        self.assertTrue(report.passed, report.failures)
        self.assertGreater(report.candidate_paired_win_rate_95pct_lower, 0.5)
        self.assertEqual(50, report.unique_openings)

    def test_reports_strength_color_and_tactical_failures(self) -> None:
        stats = ArenaStats(
            p1_wins=4,
            p2_wins=6,
            draws=0,
            avg_moves=20.0,
            player_one_color_wins=6,
            player_two_color_wins=4,
            p1_as_player_one_wins=4,
            p1_as_player_two_wins=0,
            p2_as_player_one_wins=2,
            p2_as_player_two_wins=4,
            opening_plies=4,
            unique_openings=5,
            paired_openings=5,
            p1_pair_wins=1,
            p1_pair_losses=4,
            pair_ties=0,
        )

        report = summarize_model_match(
            stats,
            elapsed_seconds=1.0,
            candidate_tactical=self._tactical(accuracy=0.5, mass=0.3),
            baseline_tactical=self._tactical(accuracy=1.0, mass=0.6),
        )

        self.assertFalse(report.passed)
        self.assertEqual(5, len(report.failures))

    @staticmethod
    def _tactical(accuracy: float, mass: float) -> TacticalSuiteStats:
        return TacticalSuiteStats(
            cases=6,
            correct=round(6 * accuracy),
            accuracy=accuracy,
            mean_expected_policy_mass=mass,
            category_accuracy={"fixture": accuracy},
        )


if __name__ == "__main__":
    unittest.main()
