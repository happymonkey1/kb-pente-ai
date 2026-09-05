from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any, cast
import unittest
from unittest.mock import ANY, patch

import torch

import main
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.train.nnet_player import NNetPlayer
from src.train.self_play_args import SelfPlayTrainerArgs


class _FakeNet:
    def parameters(self) -> tuple[object, ...]:
        return ()

    def eval(self) -> _FakeNet:
        return self

    def get_parameter_count(self) -> int:
        return 0


class _FakeOptimizer:
    pass


def _run_main(
    argv: list[str],
    events: list[tuple[object, ...]],
    loader: object,
) -> int:
    parser = main.build_parser()
    fake_net = _FakeNet()

    def write_manifest(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        trainer_args = args[8]
        assert isinstance(trainer_args, SelfPlayTrainerArgs)
        events.append(
            (
                "manifest",
                trainer_args.search_backend,
                trainer_args.native_worker_threads,
                trainer_args.native_search_cohorts,
                trainer_args.health_failure_policy.value,
            )
        )
        return {}

    def make_trainer(*args: object, **kwargs: object) -> Any:
        del kwargs
        trainer_args = args[4]
        assert isinstance(trainer_args, SelfPlayTrainerArgs)
        events.append(
            (
                "trainer",
                trainer_args.search_backend,
                trainer_args.native_worker_threads,
                trainer_args.native_search_cohorts,
                trainer_args.health_failure_policy.value,
            )
        )
        return SimpleNamespace(
            training_run_id="test-run",
            train=lambda: events.append(("train",)),
        )

    with (
        patch.object(main, "build_parser", return_value=parser),
        patch.object(main, "configure_logging"),
        patch.object(main, "select_torch_device", return_value=torch.device("cpu")),
        patch.object(main, "PenteNet", return_value=fake_net),
        patch.object(main.torch.optim, "AdamW", return_value=_FakeOptimizer()),
        patch.object(main, "JsonlMetricSink", return_value=object()),
        patch.object(main, "SelfPlayTrainer", side_effect=make_trainer),
        patch.object(main, "load_native_extension", side_effect=loader),
        patch.object(main, "write_run_manifest", side_effect=write_manifest),
        patch.object(main.sys, "argv", argv),
    ):
        return main.main()


class MainTest(unittest.TestCase):
    def test_parser_defaults_and_native_options(self) -> None:
        parser = main.build_parser()

        defaults = parser.parse_args([])
        native = parser.parse_args(
            [
                "--search-backend",
                "cpp",
                "--native-search-threads",
                "4",
                "--native-search-cohorts",
                "2",
            ]
        )

        self.assertEqual("python", defaults.search_backend)
        self.assertEqual(1, defaults.native_search_threads)
        self.assertEqual(1, defaults.native_search_cohorts)
        self.assertEqual("warn", defaults.health_failure_policy)
        self.assertEqual("cpp", native.search_backend)
        self.assertEqual(4, native.native_search_threads)
        self.assertEqual(2, native.native_search_cohorts)

        strict = parser.parse_args(
            ["--self-play-health-failure-policy", "error"]
        )
        self.assertEqual("error", strict.health_failure_policy)

    def test_cpp_training_preflights_before_manifest_and_training(self) -> None:
        events: list[tuple[object, ...]] = []

        argv = [
            "main.py",
            "--search-backend",
            "cpp",
            "--native-search-threads",
            "3",
            "--native-search-cohorts",
            "2",
            "--self-play-health-failure-policy",
            "error",
            "--self-play-iterations",
            "0",
            "--no-checkpoint",
            "--telemetry-file",
            "/tmp/kb-pente-main-test.jsonl",
        ]

        def load_extension() -> object:
            events.append(("load",))
            return object()

        self.assertEqual(0, _run_main(argv, events, load_extension))

        self.assertEqual(
            [
                ("load",),
                ("trainer", "cpp", 3, 2, "error"),
                ("manifest", "cpp", 3, 2, "error"),
                ("train",),
            ],
            events,
        )

    def test_default_python_training_skips_native_preflight(self) -> None:
        events: list[tuple[object, ...]] = []

        def fail_load() -> object:
            raise AssertionError("default Python training loaded native extension")

        self.assertEqual(
            0,
            _run_main(
                [
                    "main.py",
                    "--self-play-iterations",
                    "0",
                    "--no-checkpoint",
                    "--telemetry-file",
                    "/tmp/kb-pente-main-test-python.jsonl",
                ],
                events,
                fail_load,
            ),
        )
        self.assertEqual(
            [
                ("trainer", "python", 1, 1, "warn"),
                ("manifest", "python", 1, 1, "warn"),
                ("train",),
            ],
            events,
        )

    def test_direct_inference_with_cpp_is_extension_free(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        trainer_args = SelfPlayTrainerArgs(
            start_iteration=0,
            professional_games_training_iterations=0,
            self_play_training_iterations=0,
            temp_threshold=0,
            mcts_args=MCTSArgs(num_simulations=1),
            watch_training_raw_dataset_filepath="unused",
            watch_training_processed_dataset_filepath="unused",
            force_watch_training_raw_dataset_processing=False,
            search_backend="cpp",
            native_worker_threads=3,
        )
        program_args = argparse.Namespace(
            model="checkpoint.pth.tar",
            infer_mcts=False,
            seed=17,
            infer_games=1,
            arena_opening_plies=0,
        )

        captured_players: list[object] = []

        class FakeArena:
            def __init__(self, player1: object, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self.player1 = player1
                captured_players.append(player1)

            def play_games(self, num_games: int) -> SimpleNamespace:
                self.num_games = num_games
                return SimpleNamespace(p1_wins=1, p2_wins=0, draws=0, avg_moves=1.0)

        with (
            patch.object(main, "Arena", FakeArena),
            patch.object(
                main,
                "load_native_extension",
                side_effect=AssertionError("direct inference loaded native extension"),
            ),
        ):
            result = main.run_inference(
                program_args,
                game,
                cast(Any, _FakeNet()),
                trainer_args,
            )

        self.assertEqual(0, result)
        player = captured_players[0]
        self.assertIsInstance(player, NNetPlayer)
        assert isinstance(player, NNetPlayer)
        self.assertIsNone(player.mcts)

    def test_mcts_inference_uses_selected_builder_configuration(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        trainer_args = SelfPlayTrainerArgs(
            start_iteration=0,
            professional_games_training_iterations=0,
            self_play_training_iterations=0,
            temp_threshold=0,
            mcts_args=MCTSArgs(num_simulations=1),
            watch_training_raw_dataset_filepath="unused",
            watch_training_processed_dataset_filepath="unused",
            force_watch_training_raw_dataset_processing=False,
            search_backend="cpp",
            native_worker_threads=5,
        )
        program_args = argparse.Namespace(
            model="checkpoint.pth.tar",
            infer_mcts=True,
            seed=23,
            infer_games=1,
            arena_opening_plies=0,
        )

        class FakeArena:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs

            def play_games(self, num_games: int) -> SimpleNamespace:
                del num_games
                return SimpleNamespace(p1_wins=1, p2_wins=0, draws=0, avg_moves=1.0)

        with patch.object(main, "Arena", FakeArena), patch.object(
            main,
            "build_player",
            return_value=object(),
        ) as build:
            result = main.run_inference(
                program_args,
                game,
                cast(Any, _FakeNet()),
                trainer_args,
            )

        self.assertEqual(0, result)
        build.assert_called_once_with(
            ANY,
            None,
            "network",
            search_backend="cpp",
            game=game,
            mcts_args=trainer_args.mcts_args,
            seed=23,
            native_worker_threads=5,
        )


if __name__ == "__main__":
    unittest.main()
