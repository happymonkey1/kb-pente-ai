from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.model.model_v1 import PenteNetConfig
from src.verification.search_quality_reports import (
    BackendSearchQualityReport,
    ColorResults,
    DirectTacticalMetrics,
    GameResults,
    MatchReport,
    PairResults,
    SearchQualityConfig,
    SearchTacticalReport,
    StatisticalParityCriteria,
    StatisticalParityReport,
    WilsonInterval,
)


def _load_script():
    path = Path(__file__).parents[2] / "script" / "verify-native-search-quality.py"
    spec = importlib.util.spec_from_file_location("verify_native_search_quality", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load native-search-quality script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_script()


class _FakeModel:
    def __init__(self) -> None:
        self.config = PenteNetConfig(
            board_size=9,
            action_size=81,
            num_res_blocks=1,
            num_channels=16,
            hidden_fc_size=64,
        )


def _reports() -> tuple[
    BackendSearchQualityReport,
    BackendSearchQualityReport,
    StatisticalParityReport,
]:
    config = SearchQualityConfig(
        games=2,
        opening_plies=4,
        seed=53,
        mcts_args=MCTSArgs(num_simulations=1),
        native_worker_threads=1,
        board_size=9,
        ruleset="freestyle",
    )
    game = GameResults(
        games=2,
        current_wins=1,
        opponent_wins=1,
        draws=0,
        decisive_games=2,
        current_decisive_win_rate=0.5,
        current_decisive_win_rate_95pct=WilsonInterval(0.1, 0.9),
        current_score=0.5,
        current_elo=0.0,
    )
    color = ColorResults(1, 0, 1, 1)
    pair = PairResults(4, 1, 1, 0, 0, 1, 0.0, WilsonInterval(0.0, 1.0))
    match = MatchReport("previous", game, color, pair, 1.0, 0.1, 20.0)
    direct = DirectTacticalMetrics(6, 3, 0.5, 0.2, (("line_win", 0.5),))
    tactical = SearchTacticalReport((), 0, 0.0)
    python = BackendSearchQualityReport(
        "python", config, match, replace(match, opponent="random"),
        replace(match, opponent="heuristic"), direct, direct, tactical, 0.3,
    )
    cpp = replace(python, backend="cpp", elapsed_seconds=0.4)
    parity = StatisticalParityReport("python", "cpp", (), 0.0, True, ())
    return python, cpp, parity


class SearchQualityCliTest(unittest.TestCase):
    def test_parser_defaults_and_help_do_not_preflight(self) -> None:
        parser = _SCRIPT.build_parser()
        defaults = parser.parse_args(["current", "previous"])
        self.assertEqual(200, defaults.games)
        self.assertEqual(16, defaults.simulations)
        self.assertTrue(defaults.require_wilson_overlap)
        overrides = parser.parse_args(
            [
                "current",
                "previous",
                "--games",
                "20",
                "--simulations",
                "3",
                "--maximum-tactical-accuracy-difference",
                "0.25",
                "--no-require-wilson-overlap",
            ],
        )
        self.assertEqual(20, overrides.games)
        self.assertEqual(3, overrides.simulations)
        self.assertEqual(0.25, overrides.maximum_tactical_accuracy_difference)
        self.assertFalse(overrides.require_wilson_overlap)
        with patch.object(_SCRIPT, "load_native_extension") as preflight:
            parser.format_help()
            preflight.assert_not_called()

    def test_validation_happens_before_native_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.pth.tar"
            previous = Path(directory) / "previous.pth.tar"
            current.write_bytes(b"current")
            previous.write_bytes(b"previous")
            with (
                patch.object(
                    _SCRIPT,
                    "load_native_extension",
                    side_effect=AssertionError("preflight ran before validation"),
                ),
                self.assertRaisesRegex(ValueError, "even for paired openings"),
            ):
                _SCRIPT.main([str(current), str(previous), "--games", "3"])

            with (
                patch.object(
                    _SCRIPT,
                    "load_native_extension",
                    side_effect=AssertionError("preflight ran before validation"),
                ),
                self.assertRaisesRegex(ValueError, "simulations must be positive"),
            ):
                _SCRIPT.main([str(current), str(previous), "--simulations", "0"])

    def test_load_model_uses_explicit_architecture_and_ruleset(self) -> None:
        parser = _SCRIPT.build_parser()
        args = parser.parse_args(
            [
                "current",
                "previous",
                "--board-size",
                "9",
                "--ruleset",
                "freestyle",
                "--model-blocks",
                "2",
                "--model-channels",
                "8",
                "--model-hidden-size",
                "32",
            ],
        )
        game = PenteGame(9, ruleset=PenteRuleset.FREESTYLE)
        calls: list[dict[str, object]] = []

        class FakePenteNet:
            def __init__(self, device: torch.device, **kwargs: object) -> None:
                calls.append({"device": device, **kwargs})

            @staticmethod
            def load_checkpoint_from_path(
                path: str,
                model: object,
                expected_ruleset: str,
            ) -> int:
                calls.append(
                    {
                        "path": path,
                        "model": model,
                        "expected_ruleset": expected_ruleset,
                    },
                )
                return 7

        with patch.object(_SCRIPT, "PenteNet", FakePenteNet):
            _, iteration = _SCRIPT._load_model(
                Path("current.pth.tar"),
                torch.device("cpu"),
                game,
                args,
            )
        self.assertEqual(7, iteration)
        self.assertEqual(
            {
                "device": torch.device("cpu"),
                "board_size": 9,
                "action_size": 81,
                "num_res_blocks": 2,
                "num_channels": 8,
                "hidden_fc_size": 32,
            },
            calls[0],
        )
        self.assertEqual("freestyle", calls[1]["expected_ruleset"])

    def test_preflight_once_propagates_identical_calls_and_serializes_metadata(self) -> None:
        python_report, cpp_report, parity = _reports()
        events: list[str] = []
        api_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        parity_calls: list[tuple[object, ...]] = []
        load_calls: list[tuple[object, ...]] = []
        fake_models = [_FakeModel(), _FakeModel()]

        def load_model(*args: object, **kwargs: object) -> tuple[_FakeModel, int]:
            del kwargs
            events.append("load")
            load_calls.append(args)
            return fake_models[len(load_calls) - 1], len(events)

        def evaluate(*args: object, **kwargs: object) -> BackendSearchQualityReport:
            events.append("evaluate")
            api_calls.append((args, kwargs))
            return python_report if len(api_calls) == 1 else cpp_report

        def summarize(*args: object) -> StatisticalParityReport:
            parity_calls.append(args)
            return parity

        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.pth.tar"
            previous = Path(directory) / "previous.pth.tar"
            current.write_bytes(b"current")
            previous.write_bytes(b"previous")
            printed: list[str] = []
            with (
                patch.object(_SCRIPT, "load_native_extension", side_effect=lambda: events.append("preflight")),
                patch.object(_SCRIPT, "_load_model", side_effect=load_model),
                patch.object(_SCRIPT, "evaluate_search_quality", side_effect=evaluate),
                patch.object(
                    _SCRIPT,
                    "summarize_statistical_parity",
                    side_effect=summarize,
                ),
                patch.object(_SCRIPT, "_sha256", side_effect=lambda path: f"hash-{path.name}"),
                patch.object(_SCRIPT, "source_fingerprint", return_value="source-hash"),
                patch.object(_SCRIPT, "_git_commit", return_value="commit"),
                patch.object(_SCRIPT, "_runtime_metadata", return_value={"device": "cpu"}),
                patch("builtins.print", side_effect=printed.append),
            ):
                self.assertEqual(
                    0,
                    _SCRIPT.main(
                        [
                            str(current),
                            str(previous),
                            "--board-size",
                            "9",
                            "--ruleset",
                            "freestyle",
                            "--games",
                            "2",
                            "--simulations",
                            "1",
                            "--maximum-decisive-win-rate-difference",
                            "0.2",
                            "--maximum-paired-win-rate-difference",
                            "0.3",
                            "--maximum-tactical-accuracy-difference",
                            "0.4",
                            "--no-require-wilson-overlap",
                        ],
                    ),
                )

        self.assertEqual(["preflight", "load", "load", "evaluate", "evaluate"], events)
        self.assertEqual([current.resolve(), previous.resolve()], [call[0] for call in load_calls])
        self.assertEqual(2, len(api_calls))
        self.assertIs(api_calls[0][0][0], fake_models[0])
        self.assertIs(api_calls[0][0][1], fake_models[1])
        self.assertIs(api_calls[0][0][3], api_calls[1][0][3])
        self.assertEqual("python", api_calls[0][1]["search_backend"])
        self.assertEqual("cpp", api_calls[1][1]["search_backend"])
        output = json.loads(printed[0])
        self.assertEqual(str(current.resolve()), output["checkpoints"]["current"]["path"])
        self.assertEqual("hash-current.pth.tar", output["checkpoints"]["current"]["sha256"])
        self.assertEqual("source-hash", output["repository"]["source_fingerprint_sha256"])
        self.assertEqual("commit", output["repository"]["commit"])
        self.assertEqual("cpu", output["runtime"]["device"])
        self.assertEqual("python", output["reports"]["python"]["backend"])
        self.assertEqual("cpp", output["reports"]["cpp"]["backend"])
        self.assertTrue(output["parity"]["passed"])
        self.assertEqual(1, len(parity_calls))
        self.assertEqual(
            StatisticalParityCriteria(
                maximum_decisive_win_rate_difference=0.2,
                maximum_paired_win_rate_difference=0.3,
                maximum_tactical_accuracy_difference=0.4,
                require_wilson_overlap=False,
            ),
            parity_calls[0][2],
        )
        self.assertEqual(
            {
                "maximum_decisive_win_rate_difference": 0.2,
                "maximum_paired_win_rate_difference": 0.3,
                "maximum_tactical_accuracy_difference": 0.4,
                "require_wilson_overlap": False,
            },
            output["criteria"],
        )
        self.assertEqual(printed[0], _SCRIPT._stable_json(json.loads(printed[0])))
        self.assertEqual(1, output["schema_version"])

    def test_failed_parity_returns_nonzero(self) -> None:
        python_report, cpp_report, parity = _reports()
        failed = replace(parity, passed=False, failures=("rate drift",))
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.pth.tar"
            previous = Path(directory) / "previous.pth.tar"
            current.write_bytes(b"current")
            previous.write_bytes(b"previous")
            with (
                patch.object(_SCRIPT, "load_native_extension") as preflight,
                patch.object(_SCRIPT, "_load_model", side_effect=[(_FakeModel(), 1), (_FakeModel(), 0)]),
                patch.object(_SCRIPT, "evaluate_search_quality", side_effect=[python_report, cpp_report]),
                patch.object(_SCRIPT, "summarize_statistical_parity", return_value=failed),
                patch.object(_SCRIPT, "_sha256", return_value="hash"),
                patch.object(_SCRIPT, "source_fingerprint", return_value="source"),
                patch.object(_SCRIPT, "_git_commit", return_value="commit"),
                patch.object(_SCRIPT, "_runtime_metadata", return_value={"device": "cpu"}),
                patch("builtins.print"),
            ):
                self.assertEqual(
                    1,
                    _SCRIPT.main(
                        [
                            str(current),
                            str(previous),
                            "--board-size",
                            "9",
                            "--ruleset",
                            "freestyle",
                            "--games",
                            "2",
                            "--simulations",
                            "1",
                        ],
                    ),
                )
                preflight.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
