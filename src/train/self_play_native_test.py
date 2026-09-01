from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.game.game import TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.batched import BatchedSearchAccumulator
from src.mcts.mcts_v2 import MCTSArgs, SearchTelemetry
from src.mcts.native_backend import (
    NativeBackendUnavailableError,
    NativeWave,
)
from src.train.self_play_args import SelfPlayTrainerArgs
from src.train.self_play_generation import SelfPlayGenerator
from src.train.self_play_metrics import collect_self_play_metrics


class _FirstLegalEvaluator:
    device = torch.device("cpu")

    def __init__(self, game: PenteGame) -> None:
        self.game = game

    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        policies, values = self.evaluate_batch([position])
        return policies[0], float(values[0])

    def evaluate_batch(
        self,
        positions: Sequence[PenteBoard],
    ) -> tuple[np.ndarray, np.ndarray]:
        policies = np.zeros(
            (len(positions), self.game.get_action_size()),
            dtype=np.float32,
        )
        for index, position in enumerate(positions):
            legal = np.flatnonzero(
                self.game.get_valid_moves(position, position.current_player)
            )
            policies[index, int(legal[0])] = 1.0
        return policies, np.zeros(len(positions), dtype=np.float32)


@dataclass
class _FakeSlot:
    position: PenteBoard
    simulations: int = 0


class _FakeNativeBackend:
    def __init__(
        self,
        game: PenteGame,
        args: MCTSArgs,
        max_active_games: int,
        worker_threads: int,
        *,
        fail_first_wave: bool = False,
        failed_wave_terminal_leaves: int = 0,
        mismatch_terminal: bool = False,
    ) -> None:
        self.game = game
        self.args = args
        self.capacity = max_active_games
        self.thread_count = worker_threads
        self.fail_first_wave = fail_first_wave
        self.failed_wave_terminal_leaves = failed_wave_terminal_leaves
        self.mismatch_terminal = mismatch_terminal
        self.slots: dict[int, _FakeSlot] = {}
        self.added_slots: list[int] = []
        self.removed_slots: list[int] = []
        self.add_active_counts: list[int] = []
        self._pending: tuple[int, ...] | None = None
        self._token = 0

    @property
    def active_count(self) -> int:
        return len(self.slots)

    def add_root(
        self,
        position: PenteBoard,
        **_kwargs: object,
    ) -> int:
        if self.active_count >= self.capacity:
            raise AssertionError("fake native capacity exhausted")
        slot = next(index for index in range(self.capacity) if index not in self.slots)
        self.add_active_counts.append(self.active_count)
        self.slots[slot] = _FakeSlot(position)
        self.added_slots.append(slot)
        return slot

    def slot_complete(self, slot: int) -> bool:
        return self.slots[slot].simulations >= self.args.num_simulations

    def slot_simulations(self, slot: int) -> int:
        return self.slots[slot].simulations

    def root_policy(self, slot: int) -> np.ndarray:
        state = self.slots[slot]
        legal = np.flatnonzero(
            self.game.get_valid_moves(state.position, state.position.current_player)
        )
        policy = np.zeros(self.game.get_action_size(), dtype=np.float32)
        policy[int(legal[0])] = 1.0
        return policy

    def slot_telemetry(self, slot: int) -> SearchTelemetry:
        state = self.slots[slot]
        legal_actions = int(
            np.count_nonzero(
                self.game.get_valid_moves(
                    state.position,
                    state.position.current_player,
                )
            )
        )
        return SearchTelemetry(
            simulations=state.simulations,
            evaluator_calls=state.simulations,
            evaluated_positions=state.simulations,
            invalid_policy_fallbacks=0,
            zero_visit_fallbacks=0,
            max_depth=1,
            root_legal_actions=legal_actions,
            root_edge_visits=state.simulations,
            root_children_visited=min(state.simulations, legal_actions),
            root_visit_entropy=0.0,
            root_max_visit_share=1.0 if state.simulations else 0.0,
            root_collapse_eligible=False,
            root_search_collapsed=False,
            mean_inference_batch_size=1.0 if state.simulations else 0.0,
        )

    def evaluate_wave(self) -> NativeWave:
        if self._pending is None:
            self._token += 1
            self._pending = tuple(
                slot
                for slot in self.slots
                if not self.slot_complete(slot)
            )
            if self.fail_first_wave:
                self.fail_first_wave = False
                for slot in self._pending:
                    self.slots[slot].simulations += self.failed_wave_terminal_leaves
                raise RuntimeError("transient fake native wave failure")

        pending = self._pending
        assert pending is not None
        self._pending = None
        for slot in pending:
            self.slots[slot].simulations += 1
        size = len(pending)
        return NativeWave(
            token=self._token,
            size=size,
            raw_size=size,
            host_to_device_seconds=0.25,
            model_inference_seconds=0.5,
            device_to_host_seconds=0.25,
            inference_wait_seconds=0.125,
        )

    def native_timing_telemetry(self) -> dict[str, object]:
        def stage(workers: int, busy: float) -> dict[str, object]:
            return {
                "wall_seconds": 1.0,
                "worker": {
                    "workers": workers,
                    "wall_seconds": 1.0,
                    "callback_busy_seconds": busy,
                },
            }

        return {
            "latest_generation": {
                "select": stage(self.thread_count, 0.5),
                "dedup": stage(0, 0.0),
                "features": stage(self.thread_count, 0.5),
                "backup": stage(self.thread_count, 0.5),
            }
        }

    def advance_root(
        self,
        slot: int,
        action: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        if self._pending is not None:
            raise AssertionError("fake native root advanced with pending wave")
        state = self.slots[slot]
        state.position, _ = self.game.apply_action(
            state.position,
            state.position.current_player,
            action,
        )
        state.simulations = 0
        return {}

    def root_terminal(self, slot: int) -> TerminalResult:
        state = self.slots[slot]
        result = self.game.check_game_end(state.position)
        if self.mismatch_terminal and not result.is_terminal:
            return TerminalResult.draw()
        return result

    def remove(self, slot: int) -> None:
        self.removed_slots.append(slot)
        del self.slots[slot]


def _fake_factory(
    created: list[_FakeNativeBackend],
    *,
    fail_first_wave: bool = False,
    failed_wave_terminal_leaves: int = 0,
    mismatch_terminal: bool = False,
) -> Callable[..., Any]:
    def factory(
        game: PenteGame,
        _evaluator: object,
        args: MCTSArgs,
        **kwargs: object,
    ) -> _FakeNativeBackend:
        backend = _FakeNativeBackend(
            game,
            args,
            int(cast(int, kwargs["max_active_games"])),
            int(cast(int, kwargs["worker_threads"])),
            fail_first_wave=fail_first_wave,
            failed_wave_terminal_leaves=failed_wave_terminal_leaves,
            mismatch_terminal=mismatch_terminal,
        )
        created.append(backend)
        return backend

    return factory


class SelfPlayNativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        self.evaluator = _FirstLegalEvaluator(self.game)
        self.args = MCTSArgs(num_simulations=1, root_noise_epsilon=0.0)

    def make_generator(
        self,
        *,
        seed: int = 41,
        backend: str | None = None,
        factory: Callable[..., Any] | None = None,
        native_worker_threads: int = 1,
    ) -> SelfPlayGenerator:
        options: dict[str, Any] = {
            "native_worker_threads": native_worker_threads,
            "_native_backend_factory": factory,
        }
        if backend is not None:
            options["search_backend"] = backend
        return SelfPlayGenerator(
            self.game,
            self.evaluator,
            self.args,
            temp_threshold=0,
            rng=np.random.default_rng(seed),
            **options,
        )

    @staticmethod
    def game_signature(game: object) -> tuple[object, ...]:
        played = cast(Any, game)
        examples = tuple(
            (
                example.position.state_key(),
                example.policy.tobytes(),
                example.value,
            )
            for example in played.examples
        )
        return played.actions, played.winner, played.win_reason, examples

    def test_default_and_explicit_python_paths_match_without_loading_native(self) -> None:
        with patch(
            "src.mcts.native_backend.load_native_extension",
            side_effect=AssertionError("Python self-play loaded native extension"),
        ):
            default_games, default_batches = self.make_generator().play_games(2, 2)
            explicit_games, explicit_batches = self.make_generator(
                backend="python"
            ).play_games(2, 2)

        self.assertEqual(
            [self.game_signature(game) for game in default_games],
            [self.game_signature(game) for game in explicit_games],
        )
        self.assertEqual(default_batches, explicit_batches)
        self.assertTrue(
            all(
                batch.native_select_seconds == 0.0
                and batch.native_deduplication_seconds == 0.0
                and batch.native_feature_encode_seconds == 0.0
                and batch.native_backup_seconds == 0.0
                and batch.model_inference_seconds == 0.0
                and batch.native_worker_threads == 0
                for batch in default_batches
            )
        )

    def test_missing_native_extension_is_actionable(self) -> None:
        generator = self.make_generator(backend="cpp")
        with patch(
            "src.mcts.native_backend.importlib.import_module",
            side_effect=ModuleNotFoundError("kb_pente_native"),
        ):
            with self.assertRaisesRegex(
                NativeBackendUnavailableError,
                "uv pip install --no-build-isolation ./native/torch",
            ):
                generator.play_game()

    def test_fake_native_churn_reuses_slots_and_drains_capacity(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created),
            native_worker_threads=2,
        )

        games, batches = generator.play_games(3, 2)

        self.assertEqual(3, len(games))
        self.assertEqual(1, len(created))
        backend = created[0]
        self.assertEqual(2, backend.capacity)
        self.assertEqual(0, backend.active_count)
        self.assertEqual(3, len(backend.removed_slots))
        self.assertTrue(all(count < backend.capacity for count in backend.add_active_counts))
        self.assertGreaterEqual(len(backend.added_slots), 3)
        self.assertTrue(batches)

        for played in games:
            self.assertEqual(len(played.actions), len(played.root_telemetry))
            self.assertEqual(
                list(range(1, len(played.actions) + 1)),
                [telemetry.simulations for telemetry in played.root_telemetry],
            )
            self.assertEqual(
                [telemetry.simulations for telemetry in played.root_telemetry],
                [telemetry.evaluator_calls for telemetry in played.root_telemetry],
            )

        metrics = collect_self_play_metrics(games, batches, elapsed_seconds=1.0)
        self.assertEqual(2, metrics["native_worker_threads"])
        self.assertEqual(25.0, metrics["native_worker_busy_percent"])
        self.assertGreater(metrics["mcts_select_seconds"], 0.0)
        self.assertGreater(metrics["mcts_dedup_seconds"], 0.0)
        self.assertGreater(metrics["mcts_feature_encode_seconds"], 0.0)
        self.assertGreater(metrics["mcts_backup_seconds"], 0.0)
        self.assertGreater(metrics["model_inference_seconds"], 0.0)
        self.assertGreater(metrics["host_to_device_seconds"], 0.0)
        self.assertGreater(metrics["device_to_host_seconds"], 0.0)
        self.assertGreater(metrics["inference_wait_seconds"], 0.0)
        self.assertGreater(metrics["native_worker_busy_seconds"], 0.0)
        self.assertGreater(metrics["native_worker_capacity_seconds"], 0.0)

    def test_native_terminal_mismatch_releases_slot_before_failing(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created, mismatch_terminal=True),
        )

        with self.assertRaisesRegex(RuntimeError, "Native/Python terminal mismatch"):
            generator.play_game()

        self.assertEqual(1, len(created))
        self.assertEqual(0, created[0].active_count)
        self.assertEqual(1, len(created[0].removed_slots))

    def test_native_retry_counts_selection_once(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(
                created,
                fail_first_wave=True,
                failed_wave_terminal_leaves=1,
            ),
        )

        coordinator = generator._coordinator(1)
        active = generator._new_active_game()
        coordinator.start(active)
        accumulator = BatchedSearchAccumulator(1)
        with self.assertRaisesRegex(RuntimeError, "transient fake native wave failure"):
            coordinator.evaluate_wave([active], accumulator)
        coordinator.evaluate_wave([active], accumulator)
        coordinator.remove(active)

        self.assertEqual(1, len(created))
        self.assertEqual(0, created[0].active_count)
        batch = accumulator.telemetry()
        self.assertEqual(2, batch.selected_leaves)
        self.assertEqual(1, batch.evaluation_requests)
        self.assertEqual(1, batch.unique_evaluations)

    def test_trainer_args_normalize_integer_worker_count_and_validate_backend(self) -> None:
        args = SelfPlayTrainerArgs(
            start_iteration=0,
            professional_games_training_iterations=0,
            self_play_training_iterations=0,
            temp_threshold=0,
            mcts_args=self.args,
            watch_training_raw_dataset_filepath="unused",
            watch_training_processed_dataset_filepath="unused",
            force_watch_training_raw_dataset_processing=False,
            native_worker_threads=cast(Any, np.int64(3)),
        )
        self.assertIs(int, type(args.native_worker_threads))
        self.assertEqual(3, args.native_worker_threads)
        with self.assertRaisesRegex(ValueError, "Search backend"):
            SelfPlayTrainerArgs(
                start_iteration=0,
                professional_games_training_iterations=0,
                self_play_training_iterations=0,
                temp_threshold=0,
                mcts_args=self.args,
                watch_training_raw_dataset_filepath="unused",
                watch_training_processed_dataset_filepath="unused",
                force_watch_training_raw_dataset_processing=False,
                search_backend=cast(Any, "native"),
            )


if __name__ == "__main__":
    unittest.main()
