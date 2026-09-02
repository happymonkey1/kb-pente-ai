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
    NativeWaveSubmission,
)
from src.train.self_play_args import SelfPlayTrainerArgs
from src.train.self_play_generation import PlayedGame, SelfPlayGenerator
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
        fail_first_wait: bool = False,
        failed_wave_terminal_leaves: int = 0,
        mismatch_terminal: bool = False,
        call_log: list[tuple[str, int]] | None = None,
        backend_label: int = 0,
    ) -> None:
        self.game = game
        self.args = args
        self.capacity = max_active_games
        self.thread_count = worker_threads
        self.fail_first_wave = fail_first_wave
        self.fail_first_wait = fail_first_wait
        self.failed_wave_terminal_leaves = failed_wave_terminal_leaves
        self.mismatch_terminal = mismatch_terminal
        self.call_log = call_log
        self.backend_label = backend_label
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

    def submit_wave(self) -> NativeWaveSubmission:
        if self.call_log is not None:
            self.call_log.append(("submit", self.backend_label))
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
        return NativeWaveSubmission(self._token, len(pending), len(pending))

    def wait_wave(self, submission: NativeWaveSubmission) -> NativeWave:
        if self.call_log is not None:
            self.call_log.append(("wait", self.backend_label))
        pending = self._pending
        assert pending is not None
        if (
            submission.token != self._token
            or submission.size != len(pending)
            or submission.raw_size != len(pending)
        ):
            raise RuntimeError("invalid fake native wave submission")
        if self.fail_first_wait:
            self.fail_first_wait = False
            raise RuntimeError("transient fake native wait failure")
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

    def evaluate_wave(self) -> NativeWave:
        return self.wait_wave(self.submit_wave())

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
    fail_first_wait: bool = False,
    fail_first_wait_label: int | None = None,
    failed_wave_terminal_leaves: int = 0,
    mismatch_terminal: bool = False,
    call_log: list[tuple[str, int]] | None = None,
) -> Callable[..., Any]:
    def factory(
        game: PenteGame,
        _evaluator: object,
        args: MCTSArgs,
        **kwargs: object,
    ) -> _FakeNativeBackend:
        backend_label = len(created)
        backend = _FakeNativeBackend(
            game,
            args,
            int(cast(int, kwargs["max_active_games"])),
            int(cast(int, kwargs["worker_threads"])),
            fail_first_wave=fail_first_wave,
            fail_first_wait=(
                fail_first_wait
                and (
                    fail_first_wait_label is None
                    or fail_first_wait_label == backend_label
                )
            ),
            failed_wave_terminal_leaves=failed_wave_terminal_leaves,
            mismatch_terminal=mismatch_terminal,
            call_log=call_log,
            backend_label=backend_label,
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
        native_search_cohorts: int = 1,
    ) -> SelfPlayGenerator:
        options: dict[str, Any] = {
            "native_worker_threads": native_worker_threads,
            "native_search_cohorts": native_search_cohorts,
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

    def test_two_native_cohorts_split_capacity_workers_and_pipeline(self) -> None:
        created: list[_FakeNativeBackend] = []
        call_log: list[tuple[str, int]] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created, call_log=call_log),
            native_worker_threads=5,
            native_search_cohorts=2,
        )

        games, batches = generator.play_games(5, 5)

        self.assertEqual([(3, 3), (2, 2)], [
            (backend.capacity, backend.thread_count) for backend in created
        ])
        self.assertEqual(
            [
                ("submit", 0),
                ("submit", 1),
                ("wait", 0),
                ("submit", 0),
                ("wait", 1),
            ],
            call_log[:5],
        )
        self.assertEqual(2, len(created))
        self.assertTrue(all(backend.active_count == 0 for backend in created))
        self.assertEqual(5, sum(len(backend.removed_slots) for backend in created))
        self.assertEqual(5, len(games))

        metrics = collect_self_play_metrics(games, batches, elapsed_seconds=1.0)
        self.assertEqual(5, metrics["active_game_target"])
        self.assertEqual(3, metrics["inference_batch_target"])
        self.assertEqual(5, metrics["native_worker_threads"])
        self.assertEqual(3, metrics["native_worker_threads_per_cohort"])
        self.assertEqual(2, metrics["native_search_cohorts"])
        self.assertEqual(2, metrics["native_pipeline_max_in_flight"])

    def test_two_native_cohort_seeded_runs_have_stable_signatures(self) -> None:
        created_one: list[_FakeNativeBackend] = []
        created_two: list[_FakeNativeBackend] = []
        generator_one = self.make_generator(
            seed=73,
            backend="cpp",
            factory=_fake_factory(created_one),
            native_worker_threads=4,
            native_search_cohorts=2,
        )
        generator_two = self.make_generator(
            seed=73,
            backend="cpp",
            factory=_fake_factory(created_two),
            native_worker_threads=4,
            native_search_cohorts=2,
        )

        games_one, _ = generator_one.play_games(4, 3)
        games_two, _ = generator_two.play_games(4, 3)

        self.assertEqual(
            [self.game_signature(game) for game in games_one],
            [self.game_signature(game) for game in games_two],
        )

    def test_two_native_cohort_failure_drains_all_pending_waves(self) -> None:
        created: list[_FakeNativeBackend] = []
        call_log: list[tuple[str, int]] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(
                created,
                fail_first_wait=True,
                fail_first_wait_label=0,
                call_log=call_log,
            ),
            native_worker_threads=4,
            native_search_cohorts=2,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "transient fake native wait failure",
        ):
            generator.play_games(4, 4)

        self.assertEqual(
            [
                ("submit", 0),
                ("submit", 1),
                ("wait", 0),
                ("submit", 0),
                ("wait", 0),
                ("wait", 1),
            ],
            call_log,
        )
        self.assertEqual(2, len(created))
        self.assertTrue(all(backend._pending is None for backend in created))

    def test_two_native_cohort_configuration_and_runtime_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.make_generator(backend="cpp", native_search_cohorts=True)
        with self.assertRaises(ValueError):
            self.make_generator(backend="cpp", native_search_cohorts=0)
        with self.assertRaises(ValueError):
            self.make_generator(backend="cpp", native_search_cohorts=3)
        with self.assertRaises(ValueError):
            self.make_generator(native_search_cohorts=2)
        with self.assertRaises(ValueError):
            self.make_generator(
                backend="cpp",
                native_search_cohorts=2,
                native_worker_threads=1,
            )

        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created),
            native_search_cohorts=2,
            native_worker_threads=2,
        )
        with self.assertRaises(ValueError):
            generator.play_games(1, 1)

        real_generator = self.make_generator(
            backend="cpp",
            native_search_cohorts=2,
            native_worker_threads=2,
        )
        with self.assertRaisesRegex(ValueError, "CUDA evaluator"):
            real_generator.play_games(2, 2)

    def test_pipeline_telemetry_defaults_preserve_one_cohort_metrics(self) -> None:
        accumulator = BatchedSearchAccumulator(8)
        accumulator.inference_batch_sizes.append(8)
        metrics = collect_self_play_metrics(
            [self._minimal_played_game()],
            [accumulator.telemetry()],
            elapsed_seconds=1.0,
        )

        self.assertEqual(1, metrics["native_search_cohorts"])
        self.assertEqual(8, metrics["active_game_target"])
        self.assertEqual(8, metrics["inference_batch_target"])
        self.assertEqual(0, metrics["native_worker_threads"])
        self.assertEqual(0, metrics["native_worker_threads_per_cohort"])
        self.assertEqual(0, metrics["native_pipeline_submissions"])
        self.assertEqual(0, metrics["native_pipeline_waits"])
        self.assertEqual(0, metrics["native_pipeline_max_in_flight"])
        self.assertEqual(1.0, metrics["steady_state_mean_batch_occupancy"])

    def test_pipeline_telemetry_reports_total_and_per_cohort_targets(self) -> None:
        batches = []
        for _ in range(2):
            accumulator = BatchedSearchAccumulator(
                256,
                native_worker_threads=4,
                native_search_cohorts=2,
                native_total_active_game_target=512,
                native_total_worker_threads=8,
            )
            accumulator.inference_batch_sizes.append(256)
            accumulator.record_pipeline_wave(in_flight=2)
            batches.append(accumulator.telemetry())

        metrics = collect_self_play_metrics(
            [self._minimal_played_game()],
            batches,
            elapsed_seconds=1.0,
        )

        self.assertEqual(2, metrics["native_search_cohorts"])
        self.assertEqual(512, metrics["active_game_target"])
        self.assertEqual(256, metrics["inference_batch_target"])
        self.assertEqual(8, metrics["native_worker_threads"])
        self.assertEqual(4, metrics["native_worker_threads_per_cohort"])
        self.assertEqual(2, metrics["native_pipeline_submissions"])
        self.assertEqual(2, metrics["native_pipeline_waits"])
        self.assertEqual(2, metrics["native_pipeline_max_in_flight"])
        self.assertEqual(1.0, metrics["steady_state_mean_batch_occupancy"])

    def test_pipeline_telemetry_rejects_invalid_dynamic_fields(self) -> None:
        with self.assertRaises(TypeError):
            BatchedSearchAccumulator(1, native_search_cohorts=True)
        with self.assertRaises(ValueError):
            BatchedSearchAccumulator(1, native_total_worker_threads=-1)
        with self.assertRaises(ValueError):
            BatchedSearchAccumulator(1, native_pipeline_max_in_flight=2)
        accumulator = BatchedSearchAccumulator(1)
        with self.assertRaises(TypeError):
            accumulator.record_pipeline_wave(in_flight=True)
        with self.assertRaises(ValueError):
            accumulator.record_pipeline_wave(in_flight=-1)
        before = accumulator.telemetry()
        with self.assertRaises(ValueError):
            accumulator.record_pipeline_wave(in_flight=2)
        self.assertEqual(before, accumulator.telemetry())

        with self.assertRaises(ValueError):
            accumulator.record_native_wave(
                NativeWave(0, 0, 0, 0.0, 0.0, 0.0, 0.0),
                None,
                worker_threads=0,
                selected_leaves=0,
                pipeline_in_flight=2,
            )
        self.assertEqual(before, accumulator.telemetry())

    @staticmethod
    def _minimal_played_game() -> PlayedGame:
        return PlayedGame(
            examples=[],
            actions=(),
            winner=None,
            win_reason=None,
            root_telemetry=(
                SearchTelemetry(
                    simulations=0,
                    evaluator_calls=0,
                    evaluated_positions=0,
                    invalid_policy_fallbacks=0,
                    zero_visit_fallbacks=0,
                    max_depth=0,
                    root_legal_actions=0,
                    root_edge_visits=0,
                    root_children_visited=0,
                    root_visit_entropy=0.0,
                    root_max_visit_share=0.0,
                    root_collapse_eligible=False,
                    root_search_collapsed=False,
                    mean_inference_batch_size=0.0,
                ),
            ),
        )

    def test_native_submit_defers_telemetry_until_wait(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created),
        )
        coordinator = cast(Any, generator._coordinator(1))
        active = generator._new_active_game()
        coordinator.start(active)
        accumulator = BatchedSearchAccumulator(1)

        coordinator.submit_wave([active], accumulator)

        self.assertIsNotNone(coordinator._pending_wave)
        self.assertEqual(0, accumulator.simulation_waves)
        self.assertEqual(0, accumulator.evaluator_calls)
        self.assertEqual(0, created[0].slot_simulations(0))

        coordinator.wait_wave()

        self.assertIsNone(coordinator._pending_wave)
        self.assertEqual(1, accumulator.simulation_waves)
        self.assertEqual(1, accumulator.evaluator_calls)
        self.assertEqual(1, accumulator.native_pipeline_submissions)
        self.assertEqual(1, accumulator.native_pipeline_waits)
        self.assertEqual(1, accumulator.native_pipeline_max_in_flight)

    def test_native_coordinator_passes_generator_worker_count_to_backend(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created),
            native_worker_threads=3,
        )
        coordinator = cast(Any, generator._coordinator(1))
        active = generator._new_active_game()
        coordinator.start(active)

        self.assertEqual(3, created[0].thread_count)
        coordinator.remove(active)

    def test_native_coordinator_records_configured_pipeline_in_flight(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created),
        )
        coordinator = cast(Any, generator._coordinator(1))
        active = generator._new_active_game()
        coordinator.start(active)
        accumulator = BatchedSearchAccumulator(1, native_search_cohorts=2)

        coordinator.submit_wave([active], accumulator, in_flight=2)
        self.assertEqual(2, coordinator._pending_wave.in_flight)
        coordinator.wait_wave()

        telemetry = accumulator.telemetry()
        self.assertEqual(2, telemetry.native_pipeline_max_in_flight)
        self.assertEqual(1, telemetry.native_pipeline_submissions)
        self.assertEqual(1, telemetry.native_pipeline_waits)
        coordinator.remove(active)

    def test_native_coordinator_rejects_duplicate_submit_without_replacing_owner(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created),
        )
        coordinator = cast(Any, generator._coordinator(1))
        active = generator._new_active_game()
        coordinator.start(active)
        accumulator = BatchedSearchAccumulator(1)
        coordinator.submit_wave([active], accumulator)
        pending = coordinator._pending_wave

        with self.assertRaisesRegex(RuntimeError, "pending wave"):
            coordinator.submit_wave([active], BatchedSearchAccumulator(1))

        self.assertIs(pending, coordinator._pending_wave)
        coordinator.wait_wave()

    def test_native_pending_wave_owns_immutable_slots_counts_accumulator_and_handle(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created),
        )
        coordinator = cast(Any, generator._coordinator(1))
        active = generator._new_active_game()
        coordinator.start(active)
        active_games = [active]
        accumulator = BatchedSearchAccumulator(1)

        coordinator.submit_wave(active_games, accumulator)
        pending = coordinator._pending_wave
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual((0,), pending.active_slots)
        self.assertEqual(((0, 0),), pending.before_completed_simulations)
        self.assertIs(accumulator, pending.accumulator)
        self.assertEqual(
            (1, 1, 1),
            (
                pending.submission.token,
                pending.submission.size,
                pending.submission.raw_size,
            ),
        )

        active_games.clear()
        active.native_slot = None
        coordinator.wait_wave()
        self.assertEqual(1, accumulator.simulation_waves)

    def test_native_coordinator_clears_owner_and_reuses_after_success(self) -> None:
        created: list[_FakeNativeBackend] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(created),
        )
        coordinator = cast(Any, generator._coordinator(1))
        active = generator._new_active_game()
        coordinator.start(active)
        first_accumulator = BatchedSearchAccumulator(1)
        coordinator.submit_wave([active], first_accumulator)
        coordinator.wait_wave()
        self.assertIsNone(coordinator._pending_wave)

        coordinator.remove(active)
        replacement = generator._new_active_game()
        coordinator.start(replacement)
        second_accumulator = BatchedSearchAccumulator(1)
        coordinator.submit_wave([replacement], second_accumulator)
        coordinator.wait_wave()

        self.assertIsNone(coordinator._pending_wave)
        self.assertEqual(1, first_accumulator.simulation_waves)
        self.assertEqual(1, second_accumulator.simulation_waves)
        self.assertEqual(2, len(created[0].added_slots))

    def test_native_coordinator_wait_failure_keeps_owner_for_retry(self) -> None:
        created: list[_FakeNativeBackend] = []
        call_log: list[tuple[str, int]] = []
        generator = self.make_generator(
            backend="cpp",
            factory=_fake_factory(
                created,
                fail_first_wait=True,
                call_log=call_log,
            ),
        )
        coordinator = cast(Any, generator._coordinator(1))
        active = generator._new_active_game()
        coordinator.start(active)
        accumulator = BatchedSearchAccumulator(1)
        coordinator.submit_wave([active], accumulator)
        pending = coordinator._pending_wave

        with self.assertRaisesRegex(RuntimeError, "wait failure"):
            coordinator.wait_wave()
        retry_pending = coordinator._pending_wave
        self.assertIsNot(pending, retry_pending)
        self.assertEqual(pending.active_slots, retry_pending.active_slots)
        self.assertEqual(
            pending.before_completed_simulations,
            retry_pending.before_completed_simulations,
        )
        self.assertIs(pending.accumulator, retry_pending.accumulator)
        self.assertEqual(pending.in_flight, retry_pending.in_flight)
        self.assertEqual(pending.submission, retry_pending.submission)
        self.assertTrue(retry_pending.needs_resubmit)
        self.assertEqual(0, accumulator.simulation_waves)
        self.assertEqual(0, created[0].slot_simulations(0))

        coordinator.evaluate_wave([active], accumulator)
        self.assertIsNone(coordinator._pending_wave)
        self.assertEqual(
            [("submit", 0), ("wait", 0), ("submit", 0), ("wait", 0)],
            call_log,
        )
        self.assertEqual(1, accumulator.simulation_waves)
        self.assertEqual(1, created[0].slot_simulations(0))

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

    def test_trainer_args_validate_two_native_cohort_requirements(self) -> None:
        def make_args(**overrides: Any) -> SelfPlayTrainerArgs:
            options: dict[str, Any] = {
                "start_iteration": 0,
                "professional_games_training_iterations": 0,
                "self_play_training_iterations": 0,
                "temp_threshold": 0,
                "mcts_args": self.args,
                "watch_training_raw_dataset_filepath": "unused",
                "watch_training_processed_dataset_filepath": "unused",
                "force_watch_training_raw_dataset_processing": False,
                "batch_games": 2,
                "active_games": 2,
                "search_backend": "cpp",
                "native_worker_threads": 2,
                "native_search_cohorts": 2,
            }
            options.update(overrides)
            return SelfPlayTrainerArgs(**options)

        normalized = make_args(native_search_cohorts=cast(Any, np.int64(2)))
        self.assertIs(int, type(normalized.native_search_cohorts))
        self.assertEqual(2, normalized.native_search_cohorts)
        with self.assertRaises(TypeError):
            make_args(native_search_cohorts=True)
        with self.assertRaises(ValueError):
            make_args(native_search_cohorts=0)
        with self.assertRaises(ValueError):
            make_args(native_search_cohorts=3)
        with self.assertRaises(ValueError):
            make_args(search_backend="python")
        with self.assertRaises(ValueError):
            make_args(native_worker_threads=1)
        with self.assertRaises(ValueError):
            make_args(active_games=1)
        with self.assertRaises(ValueError):
            make_args(batch_games=1)


if __name__ == "__main__":
    unittest.main()
