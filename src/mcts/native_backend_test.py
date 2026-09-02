from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np
import torch

from src.game.game import TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.mcts.native_backend import (
    NativeBackendUnavailableError,
    NativeSearchBackend,
    NativeWave,
    NativeWaveSubmission,
    load_native_extension,
)


def _native_search_telemetry(completions: int, simulations: int) -> dict[str, object]:
    return {
        "completed_simulations": simulations,
        "evaluator_completions": completions,
        "terminal_simulations": 0,
        "selected_leaves": completions,
        "max_selected_path_depth": 0,
        "root_legal_actions": 25,
        "root_edge_visits": simulations,
        "root_children_visited": simulations,
        "root_visit_entropy": 0.0,
        "root_max_visit_share": 1.0,
        "root_collapse_eligible": False,
        "root_search_collapsed": False,
        "invalid_policy_fallbacks": 0,
        "zero_visit_fallbacks": 0,
    }


@dataclass
class _FakeSelection:
    features: torch.Tensor
    token: int
    size: int
    raw_size: int


class _FakeBatch:
    def __init__(self, **options: Any) -> None:
        self.options = options
        capacity = int(options["active_games"])
        board_size = int(options["board_size"])
        pin_memory = bool(options["pin_memory"])
        self.features = torch.empty(
            (capacity, 4, board_size, board_size),
            pin_memory=pin_memory,
        )
        self.policies = torch.empty((capacity, 361), pin_memory=pin_memory)
        self.values = torch.empty((capacity,), pin_memory=pin_memory)
        self._simulations = int(options["simulations"])
        self._token = 0
        self._pending = False
        self._selection_sizes: list[int] = []
        self._slots: list[dict[str, Any] | None] = [None] * capacity
        self.select_calls = 0
        self.backup_calls = 0
        self.slot_snapshot_calls = 0
        self.slot_telemetry_calls = 0
        self.slot_complete_calls = 0
        self.root_terminal_calls = 0
        self._dedup = {
            "cumulative": {
                "selection_waves": 0,
                "raw_evaluation_requests": 0,
                "unique_evaluations": 0,
                "eliminated_duplicate_evaluations": 0,
                "duplicate_leaf_rate": 0.0,
            },
            "last_wave": {
                "selection_waves": 0,
                "raw_evaluation_requests": 0,
                "unique_evaluations": 0,
                "eliminated_duplicate_evaluations": 0,
                "duplicate_leaf_rate": 0.0,
            },
        }

    @property
    def active_count(self) -> int:
        return sum(slot is not None for slot in self._slots)

    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def thread_count(self) -> int:
        return int(self.options["threads"])

    def add(self, *_args: object, **_kwargs: object) -> int:
        slot = self._slots.index(None)
        self._slots[slot] = {"completions": 0, "simulations": 0, "terminal": None}
        return slot

    def select(self) -> _FakeSelection:
        self.select_calls += 1
        self._token += 1
        self._pending = True
        active = [slot for slot in self._slots if slot is not None]
        size = 1 if active else 0
        if self._selection_sizes:
            size = min(self._selection_sizes.pop(0), len(active))
        raw_size = len(active)
        if size:
            self.features[:size].fill_(float(self._token))
        self._dedup["last_wave"] = {
            "selection_waves": 1,
            "raw_evaluation_requests": raw_size,
            "unique_evaluations": size,
            "eliminated_duplicate_evaluations": raw_size - size,
            "duplicate_leaf_rate": (raw_size - size) / raw_size if raw_size else 0.0,
        }
        cumulative = self._dedup["cumulative"]
        assert isinstance(cumulative, dict)
        for key in (
            "selection_waves",
            "raw_evaluation_requests",
            "unique_evaluations",
            "eliminated_duplicate_evaluations",
        ):
            cumulative[key] = int(cumulative[key]) + int(self._dedup["last_wave"][key])
        cumulative["duplicate_leaf_rate"] = (
            float(cumulative["eliminated_duplicate_evaluations"])
            / float(cumulative["raw_evaluation_requests"])
            if cumulative["raw_evaluation_requests"]
            else 0.0
        )
        return _FakeSelection(self.features[:size], self._token, size, raw_size)

    def backup(self, token: int, rows: int) -> None:
        self.backup_calls += 1
        if not self._pending or token != self._token or rows < 1:
            raise RuntimeError("invalid fake backup")
        if torch.count_nonzero(self.policies[0, 25:]).item() != 0:
            raise RuntimeError("adapter did not clear inactive policy staging")
        for slot in self._slots:
            if slot is not None and int(slot["simulations"]) < self._simulations:
                slot["completions"] = int(slot["completions"]) + 1
                slot["simulations"] = int(slot["simulations"]) + 1
        self._pending = False

    def complete(self) -> bool:
        return all(
            slot is None or int(slot["simulations"]) >= self._simulations
            for slot in self._slots
        )

    def root_policy(self, slot: int) -> torch.Tensor:
        result = torch.zeros(25)
        result[0] = 1.0
        return result

    def root_terminal(self, slot: int) -> dict[str, Any]:
        self.root_terminal_calls += 1
        state = self._slots[slot]
        assert state is not None
        terminal = state["terminal"]
        return terminal or {"status": "in_progress", "reason": "none", "winner": None}

    def slot_telemetry(self, slot: int) -> dict[str, Any]:
        self.slot_telemetry_calls += 1
        state = self._slots[slot]
        if state is None:
            raise IndexError(slot)
        return _native_search_telemetry(
            int(state["completions"]),
            int(state["simulations"]),
        )

    def slot_complete(self, slot: int) -> bool:
        self.slot_complete_calls += 1
        state = self._slots[slot]
        return state is not None and int(state["simulations"]) >= self._simulations

    def slot_snapshots(self) -> tuple[tuple[int, bool, int, int], ...]:
        self.slot_snapshot_calls += 1
        return tuple(
            (
                slot,
                int(state["simulations"]) >= self._simulations,
                int(state["simulations"]),
                int(state["completions"]),
            )
            for slot, state in enumerate(self._slots)
            if state is not None
        )

    def advance_root(self, slot: int, _action: int, **_kwargs: object) -> dict[str, Any]:
        state = self._slots[slot]
        assert state is not None
        state["simulations"] = 0
        state["completions"] = 0
        return {"reused_subtree": True}

    def observe_action(self, slot: int, _action: int, **_kwargs: object) -> dict[str, Any]:
        state = self._slots[slot]
        assert state is not None
        state["simulations"] = 0
        state["completions"] = 0
        return {"reused_subtree": True}

    def remove(self, slot: int) -> None:
        self._slots[slot] = None

    def replace_root(self, slot: int, *_args: object, **_kwargs: object) -> None:
        self._slots[slot] = {"completions": 0, "simulations": 0, "terminal": None}

    def deduplication_telemetry(self) -> dict[str, Any]:
        return self._dedup

    def timing_telemetry(self) -> dict[str, Any]:
        worker = {"items": 1, "workers": self.thread_count, "wall_seconds": 0.01, "callback_busy_seconds": 0.01, "busy_fraction": 0.5}
        stage = {"successful_operations": self._token, "token": self._token, "wall_seconds": 0.01, "worker": worker}
        return {"cumulative": {"token": self._token, "select": stage, "dedup": stage, "features": stage, "backup": stage}, "latest_generation": {"token": self._token, "select": stage, "dedup": stage, "features": stage, "backup": stage}}


class _FakeExtension:
    SearchBatch = _FakeBatch


class _FailingBackupBatch(_FakeBatch):
    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._fail_backup = True

    def backup(self, token: int, rows: int) -> None:
        if self._fail_backup:
            self._fail_backup = False
            raise RuntimeError("transient fake backup failure")
        super().backup(token, rows)


class _FailingBackupExtension:
    SearchBatch = _FailingBackupBatch


class _FailingObserveBatch(_FakeBatch):
    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._fail_observe = True

    def observe_action(self, slot: int, _action: int, **kwargs: object) -> dict[str, Any]:
        if self._fail_observe:
            self._fail_observe = False
            raise RuntimeError("transient fake observed-action failure")
        return super().observe_action(slot, _action, **kwargs)


class _FailingObserveExtension:
    SearchBatch = _FailingObserveBatch


class _TerminalAdvanceBatch(_FakeBatch):
    def advance_root(
        self,
        slot: int,
        _action: int,
        **_kwargs: object,
    ) -> dict[str, Any]:
        result = super().advance_root(slot, _action, **_kwargs)
        state = self._slots[slot]
        assert state is not None
        state["terminal"] = {"status": "draw", "reason": "none", "winner": None}
        return result


class _TerminalAdvanceExtension:
    SearchBatch = _TerminalAdvanceBatch


class _FakeEvaluator:
    device = torch.device("cpu")

    def evaluate_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = inputs.shape[0]
        return torch.ones((rows, 25), dtype=torch.float32) / 25.0, torch.zeros(rows)


class _CudaEvaluator:
    device = torch.device("cuda")

    def __init__(self) -> None:
        self.inputs: list[tuple[tuple[int, ...], int, torch.Tensor]] = []

    def evaluate_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.inputs.append((tuple(inputs.shape), inputs.data_ptr(), inputs.detach().clone()))
        rows = inputs.shape[0]
        return (
            torch.ones((rows, 25), dtype=torch.float32, device=inputs.device) / 25.0,
            torch.zeros(rows, dtype=torch.float32, device=inputs.device),
        )


class _RetryingCudaEvaluator(_CudaEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def evaluate_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient CUDA evaluator failure")
        return super().evaluate_features(inputs)


class _VariableCudaBatch(_FakeBatch):
    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._selection_sizes = [1, 2]


class _VariableCudaExtension:
    SearchBatch = _VariableCudaBatch


class _ZeroRowBatch(_FakeBatch):
    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._selection_sizes = [0]

    def select(self) -> _FakeSelection:
        selection = super().select()
        if selection.size == 0:
            self._pending = False
        return selection


class _ZeroRowExtension:
    SearchBatch = _ZeroRowBatch


class _FakeCudaStream:
    def __init__(self) -> None:
        self.synchronize_calls = 0
        self.fail_synchronize = False

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.fail_synchronize:
            raise RuntimeError("CUDA drain failure")


class _FakeCudaEvent:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.recorded_streams: list[_FakeCudaStream] = []
        self.synchronize_calls = 0

    def record(self, stream: _FakeCudaStream) -> None:
        self.recorded_streams.append(stream)

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def elapsed_time(self, _finished: _FakeCudaEvent) -> float:
        return 1.0


class _FailingQueryCudaEvent(_FakeCudaEvent):
    def __init__(self) -> None:
        super().__init__()
        self.fail_query = True

    def query(self) -> bool:
        if self.fail_query:
            self.fail_query = False
            raise RuntimeError("transient CUDA query failure")
        return super().query()


class _FailingOnceEvaluator(_FakeEvaluator):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient CUDA evaluator failure")
        return super().evaluate_features(inputs)


class NativeBackendTests(TestCase):
    def setUp(self) -> None:
        self.game = PenteGame(board_size=5, ruleset=PenteRuleset.FREESTYLE)
        self.args = MCTSArgs(num_simulations=2, root_noise_epsilon=0.0)

    def make_backend(self, capacity: int = 2) -> NativeSearchBackend:
        return NativeSearchBackend(
            self.game,
            _FakeEvaluator(),
            self.args,
            max_active_games=capacity,
            worker_threads=2,
            seed=103,
            pin_memory=False,
            extension=_FakeExtension,
        )

    def make_cuda_backend(
        self,
        evaluator: _CudaEvaluator,
        *,
        capacity: int = 2,
        extension: Any = _FakeExtension,
        simulations: int | None = None,
    ) -> NativeSearchBackend:
        return NativeSearchBackend(
            self.game,
            evaluator,
            (
                self.args
                if simulations is None
                else MCTSArgs(num_simulations=simulations)
            ),
            max_active_games=capacity,
            worker_threads=2,
            seed=103,
            extension=extension,
        )

    def make_mocked_cuda_backend(
        self,
        evaluator: _FakeEvaluator | None = None,
    ) -> tuple[NativeSearchBackend, _FakeCudaStream, tuple[_FakeCudaEvent, ...]]:
        backend = self.make_backend(capacity=1)
        backend.device = torch.device("cuda")
        backend._pin_memory = True
        backend._cuda_feature_staging = torch.empty(
            (1, 4, 5, 5),
            dtype=torch.float32,
        )
        events = tuple(_FakeCudaEvent() for _ in range(6))
        backend._cuda_timing_events = cast(Any, events)
        if evaluator is not None:
            backend.evaluator = evaluator
        return backend, _FakeCudaStream(), events

    def test_extension_loading_is_explicit_and_actionable(self) -> None:
        with patch(
            "src.mcts.native_backend.importlib.import_module",
            side_effect=ModuleNotFoundError("kb_pente_native"),
        ):
            with self.assertRaisesRegex(NativeBackendUnavailableError, "uv pip install"):
                load_native_extension()

    def test_duplicate_roots_multiwave_policy_telemetry_and_staging(self) -> None:
        backend = self.make_backend()
        root = self.game.init_board()
        first = backend.add_root(root)
        second = backend.add_root(root)
        pointers = (
            backend._batch.features.data_ptr(),
            backend._batch.policies.data_ptr(),
            backend._batch.values.data_ptr(),
        )
        waves = []
        while not backend.complete():
            waves.append(backend.evaluate_wave())
            self.assertEqual(
                pointers,
                (
                    backend._batch.features.data_ptr(),
                    backend._batch.policies.data_ptr(),
                    backend._batch.values.data_ptr(),
                ),
            )
        self.assertEqual([wave.size for wave in waves], [1, 1])
        self.assertEqual([wave.raw_size for wave in waves], [2, 2])
        self.assertEqual(backend.deduplication_telemetry()["cumulative"]["unique_evaluations"], 2)
        self.assertEqual(backend.slot_telemetry(first).evaluator_calls, 2)
        self.assertEqual(backend.slot_telemetry(first).mean_inference_batch_size, 1.0)
        self.assertEqual(backend.slot_telemetry(second), backend.slot_telemetry(first))
        np.testing.assert_allclose(backend.root_policy(first), np.eye(1, 25, 0)[0])

    def test_cpu_backend_does_not_allocate_cuda_resources(self) -> None:
        backend = self.make_backend()

        self.assertIsNone(backend._cuda_feature_staging)
        self.assertIsNone(backend._cuda_timing_events)

    def test_cpu_submit_poll_and_wait_preserve_wave_and_timing_contract(self) -> None:
        backend = self.make_backend(capacity=1)
        backend.add_root(self.game.init_board())

        first_submission = backend.submit_wave()
        self.assertIsInstance(first_submission, NativeWaveSubmission)
        self.assertEqual(
            (1, 1, 1),
            (first_submission.token, first_submission.size, first_submission.raw_size),
        )
        self.assertFalse(backend.complete())
        with self.assertRaises(AttributeError):
            first_submission.token = 2  # type: ignore[misc]

        first_wave = backend.poll_wave(first_submission)
        self.assertIsInstance(first_wave, NativeWave)
        assert first_wave is not None
        self.assertEqual(
            (1, 1, 1),
            (first_wave.token, first_wave.size, first_wave.raw_size),
        )
        self.assertEqual(1, backend._batch.select_calls)
        self.assertEqual(1, backend._batch.backup_calls)
        self.assertEqual(1, backend.inference_timing()["calls"])

        second_submission = backend.submit_wave()
        second_wave = backend.wait_wave(second_submission)
        self.assertEqual(2, second_wave.token)
        self.assertTrue(backend.complete())
        self.assertEqual(2, backend._batch.select_calls)
        self.assertEqual(2, backend._batch.backup_calls)
        self.assertEqual(2, backend.inference_timing()["calls"])

    def test_submitted_wave_rejects_duplicate_submit_and_lifecycle_mutations(
        self,
    ) -> None:
        backend = self.make_backend(capacity=1)
        slot = backend.add_root(self.game.init_board())
        submission = backend.submit_wave()

        with self.assertRaisesRegex(RuntimeError, "already submitted"):
            backend.submit_wave()
        with self.assertRaisesRegex(RuntimeError, "pending"):
            backend.add_root(self.game.init_board())
        with self.assertRaisesRegex(RuntimeError, "pending"):
            backend.root_policy(slot)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            backend.remove(slot)

        backend.wait_wave(submission)

    def test_submission_handles_reject_mismatched_stale_and_foreign_waves(self) -> None:
        backend = self.make_backend(capacity=1)
        backend.add_root(self.game.init_board())
        submission = backend.submit_wave()

        mismatched = NativeWaveSubmission(
            submission.token + 1,
            submission.size,
            submission.raw_size,
        )
        object.__setattr__(mismatched, "_owner", backend._wave_owner)
        with self.assertRaisesRegex(ValueError, "does not match"):
            backend.poll_wave(mismatched)

        foreign_backend = self.make_backend(capacity=1)
        foreign_backend.add_root(self.game.init_board())
        foreign_submission = foreign_backend.submit_wave()
        with self.assertRaisesRegex(ValueError, "another backend"):
            backend.poll_wave(foreign_submission)

        backend.wait_wave(submission)
        with self.assertRaisesRegex(ValueError, "stale"):
            backend.wait_wave(submission)
        foreign_backend.wait_wave(foreign_submission)

    def test_submit_retry_reuses_selection_token_after_inference_failure(self) -> None:
        evaluator = _FailingOnceEvaluator()
        backend = NativeSearchBackend(
            self.game,
            evaluator,
            MCTSArgs(num_simulations=1),
            max_active_games=1,
            worker_threads=1,
            pin_memory=False,
            extension=_FakeExtension,
        )
        backend.add_root(self.game.init_board())

        with self.assertRaisesRegex(RuntimeError, "transient CUDA"):
            backend.submit_wave()
        self.assertEqual(1, backend._batch.select_calls)
        self.assertEqual(0, backend.inference_timing()["calls"])

        retry = backend.submit_wave()
        self.assertEqual((1, 1, 1), (retry.token, retry.size, retry.raw_size))
        backend.wait_wave(retry)
        self.assertEqual(1, backend._batch.select_calls)
        self.assertEqual(2, evaluator.calls)
        self.assertEqual(1, backend.inference_timing()["calls"])

    def test_zero_row_submission_holds_lifecycle_until_wait(self) -> None:
        backend = NativeSearchBackend(
            self.game,
            _FakeEvaluator(),
            self.args,
            max_active_games=1,
            worker_threads=1,
            pin_memory=False,
            extension=_ZeroRowExtension,
        )

        submission = backend.submit_wave()
        self.assertEqual(
            (1, 0, 0),
            (submission.token, submission.size, submission.raw_size),
        )
        self.assertFalse(backend.complete())
        with self.assertRaisesRegex(RuntimeError, "pending"):
            backend.add_root(self.game.init_board())

        wave = backend.wait_wave(submission)
        self.assertEqual(NativeWave(1, 0, 0, 0.0, 0.0, 0.0, 0.0), wave)
        self.assertTrue(backend.complete())
        self.assertEqual(0, backend._batch.backup_calls)
        backend.add_root(self.game.init_board())

    def test_mocked_cuda_poll_uses_completion_event_without_stream_wait(self) -> None:
        backend, stream, events = self.make_mocked_cuda_backend()
        backend.add_root(self.game.init_board())
        events[5].ready = False

        with (
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch.object(backend, "_validate_outputs"),
            patch(
                "src.mcts.native_backend.torch.cuda.current_stream",
                return_value=stream,
            ),
        ):
            submission = backend.submit_wave()
            self.assertIsNone(backend.poll_wave(submission))
            self.assertEqual(0, stream.synchronize_calls)
            self.assertEqual(0, events[5].synchronize_calls)
            self.assertEqual(0, backend.inference_timing()["calls"])

            events[5].ready = True
            wave = backend.poll_wave(submission)

        self.assertIsInstance(wave, NativeWave)
        self.assertEqual(1, events[5].synchronize_calls)
        self.assertEqual(0, stream.synchronize_calls)
        self.assertEqual(1, backend.inference_timing()["calls"])
        self.assertEqual(1, backend._batch.backup_calls)

    def test_mocked_cuda_query_failure_drains_before_retry(self) -> None:
        backend, stream, events = self.make_mocked_cuda_backend()
        backend.add_root(self.game.init_board())
        events = (*events[:5], _FailingQueryCudaEvent())
        backend._cuda_timing_events = cast(Any, events)

        with (
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch.object(backend, "_validate_outputs"),
            patch(
                "src.mcts.native_backend.torch.cuda.current_stream",
                return_value=stream,
            ),
        ):
            submission = backend.submit_wave()
            with self.assertRaisesRegex(RuntimeError, "query failure"):
                backend.poll_wave(submission)
            self.assertEqual(1, stream.synchronize_calls)
            retry = backend.submit_wave()
            backend.wait_wave(retry)

        self.assertEqual(1, backend._batch.select_calls)
        self.assertEqual(1, backend.inference_timing()["calls"])
        self.assertEqual(1, backend._batch.backup_calls)

    def test_mocked_cuda_query_drain_failure_prevents_reuse(self) -> None:
        backend, stream, events = self.make_mocked_cuda_backend()
        backend.add_root(self.game.init_board())
        events = (*events[:5], _FailingQueryCudaEvent())
        backend._cuda_timing_events = cast(Any, events)
        stream.fail_synchronize = True

        with (
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch.object(backend, "_validate_outputs"),
            patch(
                "src.mcts.native_backend.torch.cuda.current_stream",
                return_value=stream,
            ),
        ):
            submission = backend.submit_wave()
            with self.assertRaisesRegex(RuntimeError, "could not be drained"):
                backend.poll_wave(submission)
            with self.assertRaisesRegex(
                RuntimeError,
                "unavailable after a drain failure",
            ):
                backend.submit_wave()

        self.assertEqual(1, stream.synchronize_calls)
        self.assertEqual(1, backend._batch.select_calls)

    def test_mocked_cuda_enqueue_failure_drains_before_retry_and_reuses_resources(
        self,
    ) -> None:
        evaluator = _FailingOnceEvaluator()
        backend, stream, events = self.make_mocked_cuda_backend(evaluator)
        backend.add_root(self.game.init_board())
        event_ids = tuple(id(event) for event in events)

        with (
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch.object(backend, "_validate_outputs"),
            patch(
                "src.mcts.native_backend.torch.cuda.current_stream",
                return_value=stream,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "transient CUDA"):
                backend.submit_wave()
            self.assertEqual(1, stream.synchronize_calls)
            retry = backend.submit_wave()
            backend.wait_wave(retry)

        self.assertEqual(1, backend._batch.select_calls)
        self.assertEqual(2, evaluator.calls)
        self.assertEqual(1, backend.inference_timing()["calls"])
        self.assertEqual(event_ids, tuple(id(event) for event in events))
        self.assertEqual(1, stream.synchronize_calls)

    @skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_reuses_feature_storage_events_and_variable_row_views(self) -> None:
        evaluator = _CudaEvaluator()
        backend = self.make_cuda_backend(evaluator, extension=_VariableCudaExtension)
        backend.add_root(self.game.init_board())
        backend.add_root(self.game.init_board())
        device_features = backend._cuda_feature_staging
        timing_events = backend._cuda_timing_events
        self.assertIsNotNone(device_features)
        self.assertIsNotNone(timing_events)
        assert device_features is not None
        assert timing_events is not None
        self.assertEqual((2, 4, 5, 5), tuple(device_features.shape))
        self.assertEqual(6, len(timing_events))
        event_ids = tuple(id(event) for event in timing_events)

        first = backend.evaluate_wave()
        second = backend.evaluate_wave()

        self.assertEqual((1, 2), (first.size, second.size))
        self.assertTrue(backend.complete())
        self.assertIs(device_features, backend._cuda_feature_staging)
        self.assertIs(timing_events, backend._cuda_timing_events)
        self.assertEqual(event_ids, tuple(id(event) for event in timing_events))
        self.assertEqual(2, backend.inference_timing()["calls"])
        self.assertGreaterEqual(first.host_to_device_seconds, 0.0)
        self.assertGreaterEqual(first.model_inference_seconds, 0.0)
        self.assertGreaterEqual(first.device_to_host_seconds, 0.0)
        self.assertGreaterEqual(first.inference_wait_seconds, 0.0)
        self.assertGreaterEqual(second.host_to_device_seconds, 0.0)
        self.assertGreaterEqual(second.model_inference_seconds, 0.0)
        self.assertGreaterEqual(second.device_to_host_seconds, 0.0)
        self.assertGreaterEqual(second.inference_wait_seconds, 0.0)

        self.assertEqual(
            [(1, 4, 5, 5), (2, 4, 5, 5)],
            [entry[0] for entry in evaluator.inputs],
        )
        for (shape, pointer, snapshot), token in zip(evaluator.inputs, (1.0, 2.0)):
            self.assertEqual((int(token), 4, 5, 5), shape)
            self.assertEqual(device_features.data_ptr(), pointer)
            torch.testing.assert_close(snapshot, torch.full_like(snapshot, token))

    @skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_evaluator_failure_can_retry_with_reused_resources(self) -> None:
        evaluator = _RetryingCudaEvaluator()
        backend = self.make_cuda_backend(evaluator, capacity=1, simulations=1)
        backend.add_root(self.game.init_board())
        device_features = backend._cuda_feature_staging
        timing_events = backend._cuda_timing_events

        with self.assertRaisesRegex(RuntimeError, "transient CUDA"):
            backend.evaluate_wave()
        self.assertFalse(backend.complete())
        backend.evaluate_wave()

        self.assertTrue(backend.complete())
        self.assertEqual(2, evaluator.calls)
        self.assertIs(device_features, backend._cuda_feature_staging)
        self.assertIs(timing_events, backend._cuda_timing_events)

    def test_successful_wave_uses_one_bulk_snapshot_without_slot_fanout(self) -> None:
        backend = self.make_backend()
        backend.add_root(self.game.init_board())
        backend.add_root(self.game.init_board())

        backend.evaluate_wave()

        self.assertEqual(1, backend._batch.slot_snapshot_calls)
        self.assertEqual(0, backend._batch.slot_telemetry_calls)
        self.assertEqual(0, backend._batch.slot_complete_calls)
        self.assertEqual(1, backend.slot_simulations(0))
        self.assertEqual(1, backend.slot_simulations(1))
        self.assertFalse(backend.slot_complete(0))
        self.assertFalse(backend.slot_complete(1))
        self.assertEqual(0, backend._batch.slot_complete_calls)

    def test_failed_wave_keeps_pending_token_for_retry(self) -> None:
        class InvalidThenValidEvaluator(_FakeEvaluator):
            def __init__(self) -> None:
                self.calls = 0

            def evaluate_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                self.calls += 1
                if self.calls == 1:
                    return torch.zeros((inputs.shape[0], 24)), torch.zeros(inputs.shape[0])
                return super().evaluate_features(inputs)

        backend = NativeSearchBackend(
            self.game,
            InvalidThenValidEvaluator(),
            MCTSArgs(num_simulations=1),
            max_active_games=1,
            worker_threads=1,
            pin_memory=False,
            extension=_FakeExtension,
        )
        backend.add_root(self.game.init_board())
        with self.assertRaises(ValueError):
            backend.evaluate_wave()
        self.assertFalse(backend.complete())
        backend.evaluate_wave()
        self.assertTrue(backend.complete())

    def test_failed_backup_does_not_publish_slot_snapshot_to_cache(self) -> None:
        backend = NativeSearchBackend(
            self.game,
            _FakeEvaluator(),
            MCTSArgs(num_simulations=1),
            max_active_games=1,
            worker_threads=1,
            pin_memory=False,
            extension=_FailingBackupExtension,
        )
        slot = backend.add_root(self.game.init_board())

        with self.assertRaisesRegex(RuntimeError, "transient"):
            backend.evaluate_wave()

        self.assertEqual(0, backend._batch.slot_snapshot_calls)
        self.assertEqual(0, backend.slot_simulations(slot))
        self.assertFalse(backend.slot_complete(slot))

        backend.evaluate_wave()
        self.assertEqual(1, backend._batch.slot_snapshot_calls)
        self.assertEqual(1, backend.slot_simulations(slot))

    def test_failed_backup_keeps_pending_token_for_retry(self) -> None:
        backend = NativeSearchBackend(
            self.game,
            _FakeEvaluator(),
            MCTSArgs(num_simulations=1),
            max_active_games=1,
            worker_threads=1,
            pin_memory=False,
            extension=_FailingBackupExtension,
        )
        backend.add_root(self.game.init_board())
        with self.assertRaisesRegex(RuntimeError, "transient"):
            backend.evaluate_wave()
        self.assertFalse(backend.complete())
        backend.evaluate_wave()
        self.assertTrue(backend.complete())

    def test_observe_action_updates_root_and_resets_batch_sizes_after_success(self) -> None:
        backend = self.make_backend(capacity=1)
        root = self.game.init_board()
        slot = backend.add_root(root)
        backend._batch_sizes[slot] = [7]
        expected, _ = self.game.apply_action(root, root.current_player, 0)

        result = backend.observe_action(
            slot,
            0,
            temperature=0.0,
            add_root_noise=False,
        )

        self.assertTrue(result["reused_subtree"])
        self.assertEqual(backend._roots[slot].state_key(), expected.state_key())
        self.assertEqual(backend._batch_sizes[slot], [])
        self.assertEqual(backend.slot_simulations(slot), 0)
        self.assertFalse(backend.slot_complete(slot))

    def test_terminal_lifecycle_is_complete_before_public_terminal_read(self) -> None:
        backend = NativeSearchBackend(
            self.game,
            _FakeEvaluator(),
            self.args,
            max_active_games=1,
            worker_threads=1,
            pin_memory=False,
            extension=_TerminalAdvanceExtension,
        )
        slot = backend.add_root(self.game.init_board())
        calls_before_advance = backend._batch.root_terminal_calls

        backend.advance_root(slot, 0)

        self.assertTrue(backend.slot_complete(slot))
        self.assertEqual(backend.root_terminal(slot), TerminalResult.draw())
        self.assertEqual(
            calls_before_advance + 1,
            backend._batch.root_terminal_calls,
        )

    def test_failed_observed_action_does_not_update_python_mirrors(self) -> None:
        backend = NativeSearchBackend(
            self.game,
            _FakeEvaluator(),
            self.args,
            max_active_games=1,
            worker_threads=1,
            pin_memory=False,
            extension=_FailingObserveExtension,
        )
        root = self.game.init_board()
        slot = backend.add_root(root)
        backend._batch_sizes[slot] = [11]
        root_key = backend._roots[slot].state_key()

        with self.assertRaisesRegex(RuntimeError, "observed-action"):
            backend.observe_action(slot, 0)
        self.assertEqual(backend._roots[slot].state_key(), root_key)
        self.assertEqual(backend._batch_sizes[slot], [11])

        backend.observe_action(slot, 0)
        self.assertNotEqual(backend._roots[slot].state_key(), root_key)
        self.assertEqual(backend._batch_sizes[slot], [])

    def test_root_validation_terminal_conversion_and_lifecycle(self) -> None:
        backend = self.make_backend(capacity=1)
        root = self.game.init_board()
        writable = np.zeros((5, 5), dtype=np.int8)
        writable_root = PenteBoard(writable, np.zeros(2, dtype=np.int16))
        writable_root.board.setflags(write=True)
        with self.assertRaisesRegex(ValueError, "immutable"):
            backend.add_root(writable_root)
        slot = backend.add_root(root)
        while not backend.complete():
            backend.evaluate_wave()
        backend._batch._slots[slot]["terminal"] = {
            "status": "win",
            "reason": "capture",
            "winner": -1,
        }
        self.assertEqual(
            backend.root_terminal(slot),
            TerminalResult.win(-1, "capture"),
        )
        backend.remove(slot)
        self.assertEqual(backend.active_count, 0)
