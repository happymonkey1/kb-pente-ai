from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest import TestCase
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
        self.features = torch.empty((capacity, 4, board_size, board_size))
        self.policies = torch.empty((capacity, 361))
        self.values = torch.empty((capacity,))
        self._simulations = int(options["simulations"])
        self._token = 0
        self._pending = False
        self._slots: list[dict[str, Any] | None] = [None] * capacity
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
        self._token += 1
        self._pending = True
        active = [slot for slot in self._slots if slot is not None]
        size = 1 if active else 0
        raw_size = len(active)
        if size:
            self.features[0].fill_(0.0)
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
        if not self._pending or token != self._token or rows != 1:
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
        state = self._slots[slot]
        assert state is not None
        terminal = state["terminal"]
        return terminal or {"status": "in_progress", "reason": "none", "winner": None}

    def slot_telemetry(self, slot: int) -> dict[str, Any]:
        state = self._slots[slot]
        if state is None:
            raise IndexError(slot)
        return _native_search_telemetry(
            int(state["completions"]),
            int(state["simulations"]),
        )

    def slot_complete(self, slot: int) -> bool:
        state = self._slots[slot]
        return state is not None and int(state["simulations"]) >= self._simulations

    def advance_root(self, slot: int, _action: int, **_kwargs: object) -> dict[str, Any]:
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


class _FakeEvaluator:
    device = torch.device("cpu")

    def evaluate_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = inputs.shape[0]
        return torch.ones((rows, 25), dtype=torch.float32) / 25.0, torch.zeros(rows)


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
