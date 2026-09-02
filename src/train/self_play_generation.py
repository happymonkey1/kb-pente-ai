from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Callable, Protocol, SupportsIndex, cast

import numpy as np

from src.game.game import TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.mcts.batched import (
    BatchedSearchAccumulator,
    BatchedSearchTelemetry,
    SearchSession,
    evaluate_search_wave,
)
from src.mcts.mcts_v2 import MCTS, MCTSArgs, PolicyValueEvaluator, SearchTelemetry
from src.mcts.native_backend import (
    NativeSearchBackend,
    NativeWaveSubmission,
    TensorEvaluator,
)
from src.train.self_play_args import SearchBackend
from src.train.training_example import TrainingExample


@dataclass(frozen=True, slots=True)
class PlayedGame:
    examples: list[TrainingExample]
    actions: tuple[int, ...]
    winner: int | None
    win_reason: str | None
    root_telemetry: tuple[SearchTelemetry, ...]


@dataclass(slots=True)
class ActiveSelfPlayGame:
    position: PenteBoard
    mcts: MCTS | None
    search_session: SearchSession | None
    pending: list[tuple[PenteBoard, np.ndarray]]
    actions: list[int]
    root_telemetry: list[SearchTelemetry]
    native_slot: int | None = None
    launch_index: int = -1


@dataclass(frozen=True, slots=True)
class _CompletedSelfPlayGame:
    launch_index: int
    played_game: PlayedGame


@dataclass(slots=True)
class _LaunchCounter:
    value: int = 0


@dataclass(frozen=True, slots=True)
class _PendingNativeWave:
    active_slots: tuple[int, ...]
    before_completed_simulations: tuple[tuple[int, int], ...]
    accumulator: BatchedSearchAccumulator
    submission: NativeWaveSubmission
    in_flight: int = 1


_NativeBackendFactory = Callable[..., NativeSearchBackend]


class _SearchCoordinator(Protocol):
    """Bridge search-specific state while the generator owns game transitions."""

    def start(self, active: ActiveSelfPlayGame) -> None:
        """Attach a fresh search to an active game."""

    def is_complete(self, active: ActiveSelfPlayGame) -> bool:
        """Return whether the active game's current root is complete."""

    def policy(self, active: ActiveSelfPlayGame) -> np.ndarray:
        """Return the completed root policy."""

    def telemetry(self, active: ActiveSelfPlayGame) -> SearchTelemetry:
        """Return telemetry for the completed current root."""

    def advance(
        self,
        active: ActiveSelfPlayGame,
        action: int,
        next_temperature: float,
        python_result: TerminalResult,
    ) -> TerminalResult | None:
        """Advance a selected action and optionally return native terminal state."""

    def remove(self, active: ActiveSelfPlayGame) -> None:
        """Release the search owned by a terminal active game."""

    def evaluate_wave(
        self,
        active: list[ActiveSelfPlayGame],
        accumulator: BatchedSearchAccumulator,
    ) -> None:
        """Evaluate one wave across all active games."""


def _fill_active_games(
    generator: SelfPlayGenerator,
    coordinator: _SearchCoordinator,
    active: list[ActiveSelfPlayGame],
    capacity: int,
    game_count: int,
    launched_games: _LaunchCounter,
) -> None:
    while launched_games.value < game_count and len(active) < capacity:
        active_game = generator._new_active_game(launched_games.value)
        coordinator.start(active_game)
        active.append(active_game)
        launched_games.value += 1


class _PythonSearchCoordinator:
    def __init__(self, generator: SelfPlayGenerator) -> None:
        self._generator = generator

    def start(self, active: ActiveSelfPlayGame) -> None:
        if active.mcts is not None or active.search_session is not None:
            raise RuntimeError("Python search coordinator received an attached game")
        search = MCTS(
            self._generator.game,
            self._generator.evaluator,
            self._generator.mcts_args,
            np.random.default_rng(
                int(self._generator.rng.integers(0, 2**63))
            ),
        )
        active.mcts = search
        active.search_session = SearchSession(
            search,
            active.position,
            self._generator._temperature(active.position),
            add_root_noise=True,
        )

    def is_complete(self, active: ActiveSelfPlayGame) -> bool:
        return self._session(active).is_complete

    def policy(self, active: ActiveSelfPlayGame) -> np.ndarray:
        return self._session(active).policy()

    def telemetry(self, active: ActiveSelfPlayGame) -> SearchTelemetry:
        return self._mcts(active).telemetry(active.position)

    def advance(
        self,
        active: ActiveSelfPlayGame,
        _action: int,
        _next_temperature: float,
        python_result: TerminalResult,
    ) -> TerminalResult | None:
        if not python_result.is_terminal:
            active.search_session = SearchSession(
                self._mcts(active),
                active.position,
                self._generator._temperature(active.position),
                add_root_noise=True,
            )
        return None

    def remove(self, active: ActiveSelfPlayGame) -> None:
        active.search_session = None
        active.mcts = None

    def evaluate_wave(
        self,
        active: list[ActiveSelfPlayGame],
        accumulator: BatchedSearchAccumulator,
    ) -> None:
        evaluate_search_wave(
            [self._session(game) for game in active],
            accumulator,
            self._generator.deduplicate_evaluations,
        )

    @staticmethod
    def _mcts(active: ActiveSelfPlayGame) -> MCTS:
        if active.mcts is None:
            raise RuntimeError("Python active game has no MCTS")
        return active.mcts

    @staticmethod
    def _session(active: ActiveSelfPlayGame) -> SearchSession:
        if active.search_session is None:
            raise RuntimeError("Python active game has no search session")
        return active.search_session


class _NativeSearchCoordinator:
    def __init__(
        self,
        generator: SelfPlayGenerator,
        max_active_games: int,
        worker_threads: int,
        backend_factory: _NativeBackendFactory | None,
    ) -> None:
        factory = NativeSearchBackend if backend_factory is None else backend_factory
        self._generator = generator
        self._backend = factory(
            generator.game,
            cast(TensorEvaluator, generator.evaluator),
            generator.mcts_args,
            max_active_games=max_active_games,
            worker_threads=worker_threads,
            seed=int(generator.rng.integers(0, 2**63)),
        )
        self._cumulative_telemetry: dict[int, SearchTelemetry | None] = {}
        self._telemetry_pending: dict[int, bool] = {}
        self._completed_simulations: dict[int, int] = {}
        self._pending_wave: _PendingNativeWave | None = None

    @property
    def has_pending_wave(self) -> bool:
        return self._pending_wave is not None

    def start(self, active: ActiveSelfPlayGame) -> None:
        if active.native_slot is not None:
            raise RuntimeError("Native search coordinator received an attached game")
        slot = self._backend.add_root(
            active.position,
            temperature=self._generator._temperature(active.position),
            add_root_noise=True,
        )
        active.native_slot = _integral_count(slot, "Native slot")
        self._cumulative_telemetry[active.native_slot] = None
        self._telemetry_pending[active.native_slot] = True
        self._completed_simulations[active.native_slot] = 0

    def is_complete(self, active: ActiveSelfPlayGame) -> bool:
        return self._backend.slot_complete(self._slot(active))

    def policy(self, active: ActiveSelfPlayGame) -> np.ndarray:
        return self._backend.root_policy(self._slot(active))

    def telemetry(self, active: ActiveSelfPlayGame) -> SearchTelemetry:
        slot = self._slot(active)
        if self._telemetry_pending.get(slot, False):
            current = self._backend.slot_telemetry(slot)
            self._cumulative_telemetry[slot] = _accumulate_search_telemetry(
                self._cumulative_telemetry[slot],
                current,
            )
            self._telemetry_pending[slot] = False
        report = self._cumulative_telemetry.get(slot)
        if report is None:
            raise RuntimeError("Native active game has no search telemetry")
        return report

    def advance(
        self,
        active: ActiveSelfPlayGame,
        action: int,
        next_temperature: float,
        _python_result: TerminalResult,
    ) -> TerminalResult:
        slot = self._slot(active)
        self._backend.advance_root(
            slot,
            action,
            temperature=next_temperature,
            add_root_noise=True,
        )
        result = self._backend.root_terminal(slot)
        if not result.is_terminal:
            self._telemetry_pending[slot] = True
            self._completed_simulations[slot] = 0
        return result

    def remove(self, active: ActiveSelfPlayGame) -> None:
        slot = self._slot(active)
        self._backend.remove(slot)
        self._cumulative_telemetry.pop(slot, None)
        self._telemetry_pending.pop(slot, None)
        self._completed_simulations.pop(slot, None)
        active.native_slot = None

    def evaluate_wave(
        self,
        active: list[ActiveSelfPlayGame],
        accumulator: BatchedSearchAccumulator,
        in_flight: int = 1,
    ) -> None:
        if self._pending_wave is None:
            self.submit_wave(active, accumulator, in_flight=in_flight)
        else:
            active_slots = tuple(self._slot(game) for game in active)
            if (
                active_slots != self._pending_wave.active_slots
                or accumulator is not self._pending_wave.accumulator
            ):
                raise RuntimeError(
                    "Native evaluate_wave does not match the pending wave"
                )
        self.wait_wave()

    def submit_wave(
        self,
        active: list[ActiveSelfPlayGame],
        accumulator: BatchedSearchAccumulator,
        in_flight: int = 1,
    ) -> None:
        if self._pending_wave is not None:
            raise RuntimeError("Native search coordinator already has a pending wave")
        parsed_in_flight = _integral_count(
            in_flight,
            "Native pipeline in-flight count",
        )
        active_slots = tuple(self._slot(game) for game in active)
        before = tuple(
            (slot, self._completed_simulations[slot]) for slot in active_slots
        )
        submission = self._backend.submit_wave()
        self._pending_wave = _PendingNativeWave(
            active_slots,
            before,
            accumulator,
            submission,
            parsed_in_flight,
        )

    def wait_wave(self) -> None:
        pending = self._pending_wave
        if pending is None:
            raise RuntimeError("Native search coordinator has no pending wave")

        backend_succeeded = False
        try:
            wave = self._backend.wait_wave(pending.submission)
            backend_succeeded = True
            timing = self._backend.native_timing_telemetry()
            worker_threads = _integral_count(
                self._backend.thread_count,
                "Native worker threads",
            )
            before = dict(pending.before_completed_simulations)
            selected_leaves = 0
            after: dict[int, int] = {}
            for slot in pending.active_slots:
                current = self._backend.slot_simulations(slot)
                selected_leaves += max(0, current - before[slot])
                after[slot] = current
            pending.accumulator.record_native_wave(
                wave,
                timing,
                worker_threads,
                selected_leaves=selected_leaves,
                pipeline_in_flight=pending.in_flight,
            )
            self._completed_simulations.update(after)
        finally:
            if backend_succeeded:
                self._pending_wave = None

    @staticmethod
    def _slot(active: ActiveSelfPlayGame) -> int:
        if active.native_slot is None:
            raise RuntimeError("Native active game has no slot")
        return active.native_slot


class SelfPlayGenerator:
    def __init__(
        self,
        game: PenteGame,
        evaluator: PolicyValueEvaluator,
        mcts_args: MCTSArgs,
        temp_threshold: int,
        rng: np.random.Generator,
        deduplicate_evaluations: bool = True,
        search_backend: SearchBackend = "python",
        native_worker_threads: int = 1,
        *,
        _native_backend_factory: _NativeBackendFactory | None = None,
    ) -> None:
        if search_backend not in ("python", "cpp"):
            raise ValueError("Search backend must be 'python' or 'cpp'")
        native_worker_threads = _integral_count(
            native_worker_threads,
            "Native worker threads",
            minimum=1,
        )
        self.game = game
        self.evaluator = evaluator
        self.mcts_args = mcts_args
        self.temp_threshold = temp_threshold
        self.rng = rng
        self.deduplicate_evaluations = deduplicate_evaluations
        self.search_backend = search_backend
        self.native_worker_threads = native_worker_threads
        self._native_backend_factory = _native_backend_factory

    def play_game(self) -> PlayedGame:
        games, _ = self.play_games(1)
        return games[0]

    def play_games(
        self,
        game_count: int,
        max_active_games: int | None = None,
    ) -> tuple[list[PlayedGame], list[BatchedSearchTelemetry]]:
        if game_count < 1:
            raise ValueError("At least one self-play game is required")
        active_limit = game_count if max_active_games is None else max_active_games
        if active_limit < 1:
            raise ValueError("At least one active self-play game is required")

        coordinator = self._coordinator(active_limit)
        launched_games = _LaunchCounter()
        active: list[ActiveSelfPlayGame] = []
        _fill_active_games(
            self,
            coordinator,
            active,
            active_limit,
            game_count,
            launched_games,
        )

        completed: list[_CompletedSelfPlayGame] = []
        batch_accumulators: dict[int, BatchedSearchAccumulator] = {}
        while active:
            remaining, completed_roots = self._advance_completed_roots(
                coordinator,
                active,
            )
            completed.extend(completed_roots)
            _fill_active_games(
                self,
                coordinator,
                remaining,
                active_limit,
                game_count,
                launched_games,
            )
            active = remaining

            if active:
                root_count = len(active)
                accumulator = batch_accumulators.get(root_count)
                if accumulator is None:
                    accumulator = BatchedSearchAccumulator(root_count)
                    batch_accumulators[root_count] = accumulator
                coordinator.evaluate_wave(active, accumulator)

        return [completed_root.played_game for completed_root in completed], [
            accumulator.telemetry()
            for accumulator in batch_accumulators.values()
        ]

    def _advance_completed_roots(
        self,
        coordinator: _SearchCoordinator,
        active: list[ActiveSelfPlayGame],
    ) -> tuple[list[ActiveSelfPlayGame], list[_CompletedSelfPlayGame]]:
        remaining: list[ActiveSelfPlayGame] = []
        completed: list[_CompletedSelfPlayGame] = []
        for active_game in active:
            if not coordinator.is_complete(active_game):
                remaining.append(active_game)
                continue

            policy = coordinator.policy(active_game)
            active_game.pending.append((active_game.position, policy))
            active_game.root_telemetry.append(coordinator.telemetry(active_game))
            action = int(self.rng.choice(len(policy), p=policy))
            active_game.actions.append(action)
            next_position, _ = self.game.apply_action(
                active_game.position,
                active_game.position.current_player,
                action,
            )
            active_game.position = next_position
            result = self.game.check_game_end(next_position)
            native_result = coordinator.advance(
                active_game,
                action,
                self._temperature(next_position),
                result,
            )
            if native_result is not None and native_result != result:
                coordinator.remove(active_game)
                raise RuntimeError(
                    "Native/Python terminal mismatch: "
                    f"native={native_result!r}, python={result!r}"
                )
            if result.is_terminal:
                coordinator.remove(active_game)
                completed.append(
                    _CompletedSelfPlayGame(
                        active_game.launch_index,
                        PlayedGame(
                            examples=finalize_training_examples(
                                active_game.pending,
                                result,
                            ),
                            actions=tuple(active_game.actions),
                            winner=result.winner,
                            win_reason=result.reason,
                            root_telemetry=tuple(active_game.root_telemetry),
                        ),
                    )
                )
            else:
                remaining.append(active_game)
        return remaining, completed

    def _coordinator(self, active_limit: int) -> _SearchCoordinator:
        if self.search_backend == "python":
            return _PythonSearchCoordinator(self)
        return _NativeSearchCoordinator(
            self,
            active_limit,
            self.native_worker_threads,
            self._native_backend_factory,
        )

    def _new_active_game(self, launch_index: int = -1) -> ActiveSelfPlayGame:
        return ActiveSelfPlayGame(
            position=self.game.init_board(),
            mcts=None,
            search_session=None,
            pending=[],
            actions=[],
            root_telemetry=[],
            launch_index=launch_index,
        )

    def _temperature(self, position: PenteBoard) -> float:
        assert position.ply is not None
        return 1.0 if position.ply < self.temp_threshold else 0.0


def _accumulate_search_telemetry(
    previous: SearchTelemetry | None,
    current: SearchTelemetry,
) -> SearchTelemetry:
    if previous is None:
        return current
    total_calls = previous.evaluator_calls + current.evaluator_calls
    if total_calls:
        mean_batch = (
            previous.mean_inference_batch_size * previous.evaluator_calls
            + current.mean_inference_batch_size * current.evaluator_calls
        ) / total_calls
    else:
        mean_batch = 0.0
    return SearchTelemetry(
        simulations=previous.simulations + current.simulations,
        evaluator_calls=total_calls,
        evaluated_positions=(
            previous.evaluated_positions + current.evaluated_positions
        ),
        invalid_policy_fallbacks=(
            previous.invalid_policy_fallbacks + current.invalid_policy_fallbacks
        ),
        zero_visit_fallbacks=(
            previous.zero_visit_fallbacks + current.zero_visit_fallbacks
        ),
        max_depth=max(previous.max_depth, current.max_depth),
        root_legal_actions=current.root_legal_actions,
        root_edge_visits=current.root_edge_visits,
        root_children_visited=current.root_children_visited,
        root_visit_entropy=current.root_visit_entropy,
        root_max_visit_share=current.root_max_visit_share,
        root_collapse_eligible=current.root_collapse_eligible,
        root_search_collapsed=current.root_search_collapsed,
        mean_inference_batch_size=float(mean_batch),
    )


def _integral_count(
    value: object,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        requirement = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {requirement}")
    return result


def finalize_training_examples(
    pending: list[tuple[PenteBoard, np.ndarray]],
    result: TerminalResult,
) -> list[TrainingExample]:
    if not result.is_terminal:
        raise ValueError("Cannot finalize examples before the game is terminal")
    return [
        TrainingExample(
            position=position,
            policy=policy,
            value=result.value_for(position.current_player),
        )
        for position, policy in pending
    ]


__all__ = [
    "ActiveSelfPlayGame",
    "PlayedGame",
    "SelfPlayGenerator",
    "finalize_training_examples",
]
