"""Optional native SearchBatch adapter and tensor inference boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import importlib
import math
import operator
import time
from typing import Any, Mapping, NoReturn, Protocol, SupportsIndex, cast

import numpy as np
import torch

from src.game.game import GameStatus, TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTSArgs, SearchTelemetry


class NativeBackendUnavailableError(RuntimeError):
    """Raised when the explicitly requested native extension is unavailable."""


class TensorEvaluator(Protocol):
    """Evaluate encoded feature tensors at the model/device boundary."""

    device: torch.device

    def evaluate_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached float32 policy probabilities and scalar values."""


@dataclass(frozen=True, slots=True)
class NativeInferenceTiming:
    calls: int = 0
    host_to_device_seconds: float = 0.0
    model_inference_seconds: float = 0.0
    device_to_host_seconds: float = 0.0
    inference_wait_seconds: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "host_to_device_seconds": self.host_to_device_seconds,
            "model_inference_seconds": self.model_inference_seconds,
            "device_to_host_seconds": self.device_to_host_seconds,
            "inference_wait_seconds": self.inference_wait_seconds,
        }


@dataclass(frozen=True, slots=True)
class NativeWave:
    token: int
    size: int
    raw_size: int
    host_to_device_seconds: float
    model_inference_seconds: float
    device_to_host_seconds: float
    inference_wait_seconds: float


@dataclass(frozen=True, slots=True)
class NativeWaveSubmission:
    """Immutable identity for one submitted native search wave."""

    token: int
    size: int
    raw_size: int
    _owner: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _nonnegative_int(self.token, "wave token"))
        object.__setattr__(self, "size", _nonnegative_int(self.size, "wave size"))
        object.__setattr__(
            self,
            "raw_size",
            _nonnegative_int(self.raw_size, "wave raw size"),
        )
        if self.raw_size < self.size:
            raise ValueError("wave raw size cannot be smaller than wave size")


@dataclass(frozen=True, slots=True)
class _Selection:
    features: torch.Tensor
    token: int
    size: int
    raw_size: int


@dataclass(frozen=True, slots=True)
class _NativeSlotSnapshot:
    complete: bool
    simulations: int
    evaluator_completions: int


class _WaveState(Enum):
    IDLE = auto()
    EXECUTING = auto()
    RETRYABLE = auto()
    SUBMITTED = auto()
    READY = auto()


@dataclass(frozen=True, slots=True)
class _CudaWaveSubmission:
    stream: torch.cuda.Stream
    completion: torch.cuda.Event


def load_native_extension() -> Any:
    """Load ``kb_pente_native`` without compiling it at runtime."""

    try:
        return importlib.import_module("kb_pente_native")
    except ImportError as error:
        raise NativeBackendUnavailableError(
            "The C++ search backend was requested, but kb_pente_native is "
            "not available. Install the optional extension with "
            "uv pip install --no-build-isolation ./native/torch, then retry. "
            "The Python search backend does not require this extension."
        ) from error


class NativeSearchBackend:
    """Own one native SearchBatch and its model inference staging."""

    def __init__(
        self,
        game: PenteGame,
        evaluator: TensorEvaluator,
        args: MCTSArgs,
        *,
        max_active_games: int,
        worker_threads: int,
        seed: int = 0,
        pin_memory: bool | None = None,
        extension: Any | None = None,
    ) -> None:
        if not isinstance(game, PenteGame):
            raise TypeError("NativeSearchBackend requires a PenteGame")
        if not isinstance(args, MCTSArgs):
            raise TypeError("args must be an MCTSArgs instance")
        self.game = game
        self.evaluator = evaluator
        self.args = args
        self.device = torch.device(evaluator.device)
        self._board_size = game.get_board_size()
        self._action_size = game.get_action_size()
        self._capacity = _positive_int(max_active_games, "max_active_games")
        self._worker_threads = _positive_int(worker_threads, "worker_threads")
        self._seed = _nonnegative_int(seed, "seed")
        self._pin_memory = (
            self.device.type == "cuda"
            if pin_memory is None
            else _bool(pin_memory, "pin_memory")
        )

        native_module = load_native_extension() if extension is None else extension
        try:
            search_batch = native_module.SearchBatch
        except AttributeError as error:
            raise NativeBackendUnavailableError(
                "kb_pente_native is installed but does not expose SearchBatch; "
                "reinstall it with uv pip install --no-build-isolation ./native/torch."
            ) from error
        self._batch = search_batch(
            board_size=self._board_size,
            ruleset=game.ruleset.value,
            simulations=args.num_simulations,
            active_games=self._capacity,
            threads=self._worker_threads,
            seed=self._seed,
            c_puct=args.c_puct,
            root_noise_epsilon=args.root_noise_epsilon,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            pin_memory=self._pin_memory,
        )

        self._features = _tensor(self._batch.features, "features")
        self._policies = _tensor(self._batch.policies, "policies")
        self._values = _tensor(self._batch.values, "values")
        self._feature_pointer = _pointer(self._features)
        self._policy_pointer = _pointer(self._policies)
        self._value_pointer = _pointer(self._values)
        self._validate_staging()
        self._cuda_feature_staging: torch.Tensor | None = None
        self._cuda_timing_events: tuple[torch.cuda.Event, ...] | None = None
        if self.device.type == "cuda":
            self._cuda_feature_staging = torch.empty(
                (self._capacity, 4, self._board_size, self._board_size),
                dtype=torch.float32,
                device=self.device,
            )
            self._cuda_timing_events = tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(6)
            )

        self._roots: dict[int, PenteBoard] = {}
        self._batch_sizes: dict[int, list[int]] = {}
        self._slot_progress: dict[int, _NativeSlotSnapshot] = {}
        self._terminal_results: dict[int, TerminalResult] = {}
        self._pending: _Selection | None = None
        self._pending_completions: dict[int, int] = {}
        self._wave_state = _WaveState.IDLE
        self._pending_result: NativeWave | None = None
        self._pending_cuda: _CudaWaveSubmission | None = None
        self._wave_owner = object()
        self._cuda_faulted = False
        self._inference_timing = NativeInferenceTiming()

    @property
    def active_count(self) -> int:
        return _nonnegative_int(self._batch.active_count, "active_count")

    @property
    def capacity(self) -> int:
        return _nonnegative_int(self._batch.capacity, "capacity")

    @property
    def thread_count(self) -> int:
        return _nonnegative_int(self._batch.thread_count, "thread_count")

    def add_root(
        self,
        position: PenteBoard,
        temperature: float = 1.0,
        add_root_noise: bool = False,
    ) -> int:
        self._ensure_wave_idle()
        root = self._admit_position(position)
        slot = _nonnegative_int(
            self._batch.add(
                *_root_tensors(root),
                root.current_player,
                root.ply,
                last_action=root.last_action,
                temperature=temperature,
                add_root_noise=add_root_noise,
            ),
            "native slot",
        )
        if slot in self._roots:
            raise RuntimeError(f"Native batch reused an active slot {slot}")
        self._roots[slot] = root
        self._batch_sizes[slot] = []
        self._refresh_lifecycle_progress(slot)
        return slot

    def submit_wave(self) -> NativeWaveSubmission:
        """Select and submit one wave, returning its immutable identity.

        CPU inference completes during this call. CUDA inference is enqueued on
        the current stream and completes through :meth:`poll_wave` or
        :meth:`wait_wave`. A failed inference or backup leaves the native token
        pending so the next submission retries the same selection.
        """

        if self._cuda_faulted:
            raise RuntimeError(
                "Native CUDA wave resources are unavailable after a drain failure"
            )
        if self._wave_state in (
            _WaveState.EXECUTING,
            _WaveState.SUBMITTED,
            _WaveState.READY,
        ):
            raise RuntimeError("A native wave is already submitted")
        if self._wave_state is _WaveState.IDLE:
            self._wave_state = _WaveState.EXECUTING
            try:
                self._select_wave()
            except Exception:
                self._wave_state = _WaveState.IDLE
                raise
        elif self._wave_state is not _WaveState.RETRYABLE:
            raise RuntimeError("Native wave state is invalid")

        selection = self._pending
        if selection is None:
            raise RuntimeError("Native wave state has no pending selection")
        submission = NativeWaveSubmission(
            selection.token,
            selection.size,
            selection.raw_size,
        )
        object.__setattr__(submission, "_owner", self._wave_owner)

        if selection.size == 0:
            self._pending_result = NativeWave(
                selection.token,
                0,
                selection.raw_size,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            self._wave_state = _WaveState.READY
            return submission

        try:
            if self.device.type == "cuda":
                self._pending_cuda = self._enqueue_cuda_inference_and_stage(selection)
                self._wave_state = _WaveState.SUBMITTED
            else:
                timings = self._run_inference_and_stage(selection)
                self._pending_result = NativeWave(
                    selection.token,
                    selection.size,
                    selection.raw_size,
                    *timings,
                )
                self._wave_state = _WaveState.READY
        except Exception:
            self._pending_cuda = None
            self._pending_result = None
            self._wave_state = _WaveState.RETRYABLE
            raise
        return submission

    def poll_wave(self, submission: NativeWaveSubmission) -> NativeWave | None:
        """Complete a submitted wave if its CUDA staging event is ready."""

        self._validate_submission(submission)
        self._ensure_submitted_for_completion()
        if self._wave_state is _WaveState.SUBMITTED:
            if self._pending_cuda is None:
                raise RuntimeError("Native CUDA wave state has no pending submission")
            pending = self._pending_cuda
            try:
                ready = pending.completion.query()
            except Exception as error:
                self._handle_cuda_submission_error(pending.stream, error)
            if not ready:
                return None
            self._resolve_cuda_submission(pending)
        return self._complete_wave()

    def wait_wave(self, submission: NativeWaveSubmission) -> NativeWave:
        """Wait for and complete a submitted wave."""

        self._validate_submission(submission)
        self._ensure_submitted_for_completion()
        if self._wave_state is _WaveState.SUBMITTED:
            if self._pending_cuda is None:
                raise RuntimeError("Native CUDA wave state has no pending submission")
            self._resolve_cuda_submission(self._pending_cuda)
        result = self._complete_wave()
        assert result is not None
        return result

    def evaluate_wave(self) -> NativeWave:
        """Run one complete wave using the legacy synchronous API."""

        return self.wait_wave(self.submit_wave())

    def root_policy(self, slot: int) -> np.ndarray:
        self._ensure_wave_idle()
        parsed_slot = _nonnegative_int(slot, "slot")
        policy = self._batch.root_policy(parsed_slot)
        if not isinstance(policy, torch.Tensor):
            raise TypeError("Native root_policy must return a torch.Tensor")
        expected = (self._action_size,)
        if (
            policy.device.type != "cpu"
            or policy.dtype != torch.float32
            or tuple(policy.shape) != expected
            or not policy.is_contiguous()
        ):
            raise ValueError(
                "Native root_policy must be a contiguous CPU float32 tensor "
                f"with shape {expected}"
            )
        return policy.detach().numpy().copy()

    def advance_root(
        self,
        slot: int,
        action: int,
        temperature: float = 1.0,
        add_root_noise: bool = False,
    ) -> dict[str, Any]:
        self._ensure_wave_idle()
        parsed_slot = _nonnegative_int(slot, "slot")
        parsed_action = _nonnegative_int(action, "action")
        try:
            root = self._roots[parsed_slot]
        except KeyError as error:
            raise ValueError(f"Unknown native root slot: {parsed_slot}") from error
        next_position, _ = self.game.apply_action(root, root.current_player, parsed_action)
        result = self._batch.advance_root(
            parsed_slot,
            parsed_action,
            temperature=temperature,
            add_root_noise=add_root_noise,
        )
        self._roots[parsed_slot] = next_position
        self._batch_sizes[parsed_slot] = []
        self._refresh_lifecycle_progress(parsed_slot)
        return dict(result)

    def observe_action(
        self,
        slot: int,
        action: int,
        temperature: float = 1.0,
        add_root_noise: bool = False,
    ) -> dict[str, Any]:
        self._ensure_wave_idle()
        parsed_slot = _nonnegative_int(slot, "slot")
        parsed_action = _nonnegative_int(action, "action")
        try:
            root = self._roots[parsed_slot]
        except KeyError as error:
            raise ValueError(f"Unknown native root slot: {parsed_slot}") from error
        next_position, _ = self.game.apply_action(
            root,
            root.current_player,
            parsed_action,
        )
        result = self._batch.observe_action(
            parsed_slot,
            parsed_action,
            temperature=temperature,
            add_root_noise=add_root_noise,
        )
        self._roots[parsed_slot] = next_position
        self._batch_sizes[parsed_slot] = []
        self._refresh_lifecycle_progress(parsed_slot)
        return dict(result)

    def remove(self, slot: int) -> None:
        self._ensure_wave_idle()
        parsed_slot = _nonnegative_int(slot, "slot")
        self._batch.remove(parsed_slot)
        self._roots.pop(parsed_slot, None)
        self._batch_sizes.pop(parsed_slot, None)
        self._slot_progress.pop(parsed_slot, None)
        self._terminal_results.pop(parsed_slot, None)

    def replace_root(
        self,
        slot: int,
        position: PenteBoard,
        temperature: float = 1.0,
        add_root_noise: bool = False,
    ) -> None:
        self._ensure_wave_idle()
        parsed_slot = _nonnegative_int(slot, "slot")
        root = self._admit_position(position)
        self._batch.replace_root(
            parsed_slot,
            *_root_tensors(root),
            root.current_player,
            root.ply,
            last_action=root.last_action,
            temperature=temperature,
            add_root_noise=add_root_noise,
        )
        self._roots[parsed_slot] = root
        self._batch_sizes[parsed_slot] = []
        self._refresh_lifecycle_progress(parsed_slot)

    def root_terminal(self, slot: int) -> TerminalResult:
        self._ensure_wave_idle()
        parsed_slot = _nonnegative_int(slot, "slot")
        cached = self._terminal_results.get(parsed_slot)
        if cached is not None:
            return cached
        result = _convert_terminal(self._batch.root_terminal(parsed_slot))
        if result.is_terminal:
            self._terminal_results[parsed_slot] = result
            self._slot_progress[parsed_slot] = _NativeSlotSnapshot(True, 0, 0)
        return result

    def slot_telemetry(self, slot: int) -> SearchTelemetry:
        parsed_slot = _nonnegative_int(slot, "slot")
        return _convert_search_telemetry(
            self._batch.slot_telemetry(parsed_slot),
            _mean(self._batch_sizes.get(parsed_slot, [])),
        )

    def complete(self) -> bool:
        if self._wave_state is not _WaveState.IDLE:
            return False
        return bool(self._batch.complete())

    def slot_complete(self, slot: int) -> bool:
        parsed_slot = _nonnegative_int(slot, "slot")
        try:
            complete = self._slot_progress[parsed_slot].complete
        except KeyError as error:
            raise IndexError(f"Native root slot is not active: {parsed_slot}") from error
        return False if self._wave_state is not _WaveState.IDLE else complete

    def slot_simulations(self, slot: int) -> int:
        parsed_slot = _nonnegative_int(slot, "slot")
        try:
            return self._slot_progress[parsed_slot].simulations
        except KeyError as error:
            raise ValueError(f"Unknown native root slot: {parsed_slot}") from error

    def deduplication_telemetry(self) -> dict[str, Any]:
        return dict(self._batch.deduplication_telemetry())

    def native_timing_telemetry(self) -> dict[str, Any]:
        return dict(self._batch.timing_telemetry())

    def inference_timing(self) -> dict[str, int | float]:
        return self._inference_timing.to_dict()

    def worker_telemetry(self) -> dict[str, int | float]:
        worker = self.native_timing_telemetry()["cumulative"]["select"]["worker"]
        busy_fraction = float(worker["busy_fraction"])
        return {
            "worker_threads": self.thread_count,
            "worker_busy_fraction": busy_fraction,
            "worker_busy_percent": 100.0 * busy_fraction,
        }

    def timing_telemetry(self) -> dict[str, Any]:
        native = self.native_timing_telemetry()
        worker = native["cumulative"]["select"]["worker"]
        busy_fraction = float(worker["busy_fraction"])
        return {
            "native": native,
            "inference": self.inference_timing(),
            "worker": {
                "worker_threads": self.thread_count,
                "worker_busy_fraction": busy_fraction,
                "worker_busy_percent": 100.0 * busy_fraction,
            },
        }

    def _ensure_wave_idle(self) -> None:
        if self._cuda_faulted:
            raise RuntimeError(
                "Native CUDA wave resources are unavailable after a drain failure"
            )
        if self._wave_state is not _WaveState.IDLE:
            raise RuntimeError(
                "A native wave is pending; complete it before changing search roots"
            )

    def _select_wave(self) -> None:
        self._validate_staging()
        before = {
            slot: snapshot.evaluator_completions
            for slot, snapshot in self._slot_progress.items()
        }
        selected = self._batch.select()
        self._pending = self._convert_selection(selected)
        self._pending_completions = before

    def _validate_submission(self, submission: NativeWaveSubmission) -> None:
        if not isinstance(submission, NativeWaveSubmission):
            raise TypeError("Native wave submission must be a NativeWaveSubmission")
        if submission._owner is not self._wave_owner:
            raise ValueError("Native wave submission belongs to another backend")
        selection = self._pending
        if selection is None or self._wave_state is _WaveState.IDLE:
            raise ValueError("Native wave submission is stale or no longer pending")
        if (
            submission.token != selection.token
            or submission.size != selection.size
            or submission.raw_size != selection.raw_size
        ):
            raise ValueError("Native wave submission does not match the pending wave")

    def _ensure_submitted_for_completion(self) -> None:
        if self._wave_state is _WaveState.RETRYABLE:
            raise RuntimeError(
                "Native wave submission failed; call submit_wave to retry "
                "the pending wave"
            )
        if self._wave_state not in (_WaveState.SUBMITTED, _WaveState.READY):
            raise RuntimeError("Native wave is not submitted")

    def _complete_wave(self) -> NativeWave:
        if self._wave_state is not _WaveState.READY:
            raise RuntimeError("Native wave is not ready for backup")
        selection = self._pending
        result = self._pending_result
        if selection is None or result is None:
            raise RuntimeError("Native wave state has no completed result")

        if selection.size == 0:
            self._update_slot_progress(self._read_slot_snapshots())
            self._clear_pending_wave()
            return result

        try:
            self._batch.backup(selection.token, selection.size)
        except Exception:
            self._pending_result = None
            self._pending_cuda = None
            self._wave_state = _WaveState.RETRYABLE
            raise

        previous = self._pending_completions
        self._pending = None
        self._pending_completions = {}
        self._pending_result = None
        self._pending_cuda = None
        self._wave_state = _WaveState.IDLE
        snapshots = self._read_slot_snapshots()
        self._record_batch_sizes(selection, previous, snapshots)
        self._update_slot_progress(snapshots)
        return result

    def _clear_pending_wave(self) -> None:
        self._pending = None
        self._pending_completions = {}
        self._pending_result = None
        self._pending_cuda = None
        self._wave_state = _WaveState.IDLE

    def _admit_position(self, position: PenteBoard) -> PenteBoard:
        if not isinstance(position, PenteBoard):
            raise TypeError("Native roots must be PenteBoard instances")
        if (
            position.board.dtype != np.int8
            or position.board.ndim != 2
            or position.board.shape != (self._board_size, self._board_size)
            or not position.board.flags.c_contiguous
        ):
            raise ValueError(
                "Native root stones must be a contiguous int8 array with "
                f"shape ({self._board_size}, {self._board_size})"
            )
        if (
            position.captures.dtype != np.int16
            or position.captures.shape != (2,)
            or not position.captures.flags.c_contiguous
        ):
            raise ValueError(
                "Native root captures must be a contiguous int16 array with shape (2,)"
            )
        if position.board.flags.writeable or position.captures.flags.writeable:
            raise ValueError("Native root arrays must be immutable")
        if position.ply is None:
            raise ValueError("Native root ply must be present")
        if self.game.check_game_end(position).is_terminal:
            raise ValueError("Cannot add a terminal root to native search")
        return position

    def _validate_staging(self) -> None:
        for tensor, shape, pointer, name in (
            (
                self._features,
                (self._capacity, 4, self._board_size, self._board_size),
                self._feature_pointer,
                "features",
            ),
            (self._policies, (self._capacity, 361), self._policy_pointer, "policies"),
            (self._values, (self._capacity,), self._value_pointer, "values"),
        ):
            _validate_staging_tensor(tensor, shape, pointer, name, self._pin_memory)

    def _convert_selection(self, selected: Any) -> _Selection:
        features = _tensor(selected.features, "selection features")
        token = _nonnegative_int(selected.token, "selection token")
        size = _nonnegative_int(selected.size, "selection size")
        raw_size = _nonnegative_int(selected.raw_size, "selection raw size")
        if size > self._capacity or raw_size > self._capacity or raw_size < size:
            raise ValueError("Native selection size exceeds capacity or raw count")
        expected = (size, 4, self._board_size, self._board_size)
        if (
            features.device.type != "cpu"
            or features.dtype != torch.float32
            or tuple(features.shape) != expected
            or not features.is_contiguous()
        ):
            raise ValueError(
                "Native selection features must be contiguous CPU float32 NCHW "
                f"with shape {expected}"
            )
        if size and _pointer(features) != self._feature_pointer:
            raise ValueError("Native selection features do not use binding-owned staging")
        if self.device.type == "cuda" and not features.is_pinned():
            raise ValueError("CUDA native search requires pinned feature staging")
        return _Selection(features, token, size, raw_size)

    def _run_inference_and_stage(
        self,
        selection: _Selection,
    ) -> tuple[float, float, float, float]:
        if self.device.type == "cuda":
            raise RuntimeError("CUDA inference must be submitted asynchronously")
        started = time.perf_counter()
        inputs = selection.features.to(device=self.device)
        host_to_device = _elapsed(started)
        started = time.perf_counter()
        policies, values = self._call_evaluator(inputs)
        model = _elapsed(started)
        self._validate_outputs(policies, values, selection.size)
        started = time.perf_counter()
        self._copy_to_staging(policies, values, selection.size)
        device_to_host = _elapsed(started)
        self._add_timing(1, host_to_device, model, device_to_host)
        return host_to_device, model, device_to_host, 0.0

    def _enqueue_cuda_inference_and_stage(
        self,
        selection: _Selection,
    ) -> _CudaWaveSubmission:
        self._validate_staging()
        if self._cuda_feature_staging is None or self._cuda_timing_events is None:
            raise RuntimeError("CUDA inference resources were not initialized")
        stream = torch.cuda.current_stream(self.device)
        (
            h2d_started,
            h2d_finished,
            model_started,
            model_finished,
            d2h_started,
            d2h_finished,
        ) = self._cuda_timing_events
        try:
            h2d_started.record(stream)
            inputs = self._cuda_feature_staging[: selection.size]
            inputs.copy_(selection.features, non_blocking=True)
            h2d_finished.record(stream)
            model_started.record(stream)
            policies, values = self._call_evaluator(inputs)
            model_finished.record(stream)
            self._validate_outputs(policies, values, selection.size)
            d2h_started.record(stream)
            self._copy_to_staging(policies, values, selection.size)
            d2h_finished.record(stream)
        except Exception as error:
            self._handle_cuda_submission_error(stream, error)
        return _CudaWaveSubmission(stream, d2h_finished)

    def _resolve_cuda_submission(self, pending: _CudaWaveSubmission) -> None:
        selection = self._pending
        if selection is None:
            raise RuntimeError("Native CUDA wave has no pending selection")
        try:
            wait_started = time.perf_counter()
            pending.completion.synchronize()
            wait = _elapsed(wait_started)
            if self._cuda_timing_events is None:
                raise RuntimeError("CUDA inference resources were not initialized")
            (
                h2d_started,
                h2d_finished,
                model_started,
                model_finished,
                d2h_started,
                d2h_finished,
            ) = self._cuda_timing_events
            host_to_device = _cuda_elapsed(h2d_started, h2d_finished)
            model = _cuda_elapsed(model_started, model_finished)
            device_to_host = _cuda_elapsed(d2h_started, d2h_finished)
            self._add_timing(1, host_to_device, model, device_to_host, wait)
            self._pending_result = NativeWave(
                selection.token,
                selection.size,
                selection.raw_size,
                host_to_device,
                model,
                device_to_host,
                wait,
            )
            self._pending_cuda = None
            self._wave_state = _WaveState.READY
        except Exception as error:
            self._handle_cuda_submission_error(pending.stream, error)

    def _handle_cuda_submission_error(
        self,
        stream: torch.cuda.Stream,
        error: Exception,
    ) -> NoReturn:
        self._pending_result = None
        self._pending_cuda = None
        self._wave_state = _WaveState.RETRYABLE
        try:
            self._drain_cuda_stream(stream)
        except Exception as drain_error:
            raise drain_error from error
        raise error

    def _drain_cuda_stream(self, stream: torch.cuda.Stream) -> None:
        try:
            stream.synchronize()
        except Exception as error:
            self._cuda_faulted = True
            raise RuntimeError(
                "CUDA wave stream could not be drained safely for reuse"
            ) from error

    def _call_evaluator(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.evaluator.evaluate_features(inputs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Tensor evaluator must return (policy, values)")
        policies, values = result
        if not isinstance(policies, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("Tensor evaluator must return two torch.Tensor objects")
        return policies, values

    def _validate_outputs(
        self,
        policies: torch.Tensor,
        values: torch.Tensor,
        rows: int,
    ) -> None:
        expected = (rows, self._action_size)
        if policies.dtype != torch.float32 or values.dtype != torch.float32:
            raise ValueError("Native evaluator outputs must use float32 tensors")
        if not _same_device(policies.device, self.device) or not _same_device(
            values.device,
            self.device,
        ):
            raise ValueError(
                f"Native evaluator outputs must be on configured device {self.device}"
            )
        if tuple(policies.shape) != expected or tuple(values.shape) != (rows,):
            raise ValueError(
                f"Native evaluator output shapes must be {expected} and ({rows},)"
            )
        if not policies.is_contiguous() or not values.is_contiguous():
            raise ValueError("Native evaluator outputs must be contiguous")
        if self.device.type == "cpu":
            if not bool(torch.isfinite(policies).all()) or bool((policies < 0).any()):
                raise ValueError("Native evaluator policy must be finite and non-negative")
            if (
                not bool(torch.isfinite(values).all())
                or bool((values < -1).any())
                or bool((values > 1).any())
            ):
                raise ValueError("Native evaluator values must be finite and in [-1, 1]")

    def _copy_to_staging(
        self,
        policies: torch.Tensor,
        values: torch.Tensor,
        rows: int,
    ) -> None:
        self._validate_staging()
        non_blocking = self.device.type == "cuda"
        with torch.no_grad():
            self._policies[:rows, : self._action_size].copy_(
                policies,
                non_blocking=non_blocking,
            )
            if self._action_size < 361:
                self._policies[:rows, self._action_size :].zero_()
            self._values[:rows].copy_(values, non_blocking=non_blocking)

    def _read_slot_snapshots(self) -> dict[int, _NativeSlotSnapshot]:
        raw_snapshots = self._batch.slot_snapshots()
        if not isinstance(raw_snapshots, tuple):
            raise TypeError("Native slot snapshots must be an immutable tuple")

        snapshots: dict[int, _NativeSlotSnapshot] = {}
        for raw_snapshot in raw_snapshots:
            if not isinstance(raw_snapshot, tuple) or len(raw_snapshot) != 4:
                raise ValueError(
                    "Native slot snapshots must contain four tuple fields"
                )
            slot = _nonnegative_int(raw_snapshot[0], "native slot")
            complete = raw_snapshot[1]
            if not isinstance(complete, bool):
                raise TypeError("Native slot completion must be a bool")
            snapshot = _NativeSlotSnapshot(
                complete,
                _nonnegative_int(raw_snapshot[2], "native simulations"),
                _nonnegative_int(
                    raw_snapshot[3],
                    "native evaluator completions",
                ),
            )
            if slot in snapshots:
                raise ValueError(f"Native slot snapshots contain duplicate slot {slot}")
            snapshots[slot] = snapshot

        if set(snapshots) != set(self._roots):
            raise RuntimeError("Native slot snapshots do not match active roots")
        return snapshots

    def _refresh_lifecycle_progress(self, slot: int) -> None:
        terminal = _convert_terminal(self._batch.root_terminal(slot))
        self._slot_progress[slot] = _NativeSlotSnapshot(
            terminal.is_terminal,
            0,
            0,
        )
        if terminal.is_terminal:
            self._terminal_results[slot] = terminal
        else:
            self._terminal_results.pop(slot, None)

    def _update_slot_progress(
        self,
        snapshots: Mapping[int, _NativeSlotSnapshot],
    ) -> None:
        self._slot_progress = dict(snapshots)

    def _record_batch_sizes(
        self,
        selection: _Selection,
        previous: Mapping[int, int],
        current: Mapping[int, _NativeSlotSnapshot],
    ) -> None:
        for slot, before in previous.items():
            snapshot = current.get(slot)
            if snapshot is None:
                continue
            if snapshot.evaluator_completions > before:
                self._batch_sizes.setdefault(slot, []).extend(
                    [selection.size] * (snapshot.evaluator_completions - before)
                )

    def _add_timing(
        self,
        calls: int,
        host_to_device: float = 0.0,
        model: float = 0.0,
        device_to_host: float = 0.0,
        wait: float = 0.0,
    ) -> None:
        current = self._inference_timing
        self._inference_timing = NativeInferenceTiming(
            calls=current.calls + calls,
            host_to_device_seconds=current.host_to_device_seconds + _finite(host_to_device),
            model_inference_seconds=current.model_inference_seconds + _finite(model),
            device_to_host_seconds=current.device_to_host_seconds + _finite(device_to_host),
            inference_wait_seconds=current.inference_wait_seconds + _finite(wait),
        )


def _root_tensors(root: PenteBoard) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(root.board, dtype=torch.int8, device="cpu"),
        torch.tensor(root.captures, dtype=torch.int16, device="cpu"),
    )


def _tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Native {name} staging must be a torch.Tensor")
    return value


def _pointer(tensor: torch.Tensor) -> int:
    return int(tensor.data_ptr())


def _validate_staging_tensor(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    expected_pointer: int,
    name: str,
    pinned: bool,
) -> None:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise ValueError(f"Native {name} staging must be a CPU float32 tensor")
    if tuple(tensor.shape) != expected_shape or not tensor.is_contiguous():
        raise ValueError(
            f"Native {name} staging must be contiguous with shape {expected_shape}"
        )
    if _pointer(tensor) != expected_pointer:
        raise ValueError(f"Native {name} staging storage was replaced")
    if tensor.is_pinned() != pinned:
        raise ValueError(f"Native {name} staging has unexpected pinning")


def _convert_terminal(raw: Mapping[str, Any]) -> TerminalResult:
    try:
        status = GameStatus(str(raw["status"]))
    except ValueError as error:
        raise ValueError(f"Native terminal status is unknown: {raw['status']!r}") from error
    if status is GameStatus.IN_PROGRESS:
        return TerminalResult.in_progress()
    if status is GameStatus.DRAW:
        return TerminalResult.draw()
    winner = _integer(raw["winner"], "terminal winner", nonnegative=False)
    if winner not in (1, -1):
        raise ValueError("Native terminal winner must be Player 1 or Player 2")
    reason = str(raw.get("reason", "none"))
    return TerminalResult.win(winner, None if reason == "none" else reason)


def _convert_search_telemetry(
    raw: Mapping[str, Any],
    mean_batch: float,
) -> SearchTelemetry:
    completions = _nonnegative_int(raw["evaluator_completions"], "evaluator completions")
    return SearchTelemetry(
        simulations=_nonnegative_int(raw["completed_simulations"], "simulations"),
        evaluator_calls=completions,
        evaluated_positions=completions,
        invalid_policy_fallbacks=_nonnegative_int(
            raw["invalid_policy_fallbacks"],
            "invalid policy fallbacks",
        ),
        zero_visit_fallbacks=_nonnegative_int(
            raw["zero_visit_fallbacks"],
            "zero visit fallbacks",
        ),
        max_depth=_nonnegative_int(raw["max_selected_path_depth"], "max depth"),
        root_legal_actions=_nonnegative_int(raw["root_legal_actions"], "root legal actions"),
        root_edge_visits=_nonnegative_int(raw["root_edge_visits"], "root edge visits"),
        root_children_visited=_nonnegative_int(
            raw["root_children_visited"],
            "root children visited",
        ),
        root_visit_entropy=float(raw["root_visit_entropy"]),
        root_max_visit_share=float(raw["root_max_visit_share"]),
        root_collapse_eligible=bool(raw["root_collapse_eligible"]),
        root_search_collapsed=bool(raw["root_search_collapsed"]),
        mean_inference_batch_size=mean_batch,
    )


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _positive_int(value: object, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: object, name: str) -> int:
    return _integer(value, name)


def _integer(value: object, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = int(operator.index(cast(SupportsIndex, value)))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if nonnegative and parsed < 0:
        raise ValueError(f"{name} cannot be negative")
    return parsed


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _elapsed(started: float) -> float:
    return _finite(time.perf_counter() - started)


def _finite(value: float) -> float:
    return value if math.isfinite(value) and value >= 0.0 else 0.0


def _cuda_elapsed(started: torch.cuda.Event, finished: torch.cuda.Event) -> float:
    return _finite(float(started.elapsed_time(finished)) / 1000.0)


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


__all__ = [
    "NativeBackendUnavailableError",
    "NativeInferenceTiming",
    "NativeSearchBackend",
    "NativeWave",
    "NativeWaveSubmission",
    "TensorEvaluator",
    "load_native_extension",
]
