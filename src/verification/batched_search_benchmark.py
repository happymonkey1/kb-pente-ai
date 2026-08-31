from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.batched import run_batched_search
from src.mcts.mcts_v2 import MCTS, MCTSArgs
from src.model.model_v1 import PenteNet


@dataclass(frozen=True, slots=True)
class BatchedSearchBenchmarkConfig:
    board_size: int = 19
    ruleset: PenteRuleset = PenteRuleset.STANDARD
    games: int = 16
    simulations: int = 16
    repeats: int = 3
    warmup_batches: int = 2
    seed: int = 37
    model_blocks: int = 1
    model_channels: int = 16
    model_hidden_size: int = 64
    maximum_policy_difference: float = 0.05
    minimum_selected_action_agreement: float = 1.0
    minimum_speedup: float = 1.0

    def __post_init__(self) -> None:
        if self.board_size < 5:
            raise ValueError("Board size must be at least five")
        if self.games < 2:
            raise ValueError("The batch benchmark requires at least two games")
        if self.simulations < 2 or self.repeats < 1 or self.warmup_batches < 0:
            raise ValueError("Invalid simulation, repeat, or warmup count")
        if self.model_blocks < 1 or self.model_channels < 1 or self.model_hidden_size < 1:
            raise ValueError("Model dimensions must be positive")
        if self.maximum_policy_difference < 0 or self.minimum_speedup < 0:
            raise ValueError("Policy tolerance and minimum speedup cannot be negative")
        if not 0 <= self.minimum_selected_action_agreement <= 1:
            raise ValueError("Selected-action agreement must be between zero and one")


@dataclass(frozen=True, slots=True)
class BatchedSearchBenchmarkReport:
    board_size: int
    ruleset: str
    games: int
    simulations: int
    repeats: int
    seed: int
    device: str
    torch_threads: int
    model_parameters: int
    reference_seconds: float
    batched_seconds: float
    speedup: float
    reference_leaf_evaluations: int
    batched_leaf_evaluations: int
    reference_leaf_evaluations_per_second: float
    batched_leaf_evaluations_per_second: float
    min_inference_batch_size: int
    mean_inference_batch_size: float
    median_inference_batch_size: float
    p95_inference_batch_size: float
    max_inference_batch_size: int
    duplicate_leaf_rate: float
    maximum_policy_difference: float
    selected_action_agreement: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_batched_search_benchmark(
    config: BatchedSearchBenchmarkConfig | None = None,
    device: torch.device | None = None,
) -> BatchedSearchBenchmarkReport:
    selected = config or BatchedSearchBenchmarkConfig()
    selected_device = device or torch.device("cpu")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA benchmark requested without an available CUDA device")

    torch.manual_seed(selected.seed)
    game = PenteGame(selected.board_size, ruleset=selected.ruleset)
    model = PenteNet(
        selected_device,
        board_size=selected.board_size,
        action_size=game.get_action_size(),
        num_res_blocks=selected.model_blocks,
        num_channels=selected.model_channels,
        hidden_fc_size=selected.model_hidden_size,
    )
    model.eval()
    roots = _build_distinct_roots(game, selected.games, selected.seed)
    for _ in range(selected.warmup_batches):
        model.evaluate_batch(roots)

    args = MCTSArgs(num_simulations=selected.simulations)
    reference_seconds = 0.0
    batched_seconds = 0.0
    reference_evaluations = 0
    batched_evaluations = 0
    evaluation_requests = 0
    batch_sizes: list[int] = []
    maximum_policy_difference = 0.0
    matching_selected_actions = 0
    compared_policies = 0

    for repeat in range(selected.repeats):
        if repeat % 2 == 0:
            reference = _run_reference(game, model, args, roots, selected.seed + repeat)
            batched = _run_batched(game, model, args, roots, selected.seed + repeat)
        else:
            batched = _run_batched(game, model, args, roots, selected.seed + repeat)
            reference = _run_reference(game, model, args, roots, selected.seed + repeat)

        reference_seconds += reference.elapsed_seconds
        batched_seconds += batched.elapsed_seconds
        reference_evaluations += reference.evaluated_positions
        batched_evaluations += batched.evaluated_positions
        evaluation_requests += batched.evaluation_requests
        batch_sizes.extend(batched.batch_sizes)
        for reference_policy, batched_policy in zip(reference.policies, batched.policies):
            maximum_policy_difference = max(
                maximum_policy_difference,
                float(np.max(np.abs(reference_policy - batched_policy))),
            )
            matching_selected_actions += int(
                np.argmax(reference_policy) == np.argmax(batched_policy)
            )
            compared_policies += 1

    speedup = reference_seconds / batched_seconds if batched_seconds else float("inf")
    selected_action_agreement = matching_selected_actions / compared_policies
    failures = []
    if maximum_policy_difference > selected.maximum_policy_difference:
        failures.append("policy difference exceeds tolerance")
    if selected_action_agreement < selected.minimum_selected_action_agreement:
        failures.append("selected-action agreement is below threshold")
    if speedup < selected.minimum_speedup:
        failures.append("batched search speedup is below threshold")
    if max(batch_sizes, default=0) <= 1:
        failures.append("batched search never evaluated more than one leaf")

    return BatchedSearchBenchmarkReport(
        board_size=selected.board_size,
        ruleset=selected.ruleset.value,
        games=selected.games,
        simulations=selected.simulations,
        repeats=selected.repeats,
        seed=selected.seed,
        device=str(selected_device),
        torch_threads=torch.get_num_threads(),
        model_parameters=model.get_parameter_count(),
        reference_seconds=reference_seconds,
        batched_seconds=batched_seconds,
        speedup=speedup,
        reference_leaf_evaluations=reference_evaluations,
        batched_leaf_evaluations=batched_evaluations,
        reference_leaf_evaluations_per_second=(
            reference_evaluations / reference_seconds
        ),
        batched_leaf_evaluations_per_second=batched_evaluations / batched_seconds,
        min_inference_batch_size=min(batch_sizes, default=0),
        mean_inference_batch_size=float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        median_inference_batch_size=float(np.median(batch_sizes)) if batch_sizes else 0.0,
        p95_inference_batch_size=(
            float(np.percentile(batch_sizes, 95)) if batch_sizes else 0.0
        ),
        max_inference_batch_size=max(batch_sizes, default=0),
        duplicate_leaf_rate=(
            1.0 - batched_evaluations / evaluation_requests
            if evaluation_requests
            else 0.0
        ),
        maximum_policy_difference=maximum_policy_difference,
        selected_action_agreement=selected_action_agreement,
        passed=not failures,
        failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class _SearchRun:
    policies: list[np.ndarray]
    elapsed_seconds: float
    evaluated_positions: int
    evaluation_requests: int
    batch_sizes: tuple[int, ...]


def _run_reference(
    game: PenteGame,
    model: PenteNet,
    args: MCTSArgs,
    roots: list[PenteBoard],
    seed: int,
) -> _SearchRun:
    _synchronize(model.device)
    started = time.perf_counter()
    policies = []
    evaluated_positions = 0
    for index, root in enumerate(roots):
        search = MCTS(game, model, args, np.random.default_rng(seed + index))
        policies.append(search.get_action_prob(root, temp=1.0, add_root_noise=False))
        evaluated_positions += search.telemetry(root).evaluated_positions
    _synchronize(model.device)
    return _SearchRun(
        policies=policies,
        elapsed_seconds=time.perf_counter() - started,
        evaluated_positions=evaluated_positions,
        evaluation_requests=evaluated_positions,
        batch_sizes=tuple(1 for _ in range(evaluated_positions)),
    )


def _run_batched(
    game: PenteGame,
    model: PenteNet,
    args: MCTSArgs,
    roots: list[PenteBoard],
    seed: int,
) -> _SearchRun:
    searches = [
        MCTS(game, model, args, np.random.default_rng(seed + index))
        for index in range(len(roots))
    ]
    _synchronize(model.device)
    started = time.perf_counter()
    result = run_batched_search(
        searches,
        roots,
        [1.0] * len(roots),
        add_root_noise=False,
    )
    _synchronize(model.device)
    return _SearchRun(
        policies=result.policies,
        elapsed_seconds=time.perf_counter() - started,
        evaluated_positions=result.telemetry.unique_evaluations,
        evaluation_requests=result.telemetry.evaluation_requests,
        batch_sizes=result.telemetry.inference_batch_sizes,
    )


def _build_distinct_roots(
    game: PenteGame,
    count: int,
    seed: int,
) -> list[PenteBoard]:
    rng = np.random.default_rng(seed)
    roots: list[PenteBoard] = []
    keys: set[bytes] = set()
    attempts = 0
    while len(roots) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("Could not construct enough distinct benchmark roots")
        position = game.init_board()
        target_plies = 4 + len(roots) % max(2, min(12, game.get_board_size()))
        for _ in range(target_plies):
            legal = np.flatnonzero(
                game.get_valid_moves(position, position.current_player)
            )
            action = int(rng.choice(legal))
            position, _ = game.apply_action(
                position,
                position.current_player,
                action,
            )
            if game.check_game_end(position).is_terminal:
                break
        key = position.state_key()
        if game.check_game_end(position).is_terminal or key in keys:
            continue
        roots.append(position)
        keys.add(key)
    return roots


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
