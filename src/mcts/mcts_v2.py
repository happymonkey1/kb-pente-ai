from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol, Sequence

import numpy as np

from src.game.game import TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame


EPSILON = 1e-8
SEARCH_COLLAPSE_MIN_ROOT_VISITS = 8


class PolicyValueEvaluator(Protocol):
    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        """Evaluate one position from its side-to-move perspective."""

    def evaluate_batch(self, positions: Sequence[PenteBoard]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate positions from each position's side-to-move perspective."""


@dataclass(frozen=True, slots=True)
class MCTSArgs:
    num_simulations: int = 400
    c_puct: float = 1.5
    root_noise_epsilon: float = 0.25
    root_dirichlet_alpha: float = 0.03

    def __post_init__(self) -> None:
        if self.num_simulations < 1:
            raise ValueError("MCTS requires at least one simulation")
        if self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if not 0 <= self.root_noise_epsilon <= 1:
            raise ValueError("root_noise_epsilon must be between zero and one")
        if self.root_dirichlet_alpha <= 0:
            raise ValueError("root_dirichlet_alpha must be positive")


@dataclass(frozen=True, slots=True)
class SearchPathEdge:
    state_key: bytes
    action: int


@dataclass(frozen=True, slots=True)
class LeafSelection:
    position: PenteBoard
    path: tuple[SearchPathEdge, ...]
    terminal_result: TerminalResult


@dataclass(frozen=True, slots=True)
class SearchTelemetry:
    simulations: int
    evaluator_calls: int
    evaluated_positions: int
    invalid_policy_fallbacks: int
    zero_visit_fallbacks: int
    max_depth: int
    root_legal_actions: int
    root_edge_visits: int
    root_children_visited: int
    root_visit_entropy: float
    root_max_visit_share: float
    root_collapse_eligible: bool
    root_search_collapsed: bool
    mean_inference_batch_size: float


class MCTS:
    def __init__(
        self,
        game: PenteGame,
        evaluator: PolicyValueEvaluator,
        args: MCTSArgs,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.game = game
        self.evaluator = evaluator
        self.args = args
        self.rng = rng if rng is not None else np.random.default_rng()

        self.qsa: dict[tuple[bytes, int], float] = {}
        self.nsa: dict[tuple[bytes, int], int] = {}
        self.ns: dict[bytes, int] = {}
        self.ps: dict[bytes, np.ndarray] = {}
        self.terminals: dict[bytes, TerminalResult] = {}
        self.valids: dict[bytes, np.ndarray] = {}

        self.simulations = 0
        self.evaluator_calls = 0
        self.evaluated_positions = 0
        self.invalid_policy_fallbacks = 0
        self.zero_visit_fallbacks = 0
        self.max_depth = 0
        self.inference_batch_sizes: list[int] = []
        self.net_time = 0.0

    def reset(self) -> None:
        self.qsa.clear()
        self.nsa.clear()
        self.ns.clear()
        self.ps.clear()
        self.terminals.clear()
        self.valids.clear()

        self.simulations = 0
        self.evaluator_calls = 0
        self.evaluated_positions = 0
        self.invalid_policy_fallbacks = 0
        self.zero_visit_fallbacks = 0
        self.max_depth = 0
        self.inference_batch_sizes.clear()
        self.net_time = 0.0

    def get_action_prob(
        self,
        position: PenteBoard,
        temp: float = 1.0,
        add_root_noise: bool = False,
    ) -> np.ndarray:
        terminal = self.game.check_game_end(position)
        if terminal.is_terminal:
            raise ValueError("Cannot search a terminal position")
        if temp < 0:
            raise ValueError("Temperature cannot be negative")

        root_key = self.game.to_string(position)
        root_priors = self._root_priors(root_key, add_root_noise)

        for _ in range(self.args.num_simulations):
            selection = self.select_leaf(position, root_key, root_priors)
            self.expand_and_backup(selection)
            self.record_simulation()
            if root_priors is None:
                root_priors = self._root_priors(root_key, add_root_noise)

        return self.action_prob_from_counts(position, temp)

    def action_prob_from_counts(self, position: PenteBoard, temp: float) -> np.ndarray:
        if temp < 0:
            raise ValueError("Temperature cannot be negative")
        root_key = self.game.to_string(position)
        counts = np.array(
            [self.nsa.get((root_key, action), 0) for action in range(self.game.get_action_size())],
            dtype=np.float64,
        )
        if temp == 0:
            return self._maximum_visit_policy(position, counts)

        adjusted = np.power(counts, 1.0 / temp)
        if float(adjusted.sum()) <= 0:
            self.zero_visit_fallbacks += 1
            return self._expanded_prior_or_uniform(position)
        return adjusted / adjusted.sum()

    def root_priors(self, position: PenteBoard, add_noise: bool) -> np.ndarray | None:
        return self._root_priors(self.game.to_string(position), add_noise)

    def record_simulation(self) -> None:
        self.simulations += 1

    def select_leaf(
        self,
        root: PenteBoard,
        root_key: bytes | None = None,
        root_priors: np.ndarray | None = None,
    ) -> LeafSelection:
        position = root
        path: list[SearchPathEdge] = []
        expected_root_key = self.game.to_string(root) if root_key is None else root_key

        while True:
            state_key = self.game.to_string(position)
            terminal = self.terminals.get(state_key)
            if terminal is None:
                terminal = self.game.check_game_end(position)
                self.terminals[state_key] = terminal

            if terminal.is_terminal or state_key not in self.ps:
                self.max_depth = max(self.max_depth, len(path))
                return LeafSelection(position, tuple(path), terminal)

            priors = root_priors if state_key == expected_root_key and root_priors is not None else self.ps[state_key]
            action = self._select_action(state_key, priors)
            path.append(SearchPathEdge(state_key, action))
            position, _ = self.game.apply_action(position, position.current_player, action)

    def expand_and_backup(
        self,
        selection: LeafSelection,
        evaluation: tuple[np.ndarray, float] | None = None,
    ) -> None:
        if selection.terminal_result.is_terminal:
            leaf_value = selection.terminal_result.value_for(selection.position.current_player)
        else:
            if evaluation is None:
                started = time.perf_counter()
                evaluation = self.evaluator.evaluate(selection.position)
                self.net_time += time.perf_counter() - started
                self.evaluator_calls += 1
                self.evaluated_positions += 1
                self.inference_batch_sizes.append(1)
            policy, leaf_value = evaluation
            self._expand(selection.position, policy)
            self._validate_value(leaf_value)

        value = float(leaf_value)
        for edge in reversed(selection.path):
            value = -value
            edge_key = (edge.state_key, edge.action)
            visits = self.nsa.get(edge_key, 0)
            old_value = self.qsa.get(edge_key, 0.0)
            self.qsa[edge_key] = (visits * old_value + value) / (visits + 1)
            self.nsa[edge_key] = visits + 1
            self.ns[edge.state_key] += 1

    def record_batch_evaluation(
        self,
        batch_size: int,
        elapsed_seconds: float,
        evaluated_positions: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Inference batch size must be positive")
        if evaluated_positions < 1:
            raise ValueError("Evaluated positions must be positive")
        self.evaluator_calls += 1
        self.evaluated_positions += evaluated_positions
        self.inference_batch_sizes.append(batch_size)
        self.net_time += elapsed_seconds

    def telemetry(self, root: PenteBoard) -> SearchTelemetry:
        root_key = self.game.to_string(root)
        valid = self.valids.get(root_key)
        counts = np.array(
            [self.nsa.get((root_key, action), 0) for action in range(self.game.get_action_size())],
            dtype=np.float64,
        )
        visited = counts > 0
        total = float(counts.sum())
        if total > 0:
            probabilities = counts[visited] / total
            entropy = float(-np.sum(probabilities * np.log(probabilities)))
            maximum_share = float(np.max(probabilities))
        else:
            entropy = 0.0
            maximum_share = 0.0

        legal_actions = int(np.count_nonzero(valid)) if valid is not None else 0
        edge_visits = int(total)
        collapse_eligible = (
            legal_actions > 1
            and edge_visits >= SEARCH_COLLAPSE_MIN_ROOT_VISITS
        )
        search_collapsed = collapse_eligible and int(np.count_nonzero(visited)) <= 1
        mean_batch = (
            float(np.mean(self.inference_batch_sizes))
            if self.inference_batch_sizes
            else 0.0
        )
        return SearchTelemetry(
            simulations=self.simulations,
            evaluator_calls=self.evaluator_calls,
            evaluated_positions=self.evaluated_positions,
            invalid_policy_fallbacks=self.invalid_policy_fallbacks,
            zero_visit_fallbacks=self.zero_visit_fallbacks,
            max_depth=self.max_depth,
            root_legal_actions=legal_actions,
            root_edge_visits=edge_visits,
            root_children_visited=int(np.count_nonzero(visited)),
            root_visit_entropy=entropy,
            root_max_visit_share=maximum_share,
            root_collapse_eligible=collapse_eligible,
            root_search_collapsed=search_collapsed,
            mean_inference_batch_size=mean_batch,
        )

    def _select_action(self, state_key: bytes, priors: np.ndarray) -> int:
        valid = self.valids[state_key].astype(bool)
        action_size = self.game.get_action_size()
        q_values = np.zeros(action_size, dtype=np.float64)
        visit_counts = np.zeros(action_size, dtype=np.float64)

        for action in np.flatnonzero(valid):
            edge_key = (state_key, int(action))
            if edge_key in self.nsa:
                q_values[action] = self.qsa[edge_key]
                visit_counts[action] = self.nsa[edge_key]

        exploration = (
            self.args.c_puct
            * priors
            * math.sqrt(self.ns[state_key] + EPSILON)
            / (1.0 + visit_counts)
        )
        scores = q_values + exploration
        scores[~valid] = -np.inf
        return int(np.argmax(scores))

    def _expand(self, position: PenteBoard, policy: np.ndarray) -> None:
        state_key = self.game.to_string(position)
        if state_key in self.ps:
            return

        policy_array = np.asarray(policy, dtype=np.float64).reshape(-1)
        expected_shape = (self.game.get_action_size(),)
        if policy_array.shape != expected_shape:
            raise ValueError(f"Evaluator policy must have shape {expected_shape}, found {policy_array.shape}")
        if not np.isfinite(policy_array).all() or np.any(policy_array < 0):
            raise ValueError("Evaluator policy must contain finite non-negative probabilities")

        valid = self.game.get_valid_moves(position, position.current_player).astype(bool)
        masked = np.where(valid, policy_array, 0.0)
        policy_sum = float(masked.sum())
        if policy_sum <= 0:
            self.invalid_policy_fallbacks += 1
            masked = valid.astype(np.float64)
            policy_sum = float(masked.sum())
        if policy_sum <= 0:
            raise ValueError("Non-terminal position has no legal action")

        self.ps[state_key] = masked / policy_sum
        self.valids[state_key] = valid
        self.ns[state_key] = 0

    def _root_priors(self, root_key: bytes, add_noise: bool) -> np.ndarray | None:
        if root_key not in self.ps:
            return None
        priors = np.array(self.ps[root_key], copy=True)
        if not add_noise or self.args.root_noise_epsilon == 0:
            return priors

        legal_actions = np.flatnonzero(self.valids[root_key])
        noise = self.rng.dirichlet(
            np.full(legal_actions.size, self.args.root_dirichlet_alpha, dtype=np.float64)
        )
        epsilon = self.args.root_noise_epsilon
        priors[legal_actions] = (1.0 - epsilon) * priors[legal_actions] + epsilon * noise
        return priors

    def _maximum_visit_policy(self, position: PenteBoard, counts: np.ndarray) -> np.ndarray:
        policy = np.zeros(self.game.get_action_size(), dtype=np.float64)
        if float(counts.max()) > 0:
            action = int(np.flatnonzero(counts == counts.max())[0])
        else:
            self.zero_visit_fallbacks += 1
            fallback = self._expanded_prior_or_uniform(position)
            action = int(np.argmax(fallback))
        policy[action] = 1.0
        return policy

    def _expanded_prior_or_uniform(self, position: PenteBoard) -> np.ndarray:
        state_key = self.game.to_string(position)
        if state_key in self.ps:
            return np.array(self.ps[state_key], copy=True)
        valid = self.game.get_valid_moves(position, position.current_player).astype(np.float64)
        return valid / valid.sum()

    @staticmethod
    def _validate_value(value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric) or not -1.0 <= numeric <= 1.0:
            raise ValueError(f"Evaluator value must be finite and in [-1, 1], found {numeric}")
