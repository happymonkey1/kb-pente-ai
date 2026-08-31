from __future__ import annotations

from dataclasses import dataclass

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
    mcts: MCTS
    search_session: SearchSession
    pending: list[tuple[PenteBoard, np.ndarray]]
    actions: list[int]
    root_telemetry: list[SearchTelemetry]


class SelfPlayGenerator:
    def __init__(
        self,
        game: PenteGame,
        evaluator: PolicyValueEvaluator,
        mcts_args: MCTSArgs,
        temp_threshold: int,
        rng: np.random.Generator,
        deduplicate_evaluations: bool = True,
    ) -> None:
        self.game = game
        self.evaluator = evaluator
        self.mcts_args = mcts_args
        self.temp_threshold = temp_threshold
        self.rng = rng
        self.deduplicate_evaluations = deduplicate_evaluations

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
        launched_games = min(game_count, active_limit)
        active = [self._new_active_game() for _ in range(launched_games)]
        completed: list[PlayedGame] = []
        batch_accumulators: dict[int, BatchedSearchAccumulator] = {}

        while active:
            remaining: list[ActiveSelfPlayGame] = []
            for game in active:
                if not game.search_session.is_complete:
                    remaining.append(game)
                    continue

                policy = game.search_session.policy()
                game.pending.append((game.position, policy))
                game.root_telemetry.append(game.mcts.telemetry(game.position))
                action = int(self.rng.choice(len(policy), p=policy))
                game.actions.append(action)
                game.position, _ = self.game.apply_action(
                    game.position,
                    game.position.current_player,
                    action,
                )
                result = self.game.check_game_end(game.position)
                if result.is_terminal:
                    completed.append(
                        PlayedGame(
                            examples=finalize_training_examples(game.pending, result),
                            actions=tuple(game.actions),
                            winner=result.winner,
                            win_reason=result.reason,
                            root_telemetry=tuple(game.root_telemetry),
                        )
                    )
                else:
                    game.search_session = self._new_search_session(game)
                    remaining.append(game)
            while launched_games < game_count and len(remaining) < active_limit:
                remaining.append(self._new_active_game())
                launched_games += 1
            active = remaining

            if active:
                root_count = len(active)
                accumulator = batch_accumulators.get(root_count)
                if accumulator is None:
                    accumulator = BatchedSearchAccumulator(root_count)
                    batch_accumulators[root_count] = accumulator
                evaluate_search_wave(
                    [game.search_session for game in active],
                    accumulator,
                    self.deduplicate_evaluations,
                )

        return completed, [
            accumulator.telemetry()
            for accumulator in batch_accumulators.values()
        ]

    def _new_active_game(self) -> ActiveSelfPlayGame:
        position = self.game.init_board()
        search = MCTS(
            self.game,
            self.evaluator,
            self.mcts_args,
            np.random.default_rng(int(self.rng.integers(0, 2**63))),
        )
        active = ActiveSelfPlayGame(
            position=position,
            mcts=search,
            search_session=SearchSession(
                search,
                position,
                self._temperature(position),
                add_root_noise=True,
            ),
            pending=[],
            actions=[],
            root_telemetry=[],
        )
        return active

    def _new_search_session(self, active: ActiveSelfPlayGame) -> SearchSession:
        return SearchSession(
            active.mcts,
            active.position,
            self._temperature(active.position),
            add_root_noise=True,
        )

    def _temperature(self, position: PenteBoard) -> float:
        assert position.ply is not None
        return 1.0 if position.ply < self.temp_threshold else 0.0


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
