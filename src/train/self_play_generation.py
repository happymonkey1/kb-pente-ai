from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.game.game import TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.mcts.batched import BatchedSearchTelemetry, run_batched_search
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
    ) -> None:
        self.game = game
        self.evaluator = evaluator
        self.mcts_args = mcts_args
        self.temp_threshold = temp_threshold
        self.rng = rng

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
        batch_telemetry: list[BatchedSearchTelemetry] = []

        while active:
            roots = [game.position for game in active]
            temperatures = []
            for root in roots:
                assert root.ply is not None
                temperatures.append(1.0 if root.ply < self.temp_threshold else 0.0)
            search_result = run_batched_search(
                [game.mcts for game in active],
                roots,
                temperatures,
                add_root_noise=True,
            )
            batch_telemetry.append(search_result.telemetry)

            remaining: list[ActiveSelfPlayGame] = []
            for game, policy in zip(active, search_result.policies):
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
                    remaining.append(game)
            while launched_games < game_count and len(remaining) < active_limit:
                remaining.append(self._new_active_game())
                launched_games += 1
            active = remaining

        return completed, batch_telemetry

    def _new_active_game(self) -> ActiveSelfPlayGame:
        return ActiveSelfPlayGame(
            position=self.game.init_board(),
            mcts=MCTS(
                self.game,
                self.evaluator,
                self.mcts_args,
                np.random.default_rng(int(self.rng.integers(0, 2**63))),
            ),
            pending=[],
            actions=[],
            root_telemetry=[],
        )


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
