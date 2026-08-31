from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import os
import pickle
import tempfile

import numpy as np

from src.artifacts import PROFESSIONAL_DATA_SCHEMA_VERSION
from src.game.game import Game, GameStatus
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.train.training_example import TrainingExample


@dataclass(frozen=True, slots=True)
class ProfessionalDataStats:
    accepted_games: int
    rejected_games: int
    accepted_positions: int
    deduplicated_positions: int
    non_terminal_games: int
    training_games: int
    validation_games: int
    training_positions: int
    validation_positions: int
    deduplicated_training_positions: int
    deduplicated_validation_positions: int
    rejection_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProcessedProfessionalData:
    schema_version: int
    board_size: int
    ruleset: str
    validation_fraction: float
    examples: list[TrainingExample]
    validation_examples: list[TrainingExample]
    stats: ProfessionalDataStats


class ProfessionGameLoader:
    COLUMNS = "ABCDEFGHJKLMNOPQRST"

    def __init__(
        self,
        raw_filepath: str,
        processed_filepath: str,
        board_size: int = 19,
        player_count: int = 2,
        force: bool = False,
        ruleset: PenteRuleset = PenteRuleset.STANDARD,
        validation_fraction: float = 0.1,
    ) -> None:
        if player_count != 2:
            raise ValueError("Professional loader supports exactly two players")
        self.raw_filepath = raw_filepath
        self.processed_filepath = processed_filepath
        self.board_size = board_size
        self.player_count = player_count
        self.force = force
        self.ruleset = ruleset
        if not 0 <= validation_fraction < 1:
            raise ValueError("Validation fraction must be in [0, 1)")
        self.validation_fraction = validation_fraction
        self.last_stats: ProfessionalDataStats | None = None
        self.validation_examples: list[TrainingExample] = []

    def load_games(self) -> list[TrainingExample]:
        if self.force or not os.path.exists(self.processed_filepath):
            processed = self.process_games()
            self._save_processed(processed)
        else:
            processed = self._load_processed()
        self.last_stats = processed.stats
        self.validation_examples = processed.validation_examples
        return processed.examples

    def process_games(self) -> ProcessedProfessionalData:
        if not os.path.exists(self.raw_filepath):
            raise ValueError(f"Raw dataset file does not exist: {self.raw_filepath}")

        training_groups: dict[bytes, list[TrainingExample]] = defaultdict(list)
        validation_groups: dict[bytes, list[TrainingExample]] = defaultdict(list)
        accepted_games = 0
        rejected_games = 0
        accepted_positions = 0
        non_terminal_games = 0
        training_games = 0
        validation_games = 0
        training_positions = 0
        validation_positions = 0
        rejection_reasons: Counter[str] = Counter()

        with open(self.raw_filepath, "r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                fields = [field for field in raw_line.strip().split(";") if field]
                if len(fields) < 2:
                    rejected_games += 1
                    rejection_reasons["malformed_record"] += 1
                    continue

                result_text = fields[-1]
                winner = self._parse_winner(result_text)
                if winner is None:
                    rejected_games += 1
                    rejection_reasons["unknown_result"] += 1
                    continue

                game = PenteGame(self.board_size, self.player_count, self.ruleset)
                position = game.init_board()
                game_examples: list[TrainingExample] = []
                rejection: str | None = None

                move_fields = fields[:-1]
                for move_index, move_text in enumerate(move_fields):
                    try:
                        row, column = self.parse_move(move_text)
                    except (TypeError, ValueError):
                        rejection = "invalid_notation"
                        break

                    action = row * self.board_size + column
                    if not game.is_valid_move(position, position.current_player, action):
                        rejection = "illegal_move"
                        break

                    policy = np.zeros(game.get_action_size(), dtype=np.float32)
                    policy[action] = 1.0
                    game_examples.append(
                        TrainingExample(
                            position=position,
                            policy=policy,
                            value=float(winner * position.current_player),
                        )
                    )
                    position, _ = game.apply_action(position, position.current_player, action)
                    if game.check_game_end(position).is_terminal and move_index < len(move_fields) - 1:
                        rejection = "move_after_terminal"
                        break

                if rejection is not None:
                    rejected_games += 1
                    rejection_reasons[rejection] += 1
                    continue

                terminal = game.check_game_end(position)
                if terminal.status is GameStatus.WIN and terminal.winner != winner:
                    rejected_games += 1
                    rejection_reasons["winner_mismatch"] += 1
                    continue
                if terminal.status is GameStatus.DRAW:
                    rejected_games += 1
                    rejection_reasons["winner_mismatch"] += 1
                    continue
                if not terminal.is_terminal:
                    non_terminal_games += 1

                accepted_games += 1
                accepted_positions += len(game_examples)
                is_validation = self._is_validation_game(raw_line.strip())
                selected_groups = validation_groups if is_validation else training_groups
                if is_validation:
                    validation_games += 1
                    validation_positions += len(game_examples)
                else:
                    training_games += 1
                    training_positions += len(game_examples)
                for example in game_examples:
                    selected_groups[example.position.state_key()].append(example)

        examples = [self._aggregate(group) for group in training_groups.values()]
        validation_examples = [self._aggregate(group) for group in validation_groups.values()]
        stats = ProfessionalDataStats(
            accepted_games=accepted_games,
            rejected_games=rejected_games,
            accepted_positions=accepted_positions,
            deduplicated_positions=len(examples) + len(validation_examples),
            non_terminal_games=non_terminal_games,
            training_games=training_games,
            validation_games=validation_games,
            training_positions=training_positions,
            validation_positions=validation_positions,
            deduplicated_training_positions=len(examples),
            deduplicated_validation_positions=len(validation_examples),
            rejection_reasons=dict(sorted(rejection_reasons.items())),
        )
        return ProcessedProfessionalData(
            schema_version=PROFESSIONAL_DATA_SCHEMA_VERSION,
            board_size=self.board_size,
            ruleset=self.ruleset.value,
            validation_fraction=self.validation_fraction,
            examples=examples,
            validation_examples=validation_examples,
            stats=stats,
        )

    def parse_move(self, move_text: str) -> tuple[int, int]:
        if not 2 <= len(move_text) <= 3:
            raise ValueError(f"Invalid move format: {move_text}")
        column_text = move_text[0].upper()
        if column_text not in self.COLUMNS[: self.board_size]:
            raise ValueError(f"Invalid column in move: {move_text}")
        try:
            row = int(move_text[1:]) - 1
        except ValueError as error:
            raise ValueError(f"Invalid row in move: {move_text}") from error
        column = self.COLUMNS.index(column_text)
        if not 0 <= row < self.board_size:
            raise ValueError(f"Move is outside the board: {move_text}")
        return row, column

    @staticmethod
    def _parse_winner(result_text: str) -> int | None:
        if result_text == "1-0":
            return Game.PLAYER_ONE
        if result_text == "0-1":
            return Game.PLAYER_TWO
        return None

    def _is_validation_game(self, record: str) -> bool:
        if self.validation_fraction == 0:
            return False
        digest = hashlib.sha256(record.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        return bucket < self.validation_fraction

    @staticmethod
    def _aggregate(group: list[TrainingExample]) -> TrainingExample:
        position = group[0].position
        policy = np.mean(np.stack([example.policy for example in group], axis=0), axis=0)
        policy /= policy.sum()
        value = float(np.mean([example.value for example in group]))
        return TrainingExample(position, policy, value)

    def _save_processed(self, processed: ProcessedProfessionalData) -> None:
        directory = os.path.dirname(os.path.abspath(self.processed_filepath))
        os.makedirs(directory, exist_ok=True)
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as stream:
                temporary_path = stream.name
                pickle.dump(processed, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, self.processed_filepath)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _load_processed(self) -> ProcessedProfessionalData:
        with open(self.processed_filepath, "rb") as stream:
            processed = pickle.load(stream)
        if not isinstance(processed, ProcessedProfessionalData):
            raise ValueError(
                "Processed professional data is a legacy format; rebuild it with force processing enabled"
            )
        if processed.schema_version != PROFESSIONAL_DATA_SCHEMA_VERSION:
            raise ValueError(
                f"Professional data schema {processed.schema_version} is incompatible with "
                f"schema {PROFESSIONAL_DATA_SCHEMA_VERSION}"
            )
        if (
            processed.board_size != self.board_size
            or processed.ruleset != self.ruleset.value
            or processed.validation_fraction != self.validation_fraction
        ):
            raise ValueError(
                "Processed professional data does not match board size, ruleset, and validation split"
            )
        return processed
