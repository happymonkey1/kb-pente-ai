import os.path
import pickle

import numpy as np
import logging

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame

logger = logging.getLogger(__name__)

class ProfessionGameLoader:
    def __init__(self, raw_filepath: str, processed_filepath: str, board_size: int = 19, player_count: int = 2, force: bool = False):
        self.raw_filepath = raw_filepath
        self.processed_filepath = processed_filepath
        self.board_size = board_size
        self.player_count = player_count
        self.force = force

        assert player_count == 2, "Only 2-player games are supported"
        assert board_size == 19, "Only 19x19 boards are supported"

    def load_games(self):
        processed_exists = os.path.exists(self.processed_filepath)
        if self.force or not processed_exists:
            return self.__process_games()
        elif processed_exists:
            return self.__load_processed_list(self.processed_filepath)
        else:
            raise ValueError(f"Failed to load processed dataset file: {self.processed_filepath}")

    def __process_games(self):
        """
        Loads a Pente dataset from a file, parses the games, and generates
        training examples. Each example is a tuple containing the board state
        and the final game outcome from the current player's perspective.

        Returns:
            A list of training examples, where each example is a tuple of
            (board_state_array, final_outcome_for_current_player).
        """
        all_training_examples = []

        logger.info(f"Loading dataset from: {self.raw_filepath}")
        if not os.path.exists(self.raw_filepath):
            raise ValueError(f"Raw dataset file '{self.raw_filepath}' does not exist")

        with open(self.raw_filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(';')[:-1]
                move_sequence = parts[:-1]
                result_str = parts[-1]

                if result_str == "1-0":
                    final_winner = 1.0
                elif result_str == "0-1":
                    final_winner = -1.0
                else:
                    print(f"Warning: Skipping line {line_num} due to unrecognized result '{result_str}'")
                    continue

                game = PenteGame(self.board_size, self.player_count)
                board = game.init_board()
                current_player = 1

                for move_str in move_sequence:
                    try:
                        row, col = self.parse_move(move_str)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid move '{move_str}' in line {line_num}. Error: {e}")
                        continue

                    if current_player == 1:
                        value_for_current_player = final_winner
                    else:
                        value_for_current_player = -final_winner

                    action = row * self.board_size + col

                    pi = np.zeros((self.board_size * self.board_size), dtype=float)
                    pi[action] = 1

                    board, next_player = game.apply_action(board, current_player, action)

                    canonical_board = game.get_canonical_form(board, current_player)

                    symmetries = game.get_symmetries(canonical_board, pi)
                    for sym_board, sym_pi in symmetries:
                        all_training_examples.append((sym_board.copy(), sym_pi, value_for_current_player))

                    current_player = next_player


        logger.info(f"Successfully loaded and processed {len(all_training_examples)} positions.")

        deduplicated = {
            PenteBoard(board=example[0], captures=np.zeros(self.player_count, dtype=np.int8)).to_string(): example
            for example in reversed(all_training_examples)
        }

        deduplicated_examples = list(deduplicated.values())

        logger.info(f"Deduplication leaves {len(deduplicated_examples)}")

        if self.force or not os.path.exists(self.processed_filepath):
            ProfessionGameLoader.__save_processed_list(deduplicated_examples, self.processed_filepath)

        logger.info("Finished saving processed dataset")

        return deduplicated_examples

    @staticmethod
    def __save_processed_list(data, filepath: str):
        logger.info(f"Saving processed dataset to: {filepath}")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except IOError as e:
            logger.error(f"Failed to save pente processed dataset file '{filepath}': {e}")

    @staticmethod
    def __load_processed_list(filepath: str):
        logger.info(f"Loading processed dataset from: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except IOError as e:
            logger.error(f"Failed to load pente processed dataset from file '{filepath}': {e}")


    def parse_move(self, move_str: str):
        """
       Converts a single move in algebraic notation (e.g., 'K10') to 0-indexed
       (row, col) coordinates. Assumes a standard Pente board layout where
       the letter 'I' is skipped.

       Args:
           move_str: The move string, like "K10", "A1", etc.
           board_size: The size of the board (e.g., 19 for a standard board).

       Returns:
           A tuple (row, col) of 0-indexed coordinates.
       """
        if not 2 <= len(move_str) <= 3:
            raise ValueError(f"Invalid move format: {move_str}")

        # Standard Pente column letters, skipping 'I'
        COLS = "ABCDEFGHJKLMNOPQRST"

        col_char = move_str[0].upper()
        row_str = move_str[1:]

        if col_char not in COLS:
            raise ValueError(f"Invalid column character '{col_char}' in move '{move_str}'")

        col = COLS.index(col_char)
        row = int(row_str) - 1

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(f"Move {move_str} with coords ({row}, {col}) is out of bounds for size {self.board_size}")

        return row, col