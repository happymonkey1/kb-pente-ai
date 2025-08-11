import numpy as np
import logging

from src.game.pente.pente_game import PenteGame

logger = logging.getLogger(__name__)

class ProfessionGameLoader:
    def __init__(self, filepath: str, board_size: int = 19, player_count: int = 2):
        self.filepath = filepath
        self.board_size = board_size
        self.player_count = player_count

        assert player_count == 2, "Only 2-player games are supported"
        assert board_size == 19, "Only 19x19 boards are supported"

    def load_games(self):
        """
        Loads a Pente dataset from a file, parses the games, and generates
        training examples. Each example is a tuple containing the board state
        and the final game outcome from the current player's perspective.

        Returns:
            A list of training examples, where each example is a tuple of
            (board_state_array, final_outcome_for_current_player).
        """

        # TODO: generate symmetries

        all_training_examples = []

        logger.info(f"Loading dataset from: {self.filepath}")

        with open(self.filepath, 'r') as f:
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
                    all_training_examples.append((board.board.copy(), pi, value_for_current_player))

                    board, current_player = game.apply_action(board, current_player, action)

        logger.info(f"Successfully loaded and processed {len(all_training_examples)} positions.")
        return all_training_examples

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