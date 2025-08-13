from src.game.game import Game
from src.game.pente.pente_board import PenteBoard
import numpy as np
from numba import njit
from scipy.signal import convolve2d

class PenteGame(Game):
    CAPTURES_TO_WIN: int = 5

    def __init__(self, board_size: int = 19, player_count: int = 2):
        super().__init__()
        self.board_size = board_size
        self.player_count = player_count

    def init_board(self):
        return PenteBoard.new_board(self.board_size, self.player_count)

    def get_board_size(self):
        return self.board_size

    def get_player_count(self):
        return self.player_count

    def get_action_size(self):
        #return self.board_size * self.board_size + 1
        return self.board_size * self.board_size

    def apply_action(self, pente_board: 'PenteBoard', player: int, action: int) -> tuple['PenteBoard', int] :
        r, c = action // self.board_size, action % self.board_size
        if pente_board.board[r, c] != 0:
            raise ValueError(f"Invalid action: {action} is not empty")

        new_board = pente_board.apply_move(player, self.__get_opponent(player), (r, c))
        return new_board, self.__get_opponent(player)

    def get_next_player(self, player: int):
        assert self.player_count == 2, "Only 2 players are supported for get_next_player logic"

        return Game.PLAYER_TWO if player == Game.PLAYER_ONE else Game.PLAYER_ONE

    def __get_opponent(self, player: int) -> int:
        assert self.player_count == 2, "Only 2 players are support for __get_opponent logic"

        return Game.PLAYER_ONE if player == Game.PLAYER_TWO else Game.PLAYER_TWO

    def get_valid_moves(self, pente_board: 'PenteBoard', player: int) -> np.ndarray:
        legal_moves = pente_board.get_legal_moves()
        valids = np.zeros(self.get_action_size(), dtype=np.int8)

        if not legal_moves:
            return valids

        indices = [self.board_size * x + y for x, y in legal_moves]
        valids[indices] = 1

        return valids

    def is_valid_move(self, pente_board: 'PenteBoard', player: int, action: int) -> bool:
        r, c = action // self.board_size, action % self.board_size
        return pente_board.board[r, c] == 0

    def check_game_end(self, pente_board: 'PenteBoard', player: int) -> (bool, int):
        """
        Check if the board is in a terminal state.
        :param pente_board:
        :param player:
        :return: Tuple of (is_terminal, winner) where winner is 1 if player 1 won and -1 if player 1 lost
        """
        assert self.player_count == 2, "check_game_end logic only supports 2 players"
        assert pente_board.board.size == self.board_size * self.board_size, "Board size not correct"
        players = [PenteGame.PLAYER_ONE, PenteGame.PLAYER_TWO]
        for p in players:
            if pente_board.get_capture_count(p) >= PenteGame.CAPTURES_TO_WIN:
                return True, 1 if p == PenteGame.PLAYER_ONE else -1

        if not np.any(pente_board.board == 0):
            return True, 1 if player != Game.PLAYER_ONE else -1

        for p in players:
            if _has_five_in_a_row_fast(pente_board.board, p):
                return True, 1 if p == PenteGame.PLAYER_ONE else -1

        return False, 0

    def get_symmetries(self, board: 'PenteBoard', pi) -> list[np.ndarray]:
        #assert(len(pi) == self.board_size**2+1)  # 1 for pass
        pi_board = np.reshape(pi, (self.board_size, self.board_size))
        #pi_board = np.copy(pi)
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                new_board = np.rot90(board.board, i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_board = np.fliplr(new_board)
                    new_pi = np.fliplr(new_pi)
                l += [(new_board, new_pi.ravel())]

        return l

    def get_canonical_form(self, board: 'PenteBoard', player: int) -> 'PenteBoard':
        return board.get_canonical_form(player, self.player_count)

    def to_string(self, board: 'PenteBoard') -> str:
        return board.to_string()

# NOTE: can't use numba JIT with scipy kernel
def _has_five_in_a_row_fast(board: np.ndarray, player: int) -> bool:
    mask = (board == player)
    kernels = [
        np.ones((1, 5), dtype=np.uint8), np.ones((5, 1), dtype=np.uint8),
        np.eye(5, dtype=np.uint8), np.fliplr(np.eye(5, dtype=np.uint8))
    ]
    for kernel in kernels:
        if np.any(convolve2d(mask, kernel, mode='valid') == 5):
            return True
    return False

