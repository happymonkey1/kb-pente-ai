

class Game:
    PLAYER_ONE = 1
    PLAYER_TWO = -1

    def __init__(self):
        pass

    def get_player_count(self) -> int:
        pass

    def init_board(self):
        pass

    def get_board_size(self):
        pass

    def get_action_size(self):
        pass

    def apply_action(self, board, player: int, action):
        pass

    def get_next_player(self, player: int):
        pass

    def get_valid_moves(self, board, player: int):
        pass

    def is_valid_move(self, board, player: int, action: int) -> bool:
        pass

    def check_game_end(self, board, player: int):
        pass

    def get_canonical_form(self, board, player: int):
        """
        returns the canonical form of the board, which should be independent of the player.
        """
        pass

    def get_symmetries(self, board, pi):
        pass

    def to_string(self, board):
        pass