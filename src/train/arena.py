import time

from src.train.player import Player
from src.game.game import Game

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ArenaStats:
    p1_wins: int
    p2_wins: int
    draws: int

    avg_moves: int

@dataclass
class GameOverStats:
    moves: int
    winner: int

class Arena:
    def __init__(self, player1: 'Player', player2: 'Player', game: 'Game', debug: bool = False, display = None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.debug = debug
        self.display = display

        assert self.game.get_player_count() == 2, "Arena only supports 2-player games"

    def play_game(self):
        if self.debug:
            logger.info("Starting new arena game")

        self.player1.reset()
        self.player2.reset()

        current_player = Game.PLAYER_ONE
        board = self.game.init_board()

        moves = 0
        game_start_time = time.time()
        while True:
            moves += 1

            action = self.__get_player_move(board, current_player)
            board, next_player = self.game.apply_action(board, current_player, action)

            if self.display:
                self.display(board.board)

            terminal, winner = self.game.check_game_end(board, current_player)
            if terminal:
                if self.debug:
                    game_time = time.time() - game_start_time
                    logger.info(f"Game over after {game_time:.2f}s: {winner} won after {moves} moves")
                    logger.info(f"Board dump:\n {board}")
                return GameOverStats(
                    moves=moves,
                    winner=winner,
                )

            current_player = next_player

    def __get_player_move(self, board, player):
        if player == Game.PLAYER_ONE:
            return self.player1.play(self.game, board, player, self.debug)
        elif player == Game.PLAYER_TWO:
            return self.player2.play(self.game, board, player, self.debug)
        else:
            raise ValueError(f"Invalid player: {player}")

    def play_games(self, num_games):
        logger.info(f"Arena playing {num_games} games")
        p1_wins, p2_wins, draws = 0, 0, 0
        p1_starts = num_games // 2
        total_moves = 0

        for i in range(p1_starts):
            game_over_stats = self.play_game()
            moves, winner = game_over_stats.moves, game_over_stats.winner
            total_moves += moves

            if winner == Game.PLAYER_ONE:
                p1_wins += 1
            elif winner == Game.PLAYER_TWO:
                p2_wins += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        if num_games % 2 != 0:
            p1_starts += 1

        for _ in range(p1_starts):
            game_over_stats = self.play_game()
            moves, winner = game_over_stats.moves, game_over_stats.winner
            total_moves += moves

            if winner == Game.PLAYER_TWO:
                p1_wins += 1
            elif winner == Game.PLAYER_ONE:
                p2_wins += 1
            else:
                draws += 1

        avg_moves = total_moves // num_games
        return ArenaStats(p1_wins, p2_wins, draws, avg_moves)