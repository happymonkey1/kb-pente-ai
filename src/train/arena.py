import time
from collections.abc import Callable
from dataclasses import dataclass
import logging

import numpy as np

from src.game.game import Game
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.train.nnet_player import NNetPlayer
from src.train.player import Player

logger = logging.getLogger(__name__)


@dataclass
class ArenaStats:
    p1_wins: int
    p2_wins: int
    draws: int

    avg_moves: float
    player_one_color_wins: int
    player_two_color_wins: int
    p1_as_player_one_wins: int
    p1_as_player_two_wins: int
    p2_as_player_one_wins: int
    p2_as_player_two_wins: int
    opening_plies: int
    unique_openings: int
    paired_openings: int
    p1_pair_wins: int
    p1_pair_losses: int
    pair_ties: int


@dataclass
class GameOverStats:
    moves: int
    winner: int


class Arena:
    def __init__(
        self,
        player1: Player,
        player2: Player,
        game: PenteGame,
        debug: bool = False,
        display: Callable[[object], None] | None = None,
        opening_plies: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.debug = debug
        self.display = display
        if opening_plies < 0:
            raise ValueError("Opening plies cannot be negative")
        self.opening_plies = opening_plies
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.game.get_player_count() != 2:
            raise ValueError("Arena supports exactly two players")

    def play_game(self, initial_position: PenteBoard | None = None) -> GameOverStats:
        if self.debug:
            logger.info("Starting new arena game")

        self.player1.reset()
        self.player2.reset()

        board = self.game.init_board() if initial_position is None else initial_position
        if self.game.check_game_end(board).is_terminal:
            raise ValueError("Arena initial position cannot be terminal")
        current_player = board.current_player

        assert board.ply is not None
        moves = board.ply
        game_start_time = time.time()
        while True:
            moves += 1

            get_move_start_time = time.time()
            action = self.__get_player_move(board, current_player)
            get_move_time = time.time() - get_move_start_time

            if self.debug:
                logger.info(f"Retrieving move took: {get_move_time}")
                player = self.player1 if current_player == Game.PLAYER_ONE else self.player2
                if isinstance(player, NNetPlayer) and player.mcts is not None:
                    logger.info(
                        "Network predictions took %.3fs (%.2f%%)",
                        player.mcts.net_time,
                        100 * player.mcts.net_time / get_move_time,
                    )

            board, next_player = self.game.apply_action(board, current_player, action)

            if self.display:
                self.display(board.board)

            result = self.game.check_game_end(board)
            if result.is_terminal:
                if self.debug:
                    game_time = time.time() - game_start_time
                    logger.info(f"Game over after {game_time:.2f}s: {result.winner} won after {moves} moves")
                    logger.info(f"Board dump:\n {board}")
                return GameOverStats(
                    moves=moves,
                    winner=0 if result.winner is None else result.winner,
                )

            current_player = next_player

    def __get_player_move(self, board: PenteBoard, player: int) -> int:
        if player == Game.PLAYER_ONE:
            return self.player1.play(self.game, board, player, self.debug)
        elif player == Game.PLAYER_TWO:
            return self.player2.play(self.game, board, player, self.debug)
        else:
            raise ValueError(f"Invalid player: {player}")

    def play_games(self, num_games: int) -> ArenaStats:
        if num_games < 1:
            raise ValueError("Arena requires at least one game")
        logger.info(f"Arena playing {num_games} games")
        p1_wins, p2_wins, draws = 0, 0, 0
        player_one_color_wins, player_two_color_wins = 0, 0
        p1_as_player_one_wins, p1_as_player_two_wins = 0, 0
        p2_as_player_one_wins, p2_as_player_two_wins = 0, 0
        p1_starts = num_games // 2
        total_moves = 0
        first_half_p1_scores: list[float] = []
        p1_pair_wins, p1_pair_losses, pair_ties = 0, 0, 0
        paired_openings = [self._sample_opening() for _ in range(p1_starts)]
        second_half_openings = list(paired_openings)
        if num_games % 2:
            second_half_openings.append(self._sample_opening())
        openings = paired_openings + second_half_openings

        for opening in paired_openings:
            game_over_stats = self.play_game(opening)
            moves, winner = game_over_stats.moves, game_over_stats.winner
            total_moves += moves
            player_one_color_wins += winner == Game.PLAYER_ONE
            player_two_color_wins += winner == Game.PLAYER_TWO

            if winner == Game.PLAYER_ONE:
                p1_wins += 1
                p1_as_player_one_wins += 1
                first_half_p1_scores.append(1.0)
            elif winner == Game.PLAYER_TWO:
                p2_wins += 1
                p2_as_player_two_wins += 1
                first_half_p1_scores.append(0.0)
            else:
                draws += 1
                first_half_p1_scores.append(0.5)

        self.player1, self.player2 = self.player2, self.player1
        try:
            for opening_index, opening in enumerate(second_half_openings):
                game_over_stats = self.play_game(opening)
                moves, winner = game_over_stats.moves, game_over_stats.winner
                total_moves += moves
                player_one_color_wins += winner == Game.PLAYER_ONE
                player_two_color_wins += winner == Game.PLAYER_TWO

                if winner == Game.PLAYER_TWO:
                    p1_wins += 1
                    p1_as_player_two_wins += 1
                    second_half_p1_score = 1.0
                elif winner == Game.PLAYER_ONE:
                    p2_wins += 1
                    p2_as_player_one_wins += 1
                    second_half_p1_score = 0.0
                else:
                    draws += 1
                    second_half_p1_score = 0.5
                if opening_index < len(first_half_p1_scores):
                    pair_score = (
                        first_half_p1_scores[opening_index]
                        + second_half_p1_score
                    )
                    p1_pair_wins += pair_score > 1.0
                    p1_pair_losses += pair_score < 1.0
                    pair_ties += pair_score == 1.0
        finally:
            self.player1, self.player2 = self.player2, self.player1

        return ArenaStats(
            p1_wins=p1_wins,
            p2_wins=p2_wins,
            draws=draws,
            avg_moves=total_moves / num_games,
            player_one_color_wins=player_one_color_wins,
            player_two_color_wins=player_two_color_wins,
            p1_as_player_one_wins=p1_as_player_one_wins,
            p1_as_player_two_wins=p1_as_player_two_wins,
            p2_as_player_one_wins=p2_as_player_one_wins,
            p2_as_player_two_wins=p2_as_player_two_wins,
            opening_plies=self.opening_plies,
            unique_openings=len({opening.state_key() for opening in openings}),
            paired_openings=p1_starts,
            p1_pair_wins=p1_pair_wins,
            p1_pair_losses=p1_pair_losses,
            pair_ties=pair_ties,
        )

    def _sample_opening(self) -> PenteBoard:
        for _ in range(100):
            position = self.game.init_board()
            for _ in range(self.opening_plies):
                legal = np.flatnonzero(
                    self.game.get_valid_moves(position, position.current_player)
                )
                action = int(self.rng.choice(legal))
                position, _ = self.game.apply_action(
                    position,
                    position.current_player,
                    action,
                )
                if self.game.check_game_end(position).is_terminal:
                    break
            if not self.game.check_game_end(position).is_terminal:
                return position
        raise RuntimeError("Could not generate a non-terminal arena opening")
