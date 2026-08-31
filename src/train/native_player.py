"""Arena player backed by one lazily created native search tree."""

from __future__ import annotations

from collections.abc import Callable
import operator
from typing import SupportsIndex, cast

import numpy as np

from src.game.game import TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTSArgs
from src.mcts.native_backend import NativeSearchBackend
from src.model.model_v1 import PenteNet
from src.train.player import Player


_NativeBackendFactory = Callable[..., NativeSearchBackend]


class NativeMCTSPlayer(Player):
    """Choose deterministic arena moves from a reusable native MCTS tree.

    The native backend is deliberately owned by the player rather than by
    ``Arena``.  After this player advances its own move, the backend is left at
    the opponent's pristine root.  The next call observes the opponent move
    into that root before searching again, retaining the selected subtree.
    """

    def __init__(
        self,
        net: PenteNet,
        game: PenteGame,
        mcts_args: MCTSArgs,
        name: str = "NativeMCTSPlayer",
        seed: int = 0,
        native_worker_threads: int = 1,
        *,
        _native_backend_factory: _NativeBackendFactory | None = None,
    ) -> None:
        if not isinstance(game, PenteGame):
            raise TypeError("NativeMCTSPlayer requires a PenteGame")
        if not isinstance(mcts_args, MCTSArgs):
            raise TypeError("mcts_args must be an MCTSArgs instance")
        self.net = net
        self.game = game
        self.mcts_args = mcts_args
        self.name = name
        self.seed = _nonnegative_int(seed, "seed")
        self.native_worker_threads = _positive_int(
            native_worker_threads,
            "native_worker_threads",
        )
        self._native_backend_factory = (
            NativeSearchBackend
            if _native_backend_factory is None
            else _native_backend_factory
        )

        self._backend: NativeSearchBackend | None = None
        self._slot: int | None = None
        self._tracked_position: PenteBoard | None = None

    def reset(self) -> None:
        """Discard the whole backend, including an incomplete opponent root."""

        self._discard_backend()

    def play(
        self,
        game: PenteGame,
        board: PenteBoard,
        player: int,
        debug: bool = False,
    ) -> int:
        del debug
        if not isinstance(board, PenteBoard):
            self._discard_backend()
            raise TypeError("NativeMCTSPlayer requires a PenteBoard")
        if player != board.current_player:
            self._discard_backend()
            raise ValueError(f"Expected Player {board.current_player}, received Player {player}")

        try:
            self.net.eval()
            if self._backend is None:
                return self._play_first(game, board, player)
            return self._play_after_opponent(game, board, player)
        except Exception:
            self._discard_backend()
            raise

    def _play_first(self, game: PenteGame, board: PenteBoard, player: int) -> int:
        self._validate_game(game)
        if game.check_game_end(board).is_terminal:
            raise ValueError("NativeMCTSPlayer cannot search a terminal position")

        backend = self._native_backend_factory(
            game,
            self.net,
            self.mcts_args,
            max_active_games=1,
            worker_threads=self.native_worker_threads,
            seed=self.seed,
        )
        self._backend = backend
        slot = _nonnegative_int(
            backend.add_root(
                board,
                temperature=0.0,
                add_root_noise=False,
            ),
            "native slot",
        )
        self._slot = slot
        self._tracked_position = board
        self._compare_terminal(
            backend.root_terminal(slot),
            game.check_game_end(board),
        )

        return self._search_and_advance(board, player)

    def _play_after_opponent(
        self,
        game: PenteGame,
        board: PenteBoard,
        player: int,
    ) -> int:
        self._validate_game(game)
        tracked = self._tracked_position
        backend = self._backend
        slot = self._slot
        if tracked is None or backend is None or slot is None:
            raise RuntimeError("NativeMCTSPlayer has incomplete backend state")

        if board.last_action is None:
            raise ValueError("NativeMCTSPlayer requires the opponent last_action")
        opponent = tracked.current_player
        if opponent != game.get_next_player(player):
            raise ValueError("NativeMCTSPlayer tracked an unexpected player to move")

        action = _nonnegative_int(board.last_action, "opponent last_action")
        try:
            expected, _ = game.apply_action(tracked, opponent, action)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "NativeMCTSPlayer supplied position is not the exact successor"
            ) from error
        if not _same_position(expected, board):
            raise ValueError("NativeMCTSPlayer supplied position is not the exact successor")

        python_result = game.check_game_end(expected)
        backend.observe_action(
            slot,
            action,
            temperature=0.0,
            add_root_noise=False,
        )
        self._compare_terminal(backend.root_terminal(slot), python_result)
        self._tracked_position = expected
        if python_result.is_terminal:
            raise ValueError("NativeMCTSPlayer cannot search a terminal position")

        return self._search_and_advance(expected, player)

    def _search_and_advance(self, position: PenteBoard, player: int) -> int:
        backend = self._backend
        slot = self._slot
        if backend is None or slot is None:
            raise RuntimeError("NativeMCTSPlayer has no native backend")

        while not backend.slot_complete(slot):
            backend.evaluate_wave()

        policy = np.asarray(backend.root_policy(slot))
        expected_shape = (self.game.get_action_size(),)
        if policy.shape != expected_shape:
            raise ValueError(
                "Native root policy must have shape "
                f"{expected_shape}, found {policy.shape}"
            )
        legal = self.game.get_valid_moves(position, player).astype(bool)
        if not np.any(legal):
            raise ValueError("NativeMCTSPlayer found no legal action")
        if not np.isfinite(policy[legal]).all():
            raise ValueError("Native root policy must be finite on legal actions")
        masked = np.where(legal, policy, -np.inf)
        action = int(np.argmax(masked))

        next_position, _ = self.game.apply_action(position, player, action)
        backend.advance_root(
            slot,
            action,
            temperature=0.0,
            add_root_noise=False,
        )
        python_result = self.game.check_game_end(next_position)
        self._compare_terminal(backend.root_terminal(slot), python_result)
        self._tracked_position = next_position
        return action

    def _validate_game(self, game: PenteGame) -> None:
        if game is not self.game:
            raise ValueError("NativeMCTSPlayer received a different game instance")

    def _compare_terminal(
        self,
        native_result: TerminalResult,
        python_result: TerminalResult,
    ) -> None:
        if native_result != python_result:
            raise RuntimeError(
                "Native/Python terminal mismatch: "
                f"native={native_result!r}, python={python_result!r}"
            )

    def _discard_backend(self) -> None:
        self._backend = None
        self._slot = None
        self._tracked_position = None


def _same_position(expected: PenteBoard, supplied: PenteBoard) -> bool:
    """Compare every immutable field, including ``last_action``."""

    return (
        expected.current_player == supplied.current_player
        and expected.ply == supplied.ply
        and expected.last_action == supplied.last_action
        and np.array_equal(expected.board, supplied.board)
        and np.array_equal(expected.captures, supplied.captures)
    )


def _positive_int(value: object, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = int(operator.index(cast(SupportsIndex, value)))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative")
    return parsed


__all__ = ["NativeMCTSPlayer"]
