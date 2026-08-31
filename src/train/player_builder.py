"""Compatibility builder for network-backed arena players."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTS, MCTSArgs
from src.mcts.native_backend import NativeSearchBackend
from src.model.model_v1 import PenteNet
from src.train.nnet_player import NNetPlayer
from src.train.native_player import NativeMCTSPlayer
from src.train.player import Player


_NativeBackendFactory = Callable[..., NativeSearchBackend]


def build_player(
    net: PenteNet,
    mcts: MCTS | None = None,
    name: str = "NNetPlayer",
    *,
    search_backend: str = "python",
    game: PenteGame | None = None,
    mcts_args: MCTSArgs | None = None,
    seed: int = 0,
    native_worker_threads: int = 1,
    _native_backend_factory: _NativeBackendFactory | None = None,
) -> Player:
    """Build an existing Python player or the native arena player.

    The first three positional parameters intentionally mirror ``NNetPlayer``.
    Python and direct-network callers therefore receive the same player object
    and behavior; native callers opt in with keyword-only configuration.
    """

    if search_backend not in ("python", "cpp"):
        raise ValueError("Search backend must be 'python' or 'cpp'")
    if mcts is not None:
        if mcts_args is not None:
            raise ValueError("A concrete MCTS and MCTSArgs cannot both be supplied")
        if search_backend == "cpp":
            raise ValueError("Native player cannot receive a Python MCTS instance")
        return NNetPlayer(net, mcts, name)
    if game is None:
        if mcts_args is None:
            return NNetPlayer(net, None, name)
        raise ValueError("MCTSArgs requires a PenteGame")
    if mcts_args is None:
        return NNetPlayer(net, None, name)
    if search_backend == "python":
        return NNetPlayer(
            net,
            MCTS(game, net, mcts_args, np.random.default_rng(seed)),
            name,
        )
    return NativeMCTSPlayer(
        net,
        game,
        mcts_args,
        name,
        seed,
        native_worker_threads,
        _native_backend_factory=_native_backend_factory,
    )


__all__ = ["build_player"]
