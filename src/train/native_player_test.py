from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast
import unittest

import numpy as np

from src.game.game import TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.mcts.native_backend import NativeWave
from src.train.native_player import NativeMCTSPlayer
from src.train.nnet_player import NNetPlayer
from src.train.player_builder import build_player


class _FakeNet:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> _FakeNet:
        self.eval_calls += 1
        return self


@dataclass
class _FakeSlot:
    position: PenteBoard
    simulations: int = 0


class _FakeNativeBackend:
    def __init__(
        self,
        game: PenteGame,
        _evaluator: object,
        args: MCTSArgs,
        *,
        max_active_games: int,
        worker_threads: int,
        seed: int,
        fail_wave: bool = False,
        fail_observe: bool = False,
        mismatch_terminal: bool = False,
    ) -> None:
        self.game = game
        self.args = args
        self.capacity = max_active_games
        self.thread_count = worker_threads
        self.seed = seed
        self.fail_wave = fail_wave
        self.fail_observe = fail_observe
        self.mismatch_terminal = mismatch_terminal
        self.slots: dict[int, _FakeSlot] = {}
        self.add_calls: list[dict[str, object]] = []
        self.advance_calls: list[dict[str, object]] = []
        self.observe_calls: list[dict[str, object]] = []
        self.remove_calls: list[int] = []
        self.wave_calls = 0
        self.terminal_calls = 0

    def add_root(
        self,
        position: PenteBoard,
        *,
        temperature: float,
        add_root_noise: bool,
    ) -> int:
        if len(self.slots) >= self.capacity:
            raise AssertionError("fake capacity exhausted")
        self.add_calls.append(
            {
                "temperature": temperature,
                "add_root_noise": add_root_noise,
            }
        )
        self.slots[0] = _FakeSlot(position)
        return 0

    def slot_complete(self, slot: int) -> bool:
        return self.slots[slot].simulations >= self.args.num_simulations

    def evaluate_wave(self) -> NativeWave:
        self.wave_calls += 1
        if self.fail_wave:
            self.fail_wave = False
            raise RuntimeError("fake wave failed")
        for state in self.slots.values():
            state.simulations += 1
        return NativeWave(
            token=self.wave_calls,
            size=1,
            raw_size=1,
            host_to_device_seconds=0.0,
            model_inference_seconds=0.0,
            device_to_host_seconds=0.0,
            inference_wait_seconds=0.0,
        )

    def root_policy(self, slot: int) -> np.ndarray:
        state = self.slots[slot]
        legal = self.game.get_valid_moves(state.position, state.position.current_player)
        policy = np.zeros(self.game.get_action_size(), dtype=np.float32)
        policy[int(np.flatnonzero(legal)[0])] = 1.0
        return policy

    def advance_root(
        self,
        slot: int,
        action: int,
        *,
        temperature: float,
        add_root_noise: bool,
    ) -> dict[str, object]:
        state = self.slots[slot]
        state.position, _ = self.game.apply_action(
            state.position,
            state.position.current_player,
            action,
        )
        state.simulations = 0
        self.advance_calls.append(
            {
                "action": action,
                "temperature": temperature,
                "add_root_noise": add_root_noise,
            }
        )
        return {}

    def observe_action(
        self,
        slot: int,
        action: int,
        *,
        temperature: float,
        add_root_noise: bool,
    ) -> dict[str, object]:
        if self.fail_observe:
            raise RuntimeError("fake observe failed")
        state = self.slots[slot]
        state.position, _ = self.game.apply_action(
            state.position,
            state.position.current_player,
            action,
        )
        state.simulations = 0
        self.observe_calls.append(
            {
                "action": action,
                "temperature": temperature,
                "add_root_noise": add_root_noise,
            }
        )
        return {}

    def root_terminal(self, slot: int) -> TerminalResult:
        self.terminal_calls += 1
        result = self.game.check_game_end(self.slots[slot].position)
        if self.mismatch_terminal and self.terminal_calls >= 2:
            return TerminalResult.draw()
        return result


class NativeMCTSPlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        self.args = MCTSArgs(num_simulations=1, root_noise_epsilon=0.0)
        self.created: list[_FakeNativeBackend] = []

    def make_factory(
        self,
        *,
        fail_wave: bool = False,
        fail_observe: bool = False,
        mismatch_terminal: bool = False,
    ) -> Callable[..., _FakeNativeBackend]:
        def factory(
            game: PenteGame,
            evaluator: object,
            args: MCTSArgs,
            **kwargs: object,
        ) -> _FakeNativeBackend:
            backend = _FakeNativeBackend(
                game,
                evaluator,
                args,
                max_active_games=cast(int, kwargs["max_active_games"]),
                worker_threads=cast(int, kwargs["worker_threads"]),
                seed=cast(int, kwargs["seed"]),
                fail_wave=fail_wave,
                fail_observe=fail_observe,
                mismatch_terminal=mismatch_terminal,
            )
            self.created.append(backend)
            return backend

        return factory

    def make_player(
        self,
        *,
        fail_wave: bool = False,
        fail_observe: bool = False,
        mismatch_terminal: bool = False,
    ) -> NativeMCTSPlayer:
        return NativeMCTSPlayer(
            cast(Any, _FakeNet()),
            self.game,
            self.args,
            seed=17,
            native_worker_threads=3,
            _native_backend_factory=cast(
                Any,
                self.make_factory(
                    fail_wave=fail_wave,
                    fail_observe=fail_observe,
                    mismatch_terminal=mismatch_terminal,
                ),
            ),
        )

    def test_first_turn_is_lazy_deterministic_and_zero_temperature(self) -> None:
        player = self.make_player()
        board = self.game.init_board()

        self.assertEqual(0, len(self.created))
        self.assertEqual(0, player.play(self.game, board, 1))

        self.assertEqual(1, len(self.created))
        backend = self.created[0]
        self.assertEqual(1, backend.capacity)
        self.assertEqual(3, backend.thread_count)
        self.assertEqual(17, backend.seed)
        self.assertEqual(
            {"temperature": 0.0, "add_root_noise": False},
            backend.add_calls[0],
        )
        self.assertEqual(
            {"action": 0, "temperature": 0.0, "add_root_noise": False},
            backend.advance_calls[0],
        )
        self.assertEqual(1, player.net.eval_calls)

    def test_two_ply_observes_opponent_and_reuses_backend(self) -> None:
        player = self.make_player()
        first = self.game.init_board()
        self.assertEqual(0, player.play(self.game, first, 1))
        after_own, _ = self.game.apply_action(first, 1, 0)
        after_opponent, _ = self.game.apply_action(after_own, -1, 1)

        self.assertEqual(2, player.play(self.game, after_opponent, 1))
        backend = self.created[0]
        self.assertEqual(1, len(self.created))
        self.assertEqual(
            {"action": 1, "temperature": 0.0, "add_root_noise": False},
            backend.observe_calls[0],
        )
        self.assertEqual(
            {"action": 2, "temperature": 0.0, "add_root_noise": False},
            backend.advance_calls[1],
        )

    def test_reset_discards_backend_without_remove_and_next_call_rebuilds(self) -> None:
        player = self.make_player()
        board = self.game.init_board()
        player.play(self.game, board, 1)
        first_backend = self.created[0]

        player.reset()
        self.assertEqual([], first_backend.remove_calls)
        player.play(self.game, board, 1)

        self.assertEqual(2, len(self.created))
        self.assertIsNot(first_backend, self.created[1])

    def test_supplied_successor_mismatch_clears_backend(self) -> None:
        player = self.make_player()
        opening, _ = self.game.apply_action(self.game.init_board(), 1, 0)
        player.play(self.game, opening, -1)
        after_own, _ = self.game.apply_action(opening, -1, 1)
        after_opponent, _ = self.game.apply_action(after_own, 1, 2)

        forged = PenteBoard(
            after_opponent.board,
            after_opponent.captures,
            current_player=after_opponent.current_player,
            ply=after_opponent.ply,
            last_action=0,
        )
        with self.assertRaisesRegex(ValueError, "exact successor"):
            player.play(self.game, forged, -1)

        self.assertIsNone(player._backend)
        self.assertEqual([], self.created[0].observe_calls)

    def test_terminal_mismatch_and_wave_failure_clear_backend(self) -> None:
        player = self.make_player(mismatch_terminal=True)
        stones = np.zeros((5, 5), dtype=np.int8)
        stones[0, :4] = 1
        board = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=1, ply=4)
        with self.assertRaisesRegex(RuntimeError, "terminal mismatch"):
            player.play(self.game, board, 1)
        self.assertIsNone(player._backend)

        failing = self.make_player(fail_wave=True)
        with self.assertRaisesRegex(RuntimeError, "fake wave failed"):
            failing.play(self.game, self.game.init_board(), 1)
        self.assertIsNone(failing._backend)

    def test_observe_failure_discards_the_incomplete_backend(self) -> None:
        player = self.make_player(fail_observe=True)
        first = self.game.init_board()
        player.play(self.game, first, 1)
        after_own, _ = self.game.apply_action(first, 1, 0)
        after_opponent, _ = self.game.apply_action(after_own, -1, 1)

        with self.assertRaisesRegex(RuntimeError, "fake observe failed"):
            player.play(self.game, after_opponent, 1)

        self.assertIsNone(player._backend)

    def test_builder_constructs_python_mcts_only_when_requested(self) -> None:
        net = _FakeNet()
        player = cast(NNetPlayer, build_player(
            cast(Any, net),
            None,
            "python-mcts",
            search_backend="python",
            game=self.game,
            mcts_args=self.args,
            seed=23,
        ))

        self.assertIsInstance(player, NNetPlayer)
        mcts = player.mcts
        assert mcts is not None
        self.assertEqual(self.args, mcts.args)

    def test_cpp_direct_path_keeps_native_backend_lazy(self) -> None:
        net = _FakeNet()
        player = cast(NNetPlayer, build_player(
            cast(Any, net),
            None,
            "direct",
            search_backend="cpp",
            _native_backend_factory=cast(Any, self.make_factory()),
        ))

        self.assertIsInstance(player, NNetPlayer)
        self.assertIsNone(player.mcts)
        self.assertEqual([], self.created)

    def test_cpp_rejects_a_concrete_python_mcts(self) -> None:
        with self.assertRaisesRegex(ValueError, "Python MCTS"):
            build_player(
                cast(Any, _FakeNet()),
                cast(Any, object()),
                search_backend="cpp",
            )

    def test_builder_preserves_direct_and_python_mcts_players(self) -> None:
        net = cast(Any, object())
        mcts = cast(Any, object())
        direct = cast(NNetPlayer, build_player(net, None, "direct"))
        python_mcts = cast(NNetPlayer, build_player(net, mcts, "python"))

        self.assertIsInstance(direct, NNetPlayer)
        self.assertIsInstance(python_mcts, NNetPlayer)
        self.assertIs(net, direct.net)
        self.assertIs(mcts, python_mcts.mcts)
        self.assertEqual("direct", direct.name)
        self.assertEqual("python", python_mcts.name)

    def test_native_builder_stays_lazy_and_propagates_configuration(self) -> None:
        player = build_player(
            cast(Any, object()),
            None,
            "native",
            search_backend="cpp",
            game=self.game,
            mcts_args=self.args,
            seed=21,
            native_worker_threads=4,
            _native_backend_factory=cast(Any, self.make_factory()),
        )
        self.assertIsInstance(player, NativeMCTSPlayer)
        self.assertEqual([], self.created)
        player.reset()
        self.assertEqual([], self.created)


if __name__ == "__main__":
    unittest.main()
