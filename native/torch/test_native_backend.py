from __future__ import annotations

import unittest

import numpy as np
import torch

from src.game.pente.pente_game import PenteGame
from src.game.pente.pente_board import PenteBoard
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.mcts.native_backend import NativeSearchBackend
from src.model.model_v1 import PenteNet


def _draw_root():
    stones = torch.tensor(
        [
            [1, 1, -1, -1, 1],
            [-1, -1, 1, 1, -1],
            [1, 1, -1, -1, 1],
            [-1, -1, 1, 1, -1],
            [1, -1, 1, -1, 0],
        ],
        dtype=torch.int8,
    )
    return stones, torch.zeros(2, dtype=torch.int16)


class NativeBackendExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extension = __import__("kb_pente_native")
        cls.game = PenteGame(board_size=5, ruleset=PenteRuleset.FREESTYLE)
        cls.net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        cls.net.eval()

    def make_backend(self, simulations: int = 2, capacity: int = 2) -> NativeSearchBackend:
        return NativeSearchBackend(
            self.game,
            self.net,
            MCTSArgs(num_simulations=simulations, root_noise_epsilon=0.0),
            max_active_games=capacity,
            worker_threads=2,
            seed=103,
            pin_memory=False,
            extension=self.extension,
        )

    def test_duplicate_roots_multiwave_completion_and_stable_staging(self) -> None:
        backend = self.make_backend()
        root = self.game.init_board()
        first = backend.add_root(root, temperature=0.0)
        second = backend.add_root(root, temperature=0.0)

        waves = []
        while not backend.complete():
            waves.append(backend.evaluate_wave())
        self.assertEqual([wave.size for wave in waves], [1, 1])
        self.assertEqual([wave.raw_size for wave in waves], [2, 2])
        self.assertEqual(backend.slot_telemetry(first), backend.slot_telemetry(second))
        self.assertEqual(backend.slot_telemetry(first).mean_inference_batch_size, 1.0)
        self.assertEqual(
            backend.deduplication_telemetry()["cumulative"]["eliminated_duplicate_evaluations"],
            2,
        )
        policy = backend.root_policy(first)
        self.assertEqual((25,), policy.shape)
        self.assertEqual(np.float32, policy.dtype)
        self.assertAlmostEqual(1.0, float(policy.sum()), places=5)
        timing = backend.timing_telemetry()
        self.assertGreaterEqual(timing["inference"]["calls"], 2)
        self.assertEqual(2, timing["worker"]["worker_threads"])

    def test_root_advance_terminal_conversion_and_remove(self) -> None:
        backend = self.make_backend(simulations=1, capacity=1)
        root = self.game.init_board()
        slot = backend.add_root(root, temperature=0.0)
        while not backend.complete():
            backend.evaluate_wave()
        backend.advance_root(slot, 0, temperature=0.0)
        self.assertFalse(backend.slot_complete(slot))
        backend.evaluate_wave()
        self.assertTrue(backend.slot_complete(slot))

        draw_stones, draw_captures = _draw_root()
        draw = PenteBoard(
            draw_stones.numpy(),
            draw_captures.numpy(),
            current_player=1,
            ply=24,
        )
        backend.replace_root(slot, draw, temperature=0.0)
        backend.evaluate_wave()
        backend.advance_root(slot, 24, temperature=0.0)
        self.assertEqual(backend.root_terminal(slot).status.value, "draw")
        self.assertTrue(backend.slot_complete(slot))
        backend.remove(slot)
        self.assertEqual(backend.active_count, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_pinned_transfer_and_wait_timing(self) -> None:
        device = torch.device("cuda")
        net = PenteNet(
            device,
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        net.eval()
        backend = NativeSearchBackend(
            self.game,
            net,
            MCTSArgs(num_simulations=1, root_noise_epsilon=0.0),
            max_active_games=1,
            worker_threads=1,
            seed=103,
            pin_memory=True,
            extension=self.extension,
        )
        backend.add_root(self.game.init_board(), temperature=0.0)
        wave = backend.evaluate_wave()
        self.assertEqual(1, wave.size)
        timing = backend.inference_timing()
        self.assertEqual(1, timing["calls"])
        self.assertGreaterEqual(timing["host_to_device_seconds"], 0.0)
        self.assertGreaterEqual(timing["model_inference_seconds"], 0.0)
        self.assertGreaterEqual(timing["device_to_host_seconds"], 0.0)
        self.assertGreaterEqual(timing["inference_wait_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
