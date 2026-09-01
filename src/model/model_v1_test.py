import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.game.pente.pente_board import PenteBoard
from src.model.model_v1 import (
    CheckpointTrainingState,
    PenteNet,
    positions_to_tensor,
)


class PenteNetTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.device = torch.device("cpu")
        self.net = PenteNet(
            self.device,
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )

    def test_batch_encoding_matches_position_features(self) -> None:
        positions = (PenteBoard.new_board(5), PenteBoard.new_board(5).apply_move(1, (0, 0)))

        tensor = positions_to_tensor(positions, self.device)

        self.assertEqual((2, 4, 5, 5), tuple(tensor.shape))
        np.testing.assert_allclose(tensor[0].numpy(), positions[0].feature_planes())
        np.testing.assert_allclose(tensor[1].numpy(), positions[1].feature_planes())

    def test_evaluate_does_not_change_model_mode(self) -> None:
        self.net.train()

        policy, value = self.net.evaluate(PenteBoard.new_board(5))

        self.assertTrue(self.net.training)
        self.assertEqual((25,), policy.shape)
        self.assertAlmostEqual(1.0, float(policy.sum()), places=6)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_batch_norm_updates_only_in_training_mode(self) -> None:
        positions = (PenteBoard.new_board(5), PenteBoard.new_board(5).apply_move(1, (0, 0)))
        inputs = positions_to_tensor(positions, self.device)
        assert self.net.bn_in.running_mean is not None
        initial = self.net.bn_in.running_mean.clone()

        self.net.train()
        self.net(inputs)
        assert self.net.bn_in.running_mean is not None
        after_training = self.net.bn_in.running_mean.clone()
        self.assertFalse(torch.equal(initial, after_training))

        self.net.eval()
        self.net.evaluate_batch(positions)
        assert self.net.bn_in.running_mean is not None
        np.testing.assert_allclose(self.net.bn_in.running_mean.numpy(), after_training.numpy())

    def test_single_and_batch_evaluation_are_equal(self) -> None:
        position = PenteBoard.new_board(5).apply_move(1, (0, 0))
        self.net.eval()

        single_policy, single_value = self.net.evaluate(position)
        batch_policies, batch_values = self.net.evaluate_batch((position,))

        np.testing.assert_allclose(single_policy, batch_policies[0])
        self.assertAlmostEqual(single_value, float(batch_values[0]))

    def test_tensor_feature_evaluation_stays_on_torch_boundary(self) -> None:
        positions = (
            PenteBoard.new_board(5),
            PenteBoard.new_board(5).apply_move(1, (0, 0)),
        )
        inputs = positions_to_tensor(positions, self.device)
        self.net.eval()

        policies, values = self.net.evaluate_features(inputs)

        self.assertIsInstance(policies, torch.Tensor)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(torch.float32, policies.dtype)
        self.assertEqual(torch.float32, values.dtype)
        self.assertEqual(torch.device("cpu"), policies.device)
        self.assertEqual(torch.device("cpu"), values.device)
        self.assertEqual((2, 25), tuple(policies.shape))
        self.assertEqual((2,), tuple(values.shape))
        self.assertTrue(policies.is_contiguous())
        self.assertTrue(values.is_contiguous())
        self.assertFalse(policies.requires_grad)
        self.assertFalse(values.requires_grad)
        np_policies, np_values = self.net.evaluate_batch(positions)
        np.testing.assert_allclose(policies.numpy(), np_policies)
        np.testing.assert_allclose(values.numpy(), np_values)

    def test_tensor_feature_evaluation_uses_module_call(self) -> None:
        inputs = positions_to_tensor((PenteBoard.new_board(5),), self.device)
        self.net.eval()
        forward_hook = Mock(return_value=None)
        handle = self.net.register_forward_hook(forward_hook)
        try:
            self.net.evaluate_features(inputs)
        finally:
            handle.remove()

        forward_hook.assert_called_once()
        self.assertIs(self.net, forward_hook.call_args.args[0])

    def test_tensor_feature_evaluation_rejects_wrong_boundary(self) -> None:
        with self.assertRaises(ValueError):
            self.net.evaluate_features(torch.zeros((1, 4, 6, 6), dtype=torch.float32))
        with self.assertRaises(ValueError):
            self.net.evaluate_features(torch.zeros((1, 4, 5, 5), dtype=torch.float64))
        with self.assertRaises(ValueError):
            self.net.evaluate_features(torch.zeros((0, 4, 5, 5), dtype=torch.float32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_autocast_evaluation_returns_float32_numpy_arrays(self) -> None:
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
        positions = (
            PenteBoard.new_board(5),
            PenteBoard.new_board(5).apply_move(1, (0, 0)),
        )

        policies, values = net.evaluate_batch(positions)

        self.assertEqual(np.float32, policies.dtype)
        self.assertEqual(np.float32, values.dtype)
        self.assertTrue(np.isfinite(policies).all())
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(policies.sum(axis=1), np.ones(2), atol=1e-5)

    def test_checkpoint_round_trip_and_legacy_rejection(self) -> None:
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            state = {
                "iteration": 3,
                "state_dict": self.net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metadata": self.net.checkpoint_metadata("freestyle"),
                "training_run_id": "checkpoint-test",
                "replay_snapshot_generation": 2,
            }
            PenteNet.save_checkpoint(state, directory, "valid.pt")
            loaded = PenteNet.from_existing_model(self.net)
            loaded_optimizer = torch.optim.AdamW(loaded.parameters(), lr=1e-3)

            iteration = PenteNet.load_checkpoint(
                directory,
                loaded,
                "valid.pt",
                loaded_optimizer,
                expected_ruleset="freestyle",
            )

            self.assertEqual(3, iteration)
            for expected, actual in zip(self.net.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(expected, actual))

            training_state = PenteNet.load_training_checkpoint_from_path(
                os.path.join(directory, "valid.pt"),
                loaded,
                loaded_optimizer,
                expected_ruleset="freestyle",
            )
            self.assertEqual(
                CheckpointTrainingState(3, 2, "checkpoint-test"),
                training_state,
            )

            legacy_path = os.path.join(directory, "legacy.pt")
            torch.save({"iteration": 1, "state_dict": self.net.state_dict()}, legacy_path)
            with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
                PenteNet.load_checkpoint_from_path(legacy_path, loaded)
            with self.assertRaisesRegex(ValueError, "ruleset"):
                PenteNet.load_checkpoint(
                    directory,
                    loaded,
                    "valid.pt",
                    expected_ruleset="standard",
                )

    def test_failed_checkpoint_write_preserves_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "latest.pt")
            original = b"known-good-checkpoint"
            with open(path, "wb") as stream:
                stream.write(original)
            state: dict[str, object] = {
                "metadata": self.net.checkpoint_metadata("freestyle"),
            }

            with patch("src.model.model_v1.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    PenteNet.save_checkpoint(state, directory, "latest.pt")

            with open(path, "rb") as stream:
                self.assertEqual(original, stream.read())


if __name__ == "__main__":
    unittest.main()
