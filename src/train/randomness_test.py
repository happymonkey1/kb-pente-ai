import unittest

import numpy as np
import torch

from src.train.randomness import seed_training_iteration


class TrainingRandomnessTest(unittest.TestCase):
    def test_iteration_streams_are_resume_independent(self) -> None:
        first_numpy = seed_training_iteration(103, 2).integers(0, 2**31, size=8)
        first_torch = torch.randint(0, 2**31, (8,))

        seed_training_iteration(103, 0)
        np.random.default_rng(9).random(100)
        torch.rand(100)
        resumed_numpy = seed_training_iteration(103, 2).integers(0, 2**31, size=8)
        resumed_torch = torch.randint(0, 2**31, (8,))

        np.testing.assert_array_equal(first_numpy, resumed_numpy)
        torch.testing.assert_close(first_torch, resumed_torch)

    def test_iterations_have_distinct_streams(self) -> None:
        first = seed_training_iteration(103, 1).integers(0, 2**31, size=8)
        second = seed_training_iteration(103, 2).integers(0, 2**31, size=8)

        self.assertFalse(np.array_equal(first, second))

    def test_rejects_negative_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            seed_training_iteration(-1, 0)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            seed_training_iteration(0, -1)


if __name__ == "__main__":
    unittest.main()
