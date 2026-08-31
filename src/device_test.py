import unittest
from unittest.mock import patch

import torch

from src.device import select_torch_device


class SelectTorchDeviceTest(unittest.TestCase):
    def test_returns_cpu_when_cuda_is_not_requested(self) -> None:
        self.assertEqual(torch.device("cpu"), select_torch_device(False))

    @patch("src.device.torch.cuda.is_available", return_value=True)
    def test_returns_cuda_when_requested_and_available(self, is_available: object) -> None:
        self.assertEqual(torch.device("cuda"), select_torch_device(True))

    @patch("src.device.torch.cuda.is_available", return_value=False)
    def test_rejects_silent_cpu_fallback(self, is_available: object) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            select_torch_device(True)


if __name__ == "__main__":
    unittest.main()
