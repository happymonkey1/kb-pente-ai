from __future__ import annotations

import torch


def select_torch_device(require_cuda: bool) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but Torch cannot access a CUDA device. "
            "Check WSL GPU passthrough and execution sandbox permissions."
        )
    return torch.device("cuda" if require_cuda else "cpu")
