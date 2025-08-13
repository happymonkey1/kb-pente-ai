from src.game.pente.pente_board import PenteBoard

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from numba import njit

from src.game.pente.pente_game import PenteGame

logger = logging.getLogger(__name__)

@torch.compile(fullgraph=True)
def boards_to_tensor(boards: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts a batch of boards (N, H, W) to a PyTorch tensor (N, 2, H, W)."""
    p1_masks = (boards == 1)
    p2_masks = (boards == -1)
    # Stack along the channel dimension (axis=1) to get (N, 2, H, W)
    batch = np.stack([p1_masks, p2_masks], axis=1)
    return torch.from_numpy(batch.astype(np.float32)).to(device)

@torch.compile(fullgraph=True)
class PenteNet(nn.Module):
    def __init__(
        self,
        device,
        board_size=19,
        #action_size=19*19+1,
        action_size=19*19,
        num_res_blocks=6,
        num_channels=512,
        hidden_fc_size: int = 1024,
    ):
        super().__init__()
        self.device = device
        self.board_size = board_size
        self.action_size = action_size
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.hidden_fc_size = hidden_fc_size

        self.conv_in = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * board_size * board_size, self.action_size)

        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value_in = nn.Linear(board_size * board_size, self.hidden_fc_size)
        self.fc_value_out = nn.Linear(self.hidden_fc_size, 1)

        self.to(device)

    @staticmethod
    def from_existing_model(net: 'PenteNet') -> 'PenteNet':
        return PenteNet(
            net.device,
            net.board_size,
            net.action_size,
            net.num_res_blocks,
            net.num_channels,
            net.hidden_fc_size,
        )

    @torch.compile(fullgraph=True)
    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))

        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)

        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value_in(v))
        v = torch.tanh(self.fc_value_out(v))

        return p, v

    @torch.compile(fullgraph=True)
    def predict(self, pente_board: 'PenteBoard'):
        self.eval()

        with torch.no_grad():
            batch = np.expand_dims(pente_board.board, axis=0)
            tensor = boards_to_tensor(batch, self.device)
            p,v = self.forward(tensor)
            p = F.softmax(p, dim=1)

        return p, v

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth.tar"):
        """Saves model and training parameters."""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(state, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    @staticmethod
    def load_checkpoint(checkpoint_dir, net, filename="checkpoint.pth.tar", optimizer=None):
        return PenteNet.load_checkpoint_from_path(os.path.join(checkpoint_dir, filename), net, optimizer)

    @staticmethod
    def load_checkpoint_from_path(filepath: str, net, optimizer=None):
        """Loads model and training parameters."""
        if not os.path.exists(filepath):
            logger.warning(f"No checkpoint found at '{filepath}'. Starting from scratch.")
            return 0

        map_location = next(net.parameters()).device
        checkpoint = torch.load(filepath, map_location=map_location)

        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_iteration = checkpoint['iteration']

        logger.info(f"Checkpoint loaded from '{filepath}'. Resuming from iteration {start_iteration}.")
        return start_iteration

    def get_checkpoint_file_name(self, iteration: int) -> str:
        return f"checkpoint-{iteration}_{self.board_size}_{self.action_size}_{self.num_res_blocks}_{self.num_channels}_{self.hidden_fc_size}.pth.tar"

@torch.compile(fullgraph=True)
class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)