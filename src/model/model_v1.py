import torch
import torch.nn as nn
import torch.nn.functional as F

class PenteNet(nn.Module):
    def __init__(self, board_size=19):
        super().__init__()
        self.board_size = board_size
        self.input_channels = 2
        self.conv1 = nn.Conv2d(self.input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Policy head
        self.conv_policy = nn.Conv2d(256, 2, kernel_size=1)
        self.fc_policy = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head
        self.conv_value = nn.Conv2d(256, 1, kernel_size=1)
        self.fc_value1 = nn.Linear(board_size * board_size, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (B, self.input_channels, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Policy
        p = F.relu(self.conv_policy(x))
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.fc_policy(p), dim=1)  # log‑probabilities

        # Value
        v = F.relu(self.conv_value(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))

        return p, v
