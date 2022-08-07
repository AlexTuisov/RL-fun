import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl


class AtariScreenNetwork(nn.Module):
    def __init__(self, input_shape, number_of_actions):
        super().__init__()
        self.q = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, number_of_actions)
        )

    def forward(self, state):
        action = self.q(state)


