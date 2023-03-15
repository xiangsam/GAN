'''
Author: Samrito
Date: 2023-03-14 20:37:57
LastEditors: Samrito
LastEditTime: 2023-03-14 20:43:59
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummaryX import summary
import numpy as np


class Generator(nn.Module):
    def __init__(self, random_dim=10, mnist_dim=28 * 28) -> None:
        super().__init__()
        self.random_dim = random_dim
        self.mnist_dim = mnist_dim
        self.G = nn.ModuleList([
            nn.Linear(random_dim, 32),
            nn.Linear(32, 128),
            nn.Linear(128, mnist_dim)
        ])
        for model in self.G:
            nn.init.uniform_(model.weight, a=-0.05, b=0.05)
            nn.init.zeros_(model.bias)

    def forward(self, x):
        h = F.relu(self.G[0](x))
        h = F.relu(self.G[1](h))
        h = self.G[-1](h)
        return torch.sigmoid(h)

    def sample_generator(self, *size):
        rand_x = torch.from_numpy(np.random.uniform(-1, 1, size)).float()
        return rand_x


class Discriminator(nn.Module):
    def __init__(self, mnist_dim=28 * 28) -> None:
        super().__init__()
        self.mnist_dim = mnist_dim
        self.D = nn.ModuleList(
            [nn.Linear(mnist_dim, 128),
             nn.Linear(128, 32),
             nn.Linear(32, 1)])
        for model in self.D:
            nn.init.uniform_(model.weight, a=-0.05, b=0.05)
            nn.init.zeros_(model.bias)

    def forward(self, x):
        # x (bs, c, h, w)
        h = F.relu(self.D[0](x))
        h = F.relu(self.D[1](h))
        return self.D[-1](h)


if __name__ == '__main__':
    x = torch.zeros((32, 10))
    nn.init.uniform_(x, -1, 1)
    g_model = Generator(random_dim=10)
    d_model = Discriminator()
    summary(g_model, x)
    x = torch.rand((32, 28 * 28))
    summary(d_model, x)