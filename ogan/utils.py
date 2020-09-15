import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn as nn


def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m

def correlation(x, y):
    x = x - torch.mean(x, dim=1, keepdim=True)
    y = y - torch.mean(y, dim=1, keepdim=True)
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.sum(x * y, 1, keepdim=True)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def create_grid(h, w, device="cpu"):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)
