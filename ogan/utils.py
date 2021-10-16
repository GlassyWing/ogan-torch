import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m


def correlation(x, y):
    x_mu = torch.mean(x, dim=1, keepdim=True)
    y_mu = torch.mean(y, dim=1, keepdim=True)
    x_std = torch.std(x, dim=1, keepdim=True)
    y_std = torch.std(y, dim=1, keepdim=True)
    a = torch.mean((x - x_mu) * (y - y_mu), dim=1, keepdim=True)
    b = x_std * y_std
    return a / b


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


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z
