import torch.nn as nn
import numpy as np
import torch

from ogan.utils import input_mapping


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SelfModulateBatchNorm2d(nn.Module):

    def __init__(self, in_channel, style_dim):
        super().__init__()

        self._std_bn = nn.BatchNorm2d(in_channel, affine=False)
        self._beta_fc = nn.Sequential(
            nn.Linear(style_dim, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel)
        )
        self._gamma_fc = nn.Sequential(
            nn.Linear(style_dim, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel)
        )

    def forward(self, x, style):
        x = self._std_bn(x)  # (b, c, h, w)
        beta = self._beta_fc(style).unsqueeze(-1).unsqueeze(-1)
        gamma = self._gamma_fc(style).unsqueeze(-1).unsqueeze(-1)
        return x * (gamma + 1) + beta


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, z_dim):
        super().__init__()

        self._up = nn.ConvTranspose2d(in_channel,
                                      out_channel,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        self._bn = SelfModulateBatchNorm2d(out_channel, z_dim)
        self._act = Swish()

    def forward(self, x, style):
        x = self._up(x)
        x = self._bn(x, style)
        x = self._act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, z_dim):
        super().__init__()

        self._up = nn.Conv2d(in_channel,
                             out_channel,
                             kernel_size=3,
                             stride=2,
                             padding=1)
        self._bn = SelfModulateBatchNorm2d(out_channel, z_dim)
        self._act = Swish()

    def forward(self, x, style):
        x = self._up(x)
        x = self._bn(x, style)
        x = self._act(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)


class FourierMapping(nn.Module):

    def __init__(self, dims, seed):
        super().__init__()
        np.random.seed(seed)
        B = np.random.randn(*dims) * 10
        np.random.seed(None)
        self.B = torch.nn.Parameter(torch.from_numpy(B.astype(np.float32)), requires_grad=False)

    def forward(self, x):
        x = input_mapping(x, self.B)
        return x