import torch.nn as nn
import torch
from timm.models.layers import Swish
import torch.nn.functional as F

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
                                      kernel_size=5,
                                      stride=2,
                                      padding=2,
                                      output_padding=1)
        self._pixel_norm = PixelNormLayer()
        self._bn = SelfModulateBatchNorm2d(out_channel, z_dim)
        self._act = Swish()
        self._res = StyleResidualBlock(out_channel, z_dim)
        self._noise_weights = nn.Parameter(torch.ones((in_channel,)) * 0.1)

    def forward(self, x, style, noise=None):
        if noise is not None:
            x = x + self._noise_weights[None, :, None, None] * noise
            self._pixel_norm(x)
        x = self._up(x)
        x = self._bn(x, style)
        x = self._act(x)
        x = self._res(x, style)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._dn = nn.Conv2d(in_channel,
                             out_channel,
                             kernel_size=5,
                             stride=2,
                             padding=2)
        self._bn = nn.BatchNorm2d(out_channel)
        self._act = Swish()
        # self._res = ResidualBlock(out_channel)

    def forward(self, x):
        x = self._dn(x)
        x = self._bn(x)
        x = self._act(x)
        # x = self._res(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
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


class StyleResidualBlock(nn.Module):

    def __init__(self, dim, z_dim):
        super().__init__()
        self.alpha = 0.1
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2), Swish(),
            nn.Conv2d(dim, dim // 2, kernel_size=1), Swish(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1))
        self._bn = SelfModulateBatchNorm2d(dim, z_dim)
        self._act = Swish()

    def forward(self, x, style):
        x_in = x
        x = self._seq(x)
        x = self._bn(x, style)
        x = self._act(x)
        return x_in + self.alpha * x


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.alpha = 0.1
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2), Swish(),
            nn.Conv2d(dim, dim // 2, kernel_size=1), Swish(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1))
        self._bn = nn.BatchNorm2d(dim)
        self._act = Swish()

    def forward(self, x):
        x_in = x
        x = self._seq(x)
        x = self._bn(x)
        x = self._act(x)
        return x_in + self.alpha * x


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
