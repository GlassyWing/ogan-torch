import torch.nn as nn
import torch
from timm.models.layers import Swish


class SelfModulateBatchNorm2d(nn.Module):

    def __init__(self, in_channel, style_dim):
        super().__init__()

        self._std_bn = nn.BatchNorm2d(in_channel, affine=False)
        self._beta_fc = nn.Sequential(
            nn.Linear(style_dim, in_channel),
            Swish(),
            nn.Linear(in_channel, in_channel)
        )
        self._gamma_fc = nn.Sequential(
            nn.Linear(style_dim, in_channel),
            Swish(),
            nn.Linear(in_channel, in_channel)
        )

    def forward(self, x, style):
        x = self._std_bn(x)  # (b, c, h, w)
        beta = self._beta_fc(style).unsqueeze(-1).unsqueeze(-1)
        gamma = self._gamma_fc(style).unsqueeze(-1).unsqueeze(-1)
        return x * (gamma + 1) + beta


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, z_dim, need_trunc=False):
        super().__init__()

        self._conv = nn.ConvTranspose2d(in_channel,
                                        out_channel,
                                        kernel_size=5,
                                        stride=2,
                                        padding=2,
                                        output_padding=1)
        self._style_extract = nn.Linear(z_dim, z_dim)
        self._bn = SelfModulateBatchNorm2d(out_channel, z_dim)
        self._act = Swish()
        self._res = StyleResidualBlock(out_channel, z_dim)
        self._noise_weights = nn.Parameter(torch.zeros((in_channel,)))
        if need_trunc:
            self._trunc = TruncationLayer(z_dim)
        else:
            self._trunc = None

    def forward(self, x, style, noise=None):
        style = self._style_extract(style)
        if self.training and self._trunc is not None:
            style = self._trunc(style)
        if noise is not None:
            x = x + self._noise_weights[None, :, None, None] * noise
        x = self._conv(x)
        x = self._bn(x, style)
        x = self._act(x)
        x = self._res(x, style)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._conv = nn.Conv2d(in_channel,
                               out_channel,
                               kernel_size=5,
                               stride=2,
                               padding=2)
        self._res = ResidualBlock(out_channel)
        self._bn = nn.BatchNorm2d(out_channel)
        self._act = Swish()

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = self._act(x)
        x = self._res(x)
        return x


class StyleResidualBlock(nn.Module):

    def __init__(self, dim, z_dim):
        super().__init__()
        self.alpha_proj = nn.Sequential(
            nn.Linear(z_dim, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
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
        alpha = self.alpha_proj(style)
        return x_in + alpha[:, :, None, None] * x


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.alpha = 0.1
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1), Swish(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1), Swish(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim))
        self._act = Swish()

    def forward(self, x):
        x_in = x
        x = self._seq(x)
        x = self._act(x)
        return x_in + self.alpha * x


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class TruncationLayer(nn.Module):

    def __init__(self, latent_size, threshold=0.8, beta=0.995):
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', torch.zeros(latent_size))
        self.is_init = True

    def update(self, last_avg):
        if self.is_init:
            self.avg_latent.copy_(last_avg)
            self.is_init = False
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 2
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        return interp
