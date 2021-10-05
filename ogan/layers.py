import torch.nn as nn
import torch
from timm.models.layers import Swish


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
        # self._up = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        # )
        self._bn = SelfModulateBatchNorm2d(out_channel, z_dim)
        self._act = Swish()
        self._res = DecoderResidualBlock(out_channel, n_group=2)

    def forward(self, x, style):
        x = self._up(x)
        x = self._bn(x, style)
        x = self._act(x)
        x = self._res(x)
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
        self._res = ResidualBlock(out_channel)

    def forward(self, x, style):
        x = self._up(x)
        x = self._bn(x, style)
        x = self._act(x)
        x = self._res(x)
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


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.alpha =  nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2), Swish(),
            nn.Conv2d(dim, dim // 2, kernel_size=1), Swish(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim), Swish())

    def forward(self, x):
        return x + self.alpha * self._seq(x)


class DecoderResidualBlock(nn.Module):

    def __init__(self, dim, n_group):
        super().__init__()
        self.alpha = nn.Parameter(0.1 * torch.ones(size=(dim,), dtype=torch.float), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(size=(dim,), dtype=torch.float), requires_grad=True)
        # self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)

        self._seq = nn.Sequential(
            nn.Conv2d(dim, n_group * dim, kernel_size=1), Swish(),
            nn.Conv2d(n_group * dim, n_group * dim, kernel_size=5, padding=2, groups=n_group), Swish(),
            nn.Conv2d(n_group * dim, n_group * dim // 2, kernel_size=1, groups=n_group), Swish(),
            nn.Conv2d(n_group * dim // 2, n_group * dim, kernel_size=3, padding=1, groups=n_group), Swish(),
            nn.Conv2d(n_group * dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish())

    def forward(self, x):
        return x + self.alpha[None, :, None, None] * self._seq(x) + self.beta[None, :, None, None]