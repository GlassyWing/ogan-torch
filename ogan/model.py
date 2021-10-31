import torch
import torch.nn as nn

from .layers import UpsampleBlock, ConvBlock, TruncationLayer


class Encoder(nn.Module):

    def __init__(self, z_dim, img_size, num_layers, max_num_channels):
        super().__init__()

        self.map_size = img_size // 2 ** (num_layers + 1)

        pre_c = 3
        modules = []
        for i in range(num_layers + 1):
            cur_c = max_num_channels // (2 ** (num_layers - i))
            modules.append(ConvBlock(pre_c, cur_c))
            pre_c = cur_c

        self._fc = nn.Linear(max_num_channels * self.map_size * self.map_size, z_dim)

        self._seq = nn.Sequential(*modules)

    def map_size(self):
        return self.map_size

    def forward(self, x):
        x = self._seq(x).reshape(x.shape[0], -1)
        x = self._fc(x)
        return x


class Generator(nn.Module):

    def __init__(self, z_dim, map_size, num_layers, max_num_channels, max_trunc_layer=None):
        super().__init__()

        self.z_dim = z_dim
        self.map_size = map_size
        self.max_num_channels = max_num_channels
        self.max_trunc_layer = max_trunc_layer if max_trunc_layer is not None else num_layers

        self._fc = nn.Linear(z_dim, self.map_size ** 2 * max_num_channels)
        self._trunc = TruncationLayer(z_dim)

        modules = []
        pre_c = max_num_channels
        for i in range(num_layers + 1):
            cur_c = max_num_channels // 2 ** (i + 1)
            modules.append(UpsampleBlock(pre_c, cur_c, z_dim, i <= self.max_trunc_layer))
            pre_c = cur_c
        self._rec = nn.Sequential(
            nn.Conv2d(pre_c, 3, kernel_size=1),
            nn.Tanh()
        )
        self.module_list = nn.ModuleList(modules)

    def forward(self, z):
        """

        :param z: Tensor. shape = (B, z_dim)
        :return:
        """
        style = z

        if self.training and self._trunc:
            self._trunc.update(style[0].detach())
            z = self._trunc(z)

        z = self._fc(z)
        z = z.reshape(-1, self.max_num_channels, self.map_size, self.map_size)

        for i, m in enumerate(self.module_list):
            s = self.map_size * 2 ** i
            noise = torch.randn(z.shape[0], 1, s, s, device=z.device)
            z = m(z, style, noise)

        x_f = self._rec(z)
        return x_f


class OGAN(nn.Module):

    def __init__(self, z_dim, img_size, num_layers, max_num_channels, max_trunc_layer=None):
        super().__init__()

        self.encoder = Encoder(z_dim, img_size, num_layers, max_num_channels)
        self.generator = Generator(z_dim, self.encoder.map_size, num_layers, max_num_channels, max_trunc_layer)
