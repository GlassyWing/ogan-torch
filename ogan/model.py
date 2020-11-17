import torch
import torch.nn as nn

from .layers import FourierMapping

from ogan.utils import create_grid


class Encoder(nn.Module):

    def __init__(self, z_dim, img_size, num_layers):
        super().__init__()

        self.map_size = img_size // 2 ** (num_layers + 1)

        pre_c = 3
        modules = []
        for i in range(num_layers + 1):
            cur_c = z_dim // (2 ** (num_layers - i))
            modules.append(
                nn.Sequential(
                    nn.Conv2d(pre_c, cur_c, kernel_size=5, padding=2, stride=2),
                    nn.BatchNorm2d(cur_c),
                    nn.LeakyReLU()))
            pre_c = cur_c

        self._fc = nn.Linear(z_dim * self.map_size * self.map_size, z_dim)

        self._seq = nn.Sequential(*modules)

    def map_size(self):
        return self.map_size

    def forward(self, x):
        x = self._seq(x).reshape(x.shape[0], -1)
        x = self._fc(x)
        return x


class Generator(nn.Module):

    def __init__(self, z_dim, map_size, num_layers):
        super().__init__()

        self.z_dim = z_dim
        self.map_size = map_size

        self._fc = nn.Linear(z_dim, self.map_size ** 2 * z_dim)

        modules = []
        pre_c = z_dim
        for i in range(num_layers):
            cur_c = z_dim // 2 ** (i + 1)
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(pre_c, cur_c, kernel_size=5, padding=2,
                                   stride=2, output_padding=1),
                nn.BatchNorm2d(cur_c),
                nn.ReLU()
            ))
            pre_c = cur_c
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(pre_c, 3, kernel_size=5, padding=2, stride=2,
                               output_padding=1),
            nn.Tanh()
        ))
        self._seq = nn.Sequential(*modules)

    def forward(self, z):
        """

        :param z: Tensor. shape = (B, z_dim)
        :return:
        """

        # (B, m_h, m_w, z_dim)
        # z_rep = z.unsqueeze(1).repeat(1, self.map_size ** 2, 1) \
        #     .reshape(z.shape[0], self.map_size, self.map_size, z.shape[1])
        # grid = self.grid.unsqueeze(0).repeat(z.shape[0], 1, 1, 1)
        # grid = self.pos_map(grid)
        # z_sample = self.map_z(torch.cat([grid, z_rep], dim=-1).permute(0, 3, 1, 2).contiguous())
        z_sample = self._fc(z)
        z_sample = z_sample.reshape(-1, self.z_dim, self.map_size, self.map_size)

        return self._seq(z_sample)


class OGAN(nn.Module):

    def __init__(self, z_dim, img_size, num_layers):
        super().__init__()

        self.encoder = Encoder(z_dim, img_size, num_layers)
        self.generator = Generator(z_dim, self.encoder.map_size, num_layers)
