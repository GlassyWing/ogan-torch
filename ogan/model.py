import torch.nn as nn

from .layers import Swish, UpsampleBlock, ResidualBlock


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
                    nn.Conv2d(pre_c, cur_c, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(cur_c),
                    Swish()))
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
        for i in range(num_layers + 1):
            cur_c = z_dim // 2 ** (i + 1)
            modules.append(UpsampleBlock(pre_c, cur_c, z_dim))
            pre_c = cur_c
        self._rec = nn.Sequential(
            ResidualBlock(pre_c),
            nn.Conv2d(pre_c, 3, kernel_size=1),
            nn.Tanh()
        )
        self.module_list = nn.ModuleList(modules)

    def forward(self, z):
        """

        :param z: Tensor. shape = (B, z_dim)
        :return:
        """
        z_in = z
        z = self._fc(z)
        z = z.reshape(-1, self.z_dim, self.map_size, self.map_size)

        for m in self.module_list:
            z = m(z, z_in)

        x_f = self._rec(z)
        return x_f


class OGAN(nn.Module):

    def __init__(self, z_dim, img_size, num_layers):
        super().__init__()

        self.encoder = Encoder(z_dim, img_size, num_layers)
        self.generator = Generator(z_dim, self.encoder.map_size, num_layers)
