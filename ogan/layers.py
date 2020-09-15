import torch.nn as nn
import numpy as np
import torch

from ogan.utils import input_mapping


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