import os
import torch
from torch.autograd import Variable
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class TemporalEncoder(nn.Module):

    def __init__(self,
                 n_layers=1,
                 hidden_size=2048,
                 add_linear=False,
                 bidirectional=False,
                 use_residual=True):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(input_size=2048,
                          hidden_size=hidden_size,
                          bidirectional=bidirectional,
                          num_layers=n_layers)

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y
