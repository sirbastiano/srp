import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only
from einops import rearrange, repeat
import opt_einsum as oe

contract = oe.contract


class stepSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8
    ):
        super().__init__()
        self.h = d_model
        self.n = d_state
        
        self.dA = nn.Parameter(torch.randn(d_model, d_state // d_model, dtype=torch.cfloat))
        self.dB = nn.Parameter(torch.randn(d_model, d_state // d_model, dtype=torch.cfloat))
        self.dC = nn.Parameter(torch.randn(1, d_model, d_state // d_model, dtype=torch.cfloat)) 
        self.D = nn.Parameter(torch.randn(1, self.h))

        self.activation = nn.LeakyReLU(negative_slope=0.125)

        self.output_linear = nn.Linear(self.h, self.h)


    def forward(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
            + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)

        y, next_state = 2*y.real, next_state

        y = y.float()

        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        y = self.output_linear(y)

        return y, next_state


class stepCompreSSM(nn.Module):
    def __init__(self):
        super(stepCompreSSM, self).__init__()
        self.h = 2
        self.n = 8
        self.num_layers = 4


        # position embedding mixing
        self.fc1 = nn.Linear(3, 2)

        self.ssm = nn.ModuleList()
        for _ in range(self.num_layers):
            self.ssm.append(
                stepSSM(d_model=self.h, d_state=self.n)
                )
        
        self.fc2 = nn.Linear(2, 2)

    def forward(self, u, state):

        if state == None:
            # create a list of 0 states for each of the ssm layers
            print(f"No state detected! Creating new empty state")
            state = [torch.zeros(*batch_shape, self.h, self.n, dtype=cfloat) for ssm in self.ssm]

        u = self.fc1(u)

        for i, ssm in enumerate(self.ssm):
            u, state[i] = ssm(u, state[i])

        u = self.fc2(u)

        return u, state