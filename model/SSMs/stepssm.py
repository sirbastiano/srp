'''
stepSSM

This file contains code to run the pure stepped model and record statistics for quantization
TODO: This file still needs testing and debugging
'''


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


# class to record mean and std of activation data in a rolling manner as we perform inference
class RollingStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of differences from the current mean

    def update(self, x):
        # x is a tensor of activations, flatten it
        x = x.detach().cpu().float().view(-1)
        for value in x:
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            delta2 = value - self.mean
            self.M2 += delta * delta2

    def get_mean_std(self):
        if self.n < 2:
            return self.mean, float('nan')
        variance = self.M2 / (self.n - 1)
        return self.mean, variance ** 0.5

class ssm_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat))
        self.B = nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat))
        self.C = nn.Parameter(torch.randn(1, 2, 4, dtype=torch.cfloat))
        self.D = nn.Parameter(torch.randn(1, 2))

    def forward(self, u, state):
        next_state = (self.A.unsqueeze(0) * state) + (self.B * u.unsqueeze(-1))
        y = (self.C.unsqueeze(1) * next_state.unsqueeze(0)).sum(-1).permute(1, 0, 2)
        
        y, next_state = 2*y.real, next_state
        y = y.float()
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '...c h -> ... (c h)')
        
        return y, next_state

class fc_input_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 4))
        self.bias = nn.Parameter(torch.randn(2,))
    
    def forward(self, x):
        return (x @ self.weight.T) + self.bias

class fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 2))
        self.bias = nn.Parameter(torch.randn(2,))

    def forward(self, x):
        return (x @ self.weight.T) + self.bias

class stepSSM(nn.Module):
    def __init__(self)
        super(stepSSM, self).__init__()

        # model weights
        self.fc1 = fc_input_layer()
        self.ssm2 = ssm_layer()
        self.fc3 = fc_layer()
        self.ssm4 = ssm_layer()
        self.fc5 = fc_layer()
        self.ssm6 = ssm_layer()
        self.fc7 = fc_layer()
        self.ssm8 = ssm_layer()
        self.fc9 = fc_layer()
        self.fc10 = fc_layer()

        # activation function 
        self.act = nn.LeakyReLU(negative_slope=0.125)

        self.activation_stats = {}

        # add statistics recording hook
        for name, module in self.named_modules():
            if isinstance(module, (fc_input_layer, ssm_layer, fc_layer)):
                module.register_forward_hook(self.create_hook(name))

    def forward(self, u, state):
        x = self.fc1(x)
        x, state[0] = self.ssm2(x, state[0])
        x = self.act(x)
        x = self.fc3(x)
        x, state[1] = self.ssm4(x, state[1])
        x = self.act(x)
        x = self.fc5(x) 
        x, state[2] = self.ssm6(x, state[2])
        x = self.act(x)
        x = self.fc7(x)
        x, state[3] = self.ssm8(x, state[3])
        x = self.act(x)
        x = self.fc9(x)
        x = self.fc10(x)

        return x, state

    def setup_step(self, batch_size):
        states = []
        for ssm in self.ssm:
            layer_state = torch.zeros(batch_size, 2, 4, dtype=torch.complex64)
            states.append(layer_state)
        return states

    def create_hook(self, name):
        rolling_list = []
        self.activation_stats[name] = rolling_list

        def hook(self, module, input, output):
            # Convert output to tuple
            outputs = output if isinstance(output, tuple) else (output,)
            # Initialize rolling stats for each output
            while len(self.rolling_list) < len(outputs):
                self.rolling_list.append(RollingStats())
            for rolling, out in zip(self.rolling_list, outputs):
                rolling.update(out)

        return hook





