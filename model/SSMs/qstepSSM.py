'''
stepSSM

This file contains code for the most simplified stepped sar compression algorithm without any extra settings

This code will also be the base for the QAT version of the model
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
import sys

contract = oe.contract

def fake_quant_complex(self, x_real, x_imag, scale):
    ''' simulate quantization during training'''
    max_val = 2 ** (self.quant_bits - 1) - 1
    
    # Quantize 
    

class QuantizedStepSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8
    ):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.quant_bits = 8
        
        
        # Store complex parameters as real/imag pairs
        self.dA_real = nn.Parameter(torch.randn(d_model, d_state // d_model))
        self.dA_imag = nn.Parameter(torch.randn(d_model, d_state // d_model))
        self.dB_real = nn.Parameter(torch.randn(d_model, d_state // d_model))
        self.dB_imag = nn.Parameter(torch.randn(d_model, d_state // d_model))
        self.dC_real = nn.Parameter(torch.randn(1, d_model, d_state // d_model))
        self.dC_imag = nn.Parameter(torch.randn(1, d_model, d_state // d_model))

        self.D = nn.Parameter(torch.randn(1, self.h))

        # Quantization scales (learnable)
        self.register_buffer('scale_A', torch.tensor(1.0))
        self.register_buffer('scale_B', torch.tensor(1.0))
        self.register_buffer('scale_C', torch.tensor(1.0))

        self.activation = nn.LeakyReLU(negative_slope=0.125)
        self.output_linear = nn.quantized.Linear(self.h, self.h)
          
    def complex_quantize_dequantize(self, x_real, x_imag, scale):
        '''Simulate quantization during training'''
          max_val = 2 ** (self.quant_bits - 1) - 1

          # Quantize
          x_real_q = torch.clamp(torch.round(x_real * (1 / scale)), -max_val, max_val)
          x_imag_q = torch.clamp(torch.round(x_imag *( 1 / scale)), -max_val, max_val)

          # Dequantize
          x_real_dq = x_real_q * scale
          x_imag_dq = x_imag_q * scale

          return torch.complex(x_real_dq, x_imag_dq)       


    def forward(self, u, state):
        # Apply fake quantization
        dA = self.complex_quantize_dequantize(self.dA_real, self.dA_imag, self.scale_A)
        dB = self.complex_quantize_dequantize(self.dB_real, self.dB_imag, self.scale_B)
        dC = self.complex_quantize_dequantize(self.dC_real, self.dC_imag, self.scale_C)

        next_state = contract("h n, b h n -> b h n", dA, state) \
            + contract("h n, b h -> b h n", dB, u)
        y = contract("c h n, b h n -> b c h", dC, next_state)

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
            print(f"No state detected!")

        u = self.fc1(u)

        for i, ssm in enumerate(self.ssm):
            u, state[i] = ssm(u, state[i])

        u = self.fc2(u)
        
        print(f"Output of the model: {u}, {state}")
        print(f"Shapes of u : {u.shape}")
        
        state_list = [torch.isnan(s).any() for s in state]
        
            
        
        if torch.isnan(u).any() or True in state_list:
            sys.exit(1)
            

        return u, state

    def setup_step(self, batch_size):
        device = next(self.parameters()).device
        states = []
        for ssm in self.ssm:
            layer_state = torch.zeros(batch_size, self.h, self.n // self.h, dtype=torch.complex64, device=device)
            states.append(layer_state)
        return states
