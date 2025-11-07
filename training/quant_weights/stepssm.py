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
# should also work for complex numbers (hidden states are complex activations, other inputs are non complex吧)
class RollingStats:
    def __init__(self):
        self.n = 0
        self.mean = torch.tensor(0+0j, dtype=torch.cfloat)
        self.M2 = torch.tensor(0+0j, dtype=torch.cfloat) # sum of squares differences from the current mean

    def update(self, x):
        # x is a tensor of activations, flatten it
        x = x.detach().cpu().view(-1)
        for value in x:
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            delta2 = value - self.mean
            self.M2 += delta * delta2.conj()

    def get_scaling_factor(self):
        if self.n < 2:
            return self.mean, float('nan'), float('nan')

        std = (self.M2.real / (self.n - 1)) ** 0.5
        low = self.mean - 3*std
        high = self.mean + 3*std
        limit_low = max(abs(low.real.item()), abs(low.imag.item()))
        limit_high = max(abs(high.real.item()), abs(high.imag.item()))
        limit = max(self.mean - 3*std, self.mean + 3*std, key=abs)

        sf = 127 / limit # i want this to break if the limit is exactly 0

        return self.mean, std, sf

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
        y = rearrange(y, '... c h -> ... (c h)')
        
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
    def __init__(self):
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

    def forward(self, x, state):
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
        for _ in range(4):
            layer_state = torch.zeros(batch_size, 2, 4, dtype=torch.complex64)
            states.append(layer_state)
        return states

    def create_hook(self, name):
        rolling_list = []
        self.activation_stats[name] = rolling_list

        def hook(module, input, output):
            # Convert output to tuple
            outputs = output if isinstance(output, tuple) else (output,)
            # Initialize rolling stats for each output
            while len(rolling_list) < len(outputs):
                rolling_list.append(RollingStats())
            for rolling, out in zip(rolling_list, outputs):
                rolling.update(out)

        return hook

    def register_scale_factors(self):
        for name, rolling_list in self.activation_stats.items():
            for idx, rolling_stats in enumerate(rolling_list):
                if rolling_stats.n < 2:
                    print(f"Warning: Insufficient samples for {name} output {idx}")
                    raise CustomError("Not enough statistics were collected for the activations")
                
                mean, std, sf = rolling_stats.get_scaling_factor()

                # Register scale factor as buffer
                buffer_name_sf = f"{name}_output{idx}_sf"
                self.register_buffer(buffer_name_sf, torch.tensor(sf))

    def write_model_weights(self):
        '''Write the model weights to a dictionary for qssm to read in
        We can calculate the scaling factors for the model weights in this function 
        just before we write them to the dict'''
        for param_tensor in self.state_dict():
            print(f"{param_tensor}: {model.state_dict()[param_tensor].size()}")




    '''notes
    Basically I need one function which is going to convert all of the statistics that have been gradually
    accumulated into scaling factors and save them as buffers. Do saved buffers also get written to model weights- yes.

    Then I need another function that is going to go through all the saved model weights and pick out scaling factors for them

    Once I have done this maybe we need a function to save the model's weights.

    And then I need a mapping that will map those weights to the weights that qssm is expecting to read in.
    
    The question is do I write the model write weights code as part of this model internally as a method or as part
    of the outside script?
    '''

    





