'''
qssm. Final shape of the model that is going to run on the FPGA. Written by S Fieldhouse.
'''

import logging
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn

class qssm(nn.Module):
    def __init__(self):
        super().__init__()

        """ I'm assuming all weights are the original weights to be read in from the fp model.
        Then I am assuming the scaling factors are all scalars that I either read in from the fp model or that I
        perhaps write a function here to derive"""
        self.weights = nn.ParameterDict({
            "fc1.weight"     : nn.Parameter(torch.randn(2, 4))
            "fc1.bias"       : nn.Parameter(torch.randn(2,)),
            "fc1.sW"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc1.sX"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc1.sY"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm2.A"         : nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat)),
            "ssm2.B"         : nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat)),
            "ssm2.C"         : nn.Parameter(torch.randn(1, 2, 4, dtype=torch.cfloat)),
            "ssm2.D"         : nn.Parameter(torch.randn(1, 2))
            "ssm2.sA"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm2.sB"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm2.sC"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm2.sD"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm2.sH"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm2.sX"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm2.sY"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc3.weight"     : nn.Parameter(torch.randn(2,2)),
            "fc3.bias"       : nn.Parameter(torch.randn(2,)),
            "fc3.sW"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc3.sX"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc3.sY"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm4.A"         : nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat)),
            "ssm4.B"         : nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat)),
            "ssm4.C"         : nn.Parameter(torch.randn(1, 2, 4, dtype=torch.cfloat)),
            "ssm4.D"         : nn.Parameter(torch.randn(1, 2))
            "ssm4.sA"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm4.sB"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm4.sC"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm4.sD"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm4.sH"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm4.sX"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm4.sY"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc5.weight"     : nn.Parameter(torch.randn(2,2)),
            "fc5.bias"       : nn.Parameter(torch.randn(2,)),
            "fc5.sW"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc5.sX"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc5.sY"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm6.A"         : nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat)),
            "ssm6.B"         : nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat)),
            "ssm6.C"         : nn.Parameter(torch.randn(1, 2, 4, dtype=torch.cfloat)),
            "ssm6.D"         : nn.Parameter(torch.randn(1, 2))
            "ssm6.sA"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm6.sB"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm6.sC"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm6.sD"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm6.sH"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm6.sX"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm6.sY"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc7.weight"     : nn.Parameter(torch.randn(2,2)),
            "fc7.bias"       : nn.Parameter(torch.randn(2,)),
            "fc7.sW"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc7.sX"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc7.sY"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm8.A"         : nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat)),
            "ssm8.B"         : nn.Parameter(torch.randn(2, 4, dtype=torch.cfloat)),
            "ssm8.C"         : nn.Parameter(torch.randn(1, 2, 4, dtype=torch.cfloat)),
            "ssm8.D"         : nn.Parameter(torch.randn(1, 2))
            "ssm8.sA"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm8.sB"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm8.sC"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm8.sD"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm8.sH"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm8.sX"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "ssm8.sY"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc9.weight"     : nn.Parameter(torch.randn(2,2)),
            "fc9.bias"       : nn.Parameter(torch.randn(2,)),
            "fc9.sW"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc9.sX"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc9.sY"         : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc10.weight"    : nn.Parameter(torch.randn(2,2)),
            "fc10.bias"      : nn.Parameter(torch.randn(2,)),
            "fc10.sW"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc10.sX"        : nn.Parameter(torch.randn(1,), requires_grad=False),
            "fc10.sY"        : nn.Parameter(torch.randn(1,), requires_grad=False)
        })

        self.act = nn.LeakyReLU(negative_slope=0.125)

    def forward(self, x, state):
        """Assuming that state is a list of 4 tensors, each tensor has a size of (4, 2)"""
        """Also assuming that initial states are all zeros so doesn't need an initial quantization"""
        # quantize on input
        x = torch.round(x / self.weights["fc1.sX"]) # quantized input
        # fc1
        x = torch.round((self.weights["fc1.sW"]*self.weights["fc1.sX"]/self.weights["fc1.sY"]) * ((torch.round(self.weights["fc1.weight"]/self.weights["fc1.sW"]) @ x) + torch.round(self.weights["fc1.bias"]/(self.weights["fc1.sX"]*self.weights["fc1.sW"]))))
        """ fc1.sW*fc1.sX / fc1.sY == M
            also, fc1.sY should = ssm2.sX """
        # ssm2
        accA = torch.round(self.weights["ssm2.A"]/self.weights["ssm2.sA"]).unsqueeze(0) @ state[0]
        accB = torch.round(self.weights["ssm2.B"]/self.weights["ssm2.sB"]) @ x.unsqueeze(-1)
        state[0] = torch.round(self.weights["ssm2.sA"]*accA + (self.weights["ssm2.sB"]*self.weights["ssm2.sX"]/self.weights["ssm2.sH"])*accB)
    
        accC = (torch.round(self.weights["ssm2.C"]/self.weights["ssm2.sC"]) @ state[0].T).permute(1, 0, 2) # this is the updated state[0]
        accD = x.unsqueeze(-2) * torch.round(self.weights["ssm2.D"]/self.weights["ssm2.sD"])
        x = torch.round((2*(self.weights["ssm2.sC"] * self.weights["ssm2.sH"] / self.weights["ssm2.sY"]) * accC).real + ((self.weights["ssm2.sD"]*self.weights["ssm2.sX"]/self.weights["ssm2.sY"]) * accD))

        del accA, accB, accC, accD

        # LeakyReLU
        x = self.act(u)

        # fc3
        x = torch.round((self.weights["fc3.sW"]*self.weights["fc3.sX"]/self.weights["fc3.sY"]) * ((torch.round(self.weights["fc3.weight"]/self.weights["fc3.sW"]) @ x) + torch.round(self.weights["fc3.bias"]/(self.weights["fc3.sX"]*self.weights["fc3.sW"]))))
        """ fc3.sW*fc3.sX / fc3.sY == M 
            fc3.sX should == ssm2.sY and fc3.sY == ssm4.sX"""

        # ssm4
        accA = torch.round(self.weights["ssm4.A"]/self.weights["ssm4.sA"]).unsqueeze(0) @ state[0]
        accB = torch.round(self.weights["ssm4.B"]/self.weights["ssm4.sB"]) @ x.unsqueeze(-1)
        state[0] = torch.round(self.weights["ssm4.sA"]*accA + (self.weights["ssm4.sB"]*self.weights["ssm4.sX"]/self.weights["ssm4.sH"])*accB)
    
        accC = (torch.round(self.weights["ssm4.C"]/self.weights["ssm4.sC"]) @ state[0].T).permute(1, 0, 2) # this is the updated state[0]
        accD = x.unsqueeze(-2) * torch.round(self.weights["ssm4.D"]/self.weights["ssm4.sD"])
        x = torch.round((2*(self.weights["ssm4.sC"] * self.weights["ssm4.sH"] / self.weights["ssm4.sY"]) * accC).real + ((self.weights["ssm4.sD"]*self.weights["ssm4.sX"]/self.weights["ssm4.sY"]) * accD))

        del accA, accB, accC, accD

        # LeakyReLU
        x = self.act(u)

        # fc5
        x = torch.round((self.weights["fc5.sW"]*self.weights["fc5.sX"]/self.weights["fc5.sY"]) * ((torch.round(self.weights["fc5.weight"]/self.weights["fc5.sW"]) @ x) + torch.round(self.weights["fc5.bias"]/(self.weights["fc5.sX"]*self.weights["fc5.sW"]))))

        # ssm6
        accA = torch.round(self.weights["ssm6.A"]/self.weights["ssm6.sA"]).unsqueeze(0) @ state[0]
        accB = torch.round(self.weights["ssm6.B"]/self.weights["ssm6.sB"]) @ x.unsqueeze(-1)
        state[0] = torch.round(self.weights["ssm6.sA"]*accA + (self.weights["ssm6.sB"]*self.weights["ssm6.sX"]/self.weights["ssm6.sH"])*accB)
    
        accC = (torch.round(self.weights["ssm6.C"]/self.weights["ssm6.sC"]) @ state[0].T).permute(1, 0, 2) # this is the updated state[0]
        accD = x.unsqueeze(-2) * torch.round(self.weights["ssm6.D"]/self.weights["ssm6.sD"])
        x = torch.round((2*(self.weights["ssm6.sC"] * self.weights["ssm6.sH"] / self.weights["ssm6.sY"]) * accC).real + ((self.weights["ssm6.sD"]*self.weights["ssm6.sX"]/self.weights["ssm6.sY"]) * accD))

        del accA, accB, accC, accD

        # LeakyReLU
        x = self.act(u)

        # fc7
        x = torch.round((self.weights["fc7.sW"]*self.weights["fc7.sX"]/self.weights["fc7.sY"]) * ((torch.round(self.weights["fc7.weight"]/self.weights["fc7.sW"]) @ x) + torch.round(self.weights["fc7.bias"]/(self.weights["fc7.sX"]*self.weights["fc7.sW"]))))

        # ssm8
        accA = torch.round(self.weights["ssm8.A"]/self.weights["ssm8.sA"]).unsqueeze(0) @ state[0]
        accB = torch.round(self.weights["ssm8.B"]/self.weights["ssm8.sB"]) @ x.unsqueeze(-1)
        state[0] = torch.round(self.weights["ssm8.sA"]*accA + (self.weights["ssm8.sB"]*self.weights["ssm8.sX"]/self.weights["ssm8.sH"])*accB)
    
        accC = (torch.round(self.weights["ssm8.C"]/self.weights["ssm8.sC"]) @ state[0].T).permute(1, 0, 2) # this is the updated state[0]
        accD = x.unsqueeze(-2) * torch.round(self.weights["ssm8.D"]/self.weights["ssm8.sD"])
        x = torch.round((2*(self.weights["ssm8.sC"] * self.weights["ssm8.sH"] / self.weights["ssm8.sY"]) * accC).real + ((self.weights["ssm8.sD"]*self.weights["ssm8.sX"]/self.weights["ssm8.sY"]) * accD))

        del accA, accB, accC, accD

        # LeakyReLU
        x = self.act(u)

        # fc9
        x = torch.round((self.weights["fc9.sW"]*self.weights["fc9.sX"]/self.weights["fc9.sY"]) * ((torch.round(self.weights["fc9.weight"]/self.weights["fc9.sW"]) @ x) + torch.round(self.weights["fc9.bias"]/(self.weights["fc9.sX"]*self.weights["fc9.sW"]))))

        # fc10
        x = torch.round((self.weights["fc10.sW"]*self.weights["fc10.sX"]/self.weights["fc10.sY"]) * ((torch.round(self.weights["fc10.weight"]/self.weights["fc10.sW"]) @ x) + torch.round(self.weights["fc10.bias"]/(self.weights["fc10.sX"]*self.weights["fc10.sW"]))))


        # here x is a quantized output
        print(f"quantized x : {x}")

        # then we dequant to fp32 - this should be same/similar to output from original model
        x = x * self.weights["fc10.sY"]

    def read_in_weights(self, address):
        # TODO: fill in




    

