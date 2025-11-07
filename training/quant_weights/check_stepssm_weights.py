import torch
import os
import sys

from stepssm import stepSSM

model = stepSSM()

print(model.state_dict())