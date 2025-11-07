import torch
import torch.nn as nn
import os
import pickle
from sarSSM import sarSSM

checkpoint = torch.load('best.ckpt', map_location='cpu')

state_dict = checkpoint['state_dict']

model_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}

print(model_dict)

model = sarSSM(num_layers=4, d_state=8, activation_function='leakyrelu')

model.load_state_dict(model_dict)

torch.save(model.state_dict(), 'sarSSM_weights.pth')

_ = model.setup_step(1)

print(model.state_dict())

save_dict = {
    k: v for k, v in model.state_dict().items()
    if any(substr in k for substr in ['dA', 'dB', 'dC', 'D', 'output_linear', 'fc1', 'fc2'])
}

print('='*80)

print(save_dict)

# old key : new key
key_mapping = {
    'fc1.weight'  : 'fc1.weight',
    'fc1.bias'    :  'fc1.bias',
    'ssm.0.kernel.kernel.dA' : 'ssm2.A',
    'ssm.0.kernel.kernel.dB' : 'ssm2.B',
    'ssm.0.kernel.kernel.dC' : 'ssm2.C',
    'ssm.0.D'  : 'ssm2.D',
    'ssm.0.output_linear.weight' : 'fc3.weight',
    'ssm.0.output_linear.bias'   : 'fc3.bias',
    'ssm.1.kernel.kernel.dA' : 'ssm4.A',
    'ssm.1.kernel.kernel.dB' : 'ssm4.B',
    'ssm.1.kernel.kernel.dC' : 'ssm4.C',
    'ssm.1.D'  : 'ssm4.D',
    'ssm.1.output_linear.weight' : 'fc5.weight',
    'ssm.1.output_linear.bias'   : 'fc5.bias',
    'ssm.2.kernel.kernel.dA' : 'ssm6.A',
    'ssm.2.kernel.kernel.dB' : 'ssm6.B',
    'ssm.2.kernel.kernel.dC' : 'ssm6.C',
    'ssm.2.D'  : 'ssm6.D',
    'ssm.2.output_linear.weight' : 'fc7.weight',
    'ssm.2.output_linear.bias'   : 'fc7.bias',
    'ssm.3.kernel.kernel.dA' : 'ssm8.A',
    'ssm.3.kernel.kernel.dB' : 'ssm8.B',
    'ssm.3.kernel.kernel.dC' : 'ssm8.C',
    'ssm.3.D'  : 'ssm8.D',
    'ssm.3.output_linear.weight' : 'fc9.weight',
    'ssm.3.output_linear.bias'   : 'fc9.bias',
    'fc2.weight' : 'fc10.weight',
    'fc2.bias'   : 'fc10.bias'
}

print('='*80)


new_dict = {key_mapping[k]: v for k, v in save_dict.items()} # if k in key_mapping}
print(new_dict)

with open("stepssm_weights.pkl", "wb") as f:
    pickle.dump(new_dict, f)

