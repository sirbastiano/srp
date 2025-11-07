import torch

# Load the checkpoint
checkpoint = torch.load('best.ckpt', map_location='cpu')

# Extract state_dict (contains model weights)
state_dict = checkpoint['state_dict']

# Write weights to text file
with open('weights.txt', 'w') as f:
    for name, param in state_dict.items():
        f.write(f"\n{'='*80}\n")
        f.write(f"Parameter: {name}\n")
        f.write(f"Shape: {param.shape}\n")
        f.write(f"Dtype: {param.dtype}\n")
        f.write(f"Values:\n{param}\n")

print(f"Extracted {len(state_dict)} parameters to weights.txt")
