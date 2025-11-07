"""
Script to load checkpoint, instantiate sarSSM model, setup step, and export weights
"""
import torch
import os
from sarSSM import sarSSM

def print_model_weights_to_file(model, output_file='model_weights.txt'):
    """
    Print all model weights and parameters to a text file

    Args:
        model: PyTorch model
        output_file: Path to output text file
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL WEIGHTS AND PARAMETERS\n")
        f.write("="*80 + "\n\n")

        # Print model architecture summary
        f.write("MODEL ARCHITECTURE:\n")
        f.write(str(model) + "\n\n")
        f.write("="*80 + "\n\n")

        # Print all named parameters
        f.write("PARAMETERS:\n")
        f.write("-"*80 + "\n\n")

        total_params = 0
        for name, param in model.named_parameters():
            f.write(f"Parameter: {name}\n")
            f.write(f"  Shape: {param.shape}\n")
            f.write(f"  Dtype: {param.dtype}\n")
            f.write(f"  Device: {param.device}\n")
            f.write(f"  Requires grad: {param.requires_grad}\n")
            f.write(f"  Values:\n")
            f.write(f"{param.data}\n\n")
            total_params += param.numel()

        f.write("="*80 + "\n")
        f.write(f"TOTAL PARAMETERS: {total_params:,}\n")
        f.write("="*80 + "\n\n")

        # Print all named buffers
        f.write("BUFFERS:\n")
        f.write("-"*80 + "\n\n")

        for name, buffer in model.named_buffers():
            f.write(f"Buffer: {name}\n")
            f.write(f"  Shape: {buffer.shape}\n")
            f.write(f"  Dtype: {buffer.dtype}\n")
            f.write(f"  Device: {buffer.device}\n")
            f.write(f"  Values:\n")
            f.write(f"{buffer.data}\n\n")

        f.write("="*80 + "\n\n")

        # Print dynamically created tensor attributes (like dA, dB, dC from setup_step)
        f.write("DYNAMIC TENSOR ATTRIBUTES (created during setup_step):\n")
        f.write("-"*80 + "\n\n")

        for module_name, module in model.named_modules():
            # Get all attributes of this module
            for attr_name in dir(module):
                # Skip private attributes and methods
                if attr_name.startswith('_'):
                    continue

                try:
                    attr = getattr(module, attr_name)
                    # Check if it's a tensor and not already a parameter or buffer
                    if isinstance(attr, torch.Tensor):
                        # Check if it's not a parameter or buffer
                        is_param = any(attr is p for p in module.parameters(recurse=False))
                        is_buffer = attr_name in dict(module.named_buffers(recurse=False))

                        if not is_param and not is_buffer:
                            full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                            f.write(f"Tensor Attribute: {full_name}\n")
                            f.write(f"  Shape: {attr.shape}\n")
                            f.write(f"  Dtype: {attr.dtype}\n")
                            f.write(f"  Device: {attr.device}\n")
                            f.write(f"  Values:\n")
                            f.write(f"{attr}\n\n")
                except (AttributeError, RuntimeError):
                    # Skip attributes that can't be accessed
                    pass

        f.write("="*80 + "\n")


def main():
    # Configuration
    checkpoint_path = 'best.ckpt'
    output_file = 'model_weights_after_setup_step.txt'

    # Model configuration
    num_layers = 4
    d_state = 8
    activation_function = 'leakyrelu'
    batch_size = 1  # Default batch size for setup_step

    print("="*80)
    print("SARSSM MODEL WEIGHT EXPORT SCRIPT")
    print("="*80)

    # Load checkpoint
    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"   Checkpoint loaded successfully!")
    print(f"   Available keys in checkpoint: {list(checkpoint.keys())}")

    # Check for model weights
    if 'state_dict' in checkpoint:
        state_dict_keys = checkpoint['state_dict'].keys()
        print(f"\n   Found {len(state_dict_keys)} keys in state_dict")

        # Show sample of keys with different prefixes
        starting_keys = [k for k in state_dict_keys if k.startswith('starting_model.')]
        student_keys = [k for k in state_dict_keys if k.startswith('student_model.')]
        teacher_keys = [k for k in state_dict_keys if k.startswith('teacher_model.')]

        print(f"   - starting_model keys: {len(starting_keys)}")
        print(f"   - student_model keys: {len(student_keys)}")
        print(f"   - teacher_model keys: {len(teacher_keys)}")

    # Instantiate sarSSM model
    print(f"\n2. Instantiating sarSSM model with:")
    print(f"   - num_layers: {num_layers}")
    print(f"   - d_state: {d_state}")
    print(f"   - activation_function: {activation_function}")

    model = sarSSM(
        num_layers=num_layers,
        d_state=d_state,
        activation_function=activation_function
    )
    print("   Model instantiated successfully!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Load starting_model weights from checkpoint
    print(f"\n3. Loading starting_model weights from checkpoint...")
    state_dict = checkpoint['state_dict']

    # Filter to get only starting_model weights and remove the prefix
    starting_model_state_dict = {k.replace('starting_model.', ''): v
                                  for k, v in state_dict.items()
                                  if k.startswith('starting_model.')}

    if len(starting_model_state_dict) == 0:
        print("   WARNING: No starting_model weights found in checkpoint!")
        print("   Available prefixes in checkpoint:")
        prefixes = set(k.split('.')[0] for k in state_dict.keys())
        for prefix in prefixes:
            print(f"   - {prefix}")
    else:
        print(f"   Found {len(starting_model_state_dict)} starting_model parameters")
        model.load_state_dict(starting_model_state_dict)
        print("   Weights loaded successfully!")

    # Call setup_step
    print(f"\n4. Calling setup_step with batch_size={batch_size}")
    states = model.setup_step(batch_size)
    print(f"   setup_step completed successfully!")
    print(f"   Returned {len(states)} layer states")
    for i, state in enumerate(states):
        print(f"   - Layer {i} state shape: {state.shape}")

    # Export weights to file
    print(f"\n5. Exporting model weights to: {output_file}")
    print_model_weights_to_file(model, output_file)
    print(f"   Weights exported successfully!")

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)
    print(f"\nModel weights have been saved to: {output_file}")


if __name__ == "__main__":
    main()
