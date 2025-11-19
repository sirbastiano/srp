#!/usr/bin/env python3
"""
Simple test to verify tensor dimension fixes in SSM step method
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

# Since the files are in /Data/gdaga/, use that path
sys.path.insert(0, '/Data/gdaga/sarpyx_new/sarpyx')
from model.SSMs.SSM import sarSSMFinal, S4D, SSKernelDiag

def test_tensor_shapes():
    """Test tensor shapes in SSM step methods"""
    
    print("Testing tensor dimension fixes...")
    
    # Test sarSSMFinal model creation
    print("\n1. Creating sarSSMFinal model...")
    model = sarSSMFinal(
        num_layers=2,
        input_dim=3,
        model_dim=2,
        state_dim=4,
        activation_function='relu'
    )
    print(f"Model created successfully: {model.__class__.__name__}")
    
    # Test input tensor
    batch_size = 32
    seq_len = 10
    input_dim = 3
    
    # Create test input
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"\n2. Input tensor shape: {x.shape}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    try:
        output = model(x)
        print(f"Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False
    
    # Test step mode
    print("\n4. Testing step method...")
    try:
        model.setup_step()
        
        # Take first timestep as input
        u = x[:, 0, :]  # (batch_size, input_dim)
        print(f"Step input shape: {u.shape}")
        
        # Initialize state as None (should be created automatically)
        state = None
        
        # Run step
        step_output, new_state = model.step(u, state)
        print(f"Step method successful!")
        print(f"Step output shape: {step_output.shape}")
        print(f"Number of layer states: {len(new_state)}")
        print(f"First layer state shape: {new_state[0].shape}")
        
        # Run another step to test state passing
        u2 = x[:, 1, :]
        step_output2, new_state2 = model.step(u2, new_state)
        print(f"Second step successful! Output shape: {step_output2.shape}")
        
    except Exception as e:
        print(f"Step method failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed! Tensor dimension fixes are working.")
    return True

def test_s4d_layer():
    """Test S4D layer specifically"""
    print("\n=== Testing S4D layer ===")
    
    # Create S4D layer
    layer = S4D(d_model=2, d_state=4, complex=False)
    
    # Test input
    batch_size = 32
    seq_len = 10
    model_dim = 2
    
    x = torch.randn(batch_size, seq_len, model_dim)
    print(f"S4D input shape: {x.shape}")
    
    # Test forward
    try:
        output, _ = layer(x)
        print(f"S4D forward successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"S4D forward failed: {e}")
        return False
    
    # Test step mode
    try:
        layer.setup_step()
        
        u = x[:, 0, :]  # (batch_size, model_dim) 
        state = layer.default_state(batch_size)
        print(f"Default state shape: {state.shape}")
        
        step_out, new_state = layer.step(u, state)
        print(f"S4D step successful! Output: {step_out.shape}, State: {new_state.shape}")
        
    except Exception as e:
        print(f"S4D step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("Testing SSM Tensor Dimension Fixes")
    print("="*60)
    
    # Test S4D layer first
    s4d_success = test_s4d_layer()
    
    if s4d_success:
        # Test full model
        success = test_tensor_shapes()
        
        if success:
            print("\n🎉 All tensor dimension issues have been resolved!")
            sys.exit(0)
        else:
            print("\n❌ Some issues remain.")
            sys.exit(1)
    else:
        print("\n❌ S4D layer issues need to be resolved first.")
        sys.exit(1)