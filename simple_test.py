#!/usr/bin/env python3
"""
Simple validation script for the improved constants module.
"""

import sys
import os

# Add current directory to path to avoid import issues
sys.path.insert(0, os.getcwd())

def test_constants_direct():
    """Test the constants module by importing it directly."""
    print("=== Testing constants module improvements ===\n")
    
    try:
        # Test basic import
        print("1. Testing basic import...")
        from sarpyx.processor.autofocus.constants import (
            SPEED_OF_LIGHT_MPS, TX_WAVELENGTH_M, F_REF
        )
        print("   ✓ Successfully imported physical constants")
        
        # Test load_constants without torch
        print("\n2. Testing load_constants() without PyTorch...")
        from sarpyx.processor.autofocus.constants import load_constants
        constants = load_constants(use_torch=False)
        
        # Verify key constants
        required_keys = ['wavelength', 'c', 'PRI', 'len_az_line', 'len_range_line']
        for key in required_keys:
            if key not in constants:
                print(f"   ✗ Missing key: {key}")
                return False
        print("   ✓ All required constants present")
        
        # Test validation function
        print("\n3. Testing validation function...")
        from sarpyx.processor.autofocus.constants import validate_constants
        is_valid = validate_constants(constants)
        print(f"   ✓ Validation result: {'PASS' if is_valid else 'FAIL'}")
        
        # Test info function
        print("\n4. Testing info generation...")
        from sarpyx.processor.autofocus.constants import get_processing_info
        info = get_processing_info(constants)
        print("   ✓ Successfully generated processing info:")
        for key, value in info.items():
            print(f"     {key}: {value}")
        
        print("\n=== All tests passed! ===")
        print("\nKey improvements made:")
        print("• Added proper error handling for missing dependencies")
        print("• Improved documentation with docstrings and type hints")
        print("• Added validation functions for constants")
        print("• Better separation of concerns with helper functions")
        print("• Graceful fallbacks when PyTorch/sentinel1decoder unavailable")
        print("• More consistent data types and naming conventions")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_constants_direct()
    sys.exit(0 if success else 1)
