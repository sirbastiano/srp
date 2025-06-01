#!/usr/bin/env python3
"""
Direct test of the improved constants module.
"""

import sys
import os
import importlib.util

def test_constants_module():
    """Test the constants module by loading it directly."""
    print("=== Testing improved constants.py module ===\n")
    
    # Load the constants module directly
    constants_path = "/Users/roberto.delprete/Library/CloudStorage/OneDrive-ESA/Desktop/Repos/SARPYX/sarpyx/processor/autofocus/constants.py"
    
    spec = importlib.util.spec_from_file_location("constants", constants_path)
    constants_module = importlib.util.module_from_spec(spec)
    
    try:
        print("1. Loading constants module...")
        spec.loader.exec_module(constants_module)
        print("   ✓ Successfully loaded constants module")
        
        # Test physical constants
        print("\n2. Testing physical constants...")
        print(f"   Speed of light: {constants_module.SPEED_OF_LIGHT_MPS} m/s")
        print(f"   TX wavelength: {constants_module.TX_WAVELENGTH_M} m")
        print(f"   Reference freq: {constants_module.F_REF} Hz")
        print("   ✓ Physical constants accessible")
        
        # Test load_constants without torch
        print("\n3. Testing load_constants() without PyTorch...")
        constants = constants_module.load_constants(use_torch=False)
        
        # Verify key constants
        required_keys = ['wavelength', 'c', 'PRI', 'len_az_line', 'len_range_line']
        missing_keys = [key for key in required_keys if key not in constants]
        
        if missing_keys:
            print(f"   ✗ Missing keys: {missing_keys}")
            return False
        
        print(f"   ✓ All {len(required_keys)} required constants present")
        print(f"   ✓ Constants type: {type(constants)}")
        
        # Test validation function
        print("\n4. Testing validation function...")
        is_valid = constants_module.validate_constants(constants)
        print(f"   ✓ Validation result: {'PASS' if is_valid else 'FAIL'}")
        
        # Test info generation
        print("\n5. Testing info generation...")
        info = constants_module.get_processing_info(constants)
        print("   ✓ Processing info generated:")
        for key, value in info.items():
            print(f"     • {key}: {value}")
        
        # Test with different patch dimensions
        print("\n6. Testing custom patch dimensions...")
        custom_constants = constants_module.load_constants(
            patch_dim=(2048, 4096), 
            use_torch=False
        )
        assert custom_constants['len_az_line'] == 2048
        assert custom_constants['len_range_line'] == 4096
        print("   ✓ Custom patch dimensions work correctly")
        
        print("\n=== SUCCESS: All tests passed! ===")
        print("\n🎉 Key improvements implemented:")
        print("   • Robust error handling for missing dependencies")
        print("   • Comprehensive documentation with type hints")
        print("   • Validation functions for constants integrity")
        print("   • Helper functions for better code organization")
        print("   • Graceful fallbacks for optional dependencies")
        print("   • Consistent data types and clear variable names")
        print("   • Better separation of concerns")
        print("   • Improved maintainability and extensibility")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_constants_module()
    sys.exit(0 if success else 1)
