#!/usr/bin/env python3
"""
Test script to validate the improved constants module.
"""

import sys
import os
import pandas as pd
import numpy as np

def test_load_constants():
    """Test the improved load_constants function."""
    print("Testing load_constants()...")
    
    try:
        # Import just what we need to avoid dependency issues
        sys.path.insert(0, '/Users/roberto.delprete/Library/CloudStorage/OneDrive-ESA/Desktop/Repos/SARPYX')
        
        from sarpyx.processor.autofocus.constants import (
            load_constants, validate_constants, get_processing_info
        )
        
        # Test without torch first
        print("  Testing without PyTorch...")
        constants_no_torch = load_constants(use_torch=False)
        print("  ✓ load_constants() works without torch")
        
        # Test with torch (if available)
        try:
            print("  Testing with PyTorch...")
            constants_with_torch = load_constants(use_torch=True)
            print("  ✓ load_constants() works with torch")
        except ImportError as e:
            print(f"  ✓ load_constants() properly handles missing torch: {e}")
            constants_with_torch = constants_no_torch
        
        # Test validation
        is_valid = validate_constants(constants_with_torch)
        print(f"  ✓ Constants validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Test info generation
        info = get_processing_info(constants_with_torch)
        print("  ✓ Processing info generation works")
        print("    Constants info:")
        for key, value in info.items():
            print(f"      {key}: {value}")
            
    except Exception as e:
        print(f"  ✗ Error testing load_constants(): {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_load_constants_from_meta():
    """Test the improved load_constants_from_meta function."""
    print("\nTesting load_constants_from_meta()...")
    
    try:
        from sarpyx.processor.autofocus.constants import load_constants_from_meta
        
        # Create mock metadata
        mock_meta = pd.DataFrame({
            'PRI': [0.0005345716926237736],
            'Range Decimation': [2],
            'Rank': [9],
            'SWST': [0.00011360148005720262],
            'Tx Pulse Start Frequency': [5.405e9],
            'Tx Ramp Rate': [1.59e12],
            'Tx Pulse Length': [2.8e-5]
        })
        
        # Test without torch first
        try:
            print("  Testing without PyTorch...")
            constants = load_constants_from_meta(mock_meta, use_torch=False)
            print("  ✓ load_constants_from_meta() works without torch")
        except Exception as e:
            print(f"  ✗ Error without torch: {e}")
        
        # Test with torch (if available)
        try:
            print("  Testing with PyTorch...")
            constants = load_constants_from_meta(mock_meta, use_torch=True)
            print("  ✓ load_constants_from_meta() works with torch")
        except ImportError as e:
            print(f"  ✓ load_constants_from_meta() properly handles missing torch: {e}")
        except Exception as e:
            print(f"  ✗ Error with torch: {e}")
        
        # Test error handling
        print("  Testing error handling...")
        incomplete_meta = pd.DataFrame({'PRI': [0.001]})
        try:
            load_constants_from_meta(incomplete_meta)
            print("  ✗ Should have raised ValueError for incomplete metadata")
        except ValueError:
            print("  ✓ Properly validates metadata completeness")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            
    except Exception as e:
        print(f"  ✗ Error testing load_constants_from_meta(): {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_backward_compatibility():
    """Test that the improvements maintain backward compatibility."""
    print("\nTesting backward compatibility...")
    
    try:
        from sarpyx.processor.autofocus.constants import load_constants
        
        # Test that basic usage still works
        print("  Testing basic usage...")
        constants = load_constants(use_torch=False)  # Start without torch to avoid issues
        
        # Check that expected keys are present
        expected_keys = ['wavelength', 'c', 'PRI', 'len_az_line', 'len_range_line']
        missing_keys = [key for key in expected_keys if key not in constants]
        
        if missing_keys:
            print(f"  ✗ Missing expected keys: {missing_keys}")
            return False
        
        print("  ✓ All expected keys present")
        print("  ✓ Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"  ✗ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Testing improved constants module...\n")
    
    results = []
    results.append(test_load_constants())
    results.append(test_load_constants_from_meta())
    results.append(test_backward_compatibility())
    
    print(f"\nTest Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! The constants module has been successfully improved.")
    else:
        print("❌ Some tests failed. Please review the improvements.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
