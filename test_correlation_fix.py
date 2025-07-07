#!/usr/bin/env python3
"""
Test script to verify the correlation broadcasting fix.
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sarpyx.processor.core.spectrum import correlation

def test_correlation_fix():
    """Test the correlation function with different array sizes."""
    
    # Create test data that would cause the original broadcasting error
    # x1 shape: (azimuth, range) 
    azimuth_size = 100
    range_size = 25444  # From the error message analysis
    nomchip_size = 4549  # From the error message analysis
    
    print(f'Creating test data:')
    print(f'  x1 shape: ({azimuth_size}, {range_size})')
    print(f'  nomchip size: {nomchip_size}')
    
    # Create random complex test data
    x1 = np.random.randn(azimuth_size, range_size) + 1j * np.random.randn(azimuth_size, range_size)
    nomchip = np.random.randn(nomchip_size) + 1j * np.random.randn(nomchip_size)
    
    # Expected correlation length
    expected_corr_length = range_size + nomchip_size - 1
    expected_output_length = nomchip_size  # After extracting valid part
    
    print(f'Expected full correlation length: {expected_corr_length}')
    print(f'Expected output length after extraction: {expected_output_length}')
    
    try:
        # Test numpy backend
        print('\\nTesting numpy backend...')
        result_numpy = correlation(x1, nomchip, backend='numpy', verbose=True)
        print(f'Numpy result shape: {result_numpy.shape}')
        assert result_numpy.shape == (expected_output_length, azimuth_size), \
            f'Expected shape ({expected_output_length}, {azimuth_size}), got {result_numpy.shape}'
        print('✅ Numpy backend test passed!')
        
        # Test scipy backend
        print('\\nTesting scipy backend...')
        result_scipy = correlation(x1, nomchip, backend='scipy', verbose=True)
        print(f'Scipy result shape: {result_scipy.shape}')
        assert result_scipy.shape == (expected_output_length, azimuth_size), \
            f'Expected shape ({expected_output_length}, {azimuth_size}), got {result_scipy.shape}'
        print('✅ Scipy backend test passed!')
        
        # Verify results are similar (allowing for numerical differences)
        diff = np.abs(result_numpy - result_scipy)
        max_diff = np.max(diff)
        print(f'\\nMax difference between numpy and scipy results: {max_diff}')
        assert max_diff < 1e-10, f'Results differ too much: {max_diff}'
        print('✅ Results are consistent between backends!')
        
        print('\\n🎉 All tests passed! Broadcasting error is fixed.')
        return True
        
    except Exception as e:
        print(f'\\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_correlation_fix()
    sys.exit(0 if success else 1)
