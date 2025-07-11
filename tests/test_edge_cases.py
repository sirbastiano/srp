#!/usr/bin/env python3
"""Extended test script for edge cases."""

import sys
sys.path.append('/Data_large/marine/PythonProjects/SAR/sarpyx')

from sarpyx.utils.io import calculate_slice_indices

def test_edge_cases():
    """Test various edge cases."""
    
    test_cases = [
        (60510, 15000),  # Original problem case
        (100, 50),       # Small array
        (1000, 300),     # Medium array  
        (50, 25),        # Very small
        (200, 180),      # Large slice relative to array
    ]
    
    for array_height, slice_height in test_cases:
        print(f'\n🧪 Testing: array_height={array_height}, slice_height={slice_height}')
        print('=' * 80)
        
        try:
            slice_info = calculate_slice_indices(array_height, slice_height)
            
            print(f'Generated {len(slice_info)} slices:')
            
            all_valid = True
            total_height = 0
            
            for i, info in enumerate(slice_info):
                drop_start = info['drop_start']
                drop_end = info['drop_end']
                actual_height = info['actual_height']
                total_height += actual_height
                
                print(f'  Slice {i+1}: {info["actual_start"]}-{info["actual_end"]} '
                      f'(height: {actual_height}, drops: {drop_start},{drop_end})')
                
                # Check for negative drops
                if drop_start < 0 or drop_end < 0:
                    print(f'    ❌ NEGATIVE DROPS DETECTED!')
                    all_valid = False
                
                # Check bounds
                if info['actual_start'] < info['original_start']:
                    print(f'    ❌ actual_start < original_start')
                    all_valid = False
                
                if info['actual_end'] > info['original_end']:
                    print(f'    ❌ actual_end > original_end')
                    all_valid = False
            
            # Check coverage
            first_start = slice_info[0]['actual_start'] if slice_info else 0
            last_end = slice_info[-1]['actual_end'] if slice_info else 0
            coverage = last_end - first_start
            coverage_ratio = coverage / array_height
            
            print(f'Coverage: {first_start} to {last_end} = {coverage}/{array_height} ({coverage_ratio:.1%})')
            
            if all_valid:
                print('✅ PASSED')
            else:
                print('❌ FAILED')
                
        except Exception as e:
            print(f'❌ ERROR: {e}')

if __name__ == '__main__':
    test_edge_cases()
