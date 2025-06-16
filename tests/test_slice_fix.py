#!/usr/bin/env python3
"""Test script to verify the slice fix works correctly."""

import sys
sys.path.append('/Data_large/marine/PythonProjects/SAR/sarpyx')

from sarpyx.utils.io import calculate_slice_indices

def test_slice_indices():
    """Test that slice indices are calculated correctly without negative drops."""
    
    # Test case from the user's problem
    array_height = 60510
    slice_height = 15000
    
    print(f'Testing with array_height={array_height}, slice_height={slice_height}')
    print()
    
    slice_info = calculate_slice_indices(array_height, slice_height)
    
    # Create a nice table
    print('┌─────────┬─────────────┬───────────┬─────────────┬───────────┬─────────────┬─────────────┐')
    print('│ Slice # │ Orig Start  │ Orig End  │ Actual Start│ Actual End│ Drop Start  │ Drop End    │')
    print('├─────────┼─────────────┼───────────┼─────────────┼───────────┼─────────────┼─────────────┤')
    
    all_valid = True
    for info in slice_info:
        slice_num = info['slice_index'] + 1
        orig_start = info['original_start']
        orig_end = info['original_end']
        actual_start = info['actual_start']
        actual_end = info['actual_end']
        drop_start = info['drop_start']
        drop_end = info['drop_end']
        
        print(f'│{slice_num:^9}│{orig_start:^13}│{orig_end:^11}│{actual_start:^13}│{actual_end:^11}│{drop_start:^13}│{drop_end:^13}│')
        
        # Validate constraints
        if drop_start < 0 or drop_end < 0:
            print(f'ERROR: Slice {slice_num} has negative drops!')
            all_valid = False
        
        if actual_start < orig_start:
            print(f'ERROR: Slice {slice_num} actual_start < orig_start!')
            all_valid = False
        
        if actual_end > orig_end:
            print(f'ERROR: Slice {slice_num} actual_end > orig_end!')
            all_valid = False
    
    print('└─────────┴─────────────┴───────────┴─────────────┴───────────┴─────────────┴─────────────┘')
    print()
    
    if all_valid:
        print('✅ All slice indices are valid! No negative drops detected.')
    else:
        print('❌ Invalid slice indices detected!')
    
    # Check coverage
    total_covered = sum(info['actual_height'] for info in slice_info)
    first_start = slice_info[0]['actual_start'] if slice_info else 0
    last_end = slice_info[-1]['actual_end'] if slice_info else 0
    
    print(f'Coverage: {first_start} to {last_end} (total height: {last_end - first_start})')
    print(f'Array height: {array_height}')
    print(f'Coverage ratio: {(last_end - first_start) / array_height:.2%}')
    
    return all_valid

if __name__ == '__main__':
    test_slice_indices()
