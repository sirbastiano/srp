#!/usr/bin/env python3
"""
Simple test for the balanced sampling integration
"""

import sys
import os
sys.path.append('/Data_large/marine/PythonProjects/SAR/sarpyx')

from dataloader.utils import get_balanced_sample_files
from dataloader.dataloader import SampleFilter

def test_simple_balanced_sampling():
    """Test the get_balanced_sample_files function directly."""
    
    print("Testing get_balanced_sample_files function:")
    
    # Test with basic filter
    sample_filter = SampleFilter(years=[2023], parts=["PT1", "PT2"])
    
    try:
        balanced_files = get_balanced_sample_files(
            max_samples=10, 
            sample_filter=sample_filter, 
            verbose=True
        )
        
        print(f"\nBalanced files result: {len(balanced_files) if balanced_files else 0} files")
        if balanced_files:
            for i, file in enumerate(balanced_files[:5]):
                print(f"  {i+1}. {os.path.basename(file)}")
            if len(balanced_files) > 5:
                print(f"  ... and {len(balanced_files) - 5} more files")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_balanced_sampling()
    if success:
        print("\n✅ Simple test completed successfully!")
    else:
        print("\n❌ Simple test failed!")
        sys.exit(1)