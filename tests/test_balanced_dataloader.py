#!/usr/bin/env python3
"""
Test script for the balanced sampling integration in the SAR dataloader.
"""

import sys
import os
sys.path.append('/Data_large/marine/PythonProjects/SAR/sarpyx')

from dataloader.dataloader import SARZarrDataset, SampleFilter

def test_balanced_dataloader():
    """Test the balanced sampling functionality in the dataloader."""
    
    print("=" * 60)
    print("Testing Balanced Sampling in SAR Dataloader")
    print("=" * 60)
    
    # Define data directory 
    data_dir = "/Data_large/marine/PythonProjects/SAR/sarpyx/data"
    
    # Test 1: Standard dataloader without balanced sampling
    print("\n1. Testing standard dataloader (no balanced sampling):")
    print("-" * 50)
    
    sample_filter = SampleFilter(years=[2023], parts=["PT1", "PT2"])
    
    try:
        dataset_standard = SARZarrDataset(
            data_dir=data_dir,
            filters=sample_filter,
            max_products=15,
            use_balanced_sampling=False,
            verbose=True
        )
        
        files_standard = dataset_standard.get_files()
        print(f"Standard selection: {len(files_standard)} files")
        for i, file in enumerate(files_standard[:5]):  # Show first 5
            print(f"  {i+1}. {os.path.basename(file)}")
        if len(files_standard) > 5:
            print(f"  ... and {len(files_standard) - 5} more files")
            
    except Exception as e:
        print(f"Error in standard dataloader: {e}")
        return False
    
    # Test 2: Balanced sampling dataloader
    print("\n2. Testing balanced sampling dataloader:")
    print("-" * 50)
    
    try:
        dataset_balanced = SARZarrDataset(
            data_dir=data_dir,
            filters=sample_filter,
            max_products=15,
            use_balanced_sampling=True,
            verbose=True
        )
        
        files_balanced = dataset_balanced.get_files()
        print(f"Balanced selection: {len(files_balanced)} files")
        for i, file in enumerate(files_balanced[:5]):  # Show first 5
            print(f"  {i+1}. {os.path.basename(file)}")
        if len(files_balanced) > 5:
            print(f"  ... and {len(files_balanced) - 5} more files")
            
    except Exception as e:
        print(f"Error in balanced dataloader: {e}")
        return False
    
    # Test 3: Compare selections
    print("\n3. Comparison:")
    print("-" * 50)
    
    standard_set = set(files_standard)
    balanced_set = set(files_balanced)
    
    common_files = standard_set.intersection(balanced_set)
    only_standard = standard_set - balanced_set
    only_balanced = balanced_set - standard_set
    
    print(f"Files in both selections: {len(common_files)}")
    print(f"Only in standard selection: {len(only_standard)}")
    print(f"Only in balanced selection: {len(only_balanced)}")
    
    if only_balanced:
        print("\nFiles unique to balanced selection:")
        for file in list(only_balanced)[:3]:  # Show first 3
            print(f"  - {os.path.basename(file)}")
    
    # Test 4: Try different filter combinations
    print("\n4. Testing different filter combinations:")
    print("-" * 50)
    
    # Test with different year filter
    try:
        filter_2024 = SampleFilter(years=[2024], parts=["PT1"])
        dataset_2024 = SARZarrDataset(
            data_dir=data_dir,
            filters=filter_2024,
            max_products=10,
            use_balanced_sampling=True,
            verbose=True
        )
        files_2024 = dataset_2024.get_files()
        print(f"2024 balanced selection: {len(files_2024)} files")
        
    except Exception as e:
        print(f"Error with 2024 filter: {e}")
    
    print("\n" + "=" * 60)
    print("Balanced Dataloader Integration Test Complete")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_balanced_dataloader()
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)