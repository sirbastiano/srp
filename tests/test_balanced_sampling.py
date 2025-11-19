#!/usr/bin/env python3
"""
Test to verify balanced sampling with small max_products works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataloader.utils import get_balanced_sample_files, SampleFilter

# Test configuration
data_dir = "/Data_large/marine/PythonProjects/SAR/sarpyx/data"
config_path = "/Data_large/marine/PythonProjects/SAR/sarpyx/balanced_samples"

print("="*80)
print("TESTING BALANCED SAMPLING WITH SMALL max_products")
print("="*80)

# Test with different max_products values
test_values = [5, 10, 20, 50]

for max_products in test_values:
    print(f"\n{'='*80}")
    print(f"Testing with max_products = {max_products}")
    print(f"{'='*80}")
    
    sample_filter = SampleFilter(
        years=[2023],
        polarizations=["vv"]
    )
    
    balanced_files = get_balanced_sample_files(
        max_samples=max_products,
        data_dir=data_dir,
        sample_filter=sample_filter,
        config_path=config_path,
        split_type="train",
        verbose=True
    )
    
    print(f"\n✅ Result: Got {len(balanced_files)} files (requested {max_products})")
    
    if len(balanced_files) < max_products:
        print(f"⚠️  Warning: Got fewer files than requested")
    else:
        print(f"✓ Success: Got the requested number of files!")
    
    # Show first few files
    print(f"\nFirst 3 files:")
    for i, f in enumerate(balanced_files[:3]):
        print(f"  {i+1}. {os.path.basename(f)}")

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
