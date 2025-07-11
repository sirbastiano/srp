#!/usr/bin/env python3
"""
Demo script showing the difference between old multi-burst and new unified decoding.
"""

import sys
from pathlib import Path

# Add the sarpyx package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sarpyx.processor.core.decode import S1L0Decoder

def demo_unified_vs_multi_burst():
    """Demonstrate the difference between unified and multi-burst decoding."""
    print("=" * 70)
    print("SAR DECODER UNIFIED MODE DEMONSTRATION")
    print("=" * 70)
    
    print("\n📝 CHANGES MADE:")
    print("1. Modified extract_echo_bursts to concatenate all bursts")
    print("2. Updated decode_radar_file to create unified dataset")
    print("3. Modified _save_data_zarr to save single unified Zarr file")
    print("4. Updated _save_data to save unified pickle files")
    print("5. Added burst_info tracking for original burst boundaries")
    
    print("\n🔧 NEW BEHAVIOR:")
    print("- All bursts are concatenated into a single echo array")
    print("- All metadata is combined with burst_index column")
    print("- Single Zarr file saved as '*_unified.zarr'")
    print("- Burst info saved separately as '*_burst_info.json'")
    print("- Pickle files saved as '*_unified_*.pkl'")
    
    print("\n📊 OLD vs NEW OUTPUT:")
    print("OLD (Multi-burst):")
    print("  - file_burst_0.zarr")
    print("  - file_burst_1.zarr")
    print("  - file_burst_2.zarr")
    print("  - ...")
    
    print("\nNEW (Unified):")
    print("  - file_unified.zarr      (single file with all data)")
    print("  - file_burst_info.json   (original burst boundaries)")
    print("  - file_info.json         (processing metadata)")
    
    print("\n💡 BENEFITS:")
    print("✅ Simpler file management - single Zarr file per product")
    print("✅ Easier data access - no need to concatenate bursts")
    print("✅ Better compression - continuous data compresses better")
    print("✅ Maintains traceability - burst info preserved")
    print("✅ Backward compatible - same API, different output structure")
    
    print("\n🎯 USAGE EXAMPLES:")
    print("decoder = S1L0Decoder()")
    print("result = decoder.decode_file(")
    print("    input_file='radar_data.dat',")
    print("    output_dir='./output',")
    print("    save_to_zarr=True  # Creates unified Zarr file")
    print(")")
    
    print("\n📈 EXPECTED PERFORMANCE:")
    print("- Reduced I/O operations (single large file vs multiple)")
    print("- Better compression ratios due to continuous data")
    print("- Simpler downstream processing workflows")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    demo_unified_vs_multi_burst()
