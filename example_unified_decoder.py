#!/usr/bin/env python3
"""
Practical example showing how to use the unified decoder.
"""

import sys
from pathlib import Path

# Add the sarpyx package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sarpyx.processor.core.decode import S1L0Decoder

def example_usage():
    """Show practical usage examples."""
    print("=" * 70)
    print("UNIFIED DECODER - PRACTICAL USAGE EXAMPLES")
    print("=" * 70)
    
    print("\n📋 EXAMPLE 1: Quick header extraction")
    print("=" * 40)
    print("""
decoder = S1L0Decoder()
result = decoder.decode_file(
    input_file='s1a-iw-raw-s-vv-20230101.dat',
    headers_only=True,
    apply_transformations=True
)

# Access header information
print(f"Number of records: {result['num_records']}")
headers_df = result['headers']
print(f"Signal types: {headers_df['signal_type_name'].unique()}")
""")
    
    print("\n📋 EXAMPLE 2: Full decoding with unified Zarr output")
    print("=" * 50)
    print("""
decoder = S1L0Decoder()
result = decoder.decode_file(
    input_file='s1a-iw-raw-s-vv-20230101.dat',
    output_dir='./processed_data',
    save_to_zarr=True,
    apply_transformations=True
)

# Access unified data information
unified_summary = result['unified_summary']
print(f"Total echo shape: {unified_summary['total_echo_shape']}")
print(f"Original bursts: {unified_summary['original_bursts']}")

# Files created:
# - processed_data/s1a-iw-raw-s-vv-20230101_unified.zarr
# - processed_data/s1a-iw-raw-s-vv-20230101_burst_info.json
# - processed_data/s1a-iw-raw-s-vv-20230101_info.json
""")
    
    print("\n📋 EXAMPLE 3: Loading unified Zarr data")
    print("=" * 40)
    print("""
from sarpyx.utils.zarr_utils import ZarrManager

# Load unified Zarr file
zarr_manager = ZarrManager('processed_data/file_unified.zarr')

# Get data slice
echo_data, metadata, ephemeris = zarr_manager.get_slice(
    rows=(0, 1000),    # First 1000 lines
    cols=(0, 500)      # First 500 samples
)

print(f"Echo data shape: {echo_data.shape}")
print(f"Metadata records: {len(metadata)}")
""")
    
    print("\n📋 EXAMPLE 4: Accessing burst boundaries")
    print("=" * 40)
    print("""
import json

# Load burst info to understand original burst structure
with open('processed_data/file_burst_info.json', 'r') as f:
    burst_info = json.load(f)

for burst in burst_info:
    print(f"Burst {burst['burst_id']}: "
          f"lines {burst['start_index']}-{burst['end_index']}, "
          f"shape {burst['echo_shape']}")
""")
    
    print("\n📋 EXAMPLE 5: Command-line usage")
    print("=" * 35)
    print("""
# Using the module directly
python -m sarpyx.processor.core.decode \\
    -i input_file.dat \\
    -o output_directory \\
    -v  # verbose mode

# This will create:
# - output_directory/input_file_unified_echo.pkl
# - output_directory/input_file_unified_metadata.pkl
# - output_directory/input_file_ephemeris.pkl
# - output_directory/input_file_burst_info.pkl
""")
    
    print("\n💡 KEY DIFFERENCES FROM OLD VERSION:")
    print("- decode_radar_file() now returns list with single unified dataset")
    print("- burst_data[0] contains all concatenated echo data")
    print("- Metadata includes 'burst_index' column for traceability")
    print("- burst_info provides original burst boundaries")
    print("- Single Zarr file contains all data with better compression")
    
    print("\n" + "=" * 70)

def padding_support_example():
    """Demonstrate padding support for variable burst sizes."""
    print("\n📋 EXAMPLE 3: Handling variable burst sizes with padding")
    print("=" * 60)
    print("""
# When bursts have different numbers of samples per line:
# Burst 0: 22012 samples, Burst 1: 22020 samples
# The decoder automatically applies padding

decoder = S1L0Decoder()
result = decoder.decode_file(
    input_file='data_with_variable_bursts.dat',
    output_dir='./padded_output',
    save_to_zarr=True
)

# Check padding information
unified_data = result['burst_data'][0]
burst_info = unified_data['burst_info']

for burst in burst_info:
    if burst['padded']:
        print(f"Burst {burst['burst_id']}: "
              f"{burst['original_width']} → {burst['final_width']} "
              f"(+{burst['pad_width']} padding)")
    else:
        print(f"Burst {burst['burst_id']}: "
              f"{burst['final_width']} samples (no padding)")

# The unified echo array is now: (total_lines, max_width)
echo_shape = unified_data['echo'].shape
print(f"Unified echo shape: {echo_shape}")
""")

def best_practices():
    """Show best practices when working with padded data."""
    print("\n🎯 BEST PRACTICES FOR PADDED DATA")
    print("=" * 40)
    print("""
1. Always check for padding:
   if burst_info[i]['padded']:
       # Consider padding in your processing
       original_width = burst_info[i]['original_width']
       # Use only valid samples: data[:, :original_width]

2. Processing considerations:
   # Range compression: zeros don't affect valid data
   # Azimuth compression: consider burst boundaries
   
3. Analysis workflows:
   # Filter out padded regions if needed
   valid_samples = burst_info[i]['original_width']
   valid_data = echo_data[:lines, :valid_samples]

4. Burst boundary tracking:
   start_line = burst_info[i]['start_index']
   end_line = burst_info[i]['end_index']
   burst_data = unified_echo[start_line:end_line, :]
""")

if __name__ == "__main__":
    example_usage()
    padding_support_example()
    best_practices()
    print("\n🎉 Ready to use the unified decoder with padding support!")
