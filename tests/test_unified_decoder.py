#!/usr/bin/env python3
"""
Test script to verify the unified decoding functionality.
"""

import sys
from pathlib import Path

# Add the sarpyx package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sarpyx.processor.core.decode import S1L0Decoder

def test_unified_decoder():
    """Test the unified decoder functionality."""
    print("Testing unified decoder...")
    
    # Create decoder instance
    decoder = S1L0Decoder()
    
    # Test with a sample file (you'll need to provide a real file path)
    test_file = Path("path/to/your/test/file.dat")  # Replace with actual file path
    output_dir = Path("./test_output")
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        print("Please provide a valid .dat file path in the test script.")
        return False
    
    try:
        # Test headers-only mode
        print("\n1. Testing headers-only mode...")
        result = decoder.decode_file(
            input_file=test_file,
            output_dir=output_dir / "headers_only",
            headers_only=True,
            apply_transformations=True
        )
        print(f"Headers extracted: {result.get('num_records', 0)} records")
        
        # Test full decoding with unified output
        print("\n2. Testing full decoding with unified output...")
        result = decoder.decode_file(
            input_file=test_file,
            output_dir=output_dir / "unified_pickle",
            headers_only=False,
            save_to_zarr=False,
            apply_transformations=True
        )
        
        unified_summary = result.get('unified_summary', {})
        print(f"Unified data shape: {unified_summary.get('total_echo_shape', 'N/A')}")
        print(f"Total metadata records: {unified_summary.get('total_metadata_records', 'N/A')}")
        print(f"Original bursts: {unified_summary.get('original_bursts', 'N/A')}")
        
        # Test full decoding with Zarr output
        print("\n3. Testing full decoding with unified Zarr output...")
        result = decoder.decode_file(
            input_file=test_file,
            output_dir=output_dir / "unified_zarr",
            headers_only=False,
            save_to_zarr=True,
            apply_transformations=True
        )
        
        print(f"Zarr file saved to: {result.get('saved_to', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_unified_decoder()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
