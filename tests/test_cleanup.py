#!/usr/bin/env python3
"""
Test script to verify the cleanup functionality of intermediate Zarr files.
This script tests the cleanup method independently.
"""

import sys
from pathlib import Path
import shutil
import tempfile
import logging

# Add the sarpyx module to path
sys.path.insert(0, '/Data_large/marine/PythonProjects/SAR/sarpyx')

from sarpyx.processor.core.decode import S1L0Decoder

def test_cleanup_functionality():
    """Test the cleanup functionality with mock intermediate files."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock intermediate Zarr files
        file_stem = "test_file"
        
        # Create mock burst directories
        burst_0_dir = temp_path / f"{file_stem}_burst_0.zarr"
        burst_1_dir = temp_path / f"{file_stem}_burst_1.zarr"
        burst_2_dir = temp_path / f"{file_stem}_burst_2.zarr"
        
        # Create the directories
        burst_0_dir.mkdir()
        burst_1_dir.mkdir()
        burst_2_dir.mkdir()
        
        # Create some mock files inside
        (burst_0_dir / "mock_data.txt").write_text("mock data")
        (burst_1_dir / "mock_data.txt").write_text("mock data")
        (burst_2_dir / "mock_data.txt").write_text("mock data")
        
        # Create a unified zarr file (should NOT be deleted)
        unified_dir = temp_path / f"{file_stem}_unified.zarr"
        unified_dir.mkdir()
        (unified_dir / "unified_data.txt").write_text("unified data")
        
        # Create some other files (should NOT be deleted)
        (temp_path / f"{file_stem}_info.json").write_text("{}")
        (temp_path / f"{file_stem}_burst_info.json").write_text("{}")
        
        print(f"Created test files in: {temp_path}")
        print("Files before cleanup:")
        for f in sorted(temp_path.glob("*")):
            print(f"  {f.name}")
        
        # Test the cleanup
        decoder = S1L0Decoder(log_level=logging.DEBUG)
        decoder._cleanup_intermediate_zarr_files(temp_path, file_stem)
        
        print("\nFiles after cleanup:")
        remaining_files = list(temp_path.glob("*"))
        for f in sorted(remaining_files):
            print(f"  {f.name}")
        
        # Verify results
        assert not burst_0_dir.exists(), "burst_0 should have been deleted"
        assert not burst_1_dir.exists(), "burst_1 should have been deleted"
        assert not burst_2_dir.exists(), "burst_2 should have been deleted"
        assert unified_dir.exists(), "unified file should NOT have been deleted"
        assert (temp_path / f"{file_stem}_info.json").exists(), "info.json should NOT have been deleted"
        assert (temp_path / f"{file_stem}_burst_info.json").exists(), "burst_info.json should NOT have been deleted"
        
        print("\n✅ Cleanup test passed!")
        print("- Intermediate burst Zarr files were successfully removed")
        print("- Unified Zarr file was preserved")
        print("- Other files were preserved")

if __name__ == "__main__":
    test_cleanup_functionality()
