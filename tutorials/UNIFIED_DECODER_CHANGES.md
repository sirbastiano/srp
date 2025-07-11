# Unified Decoder Modifications Summary

## Overview
Modified the `decode.py` module to create a single unified Zarr file instead of multiple burst files. This change improves data management, compression efficiency, and simplifies downstream processing workflows.

## Key Changes Made

### 1. Modified `decode_radar_file()` function
**Location**: Lines 225-285
**Changes**:
- Instead of creating separate burst dictionaries, now concatenates all burst data
- Concatenates radar echo data along axis 0 (azimuth lines)
- Combines all metadata with added `burst_index` column for traceability
- Creates unified result structure with burst boundary information
- Returns single unified dataset instead of multiple burst datasets

### 2. Updated `_save_data_zarr()` method  
**Location**: Lines 497-540
**Changes**:
- Saves single unified Zarr file named `*_unified.zarr`
- Handles unified data structure (single dataset vs multiple bursts)
- Saves burst info as separate JSON file for reference
- Improved error handling with pickle fallback

### 3. Updated `_save_data()` method
**Location**: Lines 427-450
**Changes**:
- Saves unified pickle files with `*_unified_*` naming convention
- Handles single unified dataset structure
- Saves burst info for reference

### 4. Modified `decode_file()` method
**Location**: Lines 365-385
**Changes**:
- Updated to handle unified data structure
- Creates unified summary instead of individual burst summaries
- Maintains backward compatibility in API

### 5. Updated `main()` function
**Location**: Lines 870-890
**Changes**:
- Handles unified data structure in command-line interface
- Saves unified files with new naming convention

## Benefits

### 1. **Simplified File Management**
- Single Zarr file per product instead of multiple burst files
- Easier to manage and transfer data
- Reduced filesystem overhead

### 2. **Better Compression**
- Continuous data compresses more efficiently than fragmented bursts
- Single compression context across all data
- Reduced storage requirements

### 3. **Improved Performance**
- Fewer I/O operations for reading/writing
- Single file access reduces seek time
- Better memory locality for processing

### 4. **Maintained Traceability**
- `burst_info` preserves original burst boundaries
- `burst_index` column in metadata maintains traceability
- Full backward compatibility for analysis

## Output Structure

### Old (Multi-burst):
```
output/
├── file_burst_0.zarr
├── file_burst_1.zarr
├── file_burst_2.zarr
├── file_info.json
└── file_ephemeris.pkl
```

### New (Unified):
```
output/
├── file_unified.zarr         # Single file with all echo data
├── file_burst_info.json      # Original burst boundaries
├── file_info.json            # Processing metadata
└── file_ephemeris.pkl        # (if using pickle format)
```

## Data Structure Changes

### Unified Dataset Structure:
```python
{
    'echo': np.ndarray,           # Concatenated echo data from all bursts
    'metadata': pd.DataFrame,     # Combined metadata with burst_index column
    'ephemeris': pd.DataFrame,    # Ephemeris data (same for all bursts)
    'burst_info': List[Dict],     # Original burst boundary information
    'original_burst_indexes': List[int]  # Original burst indexes
}
```

### Burst Info Structure:
```python
[
    {
        'burst_id': 0,
        'echo_shape': (lines, samples),
        'metadata_records': count,
        'start_index': 0,
        'end_index': lines_burst_0
    },
    {
        'burst_id': 1,
        'echo_shape': (lines, samples),
        'metadata_records': count,
        'start_index': lines_burst_0,
        'end_index': lines_burst_0 + lines_burst_1
    },
    ...
]
```

## Usage Examples

### Basic Usage:
```python
from sarpyx.processor.core.decode import S1L0Decoder

decoder = S1L0Decoder()
result = decoder.decode_file(
    input_file='radar_data.dat',
    output_dir='./output',
    save_to_zarr=True
)

# Access unified data
unified_data = result['burst_data'][0]  # Single unified dataset
echo_shape = unified_data['echo'].shape
metadata = unified_data['metadata']
burst_boundaries = unified_data['burst_info']
```

### Loading Unified Zarr:
```python
from sarpyx.utils.zarr_utils import ZarrManager

zarr_manager = ZarrManager('output/file_unified.zarr')
echo_data, metadata, ephemeris = zarr_manager.get_slice()
```

## Backward Compatibility

- API remains the same
- Return structure maintains same keys but with unified data
- Existing code will work with minimal modifications
- Command-line interface unchanged

## Testing

Created test files:
- `test_unified_decoder.py` - Test script for verification
- `demo_unified_decoder.py` - Demonstration of changes
- `example_unified_decoder.py` - Practical usage examples

## Performance Considerations

1. **Memory Usage**: Single concatenation may require more memory temporarily
2. **Processing**: Downstream processing becomes simpler with unified data
3. **Storage**: Better compression ratios, reduced storage overhead
4. **I/O**: Fewer file operations, improved read/write performance

## Critical Issue Resolved: Variable Burst Sizes

### Problem Identified:
During testing, we discovered that different bursts can have different numbers of samples per line:
- Burst 0: 22012 samples per line
- Burst 1: 22020 samples per line
- This caused `np.concatenate()` to fail with dimension mismatch error

### Solution Implemented: Adaptive Padding
**Location**: `decode_radar_file()` function, lines 275-295
**Approach**: Zero-padding to maximum width
- Automatically detect the maximum burst width across all bursts
- Pad smaller bursts with zeros on the right side to match the maximum width
- Preserve all original data while enabling successful concatenation
- Track padding information for full traceability

### Padding Information Tracking:
Added to `burst_info` structure:
```python
{
    'burst_id': int,
    'echo_shape': tuple,           # Original shape before padding
    'original_width': int,         # Original number of samples
    'final_width': int,           # Width after padding
    'padded': bool,               # Whether padding was applied
    'pad_width': int,             # Amount of padding added (if any)
    'metadata_records': int,
    'start_index': int,
    'end_index': int
}
```

### Benefits of Padding Approach:
1. **Data Preservation**: No data loss, only zero-padding added
2. **Minimal Impact**: Padding is typically small (e.g., 8 samples out of 22012 = +0.036%)
3. **Processing Safe**: Zeros don't interfere with most SAR processing algorithms
4. **Full Traceability**: Complete information about padding preserved
5. **Automatic Handling**: No user intervention required

### Alternative Approaches Considered:
1. **Truncation**: Would cause data loss (rejected)
2. **Separate Files**: Would defeat unified approach purpose (rejected)
3. **Interpolation**: Too complex and potentially introduces artifacts (rejected)

### Processing Considerations:
- Range processing: Zero padding doesn't affect valid signal processing
- Azimuth processing: Consider burst boundaries and padding regions
- Analysis: Use `original_width` for accurate sample counting
- Filtering: Can exclude padded regions if needed for specific analysis
