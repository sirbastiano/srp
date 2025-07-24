#!/usr/bin/env python3
"""
SAR Data Focusing Script

This script processes SAR data by:
1. Loading raw data from a Zarr file
2. Processing it in slices using CoarseRDA
3. Saving intermediate results
4. Concatenating all slices into final output
5. Cleaning up temporary files

Usage:
    python focus_sar_data.py <input_zarr_file> [options]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging
import shutil
import numpy as np
import pandas as pd

# Add sarpyx to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from sarpyx.processor.core.focus import CoarseRDA
from sarpyx.utils.zarr_utils import ZarrManager, dask_slice_saver, concatenate_slices_efficient
from sarpyx.utils.io import calculate_slice_indices
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__BUFFER_SLICE_HEIGHT__ = 15000  # Buffer size for each slice in rows


def setup_directories(output_dir: Path) -> tuple[Path, Path]:
    """
    Setup output and temporary directories.
    
    Args:
        output_dir (Path): Base output directory path
        
    Returns:
        tuple[Path, Path]: (output_dir, tmp_dir) paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, tmp_dir

def focalize_slice(raw_data: Dict[str, Any], verbose: bool = False, to_dict: bool = False) -> Dict[str, Any]:
    """Process a single slice of raw SAR data.
    
    Args:
        raw_data: Raw data dictionary containing 'echo', 'metadata', and 'ephemeris'
        verbose: Enable verbose output
        to_dict: Convert metadata and ephemeris to dict format
        
    Returns:
        Processed data dictionary containing 'raw', 'rc', 'rcmc', 'az', 'metadata', and 'ephemeris'
    """
    logger.info('Processing slice...')
    
    # Initialize processor
    processor = CoarseRDA(
        raw_data=raw_data,
        verbose=verbose,
    )
    logger.info(f'🛠️ Processor initialized with raw data of shape: {raw_data["echo"].shape}')
    
    # Focus the data
    processor.data_focus()
    
    # Extract processed data
    raw = processor.raw_data
    rc = processor.range_compressed_data
    rcmc = processor.rcmc_data
    az = processor.azimuth_compressed_data
    
    # Extract metadata and ephemeris from raw_data
    metadata = raw_data['metadata']
    ephemeris = raw_data['ephemeris']
    
    # Convert to dict if requested
    if to_dict:
        if hasattr(metadata, 'to_dict'):
            metadata = metadata.to_dict('records')
        if hasattr(ephemeris, 'to_dict'):
            ephemeris = ephemeris.to_dict('records')
    
    # Validate data types - be more flexible
    if to_dict:
        assert isinstance(metadata, (dict, list)), f'Expected metadata to be dict or list, got {type(metadata)}'
        assert isinstance(ephemeris, (dict, list)), f'Expected ephemeris to be dict or list, got {type(ephemeris)}'
    else:
        # Allow DataFrames and dicts
        assert hasattr(metadata, 'iloc') or isinstance(metadata, (dict, list)), f'Expected metadata to be DataFrame, dict, or list, got {type(metadata)}'
        assert hasattr(ephemeris, 'iloc') or isinstance(ephemeris, (dict, list)), f'Expected ephemeris to be DataFrame, dict, or list, got {type(ephemeris)}'
    
    assert isinstance(raw, np.ndarray), f'Expected raw data to be ndarray, got {type(raw)}'
    assert isinstance(rc, np.ndarray), f'Expected range compressed data to be ndarray, got {type(rc)}'
    assert isinstance(rcmc, np.ndarray), f'Expected rcmc data to be ndarray, got {type(rcmc)}'
    assert isinstance(az, np.ndarray), f'Expected azimuth compressed data to be ndarray, got {type(az)}'

    # Prepare result dictionary
    result = {
        'raw': raw, 
        'rc': rc, 
        'rcmc': rcmc, 
        'az': az, 
        'metadata': metadata, 
        'ephemeris': ephemeris
    }
    return result


def process_sar_slice(
    handler: ZarrManager,
    slice_idx: int, 
    slice_info: Dict[str, Any],
    tmp_dir: Path,
    verbose: bool = True,
    unique_slice: bool = True,
) -> Path:
    """
    Process a single SAR data slice.
    
    Args:
        handler (ZarrManager): Zarr data handler
        slice_idx (int): Index of the slice to process
        slice_info (Dict[str, Any]): Slice information dictionary containing:
            - original_start (int): Original start row
            - original_end (int): Original end row
            - drop_start (int): Start index to drop for overlap handling
            - drop_end (int): End index to drop for overlap handling
            - actual_start (int): Actual start row after focusing
            - actual_end (int): Actual end row after focusing
            - is_first (bool): Is this the first slice?
            - is_last (bool): Is this the last slice?
            - original_height (int): Original height of the slice
            - actual_height (int): Actual height of the slice after focusing
        n_slices (int): Total number of slices
        tmp_dir (Path): Temporary directory for saving results
        verbose (bool): Enable verbose output
        
    Returns:
        Path: Path to saved slice file
    """
    
    # Extract slice information from the dictionary:
    start_row, end_row, drop_start, drop_end = slice_info['original_start'], slice_info['original_end'], slice_info['drop_start'], slice_info['drop_end']
    
    # Convert to integers for safety
    start_row, end_row, drop_start, drop_end = map(int, (start_row, end_row, drop_start, drop_end))
    
    # ------- 1. Getting slice data -------
    echo_data, metadata, ephemeris = handler.get_slice(rows=slice(start_row, end_row), cols=None)
    filename = handler.filename 
    raw_data = {'echo': echo_data, 'metadata': metadata, 'ephemeris': ephemeris}
    logger.info(f'📊 Sliced raw data shape: {raw_data["echo"].shape}')
   
    # ------- 2. Focusing slice data -------
    result = focalize_slice(raw_data=raw_data, verbose=verbose)
    logger.info(f'✅ Slice focused successfully.')
    
    # ------- 3. Drop for overlap handling -------
    # Apply drops to all arrays consistently
    result['raw'] = result['raw'][drop_start:-drop_end] if drop_end > 0 else result['raw'][drop_start:]
    result['rc'] = result['rc'][drop_start:-drop_end] if drop_end > 0 else result['rc'][drop_start:]
    result['rcmc'] = result['rcmc'][drop_start:-drop_end] if drop_end > 0 else result['rcmc'][drop_start:]
    result['az'] = result['az'][drop_start:-drop_end] if drop_end > 0 else result['az'][drop_start:]
    
    # Handle metadata and ephemeris slicing - check if they're DataFrames first
    if hasattr(result['metadata'], 'iloc'):  # DataFrame
        if drop_end > 0:
            result['metadata'] = result['metadata'].iloc[drop_start:-drop_end]
            result['ephemeris'] = result['ephemeris'].iloc[drop_start:-drop_end]
        else:
            result['metadata'] = result['metadata'].iloc[drop_start:]
            result['ephemeris'] = result['ephemeris'].iloc[drop_start:]
    else:  # Dict or other format - keep as is for now
        logger.warning(f'Metadata/ephemeris are not DataFrames, keeping original format')

    logger.info(f'📉 Dropped overlapping data: start={drop_start}, end={drop_end}')
    
    logger.info(f'📊 Focused data shape: {result["raw"].shape}')
    
    # Safe logging for metadata shape
    if hasattr(result['metadata'], 'shape'):
        logger.info(f'📊 Metadata shape: {result["metadata"].shape}')
    else:
        logger.info(f'📊 Metadata type: {type(result["metadata"])}')
        
    if hasattr(result['ephemeris'], 'shape'):
        logger.info(f'📊 Ephemeris shape: {result["ephemeris"].shape}')
    else:
        logger.info(f'📊 Ephemeris type: {type(result["ephemeris"])}')

    if not unique_slice:
        # Save slice in tmp folder
        zarr_path = tmp_dir / f'processor_slice_{slice_idx}.zarr'
        logger.info(f'💾 Saving slice {slice_idx + 1} to: {zarr_path}')
        dask_slice_saver(result, zarr_path, chunks='auto', clevel=5)
        logger.info(f'📂 Slice {slice_idx + 1} saved successfully.')
    else:
        # Save the unique slice directly without tmp files.
        zarr_path = tmp_dir.parent / f'{filename}.zarr'
        logger.info(f'💾 Saving entire product to: {zarr_path}')
        dask_slice_saver(result, zarr_path, chunks='auto', clevel=5)
        logger.info(f'📂 Slice {slice_idx + 1} saved successfully.')
            
    # Clean up memory
    logger.debug('🧹 Cleaning up memory...')
    del raw_data, result

    return zarr_path


def cleanup_tmp_directory(tmp_dir: Path) -> None:
    """
    Clean up temporary directory and all its contents recursively.
    
    Args:
        tmp_dir (Path): Temporary directory to clean up
    """
    if tmp_dir.exists():
        for item in tmp_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        tmp_dir.rmdir()
        print(f'🗑️ Temporary directory {tmp_dir} cleaned up.')




def main() -> None:
    """Main function to process SAR data."""
    parser = argparse.ArgumentParser(
        description='Process SAR data from Zarr file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Input Zarr file path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: ../focused_data relative to script location)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--keep-tmp',
        action='store_true',
        help='Keep temporary files after processing'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f'Input file does not exist: {input_path}')
        sys.exit(1)
    
    if not input_path.suffix == '.zarr':
        logger.error(f'Input file must be a .zarr file, got: {input_path.suffix}')
        sys.exit(1)
    
    # Setup directories
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'focused_data'
    
    output_dir, tmp_dir = setup_directories(output_dir)
    
    # Extract product name
    product_name = input_path.stem
    
    logger.info(f'🚀 Starting SAR data processing...')
    logger.info(f'📁 Input file: {input_path}')
    logger.info(f'📁 Output directory: {output_dir}')
    logger.info(f'📁 Temporary directory: {tmp_dir}')
    
    try:
        # Initialize handler
        handler = ZarrManager(str(input_path))
        length = handler.load().shape[0] # Get length of the 2d array (rows)

        # ------ Dynamic N slices based on input length ------
        H = handler.load().shape[0]  # Get total height of the 2D array
        slice_height = __BUFFER_SLICE_HEIGHT__  # Buffer size
        
        if (H // slice_height) > 1:
            slice_indices = calculate_slice_indices(
                array_height=H,
                slice_height=slice_height
            )
        else:
            # If only one slice, use the entire height TODO: bring this into calculate_slice_indices function.
            slice_indices = [
                {
                    'slice_index': 0,
                    'original_start': 0,
                    'original_end': H,
                    'actual_start': 0,
                    'actual_end': H,
                    'is_first': True,
                    'is_last': True,
                    'drop_start': 0,
                    'drop_end': 0,
                    'original_height': H,
                    'actual_height': H
                }
            ]
        
        n_slices = len(slice_indices)
        # Create a tabular display of slice information
        logger.info(f'📊 Slice breakdown for {n_slices} slices:')
        logger.info('┌─────────┬─────────────┬───────────┬─────────────┬───────────┬─────────────┬─────────────┐')
        logger.info('│ Slice # │ Orig Start  │ Orig End  │ Actual Start│ Actual End│ Drop Start  │ Drop End    │')
        logger.info('├─────────┼─────────────┼───────────┼─────────────┼───────────┼─────────────┼─────────────┤')
        for i, slice_info in enumerate(slice_indices):
            logger.info(f'│ {i+1:^7} │ {slice_info["original_start"]:^11} │ {slice_info["original_end"]:^9} │ {slice_info["actual_start"]:^11} │ {slice_info["actual_end"]:^9} │ {slice_info["drop_start"]:^11} │ {slice_info["drop_end"]:^11} │')
        logger.info('└─────────┴─────────────┴───────────┴─────────────┴───────────┴─────────────┴─────────────┘')
        
        
        # Process each slice
        tmp_files = []
        for slice_idx in range(n_slices):
            logger.info(f'🔍 Processing slice {slice_idx + 1}:')
            zarr_path = process_sar_slice(
                handler=handler,
                slice_idx=slice_idx,
                slice_info=slice_indices[slice_idx],
                tmp_dir=tmp_dir,
                verbose=args.verbose,
                unique_slice=(n_slices == 1)  # If only one slice, save directly to output
            )
            tmp_files.append(zarr_path)
            logger.info(f'✅ Slice {slice_idx + 1} processed successfully.')
        
        
        # Optional: Concatenate slices if more than 1 slice.
        if n_slices > 1:
            logger.info(f'🔗 Concatenating {len(tmp_files)} slices...')
            output_file = output_dir / f'{product_name}.zarr'
            concatenated_data = concatenate_slices_efficient(tmp_files, output_file)
            logger.info(f'✅ Concatenated data saved to: {output_file}')
        
        # Cleanup temporary files unless requested to keep them
        if not args.keep_tmp:
            cleanup_tmp_directory(tmp_dir)
            logger.debug('🧹 Memory cleanup complete.')
        else:
            logger.info(f'📁 Temporary files kept in: {tmp_dir}')

        logger.info(f'🎉 Processing completed successfully!')
        sys.exit(0)
        
    except Exception as e:
        logger.error(f'❌ Error during processing: {str(e)}')
        # Cleanup on error
        if not args.keep_tmp:
            cleanup_tmp_directory(tmp_dir)
        sys.exit(1)


if __name__ == '__main__':
    main()