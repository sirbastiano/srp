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
from typing import Dict, Any, Optional
import numpy as np
import logging
import shutil

# Add sarpyx to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from sarpyx.processor.core.focus import CoarseRDA
from sarpyx.utils.zarr_utils import ZarrManager, dask_slice_saver, concatenate_slices_efficient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def process_sar_slice(
    handler: ZarrManager, 
    slice_idx: int, 
    n_slices: int, 
    tmp_dir: Path,
    verbose: bool = False
) -> Path:
    """
    Process a single SAR data slice.
    
    Args:
        handler (ZarrManager): Zarr data handler
        slice_idx (int): Index of the slice to process
        n_slices (int): Total number of slices
        tmp_dir (Path): Temporary directory for saving results
        verbose (bool): Enable verbose output
        
    Returns:
        Path: Path to saved slice file
    """
    logger.info(f'Processing slice {slice_idx + 1}/{n_slices}...')
    
    # Get slice data
    raw_data = handler.get_slice_block(slice_idx=slice_idx, N_blocks=n_slices)
    logger.info(f'📊 Sliced raw data shape: {raw_data["echo"].shape}')

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
    metadata = raw_data['metadata'].to_dict()
    ephemeris = raw_data['ephemeris'].to_dict()
    
    # Validate data types
    assert isinstance(metadata, dict), f'Expected metadata to be dict, got {type(metadata)}'
    assert isinstance(ephemeris, dict), f'Expected ephemeris to be dict, got {type(ephemeris)}'
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
    
    # Save slice
    zarr_path = tmp_dir / f'processor_slice_{slice_idx}.zarr'
    dask_slice_saver(result, zarr_path, chunks='auto', clevel=7)
    
    # Clean up memory
    del processor, raw_data, raw, rc, rcmc, az, metadata, ephemeris, result
    
    logger.info(f'💾 Slice {slice_idx + 1} saved to: {zarr_path}')
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
        'input_file',
        type=str,
        help='Input Zarr file path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: ../focused_data relative to script location)'
    )
    
    parser.add_argument(
        '--n-slices',
        type=int,
        default=5,
        help='Number of slices to process (default: 5)'
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
    logger.info(f'🔢 Number of slices: {args.n_slices}')
    
    try:
        # Initialize handler
        handler = ZarrManager(str(input_path))
        
        # Process each slice
        tmp_files = []
        for slice_idx in range(args.n_slices):
            zarr_path = process_sar_slice(
                handler=handler,
                slice_idx=slice_idx,
                n_slices=args.n_slices,
                tmp_dir=tmp_dir,
                verbose=args.verbose
            )
            tmp_files.append(zarr_path)
        
        # Concatenate slices
        logger.info(f'🔗 Concatenating {len(tmp_files)} slices...')
        output_file = output_dir / f'{product_name}.zarr'
        concatenated_data = concatenate_slices_efficient(tmp_files, output_file)
        logger.info(f'✅ Concatenated data saved to: {output_file}')
        
        # Cleanup temporary files unless requested to keep them
        if not args.keep_tmp:
            cleanup_tmp_directory(tmp_dir)
        else:
            logger.info(f'📁 Temporary files kept in: {tmp_dir}')
        
        logger.info(f'🎉 Processing completed successfully!')
        
    except Exception as e:
        logger.error(f'❌ Error during processing: {str(e)}')
        # Cleanup on error
        if not args.keep_tmp:
            cleanup_tmp_directory(tmp_dir)
        sys.exit(1)


if __name__ == '__main__':
    main()