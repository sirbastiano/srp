#!/usr/bin/env python3
"""
Decode CLI Tool for SARPyX.

This module provides a command-line interface for decoding Sentinel-1 Level-0
products to zarr format.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging

from sarpyx.processor.core.decode import S1L0Decoder
from sarpyx.utils.io import find_dat_file
from .utils import validate_path, create_output_directory, get_product_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for decode command.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Decode Sentinel-1 Level-0 products to zarr format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode a single .dat file
  sarpyx decode --input /path/to/file.dat --output /path/to/output
  
  # Decode all .dat files in a .SAFE folder
  sarpyx decode --input /path/to/S1A_*.SAFE --output /path/to/output
  
  # Enable verbose logging
  sarpyx decode --input /path/to/file.dat --output /path/to/output --verbose
"""
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to .dat file or .SAFE folder'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./decoded_data',
        help='Output directory for decoded files (default: ./decoded_data)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser


def decode_s1_l0(input_file: Path, output_dir: Path) -> None:
    """
    Decode S1 L0 file to zarr format.

    Args:
        input_file: Path to the input .dat file.
        output_dir: Directory to save decoded output.
    """
    decoder = S1L0Decoder()
    decoder.decode_file(input_file, output_dir, save_to_zarr=True, headers_only=False)
    logger.info(f'✅ Decoding completed for {input_file.name} to {output_dir}')
    del decoder


def retrieve_input_files(
    safe_folders: List[Path], 
    verbose: bool = False
) -> Tuple[List[Path], Dict[Path, List[Path]]]:
    """
    Retrieve input files from SAFE folders.

    Args:
        safe_folders: List of SAFE folder paths.
        verbose: Whether to log verbose output.

    Returns:
        Tuple of (input_files, folders_map)
    """
    pols = ['vh', 'vv', 'hh', 'hv']
    input_files: List[Path] = []
    folders_map: Dict[Path, List[Path]] = {folder: [] for folder in safe_folders}
    
    for folder in safe_folders:
        for pol in pols:
            try:
                input_file = find_dat_file(folder, pol)
                input_files.append(input_file)
                folders_map[folder].append(input_file)
                if verbose:
                    logger.info(f'📁 Found {input_file.name} in {folder.name}')
            except FileNotFoundError:
                continue
        
        if verbose and not folders_map[folder]:
            logger.info(f'📁 No .dat file found in {folder.name}, checking next folder...')
    
    return input_files, folders_map


def main() -> None:
    """
    Main entry point for decode CLI.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Validate and create output directory
    output_dir = create_output_directory(args.output)
    
    # Validate input path
    input_path = validate_path(args.input, must_exist=True)
    
    # Determine input files
    if input_path.is_file() and input_path.suffix == '.dat':
        input_files = [input_path]
        folders_map = {input_path.parent: [input_path]}
    elif input_path.is_dir() and input_path.name.endswith('.SAFE'):
        input_files, folders_map = retrieve_input_files([input_path], verbose=args.verbose)
    else:
        logger.error(f'❌ Invalid input: {input_path}. Must be a .dat file or .SAFE folder.')
        sys.exit(1)
    
    if not input_files:
        logger.error(f'❌ No valid input files found in: {input_path}')
        sys.exit(1)
    
    # Print summary
    if args.verbose:
        print('='*60)
        print('SARPyX Decode Tool')
        print('='*60)
        print(f'Input: {input_path}')
        print(f'Output directory: {output_dir}')
        print(f'Files to process: {len(input_files)}')
        print('='*60)
    
    # Process each file
    success_count = 0
    for input_file in input_files:
        if input_file.is_file():
            logger.info(f'🔍 Processing {input_file.name}...')
            try:
                decode_s1_l0(input_file, output_dir)
                success_count += 1
            except Exception as e:
                logger.error(f'❌ Error processing {input_file.name}: {e}')
        else:
            logger.warning(f'⚠️  {input_file.name} is not a valid file, skipping...')
    
    # Final summary
    if args.verbose:
        logger.info(f'🔍 Processed {success_count}/{len(input_files)} files from {len(folders_map)} SAFE folders.')
    
    if success_count == len(input_files):
        print('✅ Decoding completed successfully!')
        sys.exit(0)
    else:
        print(f'⚠️  Decoding completed with {len(input_files) - success_count} errors.')
        sys.exit(1)


if __name__ == '__main__':
    main()
