#!/usr/bin/env python3
"""
Unzip CLI Tool for SARPyX.

This module provides a command-line interface for extracting SAR data
from zip archives.
"""

import argparse
import sys
import os
from pathlib import Path
import zipfile
import logging

from .utils import validate_path, create_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for unzip command.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Extract SAR data from zip archives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract a single zip file
  sarpyx unzip --input /path/to/file.zip --output /path/to/output
  
  # Extract all zip files in a directory
  sarpyx unzip --input /path/to/directory --output /path/to/output
  
  # Extract recursively from nested directories
  sarpyx unzip --input /path/to/directory --output /path/to/output --recursive
"""
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to zip file or directory containing zip files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./extracted_data',
        help='Output directory for extracted files (default: ./extracted_data)'
    )
    
    parser.add_argument(
        '--recursive',
        '-r',
        action='store_true',
        help='Recursively search for zip files in subdirectories'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract a zip file to the specified directory.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        
    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f'✅ Successfully extracted {zip_path.name} to {extract_to}')
        return True
    except Exception as e:
        logger.error(f'❌ Failed to extract {zip_path.name}: {e}')
        return False


def find_zip_files(input_path: Path, recursive: bool = False) -> list[Path]:
    """
    Find all zip files in the given path.
    
    Args:
        input_path: Path to search for zip files
        recursive: Whether to search recursively
        
    Returns:
        List of paths to zip files
    """
    zip_files = []
    
    if input_path.is_file() and input_path.suffix == '.zip':
        zip_files.append(input_path)
    elif input_path.is_dir():
        if recursive:
            # Recursively search for zip files
            zip_files = list(input_path.rglob('*.zip'))
        else:
            # Only search immediate directory
            zip_files = list(input_path.glob('*.zip'))
    
    return zip_files


def main() -> None:
    """
    Main entry point for unzip CLI.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Validate input path
    input_path = validate_path(args.input, must_exist=True)
    
    # Create output directory
    output_dir = create_output_directory(args.output)
    
    # Find zip files
    zip_files = find_zip_files(input_path, recursive=args.recursive)
    
    if not zip_files:
        logger.error(f'❌ No zip files found in: {input_path}')
        sys.exit(1)
    
    # Print summary
    if args.verbose:
        print('='*60)
        print('SARPyX Unzip Tool')
        print('='*60)
        print(f'Input: {input_path}')
        print(f'Output directory: {output_dir}')
        print(f'Files to extract: {len(zip_files)}')
        print(f'Recursive search: {args.recursive}')
        print('='*60)
    
    # Extract each zip file
    success_count = 0
    for idx, zip_file in enumerate(zip_files, 1):
        logger.info(f'📦 Processing {idx}/{len(zip_files)}: {zip_file.name}')
        
        # Create subdirectory based on the zip file's location relative to input
        if input_path.is_dir():
            rel_path = zip_file.parent.relative_to(input_path)
            extract_location = output_dir / rel_path
        else:
            extract_location = output_dir
        
        extract_location.mkdir(parents=True, exist_ok=True)
        
        if extract_zip(zip_file, extract_location):
            success_count += 1
    
    # Final summary
    print('='*60)
    print(f'Extraction Summary:')
    print(f'  Total files: {len(zip_files)}')
    print(f'  Successful: {success_count}')
    print(f'  Failed: {len(zip_files) - success_count}')
    print('='*60)
    
    if success_count == len(zip_files):
        print('✅ All files extracted successfully!')
        sys.exit(0)
    else:
        print(f'⚠️  Extraction completed with {len(zip_files) - success_count} errors.')
        sys.exit(1)


if __name__ == '__main__':
    main()
