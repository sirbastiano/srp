#!/usr/bin/env python3
"""
SARPyX Command Line Interface.

Main entry point for all sarpyx CLI tools.
"""

import argparse
import sys
from typing import List


def create_main_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with subcommands.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog='sarpyx',
        description='SARPyX: A comprehensive toolkit for SAR data processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  decode     Decode Sentinel-1 Level-0 products to zarr format
  focus      Focus SAR data using Range-Doppler Algorithm
  shipdet    Ship detection using SNAP GPT engine
  unzip      Extract SAR data from zip archives
  upload     Upload data to Hugging Face Hub
  worldsar   Process SAR products with SNAP GPT pipelines and tiling
  
Examples:
  sarpyx decode --input /path/to/file.dat --output /path/to/output
  sarpyx focus --input /path/to/data.zarr --output /path/to/output
  sarpyx shipdet --product-path /path/to/S1A_*.SAFE --outdir /path/to/output
  sarpyx unzip --input /path/to/file.zip --output /path/to/output
  sarpyx upload --folder /path/to/folder --repo username/dataset-name
  sarpyx worldsar --input /path/to/product --output /path/to/output \\
                 --cuts-outdir /path/to/tiles --product-wkt "POLYGON ((...))"
  
For command-specific help:
  sarpyx <command> --help
"""
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='sarpyx-cli 0.1.6'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='<command>'
    )
    
    # Decode subcommand
    decode_parser = subparsers.add_parser(
        'decode',
        help='Decode Sentinel-1 Level-0 products to zarr format',
        description='Decode S1 L0 products (.dat files or .SAFE folders) to zarr format'
    )
    _add_decode_arguments(decode_parser)
    
    # Focus subcommand
    focus_parser = subparsers.add_parser(
        'focus',
        help='Focus SAR data using Range-Doppler Algorithm',
        description='Focus SAR data from zarr files using CoarseRDA processor'
    )
    _add_focus_arguments(focus_parser)
    
    # Ship detection subcommand
    shipdet_parser = subparsers.add_parser(
        'shipdet',
        help='Ship detection using SNAP GPT engine',
        description='Detect ships in SAR data using adaptive thresholding and object discrimination'
    )
    _add_shipdet_arguments(shipdet_parser)
    
    # Unzip subcommand
    unzip_parser = subparsers.add_parser(
        'unzip',
        help='Extract SAR data from zip archives',
        description='Extract zip files containing SAR data'
    )
    _add_unzip_arguments(unzip_parser)
    
    # Upload subcommand
    upload_parser = subparsers.add_parser(
        'upload',
        help='Upload data to Hugging Face Hub',
        description='Upload processed SAR data to Hugging Face Hub'
    )
    _add_upload_arguments(upload_parser)

    # WorldSAR subcommand
    worldsar_parser = subparsers.add_parser(
        'worldsar',
        help='Process SAR products with SNAP GPT pipelines and tiling',
        description='Process SAR products from multiple missions and generate tiles'
    )
    _add_worldsar_arguments(worldsar_parser)
    
    return parser


def _add_decode_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the decode subcommand.
    
    Args:
        parser: The subparser for decode command.
    """
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


def _add_focus_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the focus subcommand.
    
    Args:
        parser: The subparser for focus command.
    """
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input zarr file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./focused_data',
        help='Output directory (default: ./focused_data)'
    )
    parser.add_argument(
        '--slice-height',
        type=int,
        default=15000,
        help='Slice height for processing (default: 15000)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--keep-tmp',
        action='store_true',
        help='Keep temporary files after processing'
    )


def _add_shipdet_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the shipdet subcommand.
    
    Args:
        parser: The subparser for shipdet command.
    """
    # Required arguments
    parser.add_argument(
        '--product-path',
        type=str,
        required=True,
        help='Path to the SAR product (.SAFE directory or zip file)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    
    # Optional processing arguments
    parser.add_argument(
        '--format',
        type=str,
        default='BEAM-DIMAP',
        choices=['BEAM-DIMAP', 'GeoTIFF', 'NetCDF4-CF', 'ENVI', 'HDF5'],
        help='Output format (default: BEAM-DIMAP)'
    )
    parser.add_argument(
        '--gpt-path',
        type=str,
        default=None,
        help='Path to GPT executable (default: None - use system PATH)'
    )
    
    # Calibration arguments
    parser.add_argument(
        '--output-complex',
        action='store_true',
        help='Output complex values for calibration (default: False)'
    )
    parser.add_argument(
        '--polarizations',
        type=str,
        nargs='+',
        default=['VV'],
        choices=['VV', 'VH', 'HH', 'HV'],
        help='Polarizations to process (default: VV)'
    )
    
    # Adaptive thresholding arguments
    parser.add_argument(
        '--pfa',
        type=float,
        default=6.5,
        help='Probability of false alarm for adaptive thresholding (default: 6.5)'
    )
    parser.add_argument(
        '--background-window-m',
        type=float,
        default=800.0,
        help='Background window size in meters (default: 800.0)'
    )
    parser.add_argument(
        '--guard-window-m',
        type=float,
        default=500.0,
        help='Guard window size in meters (default: 500.0)'
    )
    parser.add_argument(
        '--target-window-m',
        type=float,
        default=50.0,
        help='Target window size in meters (default: 50.0)'
    )
    
    # Object discrimination arguments
    parser.add_argument(
        '--min-target-m',
        type=float,
        default=50.0,
        help='Minimum target size in meters (default: 50.0)'
    )
    parser.add_argument(
        '--max-target-m',
        type=float,
        default=600.0,
        help='Maximum target size in meters (default: 600.0)'
    )
    
    # Processing options
    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        help='Skip calibration step (default: False)'
    )
    parser.add_argument(
        '--skip-discrimination',
        action='store_true',
        help='Skip object discrimination step (default: False)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output (default: False)'
    )


def _add_unzip_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the unzip subcommand.
    
    Args:
        parser: The subparser for unzip command.
    """
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


def _add_upload_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the upload subcommand.
    
    Args:
        parser: The subparser for upload command.
    """
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to the folder to upload'
    )
    parser.add_argument(
        '--repo',
        type=str,
        required=True,
        help='Repository ID in format username/repo-name'
    )
    parser.add_argument(
        '--repo-type',
        type=str,
        default='dataset',
        choices=['dataset', 'model', 'space'],
        help='Type of repository (default: dataset)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )


def _add_worldsar_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the worldsar subcommand.
    
    Args:
        parser: The subparser for worldsar command.
    """
    parser.add_argument(
        '--input',
        '-i',
        dest='product_path',
        type=str,
        required=True,
        help='Path to the input SAR product.'
    )
    parser.add_argument(
        '--output',
        '-o',
        dest='output_dir',
        type=str,
        required=True,
        help='Directory to save the processed output.'
    )
    parser.add_argument(
        '--cuts-outdir',
        '--cuts_outdir',
        dest='cuts_outdir',
        type=str,
        required=True,
        help='Where to store the tiles after extraction.'
    )
    parser.add_argument(
        '--product-wkt',
        '--product_wkt',
        dest='product_wkt',
        type=str,
        required=False,
        default=None,
        help='WKT string defining the product region of interest.'
    )
    parser.add_argument(
        '--gpt-path',
        dest='gpt_path',
        type=str,
        default=None,
        help='Override GPT executable path (default: gpt_path env var).'
    )
    parser.add_argument(
        '--grid-path',
        dest='grid_path',
        type=str,
        default=None,
        help='Override grid GeoJSON path (default: grid_path env var).'
    )
    parser.add_argument(
        '--db-dir',
        dest='db_dir',
        type=str,
        default=None,
        help='Override database output directory (default: db_dir env var).'
    )
    parser.add_argument(
        '--gpt-memory',
        dest='gpt_memory',
        type=str,
        default=None,
        help='Override GPT Java heap (e.g., 24G).'
    )
    parser.add_argument(
        '--gpt-parallelism',
        dest='gpt_parallelism',
        type=int,
        default=None,
        help='Override GPT parallelism (number of tiles).'
    )
    parser.add_argument(
        '--gpt-timeout',
        dest='gpt_timeout',
        type=int,
        default=None,
        help='Override GPT timeout in seconds for a single invocation.'
    )


def main() -> None:
    """
    Main entry point for sarpyx CLI.
    """
    parser = create_main_parser()
    args = parser.parse_args()
    
    # Check if a command was provided
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command handler
    original_argv = sys.argv.copy()
    
    try:
        if args.command == 'decode':
            from .decode import main as decode_main
            sys.argv = ['sarpyx-decode'] + [arg for arg in original_argv[2:]]
            decode_main()
        elif args.command == 'focus':
            from .focus import main as focus_main
            sys.argv = ['sarpyx-focus'] + [arg for arg in original_argv[2:]]
            focus_main()
        elif args.command == 'shipdet':
            from .shipdet import main as shipdet_main
            sys.argv = ['sarpyx-shipdet'] + [arg for arg in original_argv[2:]]
            shipdet_main()
        elif args.command == 'unzip':
            from .unzip import main as unzip_main
            sys.argv = ['sarpyx-unzip'] + [arg for arg in original_argv[2:]]
            unzip_main()
        elif args.command == 'upload':
            from .upload import main as upload_main
            sys.argv = ['sarpyx-upload'] + [arg for arg in original_argv[2:]]
            upload_main()
        elif args.command == 'worldsar':
            from .worldsar import main as worldsar_main
            sys.argv = ['sarpyx-worldsar'] + [arg for arg in original_argv[2:]]
            print("=======================================================================================================================")
            print(" __        __   ___    ____   _       ____    ____      _      ____  ")
            print(" \\ \\      / /  / _ \\  |  _ \\ | |     |  _ \\  / ___|    / \\    | 0 _ \\ ")
            print("  \\ \\ /\\ / /  | | | | | |_) || |     | | | | \\___ \\   / _ \\   | |_) |")
            print("   \\ V  V /   | |_| | |  _ < | |___  | |_| |  ___) | / ___ \\  |  _ < ")
            print("    \\_/\\_/     \\___/  |_| \\_\\|_____| |____/  |____/ /_/   \\_\\ |_| \\_\\")
            print("=======================================================================================================================")
            print("======================================         DATA PROCESSOR       ===================================================")
            print("=======================================================================================================================")
            print("=======================================================================================================================")
            worldsar_main()
        else:
            print(f'Unknown command: {args.command}', file=sys.stderr)
            sys.exit(1)
    except SystemExit as e:
        sys.exit(e.code)
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
