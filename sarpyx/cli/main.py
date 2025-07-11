#!/usr/bin/env python3
"""
SARPyX Command Line Interface.

Main entry point for all sarpyx CLI tools.
"""

import argparse
import sys
from typing import List

from .shipdet import main as shipdet_main


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
  shipdet    Ship detection using SNAP GPT engine
  
Examples:
  sarpyx shipdet --product-path /path/to/S1A_*.SAFE --outdir /path/to/output
  
For command-specific help:
  sarpyx <command> --help
"""
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='sarpyx-cli 0.1.0'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='<command>'
    )
    
    # Ship detection subcommand
    shipdet_parser = subparsers.add_parser(
        'shipdet',
        help='Ship detection using SNAP GPT engine',
        description='Detect ships in SAR data using adaptive thresholding and object discrimination'
    )
    
    # Add shipdet arguments
    _add_shipdet_arguments(shipdet_parser)
    
    return parser


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
    if args.command == 'shipdet':
        # Modify sys.argv to make it look like shipdet was called directly
        # This allows the shipdet main function to work correctly
        original_argv = sys.argv.copy()
        sys.argv = ['sarpyx-shipdet'] + [arg for arg in original_argv[2:]]
        
        try:
            shipdet_main()
        except SystemExit as e:
            sys.exit(e.code)
        finally:
            sys.argv = original_argv
    else:
        print(f'Unknown command: {args.command}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
