#!/usr/bin/env python3
"""
Ship Detection CLI Tool for SARPyX.

This module provides a command-line interface for ship detection using SNAP's
GPT engine with adaptive thresholding and object discrimination.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from sarpyx.snap.engine import GPT
from .utils import (
    validate_path,
    create_output_directory,
    check_snap_installation,
    print_processing_summary,
    get_product_info,
    validate_window_sizes,
    estimate_processing_time
)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for ship detection.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Ship detection using SNAP GPT engine with adaptive thresholding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ship detection
  sarpyx shipdet --product-path /path/to/S1A_*.SAFE --outdir /path/to/output
  
  # Custom parameters
  sarpyx shipdet --product-path /path/to/product.SAFE \\
                 --outdir /path/to/output \\
                 --pfa 8 \\
                 --background-window-m 1000 \\
                 --min-target-m 10 \\
                 --max-target-m 1000
  
  # Output in GeoTIFF format
  sarpyx shipdet --product-path /path/to/product.SAFE \\
                 --outdir /path/to/output \\
                 --format GeoTIFF
"""
    )
    
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
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments.
        
    Raises:
        SystemExit: If validation fails.
    """
    # Validate product path
    validate_path(args.product_path, must_exist=True)
    
    # Validate and create output directory
    create_output_directory(args.outdir)
    
    # Check SNAP installation
    if not check_snap_installation(args.gpt_path):
        print('Error: SNAP GPT not found. Please check your SNAP installation.', file=sys.stderr)
        print('You can specify the GPT path with --gpt-path', file=sys.stderr)
        sys.exit(1)
    
    # Validate numerical parameters
    if args.pfa <= 0:
        print('Error: PFA must be positive', file=sys.stderr)
        sys.exit(1)
    
    if args.background_window_m <= 0:
        print('Error: Background window size must be positive', file=sys.stderr)
        sys.exit(1)
    
    if args.guard_window_m <= 0:
        print('Error: Guard window size must be positive', file=sys.stderr)
        sys.exit(1)
    
    if args.target_window_m <= 0:
        print('Error: Target window size must be positive', file=sys.stderr)
        sys.exit(1)
    
    if args.min_target_m <= 0:
        print('Error: Minimum target size must be positive', file=sys.stderr)
        sys.exit(1)
    
    if args.max_target_m <= args.min_target_m:
        print('Error: Maximum target size must be greater than minimum target size', file=sys.stderr)
        sys.exit(1)
    
    # Validate window size relationships
    validate_window_sizes(
        args.target_window_m,
        args.guard_window_m,
        args.background_window_m
    )


def run_ship_detection(args: argparse.Namespace) -> Optional[str]:
    """
    Execute ship detection processing pipeline.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        Path to final output product, or None if processing failed.
    """
    try:
        # Initialize GPT tool
        if args.verbose:
            print(f'Initializing GPT tool...')
            print(f'  Product: {args.product_path}')
            print(f'  Output directory: {args.outdir}')
            print(f'  Format: {args.format}')
        
        tool = GPT(
            product=args.product_path,
            outdir=args.outdir,
            format=args.format,
            gpt_path=args.gpt_path,
        )
        
        # Step 1: Calibration
        if not args.skip_calibration:
            if args.verbose:
                print(f'Applying calibration...')
                print(f'  Polarizations: {args.polarizations}')
                print(f'  Output complex: {args.output_complex}')
            
            calibration_result = tool.Calibration(
                Pols=args.polarizations,
                output_complex=args.output_complex
            )
            
            if calibration_result is None:
                print('Error: Calibration step failed', file=sys.stderr)
                return None
            
            if args.verbose:
                print(f'  Calibration completed: {calibration_result}')
        
        # Step 2: Adaptive Thresholding
        if args.verbose:
            print(f'Applying adaptive thresholding...')
            print(f'  PFA: {args.pfa}')
            print(f'  Background window: {args.background_window_m}m')
            print(f'  Guard window: {args.guard_window_m}m')
            print(f'  Target window: {args.target_window_m}m')
        
        thresholding_result = tool.AdaptiveThresholding(
            background_window_m=args.background_window_m,
            guard_window_m=args.guard_window_m,
            target_window_m=args.target_window_m,
            pfa=args.pfa
        )
        
        if thresholding_result is None:
            print('Error: Adaptive thresholding step failed', file=sys.stderr)
            return None
        
        if args.verbose:
            print(f'  Adaptive thresholding completed: {thresholding_result}')
        
        # Step 3: Object Discrimination (optional)
        final_result = thresholding_result
        if not args.skip_discrimination:
            if args.verbose:
                print(f'Applying object discrimination...')
                print(f'  Min target size: {args.min_target_m}m')
                print(f'  Max target size: {args.max_target_m}m')
            
            discrimination_result = tool.ObjectDiscrimination(
                min_target_m=args.min_target_m,
                max_target_m=args.max_target_m
            )
            
            if discrimination_result is None:
                print('Error: Object discrimination step failed', file=sys.stderr)
                return None
            
            final_result = discrimination_result
            if args.verbose:
                print(f'  Object discrimination completed: {discrimination_result}')
        
        return final_result
        
    except Exception as e:
        print(f'Error during processing: {e}', file=sys.stderr)
        return None


def main() -> None:
    """
    Main entry point for ship detection CLI.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Print product information if verbose
    if args.verbose:
        print('='*60)
        print('SARPyX Ship Detection Tool')
        print('='*60)
        
        product_info = get_product_info(args.product_path)
        print(f'Product: {product_info["name"]}')
        print(f'Type: {product_info["type"]}')
        print(f'Size: {product_info["size"]}')
        print(f'Estimated processing time: {estimate_processing_time(args.product_path)}')
        print('')
        
        # Print processing parameters
        parameters = {
            'Output format': args.format,
            'Polarizations': args.polarizations,
            'PFA': args.pfa,
            'Background window (m)': args.background_window_m,
            'Guard window (m)': args.guard_window_m,
            'Target window (m)': args.target_window_m,
            'Min target size (m)': args.min_target_m,
            'Max target size (m)': args.max_target_m,
            'Skip calibration': args.skip_calibration,
            'Skip discrimination': args.skip_discrimination
        }
        
        print_processing_summary(args.product_path, args.outdir, parameters)
    
    # Run ship detection
    result = run_ship_detection(args)
    
    if result is not None:
        print(f'Ship detection completed successfully!')
        print(f'Output saved to: {result}')
        sys.exit(0)
    else:
        print(f'Ship detection failed!', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
