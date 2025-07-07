#!/usr/bin/env python3
"""
Example script demonstrating SAR image processing functionality.

This script shows how to process Level-0 data into focused SAR images
using the S1Decoder class.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add the processor module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processor import S1Decoder, multilook_processing


def process_swath_to_image(filename: str, swath: str, output_dir: str,
                          range_looks: int = 1, azimuth_looks: int = 1) -> None:
    """
    Process a swath to focused SAR image.
    
    Args:
        filename: Path to Level-0 data file
        swath: Swath name to process
        output_dir: Output directory for results
        range_looks: Number of range looks for multilooking
        azimuth_looks: Number of azimuth looks for multilooking
    """
    try:
        print(f'Loading data from: {filename}')
        decoder = S1Decoder(filename)
        
        # Check available swaths
        available_swaths = decoder.get_swath_names()
        print(f'Available swaths: {available_swaths}')
        
        if swath not in available_swaths:
            print(f'Error: Swath {swath} not found in data')
            return
            
        print(f'Processing swath: {swath}')
        
        # Get range-compressed data
        print('Applying range compression...')
        range_compressed = decoder.get_range_compressed_swath(swath)
        
        if range_compressed.size == 0:
            print('No data available for range compression')
            return
            
        print(f'Range-compressed data shape: {range_compressed.shape}')
        
        # Get azimuth-compressed data
        print('Applying azimuth compression...')
        azimuth_compressed = decoder.get_azimuth_compressed_swath(swath)
        
        print(f'Azimuth-compressed data shape: {azimuth_compressed.shape}')
        
        # Apply multilooking if requested
        if range_looks > 1 or azimuth_looks > 1:
            print(f'Applying multilooking ({azimuth_looks}x{range_looks})...')
            multilooked = multilook_processing(azimuth_compressed, 
                                             range_looks, azimuth_looks)
            print(f'Multilooked data shape: {multilooked.shape}')
        else:
            multilooked = np.abs(azimuth_compressed)**2
            
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays
        range_file = output_path / f'{swath}_range_compressed.npy'
        azimuth_file = output_path / f'{swath}_azimuth_compressed.npy'
        intensity_file = output_path / f'{swath}_intensity.npy'
        
        print(f'Saving range-compressed data to: {range_file}')
        np.save(range_file, range_compressed)
        
        print(f'Saving azimuth-compressed data to: {azimuth_file}')
        np.save(azimuth_file, azimuth_compressed)
        
        print(f'Saving intensity image to: {intensity_file}')
        np.save(intensity_file, multilooked)
        
        # Print statistics
        print(f'\n=== Processing Statistics ===')
        print(f'Swath: {swath}')
        print(f'Azimuth samples: {range_compressed.shape[0]}')
        print(f'Range samples: {range_compressed.shape[1]}')
        print(f'Intensity min/max: {np.min(multilooked):.2e} / {np.max(multilooked):.2e}')
        print(f'Intensity mean: {np.mean(multilooked):.2e}')
        
    except Exception as e:
        print(f'Error processing swath: {e}')
        import traceback
        traceback.print_exc()


def process_all_swaths(filename: str, output_dir: str) -> None:
    """
    Process all available swaths in the file.
    
    Args:
        filename: Path to Level-0 data file
        output_dir: Output directory for results
    """
    try:
        print(f'Loading data from: {filename}')
        decoder = S1Decoder(filename)
        
        available_swaths = decoder.get_swath_names()
        print(f'Found {len(available_swaths)} swaths: {available_swaths}')
        
        for swath in available_swaths:
            print(f'\n=== Processing swath {swath} ===')
            try:
                process_swath_to_image(filename, swath, output_dir)
            except Exception as e:
                print(f'Error processing swath {swath}: {e}')
                continue
                
        print('\nProcessing complete!')
        
    except Exception as e:
        print(f'Error: {e}')


def print_state_vectors(filename: str) -> None:
    """
    Print state vector information from the file.
    
    Args:
        filename: Path to Level-0 data file
    """
    try:
        print(f'Loading state vectors from: {filename}')
        decoder = S1Decoder(filename)
        
        state_vectors = decoder.get_state_vectors()
        state_vectors.print()
        
        # Print some orbital parameters
        print(f'\nOrbital period estimate: {state_vectors.get_orbital_period():.1f} seconds')
        
        # Example position and velocity at middle time
        time_range = state_vectors.get_time_range()
        if time_range[1] > time_range[0]:
            mid_time = (time_range[0] + time_range[1]) / 2
            position = state_vectors.get_satellite_position(mid_time)
            velocity = state_vectors.get_satellite_velocity(mid_time)
            vel_mag = state_vectors.get_satellite_velocity_magnitude(mid_time)
            
            print(f'\nSatellite state at time {mid_time:.3f}s:')
            print(f'  Position: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] m')
            print(f'  Velocity: [{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}] m/s')
            print(f'  Velocity magnitude: {vel_mag:.3f} m/s')
            
    except Exception as e:
        print(f'Error: {e}')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Process Sentinel-1 Level-0 data to SAR images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_image.py data.dat --swath IW1 --output ./results
  python process_image.py data.dat --all-swaths --output ./results
  python process_image.py data.dat --state-vectors
  python process_image.py data.dat --swath S1 --multilook 2 4
        """
    )
    
    parser.add_argument('filename', help='Path to Level-0 data file')
    parser.add_argument('--swath', help='Specific swath to process')
    parser.add_argument('--all-swaths', action='store_true',
                       help='Process all available swaths')
    parser.add_argument('--state-vectors', action='store_true',
                       help='Print state vector information')
    parser.add_argument('--output', default='./output',
                       help='Output directory (default: ./output)')
    parser.add_argument('--multilook', nargs=2, type=int, metavar=('AZ', 'RG'),
                       help='Apply multilooking (azimuth_looks range_looks)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.filename).exists():
        print(f'Error: File not found: {args.filename}')
        sys.exit(1)
        
    # Parse multilook parameters
    azimuth_looks = 1
    range_looks = 1
    if args.multilook:
        azimuth_looks, range_looks = args.multilook
        
    # Execute requested operation
    if args.state_vectors:
        print_state_vectors(args.filename)
    elif args.all_swaths:
        process_all_swaths(args.filename, args.output)
    elif args.swath:
        process_swath_to_image(args.filename, args.swath, args.output,
                             range_looks, azimuth_looks)
    else:
        print('Please specify --swath, --all-swaths, or --state-vectors')
        parser.print_help()


if __name__ == '__main__':
    main()
