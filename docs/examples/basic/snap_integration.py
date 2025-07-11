#!/usr/bin/env python3
"""
SNAP Integration Examples
Application: SAR Data Processing
Complexity: Basic

This example demonstrates basic SNAP GPT automation using SARPYX.
Covers common preprocessing workflows for Sentinel-1 data.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from sarpyx.snap import GPT, SNAPProcessor
    from sarpyx.utils import visualization
except ImportError as e:
    print(f"Error importing SARPYX: {e}")
    print("Please install SARPYX: pip install sarpyx")
    sys.exit(1)


def basic_grd_processing(input_file, output_dir):
    """
    Basic GRD processing workflow using SNAP GPT.
    
    Args:
        input_file (str): Path to Sentinel-1 GRD file
        output_dir (str): Output directory for processed products
    
    Returns:
        dict: Processing results with file paths
    """
    print("=== Basic GRD Processing Workflow ===")
    
    # Initialize SNAP processor
    processor = SNAPProcessor()
    
    # Configure processing chain
    processing_config = {
        'operations': [
            'apply_orbit_file',
            'thermal_noise_removal', 
            'calibration',
            'speckle_filtering',
            'terrain_correction'
        ],
        'calibration': {
            'output_sigma_band': True,
            'output_beta_band': False,
            'output_gamma_band': False
        },
        'speckle_filter': {
            'filter_type': 'Lee',
            'filter_size': '5x5'
        },
        'terrain_correction': {
            'dem_name': 'SRTM 3Sec',
            'pixel_spacing': 10.0,
            'output_crs': 'EPSG:4326'
        }
    }
    
    print(f"Processing: {os.path.basename(input_file)}")
    print(f"Output directory: {output_dir}")
    
    # Process data
    results = processor.process_sar_data(
        input_file=input_file,
        config=processing_config,
        output_dir=output_dir
    )
    
    print("✓ Processing completed successfully")
    return results


def batch_grd_processing(input_directory, output_directory):
    """
    Batch process multiple GRD files.
    
    Args:
        input_directory (str): Directory containing Sentinel-1 GRD files
        output_directory (str): Base output directory
    """
    print("=== Batch GRD Processing ===")
    
    # Find all GRD files
    input_path = Path(input_directory)
    grd_files = list(input_path.glob("S1*_GRD*.zip"))
    
    if not grd_files:
        print(f"No GRD files found in {input_directory}")
        return
    
    print(f"Found {len(grd_files)} GRD files to process")
    
    processor = SNAPProcessor()
    results = []
    
    for i, grd_file in enumerate(grd_files):
        print(f"\nProcessing file {i+1}/{len(grd_files)}: {grd_file.name}")
        
        # Create individual output directory
        file_output_dir = Path(output_directory) / grd_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = basic_grd_processing(str(grd_file), str(file_output_dir))
            results.append({
                'input_file': str(grd_file),
                'output_dir': str(file_output_dir),
                'status': 'success',
                'result': result
            })
            print(f"✓ Successfully processed {grd_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {grd_file.name}: {str(e)}")
            results.append({
                'input_file': str(grd_file),
                'output_dir': str(file_output_dir),
                'status': 'error',
                'error': str(e)
            })
    
    # Summary
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len(results) - successful
    
    print(f"\n=== Batch Processing Summary ===")
    print(f"Total files: {len(grd_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    return results


def custom_workflow_example(input_file, output_dir):
    """
    Example of custom SNAP workflow using GPT class directly.
    
    Args:
        input_file (str): Path to Sentinel-1 file
        output_dir (str): Output directory
    """
    print("=== Custom SNAP Workflow Example ===")
    
    # Initialize GPT processor
    gpt = GPT(product=input_file, outdir=output_dir)
    
    print("Step 1: Apply orbit file...")
    gpt.ApplyOrbitFile()
    
    print("Step 2: Remove thermal noise...")
    gpt.ThermalNoiseRemoval()
    
    print("Step 3: Radiometric calibration...")
    calibrated = gpt.Calibration(
        outputSigmaBand=True,
        outputBetaBand=False,
        outputGammaBand=False
    )
    
    print("Step 4: Speckle filtering...")
    filtered = gpt.SpeckleFiltering(
        filter='Lee',
        filterSizeX=5,
        filterSizeY=5
    )
    
    print("Step 5: Terrain correction...")
    geocoded = gpt.TerrainCorrection(
        demName='SRTM 3Sec',
        pixelSpacingInMeter=10.0
    )
    
    print("Step 6: Convert to dB...")
    final_product = gpt.LinearToFromdB(
        sourceProduct=geocoded,
        nodataValueAtSource=0.0
    )
    
    # Save final product
    output_file = os.path.join(output_dir, "processed_product.dim")
    gpt.Write(
        sourceProduct=final_product,
        file=output_file,
        formatName='BEAM-DIMAP'
    )
    
    print(f"✓ Custom workflow completed: {output_file}")
    return output_file


def visualization_example(processed_data, output_dir):
    """
    Create visualizations of processed SAR data.
    
    Args:
        processed_data: Processed SAR data array
        output_dir (str): Directory to save plots
    """
    print("=== Creating Visualizations ===")
    
    # Create output directory for plots
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Basic intensity plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Linear scale
    axes[0, 0].imshow(processed_data, cmap='gray')
    axes[0, 0].set_title('SAR Intensity (Linear)')
    axes[0, 0].axis('off')
    
    # dB scale
    db_data = 10 * np.log10(processed_data + 1e-10)
    axes[0, 1].imshow(db_data, cmap='gray', vmin=-25, vmax=5)
    axes[0, 1].set_title('SAR Intensity (dB)')
    axes[0, 1].axis('off')
    
    # Histogram
    axes[1, 0].hist(processed_data.flatten(), bins=100, alpha=0.7)
    axes[1, 0].set_xlabel('Intensity (Linear)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Intensity Distribution')
    axes[1, 0].set_yscale('log')
    
    # dB Histogram
    axes[1, 1].hist(db_data.flatten(), bins=100, alpha=0.7)
    axes[1, 1].set_xlabel('Intensity (dB)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('dB Distribution')
    
    plt.tight_layout()
    plot_file = plot_dir / "sar_overview.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {plot_file}")
    return str(plot_file)


def advanced_snap_operations(input_file, output_dir):
    """
    Demonstrate advanced SNAP operations.
    
    Args:
        input_file (str): Path to Sentinel-1 SLC file
        output_dir (str): Output directory
    """
    print("=== Advanced SNAP Operations ===")
    
    gpt = GPT(product=input_file, outdir=output_dir)
    
    # Advanced operations for SLC data
    if "SLC" in input_file:
        print("Processing SLC data with advanced operations...")
        
        # TOPS Deburst (for Sentinel-1 SLC)
        gpt.TOPSAR_Deburst()
        
        # Multilooking
        multilooked = gpt.Multilook(
            nRgLooks=5,
            nAzLooks=1,
            outputIntensity=True,
            grSquarePixel=True
        )
        
        # Polarimetric processing (if dual-pol)
        try:
            pol_params = gpt.PolarimetricParameters(
                parameters=['Sigma0_VV_db', 'Sigma0_VH_db']
            )
            print("✓ Polarimetric parameters computed")
        except:
            print("ℹ Polarimetric processing skipped (single-pol data)")
        
        # Coherence estimation (if applicable)
        try:
            coherence = gpt.CoherenceEstimation(
                cohWinAz=10,
                cohWinRg=3
            )
            print("✓ Coherence estimation completed")
        except:
            print("ℹ Coherence estimation skipped")
    
    else:
        print("Processing GRD data with standard operations...")
        basic_grd_processing(input_file, output_dir)
    
    print("✓ Advanced operations completed")


def main():
    """Main function to run SNAP integration examples."""
    parser = argparse.ArgumentParser(
        description="SNAP Integration Examples for SARPYX"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/sample_s1_grd/",
        help="Input SAR file or directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output/snap_examples",
        help="Output directory"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=['basic', 'batch', 'custom', 'advanced'],
        default='basic',
        help="Processing mode"
    )
    parser.add_argument(
        "--visualize", 
        action='store_true',
        help="Create visualization plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("SARPYX SNAP Integration Examples")
    print("=" * 40)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    print("=" * 40)
    
    try:
        if args.mode == 'basic':
            if os.path.isfile(args.input):
                results = basic_grd_processing(args.input, args.output)
            else:
                print("Basic mode requires a single input file")
                return
                
        elif args.mode == 'batch':
            if os.path.isdir(args.input):
                results = batch_grd_processing(args.input, args.output)
            else:
                print("Batch mode requires an input directory")
                return
                
        elif args.mode == 'custom':
            if os.path.isfile(args.input):
                results = custom_workflow_example(args.input, args.output)
            else:
                print("Custom mode requires a single input file")
                return
                
        elif args.mode == 'advanced':
            if os.path.isfile(args.input):
                results = advanced_snap_operations(args.input, args.output)
            else:
                print("Advanced mode requires a single input file")
                return
        
        # Optional visualization
        if args.visualize and 'results' in locals():
            try:
                # This would need actual data loading implementation
                print("Visualization requires processed data array")
                # visualization_example(processed_data, args.output)
            except Exception as e:
                print(f"Visualization failed: {e}")
        
        print("\n" + "=" * 40)
        print("✓ SNAP integration examples completed successfully!")
        print(f"Check output directory: {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Check your input data and SNAP installation")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
