#!/usr/bin/env python3
"""
Basic Sub-Look Analysis Example

Description: Demonstrates the fundamental sub-look decomposition process using SARPyX
Data: Sentinel-1 SLC product (ZIP format)
Output: Sub-look images and quality metrics
Dependencies: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# SARPyX imports
from sarpyx.sla import SubLookAnalysis
from sarpyx.utils import show_image


def setup_arguments():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description='Basic Sub-Look Analysis Example')
    parser.add_argument('--input', '-i', 
                       default='data/sentinel1/S1A_IW_SLC_sample.zip',
                       help='Input SAR product path')
    parser.add_argument('--output', '-o', 
                       default='output',
                       help='Output directory')
    parser.add_argument('--looks', '-l', 
                       type=int, default=3,
                       help='Number of sub-looks to generate')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()


def validate_input(product_path):
    """Validate that input file exists and is readable."""
    path = Path(product_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {product_path}")
    if not path.suffix.lower() == '.zip':
        print(f"Warning: Expected ZIP file, got {path.suffix}")
    return path


def configure_sla(sla, n_looks=3, verbose=False):
    """Configure SubLookAnalysis parameters."""
    sla.choice = 1                    # Azimuth processing
    sla.numberOfLooks = n_looks       # Number of sub-looks
    sla.centroidSeparations = 700     # Frequency separation (Hz)
    sla.subLookBandwidth = 700        # Bandwidth per sub-look (Hz)
    
    if verbose:
        print(f"Configuration:")
        print(f"  Processing direction: {'Azimuth' if sla.choice == 1 else 'Range'}")
        print(f"  Number of looks: {sla.numberOfLooks}")
        print(f"  Centroid separations: {sla.centroidSeparations} Hz")
        print(f"  Sub-look bandwidth: {sla.subLookBandwidth} Hz")


def process_sublooks(sla, verbose=False):
    """Execute the complete sub-look processing chain."""
    
    print("Starting sub-look processing...")
    
    # Step 1: Frequency computation
    if verbose:
        print("  1. Computing frequency bins...")
    try:
        sla.frequencyComputation()
        if verbose:
            print(f"     ✓ Computed {len(sla.freqCentr)} frequency bins")
            for i, freq in enumerate(sla.freqCentr):
                print(f"       Look {i+1}: {freq:6.1f} Hz")
    except AssertionError as e:
        raise ValueError(f"Frequency computation failed: {e}")
    
    # Step 2: Spectrum computation
    if verbose:
        print("  2. Computing spectrum...")
    sla.SpectrumComputation(VERBOSE=verbose)
    if verbose:
        print(f"     ✓ Spectrum shape: {sla.SpectrumOneDim.shape}")
    
    # Step 3: De-weighting
    if verbose:
        print("  3. Applying de-weighting...")
    sla.AncillaryDeWe(VERBOSE=verbose)
    if verbose:
        print("     ✓ De-weighting complete")
    
    # Step 4: Sub-look generation
    if verbose:
        print("  4. Generating sub-look images...")
    sla.Generation(VERBOSE=verbose)
    if verbose:
        print(f"     ✓ Generated {sla.Looks.shape[0]} sub-looks")
        print(f"       Image size: {sla.Looks.shape[1]} x {sla.Looks.shape[2]}")
    
    print("Sub-look processing complete!")
    return sla.Looks


def analyze_results(sla, verbose=False):
    """Analyze and assess the quality of sub-look results."""
    
    print("Analyzing results...")
    
    # Basic statistics
    results = {
        'n_looks': sla.numberOfLooks,
        'image_shape': sla.Looks.shape,
        'mean_amplitudes': [],
        'std_amplitudes': [],
        'correlations': []
    }
    
    # Calculate statistics for each sub-look
    for i in range(sla.numberOfLooks):
        amplitude = np.abs(sla.Looks[i])
        results['mean_amplitudes'].append(np.mean(amplitude))
        results['std_amplitudes'].append(np.std(amplitude))
    
    # Calculate cross-correlations
    for i in range(sla.numberOfLooks):
        for j in range(i+1, sla.numberOfLooks):
            corr = np.abs(np.corrcoef(
                sla.Looks[i].flatten(),
                sla.Looks[j].flatten()
            )[0, 1])
            results['correlations'].append((i, j, corr))
    
    if verbose:
        print("Quality Assessment:")
        print(f"  Mean amplitudes: {[f'{x:.3f}' for x in results['mean_amplitudes']]}")
        print(f"  Std deviations:  {[f'{x:.3f}' for x in results['std_amplitudes']]}")
        print("  Cross-correlations:")
        for i, j, corr in results['correlations']:
            print(f"    Looks {i+1}-{j+1}: {corr:.3f}")
    
    return results


def visualize_results(sla, output_dir, verbose=False):
    """Create visualizations of the sub-look analysis results."""
    
    print("Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create main results figure
    n_looks = sla.numberOfLooks
    fig, axes = plt.subplots(2, n_looks, figsize=(4*n_looks, 8))
    if n_looks == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Sub-Look Analysis Results', fontsize=16)
    
    # Display amplitude images
    for i in range(n_looks):
        amplitude = np.abs(sla.Looks[i])
        
        im1 = axes[0, i].imshow(amplitude, cmap='gray', aspect='auto')
        axes[0, i].set_title(f'Sub-look {i+1} Amplitude\n(Freq: {sla.freqCentr[i]:.0f} Hz)')
        axes[0, i].set_xlabel('Range')
        axes[0, i].set_ylabel('Azimuth')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Display phase images
        phase = np.angle(sla.Looks[i])
        im2 = axes[1, i].imshow(phase, cmap='hsv', aspect='auto', vmin=-np.pi, vmax=np.pi)
        axes[1, i].set_title(f'Sub-look {i+1} Phase')
        axes[1, i].set_xlabel('Range')
        axes[1, i].set_ylabel('Azimuth')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(output_path / 'sublook_results.png', dpi=150, bbox_inches='tight')
    
    if verbose:
        print(f"  Saved main results: {output_path / 'sublook_results.png'}")
    
    # Create comparison figure if multiple looks
    if n_looks >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Intensity images
        intensity1 = np.abs(sla.Looks[0])**2
        intensity2 = np.abs(sla.Looks[1])**2
        
        im1 = axes[0].imshow(intensity1, cmap='hot', aspect='auto')
        axes[0].set_title('Intensity - Sub-look 1')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(intensity2, cmap='hot', aspect='auto')
        axes[1].set_title('Intensity - Sub-look 2')
        plt.colorbar(im2, ax=axes[1])
        
        # Intensity difference
        intensity_diff = (intensity1 - intensity2) / (intensity1 + intensity2 + 1e-10)
        im3 = axes[2].imshow(intensity_diff, cmap='RdBu', vmin=-0.5, vmax=0.5, aspect='auto')
        axes[2].set_title('Normalized Intensity Difference')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(output_path / 'intensity_comparison.png', dpi=150, bbox_inches='tight')
        
        if verbose:
            print(f"  Saved comparison: {output_path / 'intensity_comparison.png'}")
    
    # Show plots if verbose
    if verbose:
        plt.show()
    else:
        plt.close('all')


def save_results(sla, results, output_dir, verbose=False):
    """Save processing results to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save sub-look data
    np.savez_compressed(
        output_path / 'sublook_data.npz',
        looks=sla.Looks,
        frequencies=sla.freqCentr,
        freq_min=sla.freqMin,
        freq_max=sla.freqMax
    )
    
    # Save metadata and quality metrics
    metadata = {
        'processing_parameters': {
            'choice': sla.choice,
            'numberOfLooks': sla.numberOfLooks,
            'centroidSeparations': sla.centroidSeparations,
            'subLookBandwidth': sla.subLookBandwidth
        },
        'quality_metrics': results
    }
    
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                serializable_metadata[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_metadata[key][k] = v.tolist()
                    else:
                        serializable_metadata[key][k] = v
            else:
                serializable_metadata[key] = value
        
        json.dump(serializable_metadata, f, indent=2)
    
    if verbose:
        print(f"Results saved to: {output_path}")
        print(f"  - sublook_data.npz: Sub-look complex data")
        print(f"  - metadata.json: Processing parameters and quality metrics")
        print(f"  - *.png: Visualization plots")


def main():
    """Main example function."""
    
    # Parse command line arguments
    args = setup_arguments()
    
    print("SARPyX Basic Sub-Look Analysis Example")
    print("=====================================")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Number of looks: {args.looks}")
    print()
    
    try:
        # Validate input
        product_path = validate_input(args.input)
        
        # Initialize SubLookAnalysis
        print("Initializing SubLookAnalysis...")
        sla = SubLookAnalysis(str(product_path))
        
        # Configure parameters
        configure_sla(sla, n_looks=args.looks, verbose=args.verbose)
        
        # Process sub-looks
        sublook_data = process_sublooks(sla, verbose=args.verbose)
        
        # Analyze results
        results = analyze_results(sla, verbose=args.verbose)
        
        # Create visualizations
        visualize_results(sla, args.output, verbose=args.verbose)
        
        # Save results
        save_results(sla, results, args.output, verbose=args.verbose)
        
        print("\nExample completed successfully!")
        print(f"Results available in: {args.output}/")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Generated {results['n_looks']} sub-looks")
        print(f"  Image dimensions: {results['image_shape'][1]} x {results['image_shape'][2]}")
        print(f"  Average correlation: {np.mean([c[2] for c in results['correlations']]):.3f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the input file exists or use --input to specify a different file")
        return 1
    
    except ValueError as e:
        print(f"Processing error: {e}")
        print("Try adjusting the number of looks or other parameters")
        return 1
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your input data and try again")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
