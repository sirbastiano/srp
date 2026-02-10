#!/usr/bin/env python3
"""
Visualization Gallery Examples
Application: SAR Data Visualization
Complexity: Basic

This example demonstrates various visualization techniques for SAR data
using SARPYX utilities. Covers common plotting patterns and display options.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.patches as mpatches
from pathlib import Path

try:
    from sarpyx.utils import visualization
    from sarpyx.sla import SubLookAnalysis
except ImportError as e:
    print(f"Error importing SARPYX: {e}")
    print("Please install SARPYX: pip install sarpyx")
    sys.exit(1)


def create_sample_data():
    """
    Create sample SAR-like data for demonstration.
    
    Returns:
        dict: Dictionary containing various types of sample data
    """
    print("Creating sample SAR-like data...")
    
    # Simulate SAR intensity data
    np.random.seed(42)
    height, width = 512, 512
    
    # Base intensity with spatial correlation
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)
    
    # Create realistic SAR-like pattern
    base_intensity = (
        0.5 * np.exp(-(X**2 + Y**2) / 0.8) +  # Central bright area
        0.3 * np.exp(-((X-1)**2 + (Y+0.5)**2) / 0.3) +  # Bright spot
        0.1 * np.ones_like(X)  # Background
    )
    
    # Add speckle noise (Gamma distributed)
    speckle = np.random.gamma(1, 1, (height, width))
    intensity = base_intensity * speckle
    
    # Simulated dual-pol data
    vh_intensity = intensity * 0.3 + np.random.gamma(0.5, 0.5, (height, width)) * 0.1
    vv_intensity = intensity + np.random.gamma(1, 1, (height, width)) * 0.2
    
    # Phase data (for complex visualization)
    phase = np.random.uniform(-np.pi, np.pi, (height, width))
    complex_data = np.sqrt(intensity) * np.exp(1j * phase)
    
    # Time series data
    n_dates = 10
    time_series = np.zeros((height, width, n_dates))
    for i in range(n_dates):
        # Add temporal variation
        temporal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / n_dates)
        time_series[:, :, i] = intensity * temporal_factor * np.random.gamma(1, 1, (height, width))
    
    sample_data = {
        'intensity': intensity,
        'vh_intensity': vh_intensity,
        'vv_intensity': vv_intensity,
        'complex_data': complex_data,
        'time_series': time_series,
        'phase': phase
    }
    
    print("✓ Sample data created")
    return sample_data


def basic_intensity_plots(intensity_data, output_dir):
    """
    Create basic intensity visualization plots.
    
    Args:
        intensity_data (np.ndarray): SAR intensity data
        output_dir (str): Output directory for plots
    """
    print("Creating basic intensity plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Linear scale
    im1 = axes[0, 0].imshow(intensity_data, cmap='gray')
    axes[0, 0].set_title('Linear Scale')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # dB scale
    db_data = 10 * np.log10(intensity_data + 1e-10)
    im2 = axes[0, 1].imshow(db_data, cmap='gray', vmin=-25, vmax=5)
    axes[0, 1].set_title('dB Scale')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Custom colormap
    im3 = axes[0, 2].imshow(intensity_data, cmap='viridis')
    axes[0, 2].set_title('Viridis Colormap')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Histogram - Linear
    axes[1, 0].hist(intensity_data.flatten(), bins=100, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Intensity (Linear)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Linear Histogram')
    axes[1, 0].set_yscale('log')
    
    # Histogram - dB
    axes[1, 1].hist(db_data.flatten(), bins=100, alpha=0.7, density=True)
    axes[1, 1].set_xlabel('Intensity (dB)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('dB Histogram')
    
    # Statistics text
    stats_text = f"""
    Statistics:
    Mean: {np.mean(intensity_data):.3f}
    Std: {np.std(intensity_data):.3f}
    Min: {np.min(intensity_data):.3f}
    Max: {np.max(intensity_data):.3f}
    
    dB Statistics:
    Mean: {np.mean(db_data):.1f} dB
    Std: {np.std(db_data):.1f} dB
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_xlim([0, 1])
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Statistics')
    
    plt.tight_layout()
    plot_file = Path(output_dir) / "basic_intensity_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Basic intensity plots saved: {plot_file}")
    return str(plot_file)


def polarimetric_visualization(vv_data, vh_data, output_dir):
    """
    Create polarimetric visualization plots.
    
    Args:
        vv_data (np.ndarray): VV polarization data
        vh_data (np.ndarray): VH polarization data
        output_dir (str): Output directory for plots
    """
    print("Creating polarimetric visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # VV polarization
    vv_db = 10 * np.log10(vv_data + 1e-10)
    im1 = axes[0, 0].imshow(vv_db, cmap='gray', vmin=-25, vmax=5)
    axes[0, 0].set_title('VV Polarization (dB)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # VH polarization
    vh_db = 10 * np.log10(vh_data + 1e-10)
    im2 = axes[0, 1].imshow(vh_db, cmap='gray', vmin=-25, vmax=5)
    axes[0, 1].set_title('VH Polarization (dB)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # RGB Composite (VV=R, VH=G, VV/VH=B)
    ratio = vv_data / (vh_data + 1e-10)
    ratio_norm = (ratio - np.percentile(ratio, 2)) / (np.percentile(ratio, 98) - np.percentile(ratio, 2))
    ratio_norm = np.clip(ratio_norm, 0, 1)
    
    vv_norm = (vv_db - np.percentile(vv_db, 2)) / (np.percentile(vv_db, 98) - np.percentile(vv_db, 2))
    vh_norm = (vh_db - np.percentile(vh_db, 2)) / (np.percentile(vh_db, 98) - np.percentile(vh_db, 2))
    vv_norm = np.clip(vv_norm, 0, 1)
    vh_norm = np.clip(vh_norm, 0, 1)
    
    rgb_composite = np.stack([vv_norm, vh_norm, ratio_norm], axis=-1)
    axes[0, 2].imshow(rgb_composite)
    axes[0, 2].set_title('RGB Composite\n(R:VV, G:VH, B:VV/VH)')
    axes[0, 2].axis('off')
    
    # Polarization ratio
    pol_ratio_db = 10 * np.log10(ratio + 1e-10)
    im4 = axes[1, 0].imshow(pol_ratio_db, cmap='RdBu_r', vmin=-5, vmax=15)
    axes[1, 0].set_title('VV/VH Ratio (dB)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    # Scatter plot VV vs VH
    sample_indices = np.random.choice(vv_data.size, 10000, replace=False)
    vv_sample = vv_db.flatten()[sample_indices]
    vh_sample = vh_db.flatten()[sample_indices]
    
    axes[1, 1].scatter(vh_sample, vv_sample, alpha=0.3, s=1)
    axes[1, 1].set_xlabel('VH (dB)')
    axes[1, 1].set_ylabel('VV (dB)')
    axes[1, 1].set_title('VV vs VH Scatter Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add identity line and reference lines
    axes[1, 1].plot([-30, 10], [-30, 10], 'r--', alpha=0.7, label='VV = VH')
    axes[1, 1].plot([-30, 10], [-25, 15], 'g--', alpha=0.7, label='VV = VH + 5dB')
    axes[1, 1].legend()
    
    # Histograms comparison
    axes[1, 2].hist(vv_db.flatten(), bins=50, alpha=0.7, label='VV', density=True)
    axes[1, 2].hist(vh_db.flatten(), bins=50, alpha=0.7, label='VH', density=True)
    axes[1, 2].set_xlabel('Backscatter (dB)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Polarization Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = Path(output_dir) / "polarimetric_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Polarimetric plots saved: {plot_file}")
    return str(plot_file)


def complex_data_visualization(complex_data, output_dir):
    """
    Visualize complex SAR data (amplitude and phase).
    
    Args:
        complex_data (np.ndarray): Complex SAR data
        output_dir (str): Output directory for plots
    """
    print("Creating complex data visualizations...")
    
    amplitude = np.abs(complex_data)
    phase = np.angle(complex_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Amplitude
    amp_db = 20 * np.log10(amplitude + 1e-10)
    im1 = axes[0, 0].imshow(amp_db, cmap='gray', vmin=-40, vmax=10)
    axes[0, 0].set_title('Amplitude (dB)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # Phase
    im2 = axes[0, 1].imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Phase (radians)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Complex representation (HSV: Hue=Phase, Value=Amplitude)
    # Normalize amplitude for HSV representation
    amp_norm = (amp_db - np.percentile(amp_db, 2)) / (np.percentile(amp_db, 98) - np.percentile(amp_db, 2))
    amp_norm = np.clip(amp_norm, 0, 1)
    
    # Convert to HSV
    hsv_image = np.zeros((complex_data.shape[0], complex_data.shape[1], 3))
    hsv_image[:, :, 0] = (phase + np.pi) / (2 * np.pi)  # Hue: 0-1
    hsv_image[:, :, 1] = 1.0  # Saturation: full
    hsv_image[:, :, 2] = amp_norm  # Value: normalized amplitude
    
    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    rgb_image = hsv_to_rgb(hsv_image)
    
    axes[0, 2].imshow(rgb_image)
    axes[0, 2].set_title('Complex Representation\n(Hue=Phase, Brightness=Amplitude)')
    axes[0, 2].axis('off')
    
    # Phase histogram
    axes[1, 0].hist(phase.flatten(), bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Phase (radians)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Phase Distribution')
    axes[1, 0].set_xlim([-np.pi, np.pi])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Amplitude histogram
    axes[1, 1].hist(amplitude.flatten(), bins=50, alpha=0.7, density=True)
    axes[1, 1].set_xlabel('Amplitude (Linear)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Amplitude Distribution')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Complex plane plot (subset of data)
    sample_indices = np.random.choice(complex_data.size, 5000, replace=False)
    complex_sample = complex_data.flatten()[sample_indices]
    
    axes[1, 2].scatter(np.real(complex_sample), np.imag(complex_sample), 
                      alpha=0.3, s=1, c=np.abs(complex_sample), cmap='viridis')
    axes[1, 2].set_xlabel('Real Part')
    axes[1, 2].set_ylabel('Imaginary Part')
    axes[1, 2].set_title('Complex Plane Representation')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_aspect('equal')
    
    plt.tight_layout()
    plot_file = Path(output_dir) / "complex_data_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Complex data plots saved: {plot_file}")
    return str(plot_file)


def time_series_visualization(time_series_data, output_dir):
    """
    Visualize time series SAR data.
    
    Args:
        time_series_data (np.ndarray): Time series data (height, width, time)
        output_dir (str): Output directory for plots
    """
    print("Creating time series visualizations...")
    
    height, width, n_dates = time_series_data.shape
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Show first few time steps
    for i in range(min(6, n_dates)):
        row, col = i // 3, i % 3
        
        data_slice = time_series_data[:, :, i]
        db_slice = 10 * np.log10(data_slice + 1e-10)
        
        im = axes[row, col].imshow(db_slice, cmap='gray', vmin=-25, vmax=5)
        axes[row, col].set_title(f'Time Step {i+1}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    
    # Time series statistics
    mean_ts = np.mean(time_series_data, axis=(0, 1))
    std_ts = np.std(time_series_data, axis=(0, 1))
    
    axes[2, 0].plot(range(1, n_dates+1), mean_ts, 'o-', linewidth=2, markersize=6)
    axes[2, 0].fill_between(range(1, n_dates+1), 
                           mean_ts - std_ts, mean_ts + std_ts, 
                           alpha=0.3)
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Mean Intensity')
    axes[2, 0].set_title('Temporal Mean ± Std')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Coefficient of variation
    cv = std_ts / (mean_ts + 1e-10)
    axes[2, 1].plot(range(1, n_dates+1), cv, 's-', linewidth=2, markersize=6, color='red')
    axes[2, 1].set_xlabel('Time Step')
    axes[2, 1].set_ylabel('Coefficient of Variation')
    axes[2, 1].set_title('Temporal Variability')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Temporal correlation matrix
    # Reshape for correlation calculation
    reshaped_data = time_series_data.reshape(-1, n_dates)
    correlation_matrix = np.corrcoef(reshaped_data.T)
    
    im_corr = axes[2, 2].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2, 2].set_title('Temporal Correlation Matrix')
    axes[2, 2].set_xlabel('Time Step')
    axes[2, 2].set_ylabel('Time Step')
    plt.colorbar(im_corr, ax=axes[2, 2], fraction=0.046)
    
    plt.tight_layout()
    plot_file = Path(output_dir) / "time_series_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Time series plots saved: {plot_file}")
    return str(plot_file)


def advanced_visualization_techniques(intensity_data, output_dir):
    """
    Demonstrate advanced visualization techniques.
    
    Args:
        intensity_data (np.ndarray): SAR intensity data
        output_dir (str): Output directory for plots
    """
    print("Creating advanced visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Power law normalization
    im1 = axes[0, 0].imshow(intensity_data, cmap='gray', norm=PowerNorm(gamma=0.5))
    axes[0, 0].set_title('Power Law Normalization (γ=0.5)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # 2. Log normalization
    im2 = axes[0, 1].imshow(intensity_data + 1e-10, cmap='viridis', norm=LogNorm())
    axes[0, 1].set_title('Log Normalization')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # 3. Adaptive histogram equalization
    from scipy import ndimage
    # Simple contrast enhancement
    enhanced = ndimage.uniform_filter(intensity_data, size=3)
    contrast = intensity_data / (enhanced + 1e-10)
    
    im3 = axes[0, 2].imshow(contrast, cmap='gray')
    axes[0, 2].set_title('Contrast Enhanced')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # 4. Edge detection
    from scipy.ndimage import sobel
    edges = np.sqrt(sobel(intensity_data, axis=0)**2 + sobel(intensity_data, axis=1)**2)
    
    im4 = axes[1, 0].imshow(edges, cmap='hot')
    axes[1, 0].set_title('Edge Detection (Sobel)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    # 5. Multi-scale analysis
    scales = [1, 2, 4, 8]
    multi_scale = np.zeros_like(intensity_data)
    
    for scale in scales:
        smoothed = ndimage.gaussian_filter(intensity_data, sigma=scale)
        multi_scale += smoothed / len(scales)
    
    im5 = axes[1, 1].imshow(multi_scale, cmap='viridis')
    axes[1, 1].set_title('Multi-scale Average')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    # 6. Texture analysis
    # Simple texture measure: local standard deviation
    texture = ndimage.generic_filter(intensity_data, np.std, size=7)
    
    im6 = axes[1, 2].imshow(texture, cmap='plasma')
    axes[1, 2].set_title('Texture (Local Std)')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plot_file = Path(output_dir) / "advanced_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Advanced visualization saved: {plot_file}")
    return str(plot_file)


def create_visualization_summary(output_dir):
    """
    Create a summary document of all visualization techniques.
    
    Args:
        output_dir (str): Output directory
    """
    summary_text = """
# SAR Data Visualization Summary

This document summarizes the visualization techniques demonstrated in the SARPYX examples.

## Basic Visualizations

### 1. Intensity Plots
- **Linear Scale**: Direct display of intensity values
- **dB Scale**: Logarithmic scale for better dynamic range
- **Custom Colormaps**: Enhanced visual interpretation

### 2. Statistical Displays
- **Histograms**: Distribution analysis
- **Statistics Tables**: Numerical summaries
- **Box Plots**: Quartile analysis

## Polarimetric Visualizations

### 1. Single Polarization
- **VV/VH Channels**: Individual polarization display
- **dB Scaling**: Enhanced contrast visualization

### 2. Multi-polarization
- **RGB Composites**: Combined polarization display
- **Ratio Images**: Polarization contrast analysis
- **Scatter Plots**: Inter-polarization relationships

## Complex Data Visualization

### 1. Amplitude and Phase
- **Separate Display**: Individual component analysis
- **HSV Representation**: Combined amplitude-phase display
- **Complex Plane**: I-Q diagram representation

### 2. Interferometric Products
- **Interferograms**: Phase difference display
- **Coherence Maps**: Correlation visualization
- **Unwrapped Phase**: Continuous phase representation

## Time Series Visualization

### 1. Temporal Analysis
- **Animation Sequences**: Change detection
- **Statistical Trends**: Mean and variance analysis
- **Correlation Matrices**: Temporal relationships

### 2. Change Detection
- **Difference Images**: Before/after comparison
- **Ratio Images**: Relative change analysis
- **Temporal Profiles**: Point-wise time series

## Advanced Techniques

### 1. Enhancement Methods
- **Power Law Normalization**: Gamma correction
- **Log Normalization**: Dynamic range compression
- **Adaptive Enhancement**: Local contrast improvement

### 2. Feature Extraction
- **Edge Detection**: Gradient-based features
- **Texture Analysis**: Spatial pattern recognition
- **Multi-scale Analysis**: Hierarchical feature extraction

## Best Practices

### 1. Color Schemes
- **Grayscale**: Traditional SAR display
- **Viridis/Plasma**: Perceptually uniform colormaps
- **HSV**: Complex data representation

### 2. Scaling and Normalization
- **Percentile Clipping**: Robust dynamic range
- **Local Normalization**: Adaptive contrast
- **Multi-band Scaling**: Consistent visualization

### 3. Interactive Elements
- **Zoom and Pan**: Detail exploration
- **Overlay Capabilities**: Multi-layer analysis
- **Animation Controls**: Temporal navigation

## Usage Guidelines

1. **Start with basic intensity plots** for initial data exploration
2. **Use dB scaling** for SAR amplitude data
3. **Apply statistical analysis** to understand data characteristics
4. **Choose appropriate colormaps** based on data type
5. **Implement interactive features** for detailed analysis

## Code Examples

Each visualization type includes:
- Complete implementation code
- Parameter configuration options
- Output format specifications
- Quality assessment metrics

## Output Formats

- **PNG**: High-quality static images
- **PDF**: Vector graphics for publications
- **Interactive HTML**: Web-based exploration
- **Animation GIF**: Temporal sequences
"""

    summary_file = Path(output_dir) / "visualization_summary.md"
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    print(f"✓ Visualization summary saved: {summary_file}")
    return str(summary_file)


def main():
    """Main function to run visualization examples."""
    parser = argparse.ArgumentParser(
        description="SAR Data Visualization Examples for SARPYX"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output/visualization_gallery",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--type", 
        type=str, 
        choices=['all', 'basic', 'polarimetric', 'complex', 'timeseries', 'advanced'],
        default='all',
        help="Type of visualization to create"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=300,
        help="DPI for saved images"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("SARPYX Visualization Gallery Examples")
    print("=" * 40)
    print(f"Output: {args.output}")
    print(f"Type: {args.type}")
    print(f"DPI: {args.dpi}")
    print("=" * 40)
    
    # Create sample data
    sample_data = create_sample_data()
    
    created_plots = []
    
    try:
        if args.type in ['all', 'basic']:
            plot_file = basic_intensity_plots(sample_data['intensity'], args.output)
            created_plots.append(plot_file)
        
        if args.type in ['all', 'polarimetric']:
            plot_file = polarimetric_visualization(
                sample_data['vv_intensity'], 
                sample_data['vh_intensity'], 
                args.output
            )
            created_plots.append(plot_file)
        
        if args.type in ['all', 'complex']:
            plot_file = complex_data_visualization(sample_data['complex_data'], args.output)
            created_plots.append(plot_file)
        
        if args.type in ['all', 'timeseries']:
            plot_file = time_series_visualization(sample_data['time_series'], args.output)
            created_plots.append(plot_file)
        
        if args.type in ['all', 'advanced']:
            plot_file = advanced_visualization_techniques(sample_data['intensity'], args.output)
            created_plots.append(plot_file)
        
        # Create summary document
        summary_file = create_visualization_summary(args.output)
        
        print("\n" + "=" * 40)
        print("✓ Visualization gallery completed successfully!")
        print(f"Created {len(created_plots)} plot files:")
        for plot_file in created_plots:
            print(f"  - {Path(plot_file).name}")
        print(f"Summary: {Path(summary_file).name}")
        print(f"Check output directory: {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
