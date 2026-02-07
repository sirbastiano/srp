# Tutorial 3: Visualization and Quality Assessment

Learn to effectively visualize SLA results and assess processing quality using sarpyx's built-in tools.

## Overview

This tutorial covers:
- Visualization techniques for SLA decomposition results
- Quality metrics and assessment methods  
- Interactive plotting and analysis
- Histogram analysis and statistical visualization
- Multi-band visualization strategies

**Duration**: 25 minutes  
**Prerequisites**: Tutorial 1 completed  
**Data**: Results from Tutorial 1

## 1. Setting Up Visualization Environment

```python
import numpy as np
import matplotlib.pyplot as plt
from sarpyx.sla import SubLookAnalysis
from sarpyx.utils.viz import show_image, show_histogram, image_histogram_equalization
from sarpyx.utils.io import save_matlab_mat
import seaborn as sns

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load data from Tutorial 1 (adjust path as needed)
data_path = "path/to/your/sentinel1_data.dim"
output_dir = "tutorial3_outputs"
```

## 2. Basic SLA Visualization

### 2.1 Load and Process Data

```python
# Initialize SLA processor
sla = SubLookAnalysis()

# Load and process data
sla.load_data(data_path, band_name='Sigma0_VV_slc')
sla.set_sublook_parameters(num_looks=4, overlap_factor=0.5)
sla.decompose()

# Get decomposition results
sublooks = sla.get_sublooks()
master_image = sla.get_master_image()
coherence = sla.get_coherence_matrix()

print(f"Number of sub-looks: {len(sublooks)}")
print(f"Master image shape: {master_image.shape}")
print(f"Coherence matrix shape: {coherence.shape}")
```

### 2.2 Visualize Sub-look Images

```python
# Display individual sub-looks
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (ax, sublook) in enumerate(zip(axes, sublooks[:4])):
    # Convert to dB scale for visualization
    sublook_db = 10 * np.log10(np.abs(sublook) + 1e-10)
    
    im = show_image(sublook_db, 
                   title=f'Sub-look {i+1} (dB)',
                   ax=ax,
                   cmap='viridis',
                   vmin=np.percentile(sublook_db, 5),
                   vmax=np.percentile(sublook_db, 95))
    
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(f'{output_dir}/sublooks_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2.3 Master Image Visualization

```python
# Visualize master image with different enhancement techniques
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original master image
master_db = 10 * np.log10(np.abs(master_image) + 1e-10)
im1 = show_image(master_db, 
                title='Master Image (dB)',
                ax=axes[0],
                cmap='gray')
plt.colorbar(im1, ax=axes[0])

# Histogram equalized version
master_eq = image_histogram_equalization(master_db)
im2 = show_image(master_eq,
                title='Histogram Equalized',
                ax=axes[1], 
                cmap='gray')
plt.colorbar(im2, ax=axes[1])

# Adaptive contrast enhancement
master_adaptive = np.clip((master_db - np.percentile(master_db, 2)) / 
                         (np.percentile(master_db, 98) - np.percentile(master_db, 2)), 0, 1)
im3 = show_image(master_adaptive,
                title='Adaptive Contrast',
                ax=axes[2],
                cmap='gray')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(f'{output_dir}/master_image_enhancements.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 3. Statistical Analysis and Quality Assessment

### 3.1 Intensity Distribution Analysis

```python
# Analyze intensity distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histogram of master image
show_histogram(master_db, 
              title='Master Image Intensity Distribution',
              ax=axes[0,0],
              bins=100,
              alpha=0.7)
axes[0,0].set_xlabel('Intensity (dB)')
axes[0,0].set_ylabel('Frequency')

# Compare sub-look distributions
sublook_intensities = []
for i, sublook in enumerate(sublooks):
    sublook_db = 10 * np.log10(np.abs(sublook) + 1e-10)
    sublook_intensities.append(sublook_db.flatten())
    
    if i < 4:  # Plot first 4 sub-looks
        axes[0,1].hist(sublook_db.flatten(), bins=50, alpha=0.5, 
                      label=f'Sub-look {i+1}', density=True)

axes[0,1].set_xlabel('Intensity (dB)')
axes[0,1].set_ylabel('Density')
axes[0,1].set_title('Sub-look Intensity Distributions')
axes[0,1].legend()

# Statistical comparison
stats_data = []
for i, intensities in enumerate(sublook_intensities):
    stats_data.append({
        'Sub-look': i+1,
        'Mean': np.mean(intensities),
        'Std': np.std(intensities),
        'Median': np.median(intensities),
        'IQR': np.percentile(intensities, 75) - np.percentile(intensities, 25)
    })

import pandas as pd
stats_df = pd.DataFrame(stats_data)
print("Statistical Summary of Sub-looks:")
print(stats_df)

# Box plot comparison
bp = axes[1,0].boxplot(sublook_intensities[:8], labels=[f'SL{i+1}' for i in range(8)])
axes[1,0].set_xlabel('Sub-look')
axes[1,0].set_ylabel('Intensity (dB)')
axes[1,0].set_title('Sub-look Intensity Box Plots')

# Coefficient of variation analysis
cv_values = [np.std(intensities)/np.mean(intensities) for intensities in sublook_intensities]
axes[1,1].plot(range(1, len(cv_values)+1), cv_values, 'o-')
axes[1,1].set_xlabel('Sub-look Number')
axes[1,1].set_ylabel('Coefficient of Variation')
axes[1,1].set_title('Sub-look Variability Assessment')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/statistical_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.2 Coherence Analysis

```python
# Analyze coherence between sub-looks
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Coherence magnitude visualization
coherence_mag = np.abs(coherence)
im1 = show_image(coherence_mag,
                title='Inter Sub-look Coherence Magnitude',
                ax=axes[0,0],
                cmap='hot')
plt.colorbar(im1, ax=axes[0,0])

# Coherence phase visualization  
coherence_phase = np.angle(coherence)
im2 = show_image(coherence_phase,
                title='Inter Sub-look Coherence Phase',
                ax=axes[0,1],
                cmap='hsv')
plt.colorbar(im2, ax=axes[0,1])

# Coherence statistics
mean_coherence = np.mean(coherence_mag, axis=(0,1))
axes[1,0].plot(range(len(mean_coherence)), mean_coherence, 'o-')
axes[1,0].set_xlabel('Sub-look Pair Index')
axes[1,0].set_ylabel('Mean Coherence')
axes[1,0].set_title('Average Coherence Between Sub-looks')
axes[1,0].grid(True, alpha=0.3)

# Coherence histogram
axes[1,1].hist(coherence_mag.flatten(), bins=50, alpha=0.7, density=True)
axes[1,1].set_xlabel('Coherence Magnitude')
axes[1,1].set_ylabel('Density')
axes[1,1].set_title('Coherence Distribution')
axes[1,1].axvline(np.mean(coherence_mag), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(coherence_mag):.3f}')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/coherence_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Mean coherence: {np.mean(coherence_mag):.3f}")
print(f"Coherence std: {np.std(coherence_mag):.3f}")
print(f"Min coherence: {np.min(coherence_mag):.3f}")
print(f"Max coherence: {np.max(coherence_mag):.3f}")
```

## 4. Quality Metrics and Assessment

### 4.1 Signal-to-Noise Ratio Assessment

```python
def calculate_snr(image, noise_percentile=10):
    """Calculate SNR using noise floor estimation"""
    signal_power = np.mean(np.abs(image)**2)
    noise_floor = np.percentile(np.abs(image)**2, noise_percentile)
    snr_db = 10 * np.log10(signal_power / noise_floor)
    return snr_db

# Calculate SNR for each sub-look
snr_values = []
for i, sublook in enumerate(sublooks):
    snr = calculate_snr(sublook)
    snr_values.append(snr)
    print(f"Sub-look {i+1} SNR: {snr:.2f} dB")

# Plot SNR comparison
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(snr_values)+1), snr_values, 'o-', linewidth=2, markersize=8)
plt.xlabel('Sub-look Number')
plt.ylabel('SNR (dB)')
plt.title('Signal-to-Noise Ratio per Sub-look')
plt.grid(True, alpha=0.3)
plt.axhline(np.mean(snr_values), color='red', linestyle='--', 
           label=f'Mean SNR: {np.mean(snr_values):.2f} dB')
plt.legend()
plt.savefig(f'{output_dir}/snr_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4.2 Speckle Reduction Assessment

```python
# Compare speckle characteristics
def speckle_index(image):
    """Calculate speckle index (std/mean)"""
    return np.std(np.abs(image)) / np.mean(np.abs(image))

# Calculate speckle indices
original_speckle = speckle_index(master_image)
sublook_speckles = [speckle_index(sl) for sl in sublooks]

print(f"Original image speckle index: {original_speckle:.3f}")
print(f"Average sub-look speckle index: {np.mean(sublook_speckles):.3f}")
print(f"Speckle reduction factor: {original_speckle/np.mean(sublook_speckles):.2f}")

# Visualize speckle reduction
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot speckle indices
axes[0].bar(['Original', 'Sub-looks Mean'], 
           [original_speckle, np.mean(sublook_speckles)],
           color=['blue', 'orange'], alpha=0.7)
axes[0].set_ylabel('Speckle Index')
axes[0].set_title('Speckle Reduction Assessment')

# Plot individual sub-look speckle indices
axes[1].plot(range(1, len(sublook_speckles)+1), sublook_speckles, 'o-')
axes[1].axhline(original_speckle, color='red', linestyle='--', 
               label='Original Image')
axes[1].set_xlabel('Sub-look Number')
axes[1].set_ylabel('Speckle Index')
axes[1].set_title('Sub-look Speckle Characteristics')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/speckle_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 5. Interactive Analysis and ROI Selection

### 5.1 Region of Interest Analysis

```python
# Define ROI coordinates (adjust based on your data)
roi_coords = {
    'water': [100, 150, 200, 250],    # [x1, x2, y1, y2]
    'land': [300, 400, 100, 200],
    'urban': [150, 250, 300, 400]
}

# Analyze each ROI
roi_stats = {}

for roi_name, coords in roi_coords.items():
    x1, x2, y1, y2 = coords
    
    # Extract ROI from master image
    roi_master = master_image[y1:y2, x1:x2]
    
    # Extract ROI from each sub-look
    roi_sublooks = [sl[y1:y2, x1:x2] for sl in sublooks]
    
    # Calculate statistics
    roi_stats[roi_name] = {
        'master_mean': np.mean(np.abs(roi_master)),
        'master_std': np.std(np.abs(roi_master)),
        'sublook_means': [np.mean(np.abs(sl)) for sl in roi_sublooks],
        'sublook_stds': [np.std(np.abs(sl)) for sl in roi_sublooks],
        'coherence_roi': coherence[y1:y2, x1:x2, :]
    }

# Visualize ROI analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot ROI locations on master image
master_display = 10 * np.log10(np.abs(master_image) + 1e-10)
im = axes[0,0].imshow(master_display, cmap='gray', 
                     vmin=np.percentile(master_display, 5),
                     vmax=np.percentile(master_display, 95))

# Mark ROIs
colors = ['red', 'green', 'blue']
for i, (roi_name, coords) in enumerate(roi_coords.items()):
    x1, x2, y1, y2 = coords
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, color=colors[i], linewidth=2)
    axes[0,0].add_patch(rect)
    axes[0,0].text(x1, y1-5, roi_name, color=colors[i], fontweight='bold')

axes[0,0].set_title('ROI Locations')
plt.colorbar(im, ax=axes[0,0])

# Compare ROI statistics
roi_names = list(roi_stats.keys())
master_means = [roi_stats[name]['master_mean'] for name in roi_names]
sublook_means = [[np.mean(roi_stats[name]['sublook_means'])] for name in roi_names]

x_pos = np.arange(len(roi_names))
axes[0,1].bar(x_pos - 0.2, master_means, 0.4, label='Master', alpha=0.7)
axes[0,1].bar(x_pos + 0.2, [s[0] for s in sublook_means], 0.4, label='Sub-looks', alpha=0.7)
axes[0,1].set_xlabel('ROI')
axes[0,1].set_ylabel('Mean Intensity')
axes[0,1].set_title('ROI Intensity Comparison')
axes[0,1].set_xticks(x_pos)
axes[0,1].set_xticklabels(roi_names)
axes[0,1].legend()

# ROI coherence analysis
for i, roi_name in enumerate(roi_names):
    coherence_roi = roi_stats[roi_name]['coherence_roi']
    mean_coherence_roi = np.mean(np.abs(coherence_roi))
    axes[1,0].bar(i, mean_coherence_roi, color=colors[i], alpha=0.7)

axes[1,0].set_xlabel('ROI')
axes[1,0].set_ylabel('Mean Coherence')
axes[1,0].set_title('ROI Coherence Comparison')
axes[1,0].set_xticks(range(len(roi_names)))
axes[1,0].set_xticklabels(roi_names)

# Variability analysis
variability = []
for roi_name in roi_names:
    sublook_means = roi_stats[roi_name]['sublook_means']
    cv = np.std(sublook_means) / np.mean(sublook_means)
    variability.append(cv)

axes[1,1].bar(range(len(roi_names)), variability, color=colors, alpha=0.7)
axes[1,1].set_xlabel('ROI')
axes[1,1].set_ylabel('Coefficient of Variation')
axes[1,1].set_title('Sub-look Variability by ROI')
axes[1,1].set_xticks(range(len(roi_names)))
axes[1,1].set_xticklabels(roi_names)

plt.tight_layout()
plt.savefig(f'{output_dir}/roi_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print ROI statistics
for roi_name, stats in roi_stats.items():
    print(f"\n{roi_name.upper()} ROI Statistics:")
    print(f"  Master mean: {stats['master_mean']:.6f}")
    print(f"  Sub-look mean CV: {np.std(stats['sublook_means'])/np.mean(stats['sublook_means']):.3f}")
    print(f"  Mean coherence: {np.mean(np.abs(stats['coherence_roi'])):.3f}")
```

## 6. Quality Report Generation

### 6.1 Automated Quality Assessment

```python
def generate_quality_report(sla_processor, output_path):
    """Generate comprehensive quality assessment report"""
    
    report = {
        'processing_parameters': {
            'num_looks': sla_processor.num_looks,
            'overlap_factor': sla_processor.overlap_factor,
            'window_size': getattr(sla_processor, 'window_size', 'N/A')
        },
        'data_statistics': {},
        'quality_metrics': {},
        'recommendations': []
    }
    
    # Get processed data
    sublooks = sla_processor.get_sublooks()
    master = sla_processor.get_master_image()
    coherence = sla_processor.get_coherence_matrix()
    
    # Basic statistics
    report['data_statistics'] = {
        'num_sublooks': len(sublooks),
        'image_dimensions': master.shape,
        'master_intensity_range': [float(np.min(np.abs(master))), float(np.max(np.abs(master)))],
        'master_mean_intensity': float(np.mean(np.abs(master))),
        'coherence_range': [float(np.min(np.abs(coherence))), float(np.max(np.abs(coherence)))]
    }
    
    # Quality metrics
    snr_values = [calculate_snr(sl) for sl in sublooks]
    speckle_indices = [speckle_index(sl) for sl in sublooks]
    
    report['quality_metrics'] = {
        'mean_snr_db': float(np.mean(snr_values)),
        'snr_std_db': float(np.std(snr_values)),
        'mean_speckle_index': float(np.mean(speckle_indices)),
        'speckle_reduction_factor': float(speckle_index(master) / np.mean(speckle_indices)),
        'mean_coherence': float(np.mean(np.abs(coherence))),
        'coherence_std': float(np.std(np.abs(coherence)))
    }
    
    # Generate recommendations
    if report['quality_metrics']['mean_snr_db'] < 10:
        report['recommendations'].append("Low SNR detected. Consider increasing number of looks.")
    
    if report['quality_metrics']['mean_coherence'] < 0.3:
        report['recommendations'].append("Low coherence. Check data quality and processing parameters.")
    
    if report['quality_metrics']['speckle_reduction_factor'] < 1.5:
        report['recommendations'].append("Limited speckle reduction. Consider adjusting decomposition parameters.")
    
    if len(report['recommendations']) == 0:
        report['recommendations'].append("Processing quality appears good. No major issues detected.")
    
    # Save report
    import json
    with open(f"{output_path}/quality_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

# Generate quality report
quality_report = generate_quality_report(sla, output_dir)

print("=== QUALITY ASSESSMENT REPORT ===")
print(f"Number of sub-looks: {quality_report['data_statistics']['num_sublooks']}")
print(f"Image dimensions: {quality_report['data_statistics']['image_dimensions']}")
print(f"Mean SNR: {quality_report['quality_metrics']['mean_snr_db']:.2f} dB")
print(f"Mean coherence: {quality_report['quality_metrics']['mean_coherence']:.3f}")
print(f"Speckle reduction factor: {quality_report['quality_metrics']['speckle_reduction_factor']:.2f}")

print("\nRecommendations:")
for rec in quality_report['recommendations']:
    print(f"- {rec}")
```

## 7. Export Results for Further Analysis

```python
# Save processed data in multiple formats
import os
os.makedirs(output_dir, exist_ok=True)

# Save as MATLAB format
save_matlab_mat(f"{output_dir}/sla_results.mat", {
    'sublooks': np.array(sublooks),
    'master_image': master_image,
    'coherence': coherence,
    'quality_metrics': quality_report['quality_metrics']
})

# Save individual sub-looks as separate files
for i, sublook in enumerate(sublooks):
    np.save(f"{output_dir}/sublook_{i+1}.npy", sublook)

# Save visualization images
print(f"Results saved to: {output_dir}/")
print("Generated files:")
print("- sublooks_visualization.png")
print("- master_image_enhancements.png") 
print("- statistical_analysis.png")
print("- coherence_analysis.png")
print("- snr_analysis.png")
print("- speckle_analysis.png")
print("- roi_analysis.png")
print("- quality_report.json")
print("- sla_results.mat")
print("- sublook_*.npy files")
```

## Summary

In this tutorial, you learned:

1. **Basic visualization** of SLA decomposition results
2. **Statistical analysis** techniques for quality assessment
3. **Coherence analysis** between sub-looks
4. **Quality metrics** calculation (SNR, speckle reduction)
5. **ROI-based analysis** for targeted assessment
6. **Automated quality reporting** for consistent evaluation

## Next Steps

- **Tutorial 4**: Multi-temporal analysis for change detection
- **Tutorial 5**: Polarimetric analysis with dual-pol data
- Experiment with different visualization parameters
- Apply these techniques to your own SAR datasets

## Troubleshooting

**Common Issues:**

1. **Memory errors with large datasets**: Process smaller tiles or reduce number of sub-looks
2. **Poor visualization contrast**: Adjust percentile ranges or use histogram equalization
3. **Low coherence values**: Check data quality and processing parameters
4. **Inconsistent ROI results**: Ensure ROI coordinates are within image bounds

For more help, see the [Troubleshooting Guide](../user_guide/troubleshooting.md).
