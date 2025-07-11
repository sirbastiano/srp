# Tutorial 1: Your First Sub-Look Analysis

Learn the fundamentals of sub-look decomposition using SARPyX with a real Sentinel-1 example.

## Overview

In this tutorial, you'll learn how to:
- Load and prepare Sentinel-1 SLC data
- Configure sub-look analysis parameters
- Execute the complete processing chain
- Visualize and interpret results
- Assess the quality of sub-look decomposition

**Duration**: 15 minutes  
**Prerequisites**: Basic Python knowledge  
**Data**: Sample Sentinel-1 SLC product

## What is Sub-Look Analysis?

Sub-look analysis decomposes the full SAR aperture into smaller sub-apertures, each providing:
- Different viewing angles (squint angles)
- Enhanced motion sensitivity
- Improved capability for interferometric analysis
- Reduced speckle through incoherent averaging

## Setup

### Required Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# SARPyX imports
from sarpyx.sla import SubLookAnalysis
from sarpyx.utils import show_image, show_histogram
```

### Sample Data

For this tutorial, we'll use a Sentinel-1 SLC product. If you don't have one, you can download sample data:

```python
# Sample data path (adjust to your data location)
product_path = "data/S1A_IW_SLC_1SDV_20231015T060000_20231015T060027_050123_060456_1234.zip"

# Verify the file exists
if not Path(product_path).exists():
    print(f"Please download sample data or adjust the path: {product_path}")
    # Alternative: use your own Sentinel-1 SLC product
```

## Step 1: Initialize Sub-Look Analysis

Let's start by creating a SubLookAnalysis object and examining the default parameters:

```python
# Initialize the SubLookAnalysis processor
sla = SubLookAnalysis(product_path)

# Examine default parameters
print("Default Parameters:")
print(f"Processing direction: {'Azimuth' if sla.choice == 1 else 'Range'}")
print(f"Number of looks: {sla.numberOfLooks}")
print(f"Centroid separations: {sla.centroidSeparations} Hz")
print(f"Sub-look bandwidth: {sla.subLookBandwidth} Hz")
print(f"De-weighting method: {sla.choiceDeWe}")
```

Expected output:
```
Default Parameters:
Processing direction: Azimuth
Number of looks: 3
Centroid separations: 700 Hz
Sub-look bandwidth: 700 Hz
De-weighting method: 0
```

## Step 2: Configure Processing Parameters

Let's configure the parameters for our analysis. We'll start with conservative settings:

```python
# Configuration for sub-look analysis
sla.choice = 1                    # Azimuth processing (most common)
sla.numberOfLooks = 3             # Generate 3 sub-looks
sla.centroidSeparations = 700     # 700 Hz separation between looks
sla.subLookBandwidth = 700        # 700 Hz bandwidth per sub-look

print("Configured Parameters:")
print(f"Processing: {'Azimuth' if sla.choice == 1 else 'Range'}")
print(f"Sub-looks: {sla.numberOfLooks}")
print(f"Separation: {sla.centroidSeparations} Hz")
print(f"Bandwidth: {sla.subLookBandwidth} Hz")
```

## Step 3: Frequency Computation

The first processing step computes the frequency bins for sub-look decomposition:

```python
# Compute frequency bins
print("Computing frequency bins...")
try:
    sla.frequencyComputation()
    print("✓ Frequency computation successful")
    
    # Display computed frequencies
    print(f"\nSub-look center frequencies:")
    for i, freq in enumerate(sla.freqCentr):
        print(f"  Look {i+1}: {freq:6.1f} Hz")
    
    print(f"\nFrequency ranges:")
    for i in range(sla.numberOfLooks):
        print(f"  Look {i+1}: [{sla.freqMin[i]:6.1f}, {sla.freqMax[i]:6.1f}] Hz")
        
except AssertionError as e:
    print(f"✗ Frequency computation failed: {e}")
    print("Try reducing the number of looks or bandwidth")
```

Expected output:
```
Computing frequency bins...
✓ Frequency computation successful

Sub-look center frequencies:
  Look 1:  -700.0 Hz
  Look 2:    0.0 Hz
  Look 3:  700.0 Hz

Frequency ranges:
  Look 1: [-1050.0, -350.0] Hz
  Look 2: [ -350.0,  350.0] Hz
  Look 3: [  350.0, 1050.0] Hz
```

## Step 4: Spectrum Computation

Transform the SAR data to the frequency domain:

```python
# Compute spectrum
print("Computing spectrum...")
try:
    sla.SpectrumComputation(VERBOSE=True)
    print("✓ Spectrum computation successful")
    
    # Display spectrum information
    print(f"Spectrum shape: {sla.SpectrumOneDim.shape}")
    print(f"Data type: {sla.SpectrumOneDim.dtype}")
    print(f"Frequency vector length: {len(sla.freqVect)}")
    
except Exception as e:
    print(f"✗ Spectrum computation failed: {e}")
```

With `VERBOSE=True`, you'll see plots showing the computed spectrum.

## Step 5: De-weighting

Apply de-weighting to compensate for window functions:

```python
# Apply de-weighting
print("Applying de-weighting...")
try:
    sla.AncillaryDeWe(VERBOSE=False)
    print("✓ De-weighting successful")
    
except Exception as e:
    print(f"✗ De-weighting failed: {e}")
```

## Step 6: Sub-Look Generation

Generate the final sub-look images:

```python
# Generate sub-look images
print("Generating sub-look images...")
try:
    sla.Generation(VERBOSE=True)
    print("✓ Sub-look generation successful")
    
    # Display results information
    print(f"\nResults:")
    print(f"Sub-look array shape: {sla.Looks.shape}")
    print(f"Number of sub-looks: {sla.Looks.shape[0]}")
    print(f"Image dimensions: {sla.Looks.shape[1]} x {sla.Looks.shape[2]}")
    
except Exception as e:
    print(f"✗ Sub-look generation failed: {e}")
```

With `VERBOSE=True`, you'll see plots of the generated sub-look images.

## Step 7: Visualize Results

Let's create comprehensive visualizations of our results:

```python
# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sub-Look Analysis Results', fontsize=16)

# Display original data
axes[0, 0].imshow(np.abs(sla.Box), cmap='gray', aspect='auto')
axes[0, 0].set_title('Original SAR Data')
axes[0, 0].set_xlabel('Range')
axes[0, 0].set_ylabel('Azimuth')

# Display sub-look images
for i in range(sla.numberOfLooks):
    row = 0 if i < 2 else 1
    col = i + 1 if i < 2 else i - 2
    
    # Display amplitude
    sublook_amp = np.abs(sla.Looks[i])
    im = axes[row, col].imshow(sublook_amp, cmap='gray', aspect='auto')
    axes[row, col].set_title(f'Sub-look {i+1}\n(Freq: {sla.freqCentr[i]:.0f} Hz)')
    axes[row, col].set_xlabel('Range')
    axes[row, col].set_ylabel('Azimuth')

# Display coherence map (if we have multiple looks)
if sla.numberOfLooks >= 2:
    # Calculate coherence between first two sub-looks
    look1_flat = sla.Looks[0].flatten()
    look2_flat = sla.Looks[1].flatten()
    
    # Coherence estimation (simplified)
    coherence = np.abs(np.corrcoef(look1_flat, look2_flat)[0, 1])
    
    axes[1, 2].text(0.5, 0.5, f'Coherence\nbetween\nLooks 1-2:\n{coherence:.3f}', 
                    ha='center', va='center', fontsize=14, 
                    transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Quality Metrics')
    axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

## Step 8: Quality Assessment

Let's assess the quality of our sub-look decomposition:

```python
def assess_sublook_quality(sla):
    """Assess the quality of sub-look decomposition."""
    
    metrics = {}
    
    # 1. Mean amplitude per sub-look
    mean_amplitudes = []
    for i in range(sla.numberOfLooks):
        mean_amp = np.mean(np.abs(sla.Looks[i]))
        mean_amplitudes.append(mean_amp)
    
    metrics['mean_amplitudes'] = mean_amplitudes
    
    # 2. Standard deviation (speckle level)
    std_amplitudes = []
    for i in range(sla.numberOfLooks):
        std_amp = np.std(np.abs(sla.Looks[i]))
        std_amplitudes.append(std_amp)
    
    metrics['std_amplitudes'] = std_amplitudes
    
    # 3. Cross-correlation between sub-looks
    correlations = []
    for i in range(sla.numberOfLooks):
        for j in range(i+1, sla.numberOfLooks):
            corr = np.corrcoef(
                sla.Looks[i].flatten(), 
                sla.Looks[j].flatten()
            )[0, 1]
            correlations.append((i, j, np.abs(corr)))
    
    metrics['correlations'] = correlations
    
    # 4. Frequency separation validation
    freq_separations = []
    for i in range(sla.numberOfLooks-1):
        sep = sla.freqCentr[i+1] - sla.freqCentr[i]
        freq_separations.append(sep)
    
    metrics['frequency_separations'] = freq_separations
    
    return metrics

# Assess quality
quality = assess_sublook_quality(sla)

print("Quality Assessment:")
print("==================")

print(f"\n1. Mean Amplitudes:")
for i, amp in enumerate(quality['mean_amplitudes']):
    print(f"   Sub-look {i+1}: {amp:.3f}")

print(f"\n2. Standard Deviations:")
for i, std in enumerate(quality['std_amplitudes']):
    print(f"   Sub-look {i+1}: {std:.3f}")

print(f"\n3. Cross-correlations:")
for i, j, corr in quality['correlations']:
    print(f"   Sub-looks {i+1}-{j+1}: {corr:.3f}")

print(f"\n4. Frequency Separations:")
for i, sep in enumerate(quality['frequency_separations']):
    print(f"   Between looks {i+1}-{i+2}: {sep:.1f} Hz")
    if abs(sep - sla.centroidSeparations) > 1e-6:
        print(f"   ⚠ Warning: Expected {sla.centroidSeparations} Hz")
```

## Step 9: Advanced Analysis

Let's perform some advanced analysis on our sub-look results:

```python
# Create intensity difference images
if sla.numberOfLooks >= 2:
    # Intensity images
    intensity1 = np.abs(sla.Looks[0])**2
    intensity2 = np.abs(sla.Looks[1])**2
    
    # Intensity difference (normalized)
    intensity_diff = (intensity1 - intensity2) / (intensity1 + intensity2 + 1e-10)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(intensity1, cmap='hot', aspect='auto')
    axes[0].set_title('Intensity - Sub-look 1')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(intensity2, cmap='hot', aspect='auto')
    axes[1].set_title('Intensity - Sub-look 2')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(intensity_diff, cmap='RdBu', vmin=-0.5, vmax=0.5, aspect='auto')
    axes[2].set_title('Normalized Intensity Difference')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"Intensity difference statistics:")
    print(f"  Mean: {np.mean(intensity_diff):.6f}")
    print(f"  Std:  {np.std(intensity_diff):.6f}")
    print(f"  Min:  {np.min(intensity_diff):.3f}")
    print(f"  Max:  {np.max(intensity_diff):.3f}")
```

## Step 10: Parameter Exploration

Let's explore how different parameters affect the results:

```python
def compare_parameters():
    """Compare results with different parameter settings."""
    
    configs = [
        {'numberOfLooks': 3, 'centroidSeparations': 500, 'subLookBandwidth': 500},
        {'numberOfLooks': 3, 'centroidSeparations': 700, 'subLookBandwidth': 700},
        {'numberOfLooks': 5, 'centroidSeparations': 400, 'subLookBandwidth': 400}
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"Testing configuration {i+1}: {config}")
        
        try:
            # Create new SLA instance
            test_sla = SubLookAnalysis(product_path)
            
            # Apply configuration
            for key, value in config.items():
                setattr(test_sla, key, value)
            
            # Process (without verbose output)
            test_sla.frequencyComputation()
            test_sla.SpectrumComputation(VERBOSE=False)
            test_sla.AncillaryDeWe(VERBOSE=False)
            test_sla.Generation(VERBOSE=False)
            
            # Calculate quality metric
            correlations = []
            for look_i in range(test_sla.numberOfLooks):
                for look_j in range(look_i+1, test_sla.numberOfLooks):
                    corr = np.abs(np.corrcoef(
                        test_sla.Looks[look_i].flatten(),
                        test_sla.Looks[look_j].flatten()
                    )[0, 1])
                    correlations.append(corr)
            
            avg_correlation = np.mean(correlations)
            
            results.append({
                'config': config,
                'avg_correlation': avg_correlation,
                'success': True
            })
            
            print(f"  ✓ Success! Average correlation: {avg_correlation:.3f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                'config': config,
                'success': False,
                'error': str(e)
            })
    
    return results

# Run parameter comparison
print("Parameter Exploration:")
print("=====================")
param_results = compare_parameters()

print("\nSummary:")
for i, result in enumerate(param_results):
    config = result['config']
    if result['success']:
        print(f"Config {i+1}: {config['numberOfLooks']} looks, "
              f"{config['centroidSeparations']} Hz sep, "
              f"{config['subLookBandwidth']} Hz BW → "
              f"Correlation: {result['avg_correlation']:.3f}")
    else:
        print(f"Config {i+1}: FAILED - {result['error']}")
```

## Summary

Congratulations! You've completed your first sub-look analysis. Here's what you learned:

1. **Initialization**: How to create a SubLookAnalysis object
2. **Configuration**: Setting processing parameters
3. **Processing Chain**: The four main steps of sub-look analysis
4. **Visualization**: Creating meaningful plots of results
5. **Quality Assessment**: Evaluating processing success
6. **Parameter Exploration**: Understanding parameter effects

### Key Takeaways

- **Frequency planning is critical**: Sub-look frequencies must fit within available bandwidth
- **Parameter balance**: Trade-offs between number of looks, separation, and bandwidth
- **Quality metrics**: Correlation and intensity statistics help assess results
- **Visualization helps**: Always plot results to understand what's happening

## Exercises

Try these exercises to deepen your understanding:

1. **Different sensor data**: Try the analysis with COSMO-SkyMed or other SAR data
2. **Range processing**: Set `choice = 0` and compare with azimuth results
3. **Parameter optimization**: Find optimal parameters for your specific dataset
4. **Motion detection**: Use intensity differences to identify moving targets

## Troubleshooting

### Common Issues

1. **"Sub-look spectrum outside the available bandwidth"**
   - Reduce `subLookBandwidth` or `centroidSeparations`
   - Decrease `numberOfLooks`

2. **Memory errors**
   - Use smaller spatial subsets
   - Reduce data precision if acceptable

3. **Poor quality results**
   - Check input data quality
   - Adjust window functions
   - Verify frequency parameters

## Next Steps

Continue your learning journey:
- **[Tutorial 2: SNAP Integration Basics](02_snap_integration_basics.md)**: Learn SNAP automation
- **[Tutorial 3: Visualization and Quality](03_visualization_quality.md)**: Advanced visualization techniques
- **[Tutorial 4: Multi-temporal Analysis](04_multitemporal_analysis.md)**: Time series processing

You're now ready to explore more advanced SARPyX capabilities!
