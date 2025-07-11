# Tutorial 5: Polarimetric Analysis

Learn to work with dual-pol SAR data for vegetation monitoring and polarimetric decomposition using SARPyX.

## Overview

This tutorial covers:
- Loading and processing dual-pol SAR data
- Polarimetric parameter extraction
- Vegetation index calculation using polarimetric ratios
- Co-pol and cross-pol analysis
- Advanced polarimetric decomposition techniques
- Multi-temporal polarimetric monitoring

**Duration**: 35 minutes  
**Prerequisites**: Tutorial 2 (SNAP integration)  
**Data**: Dual-pol Sentinel-1 data (VV/VH or HH/HV)

## 1. Dual-pol Data Setup and Loading

```python
import numpy as np
import matplotlib.pyplot as plt
from sarpyx.sla import SubLookAnalysis
from sarpyx.snap import GPT
from sarpyx.science.indices import *
from sarpyx.utils.viz import show_image, show_histogram
from sarpyx.utils.io import save_matlab_mat
import os

# Set up directories
data_dir = "path/to/dualpol/data"
output_dir = "tutorial5_outputs"
os.makedirs(output_dir, exist_ok=True)

# Configuration for dual-pol processing
dualpol_config = {
    'input_file': f"{data_dir}/S1A_dualpol_*.zip",  # Sentinel-1 dual-pol product
    'polarizations': ['VV', 'VH'],  # or ['HH', 'HV'] for some acquisitions
    'processing_level': 'SLC',  # Single Look Complex
    'output_format': 'BEAM-DIMAP'
}

print("Dual-pol processing configuration:")
print(f"  Polarizations: {dualpol_config['polarizations']}")
print(f"  Processing level: {dualpol_config['processing_level']}")
```

## 2. SNAP-based Dual-pol Preprocessing

### 2.1 Calibration and Multi-looking

```python
# Initialize SNAP GPT processor
gpt = GPT()

def preprocess_dualpol_data(input_file, output_dir, polarizations=['VV', 'VH']):
    """Preprocess dual-pol data using SNAP"""
    
    preprocessing_steps = []
    
    # Step 1: Calibration
    calibration_params = {
        'outputSigmaBand': True,
        'outputBetaBand': False,
        'outputGammaBand': False,
        'outputDNBand': False
    }
    
    calibration_file = f"{output_dir}/calibrated.dim"
    gpt.calibration(input_file, calibration_file, **calibration_params)
    preprocessing_steps.append(('Calibration', calibration_file))
    
    # Step 2: Multi-looking (optional, for speckle reduction)
    multilook_params = {
        'nRgLooks': 1,
        'nAzLooks': 1,
        'outputIntensity': True,
        'grSquarePixel': True
    }
    
    multilook_file = f"{output_dir}/multilooked.dim"
    gpt.multilook(calibration_file, multilook_file, **multilook_params)
    preprocessing_steps.append(('Multilooking', multilook_file))
    
    # Step 3: Terrain Correction
    terrain_params = {
        'demName': 'SRTM 3Sec',
        'pixelSpacingInMeter': 20.0,
        'nodataValueAtSea': False
    }
    
    terrain_file = f"{output_dir}/terrain_corrected.dim"
    gpt.terrain_correction(multilook_file, terrain_file, **terrain_params)
    preprocessing_steps.append(('Terrain Correction', terrain_file))
    
    return terrain_file, preprocessing_steps

# Preprocess dual-pol data
print("Preprocessing dual-pol data with SNAP...")
try:
    processed_file, processing_steps = preprocess_dualpol_data(
        dualpol_config['input_file'], 
        output_dir, 
        dualpol_config['polarizations']
    )
    
    print("Preprocessing completed successfully:")
    for step_name, file_path in processing_steps:
        print(f"  {step_name}: {file_path}")
        
except Exception as e:
    print(f"Preprocessing error: {e}")
    # Fallback: use existing processed file
    processed_file = f"{output_dir}/terrain_corrected.dim"
```

### 2.2 Load Dual-pol Bands

```python
def load_dualpol_bands(file_path, polarizations=['VV', 'VH']):
    """Load dual-pol bands for analysis"""
    
    pol_data = {}
    
    for pol in polarizations:
        try:
            # Initialize SLA for each polarization
            sla = SubLookAnalysis()
            
            # Load specific polarization band
            band_name = f'Sigma0_{pol}_slc' if 'SLC' in file_path else f'Sigma0_{pol}'
            sla.load_data(file_path, band_name=band_name)
            
            # Get data without decomposition for direct access
            pol_data[pol] = {
                'sla_processor': sla,
                'data': sla.data,
                'metadata': sla.metadata
            }
            
            print(f"Loaded {pol} polarization: {pol_data[pol]['data'].shape}")
            
        except Exception as e:
            print(f"Error loading {pol} polarization: {e}")
            continue
    
    return pol_data

# Load dual-pol data
print("Loading dual-pol bands...")
pol_data = load_dualpol_bands(processed_file, dualpol_config['polarizations'])

# Verify data loading
if len(pol_data) >= 2:
    pol_names = list(pol_data.keys())
    print(f"Successfully loaded polarizations: {pol_names}")
    
    # Get common dimensions
    shapes = [data['data'].shape for data in pol_data.values()]
    print(f"Data shapes: {dict(zip(pol_names, shapes))}")
else:
    print("Error: Need at least 2 polarizations for dual-pol analysis")
```

## 3. Basic Polarimetric Parameter Extraction

### 3.1 Co-pol and Cross-pol Separation

```python
def extract_polarimetric_parameters(pol_data):
    """Extract basic polarimetric parameters"""
    
    pol_names = list(pol_data.keys())
    
    # Determine co-pol and cross-pol channels
    if 'VV' in pol_names and 'VH' in pol_names:
        copol_name = 'VV'
        crosspol_name = 'VH'
    elif 'HH' in pol_names and 'HV' in pol_names:
        copol_name = 'HH'
        crosspol_name = 'HV'
    else:
        raise ValueError(f"Unsupported polarization combination: {pol_names}")
    
    # Extract data
    copol_data = pol_data[copol_name]['data']
    crosspol_data = pol_data[crosspol_name]['data']
    
    print(f"Co-pol channel: {copol_name}")
    print(f"Cross-pol channel: {crosspol_name}")
    
    # Calculate basic parameters
    parameters = {
        'copol_intensity': np.abs(copol_data)**2,
        'crosspol_intensity': np.abs(crosspol_data)**2,
        'total_power': np.abs(copol_data)**2 + np.abs(crosspol_data)**2,
        'copol_data': copol_data,
        'crosspol_data': crosspol_data,
        'pol_names': (copol_name, crosspol_name)
    }
    
    # Polarimetric ratios
    parameters['copol_crosspol_ratio'] = (parameters['copol_intensity'] / 
                                        (parameters['crosspol_intensity'] + 1e-10))
    
    parameters['crosspol_fraction'] = (parameters['crosspol_intensity'] / 
                                     (parameters['total_power'] + 1e-10))
    
    # Convert to dB scale
    parameters['copol_db'] = 10 * np.log10(parameters['copol_intensity'] + 1e-10)
    parameters['crosspol_db'] = 10 * np.log10(parameters['crosspol_intensity'] + 1e-10)
    parameters['ratio_db'] = 10 * np.log10(parameters['copol_crosspol_ratio'] + 1e-10)
    
    return parameters

# Extract polarimetric parameters
print("Extracting polarimetric parameters...")
pol_params = extract_polarimetric_parameters(pol_data)

print("Polarimetric parameters extracted:")
print(f"  Co-pol/Cross-pol channels: {pol_params['pol_names']}")
print(f"  Co-pol intensity range: {np.min(pol_params['copol_db']):.1f} to {np.max(pol_params['copol_db']):.1f} dB")
print(f"  Cross-pol intensity range: {np.min(pol_params['crosspol_db']):.1f} to {np.max(pol_params['crosspol_db']):.1f} dB")
print(f"  Ratio range: {np.min(pol_params['ratio_db']):.1f} to {np.max(pol_params['ratio_db']):.1f} dB")
```

### 3.2 Visualize Basic Polarimetric Components

```python
# Visualize polarimetric components
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Co-pol intensity
im1 = show_image(pol_params['copol_db'],
                title=f'{pol_params["pol_names"][0]} Intensity (dB)',
                ax=axes[0, 0],
                cmap='viridis',
                vmin=np.percentile(pol_params['copol_db'], 5),
                vmax=np.percentile(pol_params['copol_db'], 95))
plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

# Cross-pol intensity
im2 = show_image(pol_params['crosspol_db'],
                title=f'{pol_params["pol_names"][1]} Intensity (dB)',
                ax=axes[0, 1],
                cmap='viridis',
                vmin=np.percentile(pol_params['crosspol_db'], 5),
                vmax=np.percentile(pol_params['crosspol_db'], 95))
plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

# Co-pol/Cross-pol ratio
im3 = show_image(pol_params['ratio_db'],
                title='Co-pol/Cross-pol Ratio (dB)',
                ax=axes[0, 2],
                cmap='RdYlBu_r',
                vmin=np.percentile(pol_params['ratio_db'], 5),
                vmax=np.percentile(pol_params['ratio_db'], 95))
plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)

# Cross-pol fraction
im4 = show_image(pol_params['crosspol_fraction'],
                title='Cross-pol Fraction',
                ax=axes[1, 0],
                cmap='plasma',
                vmin=0, vmax=1)
plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)

# Total power
total_power_db = 10 * np.log10(pol_params['total_power'] + 1e-10)
im5 = show_image(total_power_db,
                title='Total Power (dB)',
                ax=axes[1, 1],
                cmap='hot',
                vmin=np.percentile(total_power_db, 5),
                vmax=np.percentile(total_power_db, 95))
plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)

# RGB composite (R=VV, G=VH, B=VV/VH ratio)
rgb_composite = np.zeros((*pol_params['copol_db'].shape, 3))

# Normalize channels for RGB display
copol_norm = (pol_params['copol_db'] - np.percentile(pol_params['copol_db'], 2)) / \
             (np.percentile(pol_params['copol_db'], 98) - np.percentile(pol_params['copol_db'], 2))
crosspol_norm = (pol_params['crosspol_db'] - np.percentile(pol_params['crosspol_db'], 2)) / \
                (np.percentile(pol_params['crosspol_db'], 98) - np.percentile(pol_params['crosspol_db'], 2))
ratio_norm = (pol_params['ratio_db'] - np.percentile(pol_params['ratio_db'], 2)) / \
             (np.percentile(pol_params['ratio_db'], 98) - np.percentile(pol_params['ratio_db'], 2))

rgb_composite[:, :, 0] = np.clip(copol_norm, 0, 1)       # Red: Co-pol
rgb_composite[:, :, 1] = np.clip(crosspol_norm, 0, 1)    # Green: Cross-pol
rgb_composite[:, :, 2] = np.clip(ratio_norm, 0, 1)       # Blue: Ratio

axes[1, 2].imshow(rgb_composite)
axes[1, 2].set_title('RGB Composite\n(R=Co-pol, G=Cross-pol, B=Ratio)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/polarimetric_components.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 4. Vegetation Index Calculation

### 4.1 Radar Vegetation Index (RVI)

```python
# Calculate Radar Vegetation Index using SARPyX science module
print("Calculating vegetation indices...")

# Extract required data for vegetation indices
vv_intensity = pol_params['copol_intensity']
vh_intensity = pol_params['crosspol_intensity']

# Calculate RVI
rvi_values = RVI(vh_intensity, vv_intensity)

print(f"RVI calculated:")
print(f"  Range: {np.min(rvi_values):.3f} to {np.max(rvi_values):.3f}")
print(f"  Mean: {np.mean(rvi_values):.3f}")
```

### 4.2 Normalized Difference Polarization Index (NDPoll)

```python
# Calculate NDPoll index
ndpoll_values = NDPoll(vh_intensity, vv_intensity)

print(f"NDPoll calculated:")
print(f"  Range: {np.min(ndpoll_values):.3f} to {np.max(ndpoll_values):.3f}")
print(f"  Mean: {np.mean(ndpoll_values):.3f}")
```

### 4.3 Dual Polarization Difference (DPDD)

```python
# Calculate DPDD index
dpdd_values = DPDD(vh_intensity, vv_intensity)

print(f"DPDD calculated:")
print(f"  Range: {np.min(dpdd_values):.3f} to {np.max(dpdd_values):.3f}")
print(f"  Mean: {np.mean(dpdd_values):.3f}")
```

### 4.4 Visualize Vegetation Indices

```python
# Visualize vegetation indices
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# RVI
im1 = show_image(rvi_values,
                title='Radar Vegetation Index (RVI)',
                ax=axes[0, 0],
                cmap='RdYlGn',
                vmin=np.percentile(rvi_values, 5),
                vmax=np.percentile(rvi_values, 95))
plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

# NDPoll
im2 = show_image(ndpoll_values,
                title='Normalized Difference Polarization Index',
                ax=axes[0, 1],
                cmap='RdYlGn',
                vmin=np.percentile(ndpoll_values, 5),
                vmax=np.percentile(ndpoll_values, 95))
plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

# DPDD
im3 = show_image(dpdd_values,
                title='Dual Polarization Difference (DPDD)',
                ax=axes[1, 0],
                cmap='RdYlGn',
                vmin=np.percentile(dpdd_values, 5),
                vmax=np.percentile(dpdd_values, 95))
plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

# Vegetation index comparison
# Sample points for comparison
height, width = rvi_values.shape
sample_points = [
    (height//4, width//4),      # Point 1
    (height//2, width//2),      # Point 2  
    (3*height//4, 3*width//4)   # Point 3
]

indices_comparison = {
    'RVI': [],
    'NDPoll': [],
    'DPDD': []
}

for y, x in sample_points:
    indices_comparison['RVI'].append(rvi_values[y, x])
    indices_comparison['NDPoll'].append(ndpoll_values[y, x])
    indices_comparison['DPDD'].append(dpdd_values[y, x])

# Plot comparison
x_pos = np.arange(len(sample_points))
width_bar = 0.25

axes[1, 1].bar(x_pos - width_bar, indices_comparison['RVI'], width_bar, 
              label='RVI', alpha=0.7)
axes[1, 1].bar(x_pos, indices_comparison['NDPoll'], width_bar, 
              label='NDPoll', alpha=0.7)
axes[1, 1].bar(x_pos + width_bar, indices_comparison['DPDD'], width_bar, 
              label='DPDD', alpha=0.7)

axes[1, 1].set_xlabel('Sample Points')
axes[1, 1].set_ylabel('Index Value')
axes[1, 1].set_title('Vegetation Indices Comparison')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels([f'Point {i+1}' for i in range(len(sample_points))])
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/vegetation_indices.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 5. Advanced Polarimetric Analysis

### 5.1 Polarimetric Decomposition

```python
def freeman_durden_decomposition(copol_intensity, crosspol_intensity):
    """
    Simplified Freeman-Durden decomposition for dual-pol data
    Returns volume, double-bounce, and surface scattering components
    """
    
    # Volume scattering (related to cross-pol)
    volume_scatter = 8 * crosspol_intensity / 3
    
    # Remaining power after volume scattering
    remaining_copol = copol_intensity - volume_scatter/4
    remaining_copol = np.maximum(remaining_copol, 0)  # Ensure non-negative
    
    # Simple separation of surface and double-bounce
    # (This is a simplified approach for dual-pol data)
    surface_scatter = remaining_copol * 0.7  # Assume 70% surface scattering
    double_bounce = remaining_copol * 0.3    # Assume 30% double-bounce
    
    return {
        'volume': volume_scatter,
        'surface': surface_scatter,
        'double_bounce': double_bounce,
        'total_power': volume_scatter + surface_scatter + double_bounce
    }

# Perform polarimetric decomposition
print("Performing polarimetric decomposition...")
decomp_results = freeman_durden_decomposition(
    pol_params['copol_intensity'],
    pol_params['crosspol_intensity']
)

# Calculate decomposition percentages
total_power = decomp_results['total_power']
volume_fraction = decomp_results['volume'] / (total_power + 1e-10)
surface_fraction = decomp_results['surface'] / (total_power + 1e-10)
double_bounce_fraction = decomp_results['double_bounce'] / (total_power + 1e-10)

print("Decomposition statistics:")
print(f"  Mean volume scattering: {np.mean(volume_fraction)*100:.1f}%")
print(f"  Mean surface scattering: {np.mean(surface_fraction)*100:.1f}%")
print(f"  Mean double-bounce: {np.mean(double_bounce_fraction)*100:.1f}%")
```

### 5.2 Visualize Polarimetric Decomposition

```python
# Visualize decomposition results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Volume scattering
im1 = show_image(volume_fraction,
                title='Volume Scattering Fraction',
                ax=axes[0, 0],
                cmap='Greens',
                vmin=0, vmax=1)
plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

# Surface scattering
im2 = show_image(surface_fraction,
                title='Surface Scattering Fraction',
                ax=axes[0, 1],
                cmap='Blues',
                vmin=0, vmax=1)
plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

# Double-bounce scattering
im3 = show_image(double_bounce_fraction,
                title='Double-bounce Scattering Fraction',
                ax=axes[1, 0],
                cmap='Reds',
                vmin=0, vmax=1)
plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

# RGB decomposition composite
decomp_rgb = np.zeros((*volume_fraction.shape, 3))
decomp_rgb[:, :, 0] = double_bounce_fraction  # Red: Double-bounce
decomp_rgb[:, :, 1] = volume_fraction         # Green: Volume
decomp_rgb[:, :, 2] = surface_fraction        # Blue: Surface

axes[1, 1].imshow(decomp_rgb)
axes[1, 1].set_title('RGB Decomposition\n(R=Double-bounce, G=Volume, B=Surface)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/polarimetric_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.3 Scattering Mechanism Analysis

```python
def analyze_scattering_mechanisms(decomp_results, threshold=0.5):
    """Analyze dominant scattering mechanisms"""
    
    volume_frac = decomp_results['volume'] / (decomp_results['total_power'] + 1e-10)
    surface_frac = decomp_results['surface'] / (decomp_results['total_power'] + 1e-10)
    double_bounce_frac = decomp_results['double_bounce'] / (decomp_results['total_power'] + 1e-10)
    
    # Determine dominant scattering mechanism
    scattering_map = np.zeros_like(volume_frac, dtype=int)
    
    # 0: Mixed, 1: Volume dominant, 2: Surface dominant, 3: Double-bounce dominant
    volume_dominant = volume_frac > threshold
    surface_dominant = surface_frac > threshold
    double_bounce_dominant = double_bounce_frac > threshold
    
    scattering_map[volume_dominant] = 1
    scattering_map[surface_dominant] = 2
    scattering_map[double_bounce_dominant] = 3
    
    # Calculate statistics
    total_pixels = scattering_map.size
    mixed_pixels = np.sum(scattering_map == 0)
    volume_pixels = np.sum(scattering_map == 1)
    surface_pixels = np.sum(scattering_map == 2)
    double_bounce_pixels = np.sum(scattering_map == 3)
    
    stats = {
        'mixed_percentage': mixed_pixels / total_pixels * 100,
        'volume_percentage': volume_pixels / total_pixels * 100,
        'surface_percentage': surface_pixels / total_pixels * 100,
        'double_bounce_percentage': double_bounce_pixels / total_pixels * 100,
        'scattering_map': scattering_map
    }
    
    return stats

# Analyze scattering mechanisms
print("Analyzing scattering mechanisms...")
scattering_stats = analyze_scattering_mechanisms(decomp_results, threshold=0.4)

print("Scattering mechanism statistics:")
print(f"  Mixed scattering: {scattering_stats['mixed_percentage']:.1f}%")
print(f"  Volume scattering dominant: {scattering_stats['volume_percentage']:.1f}%")
print(f"  Surface scattering dominant: {scattering_stats['surface_percentage']:.1f}%")
print(f"  Double-bounce dominant: {scattering_stats['double_bounce_percentage']:.1f}%")

# Visualize scattering mechanism map
plt.figure(figsize=(12, 8))

# Create custom colormap for scattering mechanisms
colors = ['gray', 'green', 'blue', 'red']  # Mixed, Volume, Surface, Double-bounce
labels = ['Mixed', 'Volume Dominant', 'Surface Dominant', 'Double-bounce Dominant']

plt.imshow(scattering_stats['scattering_map'], cmap='viridis', vmin=0, vmax=3)
plt.title('Dominant Scattering Mechanisms')
plt.colorbar(label='Scattering Type', ticks=[0, 1, 2, 3], 
            format=plt.FuncFormatter(lambda x, p: labels[int(x)] if int(x) < len(labels) else ''))

plt.savefig(f'{output_dir}/scattering_mechanisms.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 6. SLA Integration with Polarimetric Data

### 6.1 Polarimetric Sub-look Analysis

```python
def polarimetric_sla_analysis(pol_data, num_looks=4, overlap_factor=0.5):
    """Perform SLA on both polarizations and analyze differences"""
    
    pol_sla_results = {}
    
    for pol_name, data in pol_data.items():
        print(f"Performing SLA on {pol_name} polarization...")
        
        # Configure SLA parameters
        sla = data['sla_processor']
        sla.set_sublook_parameters(num_looks=num_looks, overlap_factor=overlap_factor)
        
        # Perform decomposition
        sla.decompose()
        
        # Store results
        pol_sla_results[pol_name] = {
            'sublooks': sla.get_sublooks(),
            'master_image': sla.get_master_image(),
            'coherence': sla.get_coherence_matrix(),
            'num_looks': num_looks
        }
        
        print(f"  {pol_name}: {len(pol_sla_results[pol_name]['sublooks'])} sub-looks generated")
    
    return pol_sla_results

# Perform polarimetric SLA
print("Performing polarimetric sub-look analysis...")
pol_sla_results = polarimetric_sla_analysis(pol_data, num_looks=4, overlap_factor=0.5)
```

### 6.2 Inter-polarization Coherence Analysis

```python
def analyze_inter_polarization_coherence(pol_sla_results):
    """Analyze coherence between polarizations"""
    
    pol_names = list(pol_sla_results.keys())
    
    if len(pol_names) < 2:
        print("Error: Need at least 2 polarizations for inter-pol coherence")
        return None
    
    pol1_name, pol2_name = pol_names[0], pol_names[1]
    pol1_sublooks = pol_sla_results[pol1_name]['sublooks']
    pol2_sublooks = pol_sla_results[pol2_name]['sublooks']
    
    # Calculate inter-polarization coherence for each sub-look pair
    inter_pol_coherence = []
    
    for i, (sl1, sl2) in enumerate(zip(pol1_sublooks, pol2_sublooks)):
        # Normalize by power
        numerator = np.mean(sl1 * np.conj(sl2))
        denominator = np.sqrt(np.mean(np.abs(sl1)**2) * np.mean(np.abs(sl2)**2))
        coherence = numerator / (denominator + 1e-10)
        
        inter_pol_coherence.append(coherence)
        print(f"  Sub-look {i+1} inter-pol coherence: {np.abs(coherence):.3f}")
    
    mean_inter_pol_coherence = np.mean([np.abs(c) for c in inter_pol_coherence])
    
    results = {
        'pol_pair': (pol1_name, pol2_name),
        'coherence_values': inter_pol_coherence,
        'mean_coherence': mean_inter_pol_coherence,
        'coherence_magnitudes': [np.abs(c) for c in inter_pol_coherence],
        'coherence_phases': [np.angle(c) for c in inter_pol_coherence]
    }
    
    return results

# Analyze inter-polarization coherence
print("Analyzing inter-polarization coherence...")
inter_pol_results = analyze_inter_polarization_coherence(pol_sla_results)

if inter_pol_results:
    print(f"Inter-polarization coherence analysis:")
    print(f"  Polarization pair: {inter_pol_results['pol_pair']}")
    print(f"  Mean coherence magnitude: {inter_pol_results['mean_coherence']:.3f}")
    print(f"  Coherence range: {np.min(inter_pol_results['coherence_magnitudes']):.3f} to {np.max(inter_pol_results['coherence_magnitudes']):.3f}")
```

### 6.3 Polarimetric Quality Assessment

```python
def assess_polarimetric_quality(pol_sla_results, inter_pol_results):
    """Assess quality of polarimetric processing"""
    
    quality_metrics = {}
    
    for pol_name, results in pol_sla_results.items():
        # Calculate SNR for each sub-look
        sublooks = results['sublooks']
        snr_values = []
        
        for sublook in sublooks:
            signal_power = np.mean(np.abs(sublook)**2)
            noise_floor = np.percentile(np.abs(sublook)**2, 10)
            snr_db = 10 * np.log10(signal_power / noise_floor)
            snr_values.append(snr_db)
        
        # Calculate speckle reduction
        master = results['master_image']
        master_speckle = np.std(np.abs(master)) / np.mean(np.abs(master))
        
        sublook_speckles = [np.std(np.abs(sl)) / np.mean(np.abs(sl)) for sl in sublooks]
        mean_sublook_speckle = np.mean(sublook_speckles)
        speckle_reduction = master_speckle / mean_sublook_speckle
        
        quality_metrics[pol_name] = {
            'mean_snr_db': np.mean(snr_values),
            'snr_std_db': np.std(snr_values),
            'speckle_reduction_factor': speckle_reduction,
            'master_speckle_index': master_speckle,
            'mean_sublook_speckle': mean_sublook_speckle
        }
    
    # Overall quality assessment
    if inter_pol_results:
        quality_metrics['inter_polarization'] = {
            'mean_coherence': inter_pol_results['mean_coherence'],
            'coherence_stability': np.std(inter_pol_results['coherence_magnitudes'])
        }
    
    return quality_metrics

# Assess polarimetric quality
print("Assessing polarimetric processing quality...")
quality_metrics = assess_polarimetric_quality(pol_sla_results, inter_pol_results)

print("Quality Assessment Results:")
for pol_name, metrics in quality_metrics.items():
    if pol_name != 'inter_polarization':
        print(f"  {pol_name} polarization:")
        print(f"    Mean SNR: {metrics['mean_snr_db']:.2f} dB")
        print(f"    Speckle reduction factor: {metrics['speckle_reduction_factor']:.2f}")
        print(f"    Master speckle index: {metrics['master_speckle_index']:.3f}")

if 'inter_polarization' in quality_metrics:
    inter_metrics = quality_metrics['inter_polarization']
    print(f"  Inter-polarization:")
    print(f"    Mean coherence: {inter_metrics['mean_coherence']:.3f}")
    print(f"    Coherence stability: {inter_metrics['coherence_stability']:.3f}")
```

## 7. Multi-temporal Polarimetric Monitoring

### 7.1 Temporal Vegetation Index Analysis

```python
def temporal_vegetation_monitoring(vegetation_indices, dates):
    """Monitor vegetation changes over time using polarimetric indices"""
    
    # Simulate temporal data (in practice, load from multiple acquisitions)
    temporal_indices = {
        'RVI': [vegetation_indices['RVI']],  # Add more time points
        'NDPoll': [vegetation_indices['NDPoll']],
        'DPDD': [vegetation_indices['DPDD']],
        'dates': dates
    }
    
    # Sample points for temporal analysis
    height, width = vegetation_indices['RVI'].shape
    sample_points = {
        'forest': (height//4, width//4),
        'agriculture': (height//2, width//2),
        'urban': (3*height//4, 3*width//4)
    }
    
    temporal_analysis = {}
    
    for point_name, (y, x) in sample_points.items():
        temporal_analysis[point_name] = {
            'coordinates': (y, x),
            'RVI_series': [idx[y, x] for idx in temporal_indices['RVI']],
            'NDPoll_series': [idx[y, x] for idx in temporal_indices['NDPoll']],
            'DPDD_series': [idx[y, x] for idx in temporal_indices['DPDD']]
        }
    
    return temporal_analysis

# Demonstrate temporal monitoring concept
vegetation_indices = {
    'RVI': rvi_values,
    'NDPoll': ndpoll_values,
    'DPDD': dpdd_values
}

dates = ['2023-06-01']  # Single date for this demo
temporal_analysis = temporal_vegetation_monitoring(vegetation_indices, dates)

print("Temporal vegetation monitoring setup:")
for point_name, data in temporal_analysis.items():
    print(f"  {point_name}: RVI={data['RVI_series'][0]:.3f}, NDPoll={data['NDPoll_series'][0]:.3f}, DPDD={data['DPDD_series'][0]:.3f}")
```

## 8. Export Results and Summary Report

### 8.1 Save All Results

```python
# Save all polarimetric analysis results
print("Saving polarimetric analysis results...")

# Save basic polarimetric parameters
save_matlab_mat(f"{output_dir}/polarimetric_parameters.mat", {
    'copol_intensity': pol_params['copol_intensity'],
    'crosspol_intensity': pol_params['crosspol_intensity'],
    'copol_crosspol_ratio': pol_params['copol_crosspol_ratio'],
    'crosspol_fraction': pol_params['crosspol_fraction'],
    'polarization_names': pol_params['pol_names']
})

# Save vegetation indices
save_matlab_mat(f"{output_dir}/vegetation_indices.mat", {
    'RVI': rvi_values,
    'NDPoll': ndpoll_values,
    'DPDD': dpdd_values
})

# Save decomposition results
save_matlab_mat(f"{output_dir}/polarimetric_decomposition.mat", {
    'volume_scattering': decomp_results['volume'],
    'surface_scattering': decomp_results['surface'],
    'double_bounce': decomp_results['double_bounce'],
    'scattering_map': scattering_stats['scattering_map']
})

# Save SLA results for each polarization
for pol_name, sla_data in pol_sla_results.items():
    np.save(f"{output_dir}/sublooks_{pol_name}.npy", np.array(sla_data['sublooks']))
    np.save(f"{output_dir}/master_{pol_name}.npy", sla_data['master_image'])
    np.save(f"{output_dir}/coherence_{pol_name}.npy", sla_data['coherence'])

print("Results saved successfully!")
```

### 8.2 Generate Comprehensive Report

```python
def generate_polarimetric_report(pol_params, vegetation_indices, decomp_results, 
                               scattering_stats, quality_metrics, output_path):
    """Generate comprehensive polarimetric analysis report"""
    
    report = {
        'processing_info': {
            'polarizations': pol_params['pol_names'],
            'image_dimensions': pol_params['copol_intensity'].shape,
            'processing_date': '2023-06-01'  # Update with actual date
        },
        'polarimetric_statistics': {
            'copol_mean_db': float(np.mean(pol_params['copol_db'])),
            'crosspol_mean_db': float(np.mean(pol_params['crosspol_db'])),
            'ratio_mean_db': float(np.mean(pol_params['ratio_db'])),
            'crosspol_fraction_mean': float(np.mean(pol_params['crosspol_fraction']))
        },
        'vegetation_indices': {
            'RVI_mean': float(np.mean(vegetation_indices['RVI'])),
            'RVI_std': float(np.std(vegetation_indices['RVI'])),
            'NDPoll_mean': float(np.mean(vegetation_indices['NDPoll'])),
            'NDPoll_std': float(np.std(vegetation_indices['NDPoll'])),
            'DPDD_mean': float(np.mean(vegetation_indices['DPDD'])),
            'DPDD_std': float(np.std(vegetation_indices['DPDD']))
        },
        'scattering_mechanisms': {
            'volume_percentage': scattering_stats['volume_percentage'],
            'surface_percentage': scattering_stats['surface_percentage'],
            'double_bounce_percentage': scattering_stats['double_bounce_percentage'],
            'mixed_percentage': scattering_stats['mixed_percentage']
        },
        'quality_assessment': quality_metrics
    }
    
    # Save report
    import json
    with open(f"{output_path}/polarimetric_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

# Generate comprehensive report
report = generate_polarimetric_report(
    pol_params, 
    {'RVI': rvi_values, 'NDPoll': ndpoll_values, 'DPDD': dpdd_values},
    decomp_results,
    scattering_stats,
    quality_metrics,
    output_dir
)

print("=== POLARIMETRIC ANALYSIS REPORT ===")
print(f"Polarizations processed: {report['processing_info']['polarizations']}")
print(f"Image dimensions: {report['processing_info']['image_dimensions']}")
print(f"\nPolarimetric Statistics:")
print(f"  Co-pol mean: {report['polarimetric_statistics']['copol_mean_db']:.2f} dB")
print(f"  Cross-pol mean: {report['polarimetric_statistics']['crosspol_mean_db']:.2f} dB")
print(f"  Co-pol/Cross-pol ratio: {report['polarimetric_statistics']['ratio_mean_db']:.2f} dB")
print(f"\nVegetation Indices:")
print(f"  RVI mean: {report['vegetation_indices']['RVI_mean']:.3f}")
print(f"  NDPoll mean: {report['vegetation_indices']['NDPoll_mean']:.3f}")
print(f"  DPDD mean: {report['vegetation_indices']['DPDD_mean']:.3f}")
print(f"\nScattering Mechanisms:")
print(f"  Volume scattering: {report['scattering_mechanisms']['volume_percentage']:.1f}%")
print(f"  Surface scattering: {report['scattering_mechanisms']['surface_percentage']:.1f}%")
print(f"  Double-bounce: {report['scattering_mechanisms']['double_bounce_percentage']:.1f}%")
```

### 8.3 Final Visualization Summary

```python
# Create comprehensive summary figure
fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# Row 1: Basic polarimetric components
im1 = axes[0, 0].imshow(pol_params['copol_db'], cmap='viridis',
                       vmin=np.percentile(pol_params['copol_db'], 5),
                       vmax=np.percentile(pol_params['copol_db'], 95))
axes[0, 0].set_title(f'{pol_params["pol_names"][0]} Intensity (dB)')
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(pol_params['crosspol_db'], cmap='viridis',
                       vmin=np.percentile(pol_params['crosspol_db'], 5),
                       vmax=np.percentile(pol_params['crosspol_db'], 95))
axes[0, 1].set_title(f'{pol_params["pol_names"][1]} Intensity (dB)')
plt.colorbar(im2, ax=axes[0, 1])

im3 = axes[0, 2].imshow(pol_params['ratio_db'], cmap='RdYlBu_r',
                       vmin=np.percentile(pol_params['ratio_db'], 5),
                       vmax=np.percentile(pol_params['ratio_db'], 95))
axes[0, 2].set_title('Co-pol/Cross-pol Ratio (dB)')
plt.colorbar(im3, ax=axes[0, 2])

# Row 2: Vegetation indices
im4 = axes[1, 0].imshow(rvi_values, cmap='RdYlGn',
                       vmin=np.percentile(rvi_values, 5),
                       vmax=np.percentile(rvi_values, 95))
axes[1, 0].set_title('Radar Vegetation Index (RVI)')
plt.colorbar(im4, ax=axes[1, 0])

im5 = axes[1, 1].imshow(ndpoll_values, cmap='RdYlGn',
                       vmin=np.percentile(ndpoll_values, 5),
                       vmax=np.percentile(ndpoll_values, 95))
axes[1, 1].set_title('NDPoll Index')
plt.colorbar(im5, ax=axes[1, 1])

im6 = axes[1, 2].imshow(dpdd_values, cmap='RdYlGn',
                       vmin=np.percentile(dpdd_values, 5),
                       vmax=np.percentile(dpdd_values, 95))
axes[1, 2].set_title('DPDD Index')
plt.colorbar(im6, ax=axes[1, 2])

# Row 3: Decomposition and analysis
axes[2, 0].imshow(decomp_rgb)
axes[2, 0].set_title('Polarimetric Decomposition\n(R=Double-bounce, G=Volume, B=Surface)')
axes[2, 0].axis('off')

# Scattering mechanism distribution
mechanisms = ['Mixed', 'Volume', 'Surface', 'Double-bounce']
percentages = [scattering_stats['mixed_percentage'],
              scattering_stats['volume_percentage'],
              scattering_stats['surface_percentage'],
              scattering_stats['double_bounce_percentage']]

axes[2, 1].pie(percentages, labels=mechanisms, autopct='%1.1f%%',
              colors=['gray', 'green', 'blue', 'red'])
axes[2, 1].set_title('Scattering Mechanism Distribution')

# Quality metrics summary
pol_names = [k for k in quality_metrics.keys() if k != 'inter_polarization']
snr_values = [quality_metrics[pol]['mean_snr_db'] for pol in pol_names]
speckle_reduction = [quality_metrics[pol]['speckle_reduction_factor'] for pol in pol_names]

x_pos = np.arange(len(pol_names))
width = 0.35

bars1 = axes[2, 2].bar(x_pos - width/2, snr_values, width, label='SNR (dB)', alpha=0.7)
bars2 = axes[2, 2].bar(x_pos + width/2, speckle_reduction, width, label='Speckle Reduction', alpha=0.7)

axes[2, 2].set_xlabel('Polarization')
axes[2, 2].set_ylabel('Value')
axes[2, 2].set_title('Quality Metrics Summary')
axes[2, 2].set_xticks(x_pos)
axes[2, 2].set_xticklabels(pol_names)
axes[2, 2].legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/polarimetric_analysis_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPolarimetric analysis complete!")
print(f"Output files saved to: {output_dir}/")
print("Generated files:")
print("- polarimetric_components.png")
print("- vegetation_indices.png")
print("- polarimetric_decomposition.png")
print("- scattering_mechanisms.png")
print("- polarimetric_analysis_summary.png")
print("- polarimetric_parameters.mat")
print("- vegetation_indices.mat")
print("- polarimetric_decomposition.mat")
print("- polarimetric_report.json")
print("- sublooks_*.npy, master_*.npy, coherence_*.npy")
```

## Summary

In this tutorial, you learned:

1. **Dual-pol data processing** with SNAP integration
2. **Basic polarimetric parameter extraction** (co-pol, cross-pol, ratios)
3. **Vegetation index calculation** (RVI, NDPoll, DPDD)
4. **Polarimetric decomposition** (Freeman-Durden approach)
5. **Scattering mechanism analysis** and classification
6. **SLA integration** with polarimetric data
7. **Quality assessment** for polarimetric processing
8. **Multi-temporal monitoring** concepts for vegetation analysis

## Next Steps

- **Tutorial 6**: Custom processing workflows for specific applications
- **Tutorial 7**: Ship detection with CFAR algorithms
- Experiment with full polarimetric data (quad-pol)
- Apply to different land cover types
- Integrate with optical data for enhanced analysis

## Troubleshooting

**Common Issues:**

1. **Polarization mismatch**: Ensure consistent polarization naming conventions
2. **SNAP processing errors**: Check SNAP installation and GPT configuration
3. **Memory limitations**: Process smaller spatial subsets for large datasets
4. **Low inter-pol coherence**: Check data quality and temporal baseline
5. **Unrealistic vegetation indices**: Verify calibration and preprocessing steps

For more help, see the [Troubleshooting Guide](../user_guide/troubleshooting.md) and [SNAP Integration Guide](../user_guide/snap_integration.md).
