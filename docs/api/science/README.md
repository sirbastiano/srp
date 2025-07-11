# Science Module API

The `sarpyx.science` module provides scientific analysis tools and vegetation indices specifically designed for SAR remote sensing applications.

## Overview

The science module focuses on extracting meaningful scientific information from SAR data, particularly for vegetation monitoring, agricultural applications, and forest analysis. It implements various radar-based indices that are sensitive to vegetation structure, biomass, and phenological changes.

## Quick Start

```python
from sarpyx.science import indices

# Load your SAR backscatter data (linear scale)
sigma_vv = load_vv_data()  # VV polarization backscatter
sigma_vh = load_vh_data()  # VH polarization backscatter

# Calculate vegetation indices
rvi = indices.calculate_rvi(sigma_vv, sigma_vh)
ndpoll = indices.calculate_ndpoll(sigma_vv, sigma_vh)
dprvi = indices.calculate_dprvi_vv(sigma_vv, sigma_vh)
```

## Vegetation Indices

### Dual-Polarization Indices (VV/VH)

#### Radar Vegetation Index (RVI)
```python
rvi = indices.calculate_rvi(sigma_vv, sigma_vh)
```
- **Formula**: `RVI = (4 * VH) / (VV + VH)`
- **Range**: 0-4, higher values indicate more vegetation
- **Best for**: General vegetation detection and biomass estimation
- **References**: [Polarimetric SAR applications](https://www.mdpi.com/2076-3417/9/4/655)

#### Normalized Difference Polarization Index (NDPoll)
```python
ndpoll = indices.calculate_ndpoll(sigma_vv, sigma_vh)
```
- **Formula**: `NDPoll = (VV - VH) / (VV + VH)`  
- **Range**: -1 to 1, negative values indicate vegetation dominance
- **Best for**: Vegetation vs. non-vegetation classification

#### Dual-Polarized Radar Vegetation Index VV (DpRVIVV)
```python
dprvi_vv = indices.calculate_dprvi_vv(sigma_vv, sigma_vh)
```
- **Formula**: `DpRVIVV = (4.0 * VH) / (VV + VH)`
- **Range**: 0-4, similar to RVI but optimized for VV/VH combinations
- **Best for**: Agricultural crop monitoring

#### Dual-Pol Diagonal Distance (DPDD)
```python
dpdd = indices.calculate_dpdd(sigma_vv, sigma_vh)
```
- **Formula**: `DPDD = (VV + VH) / √2`
- **Best for**: Total backscatter assessment in dual-pol data

#### Vertical Dual De-Polarization Index (VDDPI)
```python
vddpi = indices.calculate_vddpi(sigma_vv, sigma_vh)
```
- **Formula**: `VDDPI = (VV + VH) / VV`
- **Best for**: Vegetation structure analysis

### Quad-Polarization Indices (HH/HV/VV)

#### Dual-Polarized Radar Vegetation Index HH (DpRVIHH)
```python
dprvi_hh = indices.calculate_dprvi_hh(sigma_hh, sigma_hv)
```
- **Formula**: `DpRVIHH = (4.0 * HV) / (HH + HV)`
- **Best for**: Forest structure analysis with HH/HV data
- **References**: [Radar vegetation indices](https://www.tandfonline.com/doi/abs/10.5589/m12-043)

#### Quad-Polarized Radar Vegetation Index (QpRVI)
```python
qprvi = indices.calculate_qprvi(sigma_hh, sigma_vv, sigma_hv)
```
- **Formula**: `QpRVI = (8.0 * HV) / (HH + VV + 2.0 * HV)`
- **Best for**: Comprehensive vegetation analysis with full polarimetric data

#### Radar Forest Degradation Index (RFDI)
```python
rfdi = indices.calculate_rfdi(sigma_hh, sigma_hv)
```
- **Formula**: `RFDI = (HH - HV) / (HH + HV)`
- **Range**: -1 to 1
- **Best for**: Forest health and degradation monitoring

### Polarization Ratios and Differences

#### VH/VV Ratio
```python
vhvv_ratio = indices.calculate_vhvvr(sigma_vh, sigma_vv)
```
- **Formula**: `VHVVR = VH / VV`
- **Best for**: Depolarization analysis

#### VV/VH Ratio
```python
vvvh_ratio = indices.calculate_vvvhr(sigma_vv, sigma_vh)
```
- **Formula**: `VVVHR = VV / VH`
- **Best for**: Inverse depolarization analysis

#### VH-VV Difference
```python
vhvv_diff = indices.calculate_vhvvd(sigma_vh, sigma_vv)
```
- **Formula**: `VHVVD = VH - VV`
- **Best for**: Absolute backscatter differences

#### VV-VH Difference
```python
vvvh_diff = indices.calculate_vvvhd(sigma_vv, sigma_vh)
```
- **Formula**: `VVVHD = VV - VH`
- **Best for**: Surface roughness analysis

#### VV+VH Sum
```python
vvvh_sum = indices.calculate_vvvhs(sigma_vv, sigma_vh)
```
- **Formula**: `VVVHS = VV + VH`
- **Best for**: Total backscatter power

## Usage Examples

### Agricultural Monitoring

```python
import numpy as np
from sarpyx.science import indices

# Load Sentinel-1 dual-pol data (linear scale)
vv_data = load_backscatter_data('VV')
vh_data = load_backscatter_data('VH')

# Calculate multiple vegetation indices
rvi = indices.calculate_rvi(vv_data, vh_data)
ndpoll = indices.calculate_ndpoll(vv_data, vh_data)
dprvi = indices.calculate_dprvi_vv(vv_data, vh_data)

# Create a composite vegetation index
vegetation_composite = np.stack([rvi, ndpoll, dprvi], axis=-1)

# Apply thresholds for crop classification
crop_mask = (rvi > 0.5) & (ndpoll < -0.2)
print(f"Vegetated area: {np.sum(crop_mask) / crop_mask.size * 100:.1f}%")
```

### Forest Monitoring with Quad-Pol Data

```python
# Load quad-pol data
hh_data = load_backscatter_data('HH')
hv_data = load_backscatter_data('HV')
vv_data = load_backscatter_data('VV')

# Calculate forest-specific indices
qprvi = indices.calculate_qprvi(hh_data, vv_data, hv_data)
rfdi = indices.calculate_rfdi(hh_data, hv_data)
dprvi_hh = indices.calculate_dprvi_hh(hh_data, hv_data)

# Forest health assessment
healthy_forest = (qprvi > 1.0) & (rfdi > 0.1)
degraded_forest = (qprvi < 0.5) & (rfdi < -0.1)

print(f"Healthy forest: {np.sum(healthy_forest) / healthy_forest.size * 100:.1f}%")
print(f"Degraded forest: {np.sum(degraded_forest) / degraded_forest.size * 100:.1f}%")
```

### Multi-Temporal Analysis

```python
# Load time series of SAR data
dates = ['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']
rvi_time_series = []

for date in dates:
    vv = load_sar_data(date, 'VV')
    vh = load_sar_data(date, 'VH')
    rvi = indices.calculate_rvi(vv, vh)
    rvi_time_series.append(rvi)

# Convert to numpy array (time, height, width)
rvi_stack = np.stack(rvi_time_series, axis=0)

# Calculate temporal statistics
rvi_mean = np.mean(rvi_stack, axis=0)
rvi_std = np.std(rvi_stack, axis=0)
rvi_trend = np.polyfit(range(len(dates)), rvi_stack, deg=1)[0]

# Identify areas with increasing vegetation
increasing_vegetation = rvi_trend > 0.01
print(f"Areas with vegetation growth: {np.sum(increasing_vegetation) / increasing_vegetation.size * 100:.1f}%")
```

### Integration with Sub-Look Analysis

```python
from sarpyx.sla import SubLookAnalysis
from sarpyx.science import indices

# Perform sub-look analysis first
sla = SubLookAnalysis(product_path)
sla.choice = 1  # Azimuth processing
sla.numberOfLooks = 3
sla.SpectrumComputation()
sla.Generation()

# Extract individual sub-looks
sublook_1 = sla.Looks[0, :, :]
sublook_2 = sla.Looks[1, :, :]
sublook_3 = sla.Looks[2, :, :]

# Calculate RVI for each sub-look (if dual-pol available)
if sla.polarizations == ['VV', 'VH']:
    rvi_sublook_1 = indices.calculate_rvi(sublook_1[0], sublook_1[1])
    rvi_sublook_2 = indices.calculate_rvi(sublook_2[0], sublook_2[1])
    rvi_sublook_3 = indices.calculate_rvi(sublook_3[0], sublook_3[1])
    
    # Analyze sub-look vegetation differences
    rvi_variance = np.var([rvi_sublook_1, rvi_sublook_2, rvi_sublook_3], axis=0)
    high_variance_areas = rvi_variance > 0.1  # Areas with significant sub-look differences
```

## Data Requirements

### Input Data Format
All index calculation functions expect:
- **Linear scale backscatter coefficients** (not dB)
- **NumPy arrays** with consistent shapes
- **Calibrated data** (radiometrically corrected)

```python
# Convert from dB to linear scale if needed
sigma_db = load_sar_data_db()
sigma_linear = 10**(sigma_db / 10)

# Calculate indices
rvi = indices.calculate_rvi(sigma_vv_linear, sigma_vh_linear)
```

### Quality Control

```python
# Check for valid data ranges
def validate_backscatter(sigma):
    """Validate backscatter data quality."""
    if np.any(sigma < 0):
        print("Warning: Negative backscatter values detected")
    if np.any(sigma > 10):
        print("Warning: Unusually high backscatter values detected")
    if np.any(np.isnan(sigma)):
        print(f"Warning: {np.sum(np.isnan(sigma))} NaN values found")

validate_backscatter(sigma_vv)
validate_backscatter(sigma_vh)
```

## Error Handling

All index functions include robust error handling:

```python
# Functions return NaN for invalid calculations
sigma_vv = np.array([1.0, 0.0, 2.0])  # Contains zero
sigma_vh = np.array([0.5, 1.0, 1.0])

rvi = indices.calculate_rvi(sigma_vv, sigma_vh)
# Result: [2.0, inf, 2.0] -> [2.0, nan, 2.0] after processing

# Check for and handle NaN values
valid_mask = ~np.isnan(rvi)
rvi_clean = rvi[valid_mask]
```

## Performance Optimization

### Vectorized Operations
All functions are fully vectorized for optimal performance:

```python
# Process large arrays efficiently
large_vv = np.random.random((10000, 10000))
large_vh = np.random.random((10000, 10000))

# Vectorized calculation (fast)
rvi = indices.calculate_rvi(large_vv, large_vh)
```

### Memory Management
For very large datasets:

```python
# Process in chunks to manage memory
def process_large_dataset(vv_file, vh_file, chunk_size=1000):
    rvi_results = []
    
    for i in range(0, vv_file.shape[0], chunk_size):
        vv_chunk = vv_file[i:i+chunk_size]
        vh_chunk = vh_file[i:i+chunk_size]
        
        rvi_chunk = indices.calculate_rvi(vv_chunk, vh_chunk)
        rvi_results.append(rvi_chunk)
    
    return np.concatenate(rvi_results, axis=0)
```

## Integration with Other Modules

### With SNAP Processing

```python
from sarpyx.snap import GPT
from sarpyx.science import indices

# Process with SNAP
gpt = GPT(product=sentinel1_path, outdir=output_dir)
calibrated = gpt.Calibration(['VV', 'VH'])
terrain_corrected = gpt.TerrainCorrection()

# Load processed data and calculate indices
vv_data = load_snap_output(terrain_corrected, 'VV')
vh_data = load_snap_output(terrain_corrected, 'VH')

rvi = indices.calculate_rvi(vv_data, vh_data)
```

### With Visualization

```python
from sarpyx.utils import show_image
from sarpyx.science import indices

# Calculate and visualize indices
rvi = indices.calculate_rvi(sigma_vv, sigma_vh)
ndpoll = indices.calculate_ndpoll(sigma_vv, sigma_vh)

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

show_image(sigma_vv, 'VV Backscatter', ax=axes[0])
show_image(rvi, 'RVI', ax=axes[1], vmin=0, vmax=2)
show_image(ndpoll, 'NDPoll', ax=axes[2], vmin=-1, vmax=1)

plt.tight_layout()
plt.show()
```

## See Also

- [User Guide: Science Applications](../../user_guide/science_applications.md): Detailed application examples
- [SLA Module](../sla/README.md): Sub-aperture analysis for enhanced resolution
- [Utils Module](../utils/README.md): Visualization and utility functions
- [Examples](../../examples/README.md): Ready-to-run code examples
