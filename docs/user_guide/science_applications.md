# Science Applications

sarpyx provides specialized algorithms and indices for various scientific applications using SAR data. This guide covers vegetation monitoring, polarimetric analysis, and other Earth observation applications.

## Overview

The `sarpyx.science` module contains scientifically validated algorithms for:

- **Vegetation Monitoring**: Radar vegetation indices for crop monitoring and forest analysis
- **Polarimetric Analysis**: Multi-polarization data exploitation 
- **Land Cover Classification**: Indices supporting classification workflows
- **Change Detection**: Multi-temporal analysis capabilities
- **Environmental Monitoring**: Forest degradation and agricultural applications

## Vegetation Indices

### Radar Vegetation Index (RVI)

The Radar Vegetation Index is designed to monitor vegetation density and health using dual-polarization SAR data.

```python
from sarpyx.science.indices import calculate_rvi

# Load backscatter data (linear scale)
sigma_vv = load_sentinel1_band('VV_band.tif')  # Your loading function
sigma_vh = load_sentinel1_band('VH_band.tif')

# Calculate RVI
rvi = calculate_rvi(sigma_vv, sigma_vh)

# RVI ranges from 0 (low vegetation) to 1 (high vegetation)
print(f"RVI range: {rvi.min():.3f} to {rvi.max():.3f}")
```

**Formula**: `RVI = (4 * σ⁰_VH) / (σ⁰_VV + σ⁰_VH)`

**Applications**:
- Crop growth monitoring
- Forest biomass estimation  
- Vegetation change detection
- Agricultural phenology tracking

### Normalized Difference Polarization Index (NDPoll)

NDPoll exploits the polarization signature difference for vegetation characterization.

```python
from sarpyx.science.indices import calculate_ndpoll

# Calculate NDPoll
ndpoll = calculate_ndpoll(sigma_vv, sigma_vh)

# NDPoll ranges from -1 to +1
# Positive values typically indicate vegetated areas
# Negative values often correspond to bare soil or water
```

**Formula**: `NDPoll = (σ⁰_VV - σ⁰_VH) / (σ⁰_VV + σ⁰_VH)`

**Applications**:
- Land cover discrimination
- Urban vs. vegetated area mapping
- Water body detection

### Dual-Pol Diagonal Distance (DPDD)

DPDD provides a geometric interpretation of dual-polarization scattering.

```python
from sarpyx.science.indices import calculate_dpdd

# Calculate DPDD
dpdd = calculate_dpdd(sigma_vv, sigma_vh)

# DPDD represents the total scattering power
```

**Formula**: `DPDD = (σ⁰_VV + σ⁰_VH) / √2`

**Applications**:
- Total scattering power assessment
- Temporal change analysis
- Flood mapping support

## Advanced Vegetation Indices

### Dual-Polarized Radar Vegetation Indices

For specific polarization combinations:

```python
from sarpyx.science.indices import calculate_dprvi_vv, calculate_dprvi_hh

# VV-based DpRVI (for Sentinel-1 VV/VH data)
dprvi_vv = calculate_dprvi_vv(sigma_vv, sigma_vh)

# HH-based DpRVI (for ALOS PALSAR HH/HV data) 
dprvi_hh = calculate_dprvi_hh(sigma_hh, sigma_hv)
```

### Quad-Polarized Radar Vegetation Index (QpRVI)

For fully polarimetric data (L-band missions like ALOS PALSAR):

```python
from sarpyx.science.indices import calculate_qprvi

# Requires HH, VV, and HV channels
qprvi = calculate_qprvi(sigma_hh, sigma_vv, sigma_hv)
```

**Formula**: `QpRVI = (8 * σ⁰_HV) / (σ⁰_HH + σ⁰_VV + 2 * σ⁰_HV)`

### Forest Monitoring Indices

#### Radar Forest Degradation Index (RFDI)

Specifically designed for forest degradation monitoring:

```python
from sarpyx.science.indices import calculate_rfdi

# Requires HH and HV polarizations (L-band optimal)
rfdi = calculate_rfdi(sigma_hh, sigma_hv)

# RFDI < 0: Intact forest
# RFDI > 0: Degraded/deforested areas
```

#### Vertical Dual De-Polarization Index (VDDPI)

Measures depolarization effects in vegetation:

```python
from sarpyx.science.indices import calculate_vddpi

vddpi = calculate_vddpi(sigma_vv, sigma_vh)
```

## Complete Vegetation Analysis Workflow

```python
import numpy as np
from pathlib import Path
from sarpyx.science.indices import (
    calculate_rvi, calculate_ndpoll, calculate_dpdd,
    calculate_dprvi_vv, calculate_vddpi
)

def vegetation_analysis_workflow(vv_path, vh_path, output_dir):
    """Complete vegetation index analysis."""
    
    # Load data (implement your data loader)
    sigma_vv = load_geotiff(vv_path)
    sigma_vh = load_geotiff(vh_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Calculate all relevant indices
    indices = {
        'RVI': calculate_rvi(sigma_vv, sigma_vh),
        'NDPoll': calculate_ndpoll(sigma_vv, sigma_vh),
        'DPDD': calculate_dpdd(sigma_vv, sigma_vh),
        'DpRVI_VV': calculate_dprvi_vv(sigma_vv, sigma_vh),
        'VDDPI': calculate_vddpi(sigma_vv, sigma_vh)
    }
    
    # Save results
    for name, index in indices.items():
        output_file = output_dir / f"{name}.tif"
        save_geotiff(index, output_file, geotransform, projection)
        print(f"Saved {name} to {output_file}")
    
    # Generate statistics
    stats = {}
    for name, index in indices.items():
        valid_mask = ~np.isnan(index)
        stats[name] = {
            'mean': np.mean(index[valid_mask]),
            'std': np.std(index[valid_mask]),
            'min': np.min(index[valid_mask]),
            'max': np.max(index[valid_mask]),
            'valid_pixels': np.sum(valid_mask)
        }
    
    return indices, stats

# Example usage
indices, statistics = vegetation_analysis_workflow(
    'sentinel1_vv.tif',
    'sentinel1_vh.tif', 
    'vegetation_analysis_results/'
)

# Print summary statistics
for index_name, stats in statistics.items():
    print(f"\n{index_name}:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
```

## Multi-Temporal Analysis

### Time Series Vegetation Monitoring

```python
def temporal_vegetation_analysis(time_series_paths, output_dir):
    """Analyze vegetation changes over time."""
    
    time_series_indices = {}
    
    for date, (vv_path, vh_path) in time_series_paths.items():
        # Load data for this date
        sigma_vv = load_geotiff(vv_path)
        sigma_vh = load_geotiff(vh_path)
        
        # Calculate indices
        time_series_indices[date] = {
            'RVI': calculate_rvi(sigma_vv, sigma_vh),
            'NDPoll': calculate_ndpoll(sigma_vv, sigma_vh)
        }
    
    # Calculate temporal statistics
    dates = sorted(time_series_indices.keys())
    
    # RVI temporal analysis
    rvi_stack = np.stack([time_series_indices[date]['RVI'] for date in dates])
    
    # Temporal mean and standard deviation
    rvi_mean = np.nanmean(rvi_stack, axis=0)
    rvi_std = np.nanstd(rvi_stack, axis=0)
    
    # Trend analysis (simple linear fit)
    rvi_trend = calculate_temporal_trend(rvi_stack)
    
    # Save temporal products
    save_geotiff(rvi_mean, output_dir / "RVI_temporal_mean.tif")
    save_geotiff(rvi_std, output_dir / "RVI_temporal_std.tif")
    save_geotiff(rvi_trend, output_dir / "RVI_trend.tif")
    
    return {
        'temporal_mean': rvi_mean,
        'temporal_std': rvi_std,
        'trend': rvi_trend
    }

def calculate_temporal_trend(time_stack):
    """Calculate linear trend over time dimension."""
    n_times = time_stack.shape[0]
    time_vector = np.arange(n_times)
    
    # Vectorized linear regression
    trend = np.full(time_stack.shape[1:], np.nan)
    
    for i in range(time_stack.shape[1]):
        for j in range(time_stack.shape[2]):
            pixel_series = time_stack[:, i, j]
            valid_mask = ~np.isnan(pixel_series)
            
            if np.sum(valid_mask) > 2:  # Need at least 3 points
                x = time_vector[valid_mask]
                y = pixel_series[valid_mask]
                trend[i, j] = np.polyfit(x, y, 1)[0]  # Slope only
    
    return trend
```

## Polarimetric Scattering Analysis

### Polarization Ratio Analysis

```python
def polarization_ratios(sigma_vv, sigma_vh, sigma_hh=None, sigma_hv=None):
    """Calculate various polarization ratios."""
    
    ratios = {}
    
    # VV/VH ratio (depolarization ratio)
    ratios['VV_VH_ratio'] = calculate_vvvhr(sigma_vv, sigma_vh)
    
    # VH/VV ratio (cross-pol ratio)
    ratios['VH_VV_ratio'] = calculate_vhvvr(sigma_vh, sigma_vv)
    
    # VV+VH sum (total power)
    ratios['VV_VH_sum'] = calculate_vvvhs(sigma_vv, sigma_vh)
    
    # If quad-pol data available
    if sigma_hh is not None and sigma_hv is not None:
        ratios['HH_HV_ratio'] = sigma_hh / np.where(sigma_hv != 0, sigma_hv, np.nan)
        ratios['co_cross_ratio'] = (sigma_vv + sigma_hh) / (sigma_vh + sigma_hv)
    
    return ratios

# Example usage
from sarpyx.science.indices import calculate_vvvhr, calculate_vhvvr, calculate_vvvhs

ratios = polarization_ratios(sigma_vv, sigma_vh)
```

## Agricultural Applications

### Crop Phenology Monitoring

```python
def crop_phenology_analysis(field_boundaries, time_series_data):
    """Monitor crop development stages using radar indices."""
    
    phenology_metrics = {}
    
    for field_id, field_polygon in field_boundaries.items():
        field_time_series = extract_field_statistics(time_series_data, field_polygon)
        
        # Calculate RVI for each date
        rvi_series = []
        for date, (vv, vh) in field_time_series.items():
            field_rvi = calculate_rvi(vv, vh)
            rvi_series.append((date, np.nanmean(field_rvi)))
        
        # Identify phenological stages
        phenology_metrics[field_id] = {
            'emergence': detect_emergence(rvi_series),
            'peak_growth': detect_peak_growth(rvi_series),
            'senescence': detect_senescence(rvi_series),
            'harvest': detect_harvest(rvi_series)
        }
    
    return phenology_metrics

def detect_peak_growth(rvi_time_series):
    """Detect peak growth stage from RVI time series."""
    dates, rvi_values = zip(*rvi_time_series)
    peak_idx = np.argmax(rvi_values)
    return dates[peak_idx], rvi_values[peak_idx]
```

## Forest Applications

### Forest Biomass Estimation

```python
def forest_biomass_estimation(sigma_vv, sigma_vh, biomass_model='rvi_power'):
    """Estimate forest biomass using radar indices."""
    
    # Calculate RVI
    rvi = calculate_rvi(sigma_vv, sigma_vh)
    
    if biomass_model == 'rvi_power':
        # Power law relationship: Biomass = a * RVI^b
        a, b = 150.0, 0.8  # Model coefficients (site-specific)
        biomass = a * np.power(rvi, b)
    
    elif biomass_model == 'multi_index':
        # Multi-index approach
        ndpoll = calculate_ndpoll(sigma_vv, sigma_vh)
        dpdd = calculate_dpdd(sigma_vv, sigma_vh)
        
        # Linear combination (coefficients from regression)
        biomass = 45.2 * rvi + 32.1 * ndpoll + 0.8 * dpdd + 25.0
    
    # Apply realistic constraints
    biomass = np.clip(biomass, 0, 500)  # Typical forest range: 0-500 Mg/ha
    
    return biomass

# Example usage
estimated_biomass = forest_biomass_estimation(sigma_vv, sigma_vh, 'multi_index')
save_geotiff(estimated_biomass, 'forest_biomass_estimate.tif')
```

## Best Practices and Considerations

### Data Requirements

1. **Input Data Scale**: All functions expect linear backscatter coefficients (σ⁰), not dB values
2. **Calibration**: Ensure radiometric calibration for absolute backscatter values
3. **Noise Floor**: Consider radar noise floor, especially for low backscatter areas
4. **Temporal Consistency**: Use consistent processing for time series analysis

### Quality Control

```python
def quality_control_vegetation_indices(indices_dict):
    """Apply quality control to vegetation indices."""
    
    qc_results = {}
    
    for name, index in indices_dict.items():
        # Remove invalid values
        valid_mask = np.isfinite(index)
        
        # Apply physical constraints
        if name == 'RVI':
            # RVI should be between 0 and 1
            index = np.where((index >= 0) & (index <= 1), index, np.nan)
        elif name == 'NDPoll':
            # NDPoll should be between -1 and 1  
            index = np.where((index >= -1) & (index <= 1), index, np.nan)
        
        # Flag outliers (3-sigma rule)
        mean_val = np.nanmean(index)
        std_val = np.nanstd(index)
        outlier_mask = np.abs(index - mean_val) > 3 * std_val
        index[outlier_mask] = np.nan
        
        qc_results[name] = {
            'processed_index': index,
            'valid_pixels': np.sum(~np.isnan(index)),
            'outliers_removed': np.sum(outlier_mask)
        }
    
    return qc_results
```

### Validation and Accuracy Assessment

```python
def validate_against_ground_truth(predicted_indices, ground_truth_data):
    """Validate radar indices against field measurements."""
    
    validation_results = {}
    
    for index_name, index_values in predicted_indices.items():
        if index_name in ground_truth_data:
            gt_values = ground_truth_data[index_name]
            
            # Extract co-located values
            valid_mask = ~(np.isnan(index_values) | np.isnan(gt_values))
            pred_valid = index_values[valid_mask]
            gt_valid = gt_values[valid_mask]
            
            # Calculate validation metrics
            correlation = np.corrcoef(pred_valid, gt_valid)[0, 1]
            rmse = np.sqrt(np.mean((pred_valid - gt_valid)**2))
            bias = np.mean(pred_valid - gt_valid)
            
            validation_results[index_name] = {
                'correlation': correlation,
                'rmse': rmse,
                'bias': bias,
                'n_samples': len(pred_valid)
            }
    
    return validation_results
```

## Integration with Sub-Look Analysis

Combine vegetation indices with sub-look analysis for enhanced vegetation characterization:

```python
from sarpyx.sla import SubLookAnalysis

def enhanced_vegetation_analysis(slc_product, output_dir):
    """Combine sub-look analysis with vegetation indices."""
    
    # Perform sub-look decomposition
    sla = SubLookAnalysis(slc_product)
    sla.choice = 1  # Azimuth processing
    sla.numberOfLooks = 3
    
    sla.frequencyComputation()
    sla.SpectrumComputation() 
    sla.AncillaryDeWe()
    sla.Generation()
    
    # Extract sub-look intensity
    sublook_intensity = {}
    for i, look in enumerate(sla.Looks):
        sublook_intensity[f'look_{i+1}'] = np.abs(look)**2
    
    # Calculate vegetation indices for each sub-look
    vegetation_results = {}
    
    for look_name, intensity in sublook_intensity.items():
        # Convert to equivalent VV/VH for vegetation analysis
        # (specific conversion depends on original polarization)
        vv_equiv = intensity  # Simplified assumption
        vh_equiv = intensity * 0.1  # Typical cross-pol ratio
        
        vegetation_results[look_name] = {
            'RVI': calculate_rvi(vv_equiv, vh_equiv),
            'NDPoll': calculate_ndpoll(vv_equiv, vh_equiv)
        }
    
    return sla, vegetation_results
```

## References and Further Reading

### Key Publications

1. **RVI Development**: Kim & van Zyl (2009) - "A Time-Series Approach to Estimate Soil Moisture Using Polarimetric Radar Data"
2. **NDPoll Applications**: Trudel et al. (2012) - "Using RADARSAT-2 polarimetric and ENVISAT-ASAR dual-polarization data for estimating soil moisture"
3. **DPDD Method**: Mandal et al. (2018) - "Dual polarimetric radar vegetation index for crop growth monitoring using sentinel-1 SAR data"
4. **Forest Applications**: Antropov et al. (2017) - "Stand-level stem volume of boreal forests from spaceborne SAR imagery"

### Online Resources

- [ESA SNAP Tutorials](https://step.esa.int/main/toolboxes/snap/)
- [NASA ARSET SAR Training](https://appliedsciences.nasa.gov/join-mission/training/english/arset-synthetic-aperture-radar)
- [Sentinel-1 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar)

## Troubleshooting

### Common Issues

1. **NaN Values**: Check for zero denominators in ratio calculations
2. **Unrealistic Index Values**: Verify input data scale (linear vs. dB)
3. **Memory Issues**: Process large datasets in chunks or use memory mapping
4. **Noisy Results**: Apply spatial filtering or temporal averaging

### Performance Optimization

```python
# For large datasets, use chunked processing
def process_large_dataset_chunked(large_array_vv, large_array_vh, chunk_size=1024):
    """Process large arrays in memory-efficient chunks."""
    
    result = np.full_like(large_array_vv, np.nan)
    
    for i in range(0, large_array_vv.shape[0], chunk_size):
        for j in range(0, large_array_vv.shape[1], chunk_size):
            # Extract chunk
            vv_chunk = large_array_vv[i:i+chunk_size, j:j+chunk_size]
            vh_chunk = large_array_vh[i:i+chunk_size, j:j+chunk_size]
            
            # Process chunk
            result_chunk = calculate_rvi(vv_chunk, vh_chunk)
            
            # Store result
            result[i:i+chunk_size, j:j+chunk_size] = result_chunk
    
    return result
```
