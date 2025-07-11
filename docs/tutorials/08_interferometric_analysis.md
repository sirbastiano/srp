# Tutorial 8: Advanced Interferometric Analysis

## Overview

This tutorial covers advanced interferometric SAR (InSAR) analysis using SARPYX, including differential interferometry (DInSAR), persistent scatterer interferometry (PSI), and small baseline subset (SBAS) techniques for deformation monitoring and surface change detection.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Generate interferograms from SAR image pairs
- Apply differential interferometry for deformation analysis
- Implement persistent scatterer identification algorithms
- Perform time series analysis for long-term monitoring
- Integrate atmospheric correction techniques

## Prerequisites

- Completion of Tutorials 1-4
- Understanding of SAR interferometry principles
- Knowledge of phase unwrapping concepts
- Basic understanding of atmospheric effects in SAR

## Dataset Requirements

```python
# Required data for this tutorial
data_requirements = {
    'sar_data': 'Sentinel-1 SLC data stack (>10 images)',
    'temporal_baseline': '< 12 days for coherent pairs',
    'perpendicular_baseline': '< 200m for good coherence',
    'area': 'Stable and slowly deforming terrain',
    'dem': 'High-resolution DEM (SRTM 30m or better)',
    'reference_point': 'Stable reference area identification'
}
```

## Step 1: Data Preparation and Coregistration

### 1.1 Load and Organize SAR Data Stack

```python
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sarpyx import SLA, SNAPProcessor
from sarpyx.interferometry import InSARProcessor, PSAnalyzer
from sarpyx.utils import visualization, phase_unwrapping
from sarpyx.science import atmospheric_correction
import warnings
warnings.filterwarnings('ignore')

# Initialize processors
snap_processor = SNAPProcessor()
insar_processor = InSARProcessor()
ps_analyzer = PSAnalyzer()

# Define SAR data stack
sar_stack = {
    'master_image': 'path/to/master_S1_SLC.zip',
    'slave_images': [
        'path/to/slave1_S1_SLC.zip',
        'path/to/slave2_S1_SLC.zip',
        'path/to/slave3_S1_SLC.zip',
        # Add more slave images...
    ],
    'acquisition_dates': [
        datetime(2023, 1, 1),
        datetime(2023, 1, 13),
        datetime(2023, 1, 25),
        datetime(2023, 2, 6),
        # Corresponding dates...
    ]
}

print(f"SAR stack contains {len(sar_stack['slave_images']) + 1} images")
print(f"Time span: {sar_stack['acquisition_dates'][0]} to {sar_stack['acquisition_dates'][-1]}")

# Coregistration configuration
coreg_config = {
    'reference_band': 'VV',
    'resampling_method': 'bilinear',
    'interpolation': 'bilinear',
    'dem_correction': True,
    'dem_file': 'path/to/dem.tif',
    'output_coregistered': True
}
```

### 1.2 Perform Stack Coregistration

```python
# Coregister all images to master
print("Performing coregistration of SAR stack...")

coregistered_stack = insar_processor.coregister_stack(
    master_image=sar_stack['master_image'],
    slave_images=sar_stack['slave_images'],
    config=coreg_config
)

# Load coregistered data
master_data = coregistered_stack['master']
slave_data_list = coregistered_stack['slaves']

print(f"Master image shape: {master_data.shape}")
print(f"Number of coregistered slaves: {len(slave_data_list)}")

# Display coregistration quality
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Master image
axes[0, 0].imshow(np.abs(master_data), cmap='gray')
axes[0, 0].set_title('Master Image (Amplitude)')

# First slave image
axes[0, 1].imshow(np.abs(slave_data_list[0]), cmap='gray')
axes[0, 1].set_title('First Slave Image (Amplitude)')

# Coregistration offset map (if available)
if 'offset_map' in coregistered_stack:
    axes[1, 0].imshow(coregistered_stack['offset_map']['range'])
    axes[1, 0].set_title('Range Offset Map')
    
    axes[1, 1].imshow(coregistered_stack['offset_map']['azimuth'])
    axes[1, 1].set_title('Azimuth Offset Map')

plt.tight_layout()
plt.show()
```

## Step 2: Interferogram Generation and Analysis

### 2.1 Generate Interferograms

```python
# Generate interferograms for all pairs
interferograms = {}
coherence_maps = {}

print("Generating interferograms...")

for i, (slave_data, acq_date) in enumerate(zip(slave_data_list, sar_stack['acquisition_dates'][1:])):
    pair_name = f"master_{acq_date.strftime('%Y%m%d')}"
    
    # Generate interferogram
    ifg_result = insar_processor.generate_interferogram(
        master=master_data,
        slave=slave_data,
        window_size=(5, 5),  # Multilooking window
        compute_coherence=True
    )
    
    interferograms[pair_name] = ifg_result['interferogram']
    coherence_maps[pair_name] = ifg_result['coherence']
    
    print(f"Generated interferogram {i+1}/{len(slave_data_list)}: {pair_name}")

# Display interferograms
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
pair_names = list(interferograms.keys())[:6]  # Show first 6

for i, pair_name in enumerate(pair_names):
    if i < 6:
        row, col = i // 3, i % 3
        
        # Interferogram phase
        phase = np.angle(interferograms[pair_name])
        axes[row, col].imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[row, col].set_title(f'Interferogram: {pair_name}')
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Coherence analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, pair_name in enumerate(pair_names):
    if i < 6:
        row, col = i // 3, i % 3
        
        coherence = coherence_maps[pair_name]
        axes[row, col].imshow(coherence, cmap='viridis', vmin=0, vmax=1)
        axes[row, col].set_title(f'Coherence: {pair_name}')
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

### 2.2 Phase Unwrapping

```python
# Apply phase unwrapping to interferograms
unwrapped_phases = {}

print("Performing phase unwrapping...")

for pair_name, interferogram in interferograms.items():
    # Get corresponding coherence mask
    coherence = coherence_maps[pair_name]
    coherence_threshold = 0.3
    
    # Create quality mask
    quality_mask = coherence > coherence_threshold
    
    # Apply phase unwrapping
    wrapped_phase = np.angle(interferogram)
    unwrapped_phase = phase_unwrapping.unwrap_phase(
        wrapped_phase=wrapped_phase,
        quality_mask=quality_mask,
        algorithm='minimum_cost_flow'  # or 'branch_cut', 'quality_guided'
    )
    
    unwrapped_phases[pair_name] = unwrapped_phase
    print(f"Unwrapped phase for {pair_name}")

# Display wrapped vs unwrapped phases
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

example_pairs = list(interferograms.keys())[:4]

for i, pair_name in enumerate(example_pairs):
    # Wrapped phase
    wrapped = np.angle(interferograms[pair_name])
    axes[0, i].imshow(wrapped, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0, i].set_title(f'Wrapped: {pair_name}')
    axes[0, i].axis('off')
    
    # Unwrapped phase
    unwrapped = unwrapped_phases[pair_name]
    axes[1, i].imshow(unwrapped, cmap='jet')
    axes[1, i].set_title(f'Unwrapped: {pair_name}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
```

## Step 3: Persistent Scatterer Analysis

### 3.1 Identify Persistent Scatterers

```python
# Persistent Scatterer identification
print("Identifying Persistent Scatterers...")

# Calculate amplitude dispersion index for all pixels
amplitude_stack = np.stack([np.abs(master_data)] + [np.abs(slave) for slave in slave_data_list])
mean_amplitude = np.mean(amplitude_stack, axis=0)
std_amplitude = np.std(amplitude_stack, axis=0)

# Amplitude Dispersion Index (ADI)
adi = std_amplitude / (mean_amplitude + 1e-10)

# Coherence-based selection
mean_coherence = np.mean(np.stack(list(coherence_maps.values())), axis=0)

# PS candidate selection criteria
ps_criteria = {
    'adi_threshold': 0.25,  # Low amplitude variation
    'coherence_threshold': 0.7,  # High coherence
    'min_neighbors': 5  # Spatial density requirement
}

# Apply PS selection
ps_candidates = ps_analyzer.select_ps_candidates(
    amplitude_dispersion=adi,
    coherence=mean_coherence,
    criteria=ps_criteria
)

print(f"Selected {np.sum(ps_candidates)} PS candidates from {ps_candidates.size} pixels")
print(f"PS density: {np.sum(ps_candidates) / ps_candidates.size * 100:.2f}%")

# Visualize PS candidates
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Amplitude Dispersion Index
im1 = axes[0].imshow(adi, cmap='viridis', vmin=0, vmax=0.5)
axes[0].set_title('Amplitude Dispersion Index')
plt.colorbar(im1, ax=axes[0])

# Mean Coherence
im2 = axes[1].imshow(mean_coherence, cmap='viridis', vmin=0, vmax=1)
axes[1].set_title('Mean Coherence')
plt.colorbar(im2, ax=axes[1])

# PS Candidates
axes[2].imshow(np.abs(master_data), cmap='gray', alpha=0.7)
ps_overlay = np.ma.masked_where(~ps_candidates, ps_candidates)
axes[2].imshow(ps_overlay, cmap='Reds', alpha=0.8)
axes[2].set_title('PS Candidates Overlay')

plt.tight_layout()
plt.show()
```

### 3.2 Phase Analysis on PS Points

```python
# Extract phase time series for PS points
ps_indices = np.where(ps_candidates)
num_ps = len(ps_indices[0])

print(f"Extracting phase time series for {num_ps} PS points...")

# Create phase matrix (PS points x time)
phase_matrix = np.zeros((num_ps, len(unwrapped_phases) + 1))
phase_matrix[:, 0] = 0  # Master image reference

# Fill phase matrix with unwrapped phases
for i, (pair_name, unwrapped_phase) in enumerate(unwrapped_phases.items()):
    phase_values = unwrapped_phase[ps_indices]
    phase_matrix[:, i + 1] = phase_values

# Reference to first PS point (or average of stable area)
reference_ps_idx = 0  # Choose a stable reference point
phase_matrix_ref = phase_matrix - phase_matrix[reference_ps_idx, :]

# Convert phase to displacement (assuming C-band, ~5.6 cm wavelength)
wavelength = 0.056  # meters
displacement_matrix = phase_matrix_ref * wavelength / (4 * np.pi)  # LOS displacement

print(f"Phase matrix shape: {phase_matrix.shape}")
print(f"Displacement range: {np.min(displacement_matrix):.3f} to {np.max(displacement_matrix):.3f} m")

# Visualize time series for selected PS points
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot time series for first few PS points
time_axis = [sar_stack['acquisition_dates'][0]] + sar_stack['acquisition_dates'][1:]

for i in range(min(4, num_ps)):
    row, col = i // 2, i % 2
    displacement_ts = displacement_matrix[i, :]
    
    axes[row, col].plot(time_axis, displacement_ts * 1000, 'o-', linewidth=2, markersize=4)
    axes[row, col].set_ylabel('LOS Displacement (mm)')
    axes[row, col].set_title(f'PS Point {i+1}')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Step 4: Atmospheric Phase Screen Estimation

### 4.1 Estimate Atmospheric Contributions

```python
# Atmospheric phase screen estimation
print("Estimating atmospheric phase screens...")

atmospheric_processor = atmospheric_correction.AtmosphericProcessor()

# Configure atmospheric correction
atm_config = {
    'method': 'linear_regression',  # 'linear_regression', 'kriging', 'polynomial'
    'elevation_dependent': True,
    'dem_file': 'path/to/dem.tif',
    'reference_height': 0,  # Sea level reference
    'filter_size': 5,  # Spatial filtering
    'temporal_filtering': True
}

atmospheric_screens = {}
corrected_phases = {}

for pair_name, unwrapped_phase in unwrapped_phases.items():
    # Estimate atmospheric phase screen
    atm_screen = atmospheric_processor.estimate_atmosphere(
        unwrapped_phase=unwrapped_phase,
        ps_mask=ps_candidates,
        elevation_data=None,  # Load from DEM if needed
        config=atm_config
    )
    
    # Apply atmospheric correction
    corrected_phase = unwrapped_phase - atm_screen
    
    atmospheric_screens[pair_name] = atm_screen
    corrected_phases[pair_name] = corrected_phase

print("Atmospheric correction completed")

# Visualize atmospheric correction
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

example_pair = list(unwrapped_phases.keys())[0]

# Original unwrapped phase
axes[0, 0].imshow(unwrapped_phases[example_pair], cmap='jet')
axes[0, 0].set_title('Original Unwrapped Phase')
axes[0, 0].axis('off')

# Atmospheric screen
axes[0, 1].imshow(atmospheric_screens[example_pair], cmap='jet')
axes[0, 1].set_title('Atmospheric Phase Screen')
axes[0, 1].axis('off')

# Corrected phase
axes[0, 2].imshow(corrected_phases[example_pair], cmap='jet')
axes[0, 2].set_title('Atmospherically Corrected Phase')
axes[0, 2].axis('off')

# Phase histograms
phases_orig = unwrapped_phases[example_pair][ps_candidates].flatten()
phases_atm = atmospheric_screens[example_pair][ps_candidates].flatten()
phases_corr = corrected_phases[example_pair][ps_candidates].flatten()

axes[1, 0].hist(phases_orig, bins=50, alpha=0.7, density=True)
axes[1, 0].set_title('Original Phase Distribution')
axes[1, 0].set_xlabel('Phase (radians)')

axes[1, 1].hist(phases_atm, bins=50, alpha=0.7, density=True, color='orange')
axes[1, 1].set_title('Atmospheric Phase Distribution')
axes[1, 1].set_xlabel('Phase (radians)')

axes[1, 2].hist(phases_corr, bins=50, alpha=0.7, density=True, color='green')
axes[1, 2].set_title('Corrected Phase Distribution')
axes[1, 2].set_xlabel('Phase (radians)')

# Phase standard deviation comparison
std_orig = np.std(phases_orig)
std_corr = np.std(phases_corr)

axes[2, 0].bar(['Original', 'Corrected'], [std_orig, std_corr], color=['blue', 'green'])
axes[2, 0].set_ylabel('Phase Standard Deviation')
axes[2, 0].set_title('Phase Noise Reduction')

# Improvement factor
improvement = std_orig / std_corr
axes[2, 1].text(0.5, 0.5, f'Improvement Factor:\n{improvement:.2f}x', 
                horizontalalignment='center', verticalalignment='center',
                transform=axes[2, 1].transAxes, fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgreen'))
axes[2, 1].axis('off')

# Remove empty subplot
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()
```

### 4.2 Updated Time Series Analysis

```python
# Recalculate displacement time series with atmospheric correction
corrected_displacement_matrix = np.zeros_like(displacement_matrix)
corrected_displacement_matrix[:, 0] = 0  # Master reference

for i, (pair_name, corrected_phase) in enumerate(corrected_phases.items()):
    corrected_phase_values = corrected_phase[ps_indices]
    # Reference to first PS point
    corrected_phase_ref = corrected_phase_values - corrected_phase_values[reference_ps_idx]
    corrected_displacement_matrix[:, i + 1] = corrected_phase_ref * wavelength / (4 * np.pi)

# Compare before and after atmospheric correction
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Select representative PS points
selected_ps = [0, num_ps//4, num_ps//2, num_ps*3//4, num_ps-1, num_ps//8]

for i, ps_idx in enumerate(selected_ps[:6]):
    row, col = i // 3, i % 3
    
    # Original displacement
    axes[row, col].plot(time_axis, displacement_matrix[ps_idx, :] * 1000, 
                       'o-', label='Original', alpha=0.7, linewidth=2)
    
    # Corrected displacement
    axes[row, col].plot(time_axis, corrected_displacement_matrix[ps_idx, :] * 1000, 
                       's-', label='Atm. Corrected', alpha=0.7, linewidth=2)
    
    axes[row, col].set_ylabel('LOS Displacement (mm)')
    axes[row, col].set_title(f'PS Point {ps_idx+1}')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Step 5: Advanced Time Series Analysis

### 5.1 Linear Velocity Estimation

```python
# Estimate linear velocity for each PS point
print("Estimating linear velocities...")

def estimate_linear_velocity(displacement_ts, time_axis):
    """Estimate linear velocity using least squares"""
    # Convert time to days from first acquisition
    time_days = np.array([(t - time_axis[0]).days for t in time_axis])
    
    # Remove NaN values
    valid_idx = ~np.isnan(displacement_ts)
    if np.sum(valid_idx) < 3:
        return np.nan, np.nan, np.nan
    
    time_valid = time_days[valid_idx]
    disp_valid = displacement_ts[valid_idx]
    
    # Linear regression
    A = np.vstack([time_valid, np.ones(len(time_valid))]).T
    velocity, intercept = np.linalg.lstsq(A, disp_valid, rcond=None)[0]
    
    # Calculate R-squared
    disp_pred = velocity * time_valid + intercept
    ss_res = np.sum((disp_valid - disp_pred) ** 2)
    ss_tot = np.sum((disp_valid - np.mean(disp_valid)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return velocity, intercept, r_squared

# Calculate velocities for all PS points
velocities = []
r_squared_values = []

for i in range(num_ps):
    velocity, intercept, r_squared = estimate_linear_velocity(
        corrected_displacement_matrix[i, :], time_axis
    )
    velocities.append(velocity)
    r_squared_values.append(r_squared)

velocities = np.array(velocities)
r_squared_values = np.array(r_squared_values)

# Convert from m/day to mm/year
velocities_mm_year = velocities * 365.25 * 1000

print(f"Velocity statistics:")
print(f"Mean velocity: {np.nanmean(velocities_mm_year):.2f} mm/year")
print(f"Std velocity: {np.nanstd(velocities_mm_year):.2f} mm/year")
print(f"Min velocity: {np.nanmin(velocities_mm_year):.2f} mm/year")
print(f"Max velocity: {np.nanmax(velocities_mm_year):.2f} mm/year")
```

### 5.2 Velocity Map Generation

```python
# Create velocity map
velocity_map = np.full(master_data.shape, np.nan)
velocity_map[ps_indices] = velocities_mm_year

# Create quality map
quality_map = np.full(master_data.shape, np.nan)
quality_map[ps_indices] = r_squared_values

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Velocity map
velocity_range = np.nanpercentile(velocities_mm_year, [5, 95])
im1 = axes[0, 0].imshow(velocity_map, cmap='RdBu_r', 
                       vmin=velocity_range[0], vmax=velocity_range[1])
axes[0, 0].set_title('Linear Velocity Map (mm/year)')
cbar1 = plt.colorbar(im1, ax=axes[0, 0])
cbar1.set_label('LOS Velocity (mm/year)')

# Quality map (R-squared)
im2 = axes[0, 1].imshow(quality_map, cmap='viridis', vmin=0, vmax=1)
axes[0, 1].set_title('Velocity Estimation Quality (R²)')
cbar2 = plt.colorbar(im2, ax=axes[0, 1])

# Velocity histogram
valid_velocities = velocities_mm_year[~np.isnan(velocities_mm_year)]
axes[1, 0].hist(valid_velocities, bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Velocity (mm/year)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Velocity Distribution')
axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)

# Velocity vs Quality scatter plot
valid_idx = ~(np.isnan(velocities_mm_year) | np.isnan(r_squared_values))
axes[1, 1].scatter(r_squared_values[valid_idx], velocities_mm_year[valid_idx], 
                  alpha=0.6, s=20)
axes[1, 1].set_xlabel('R-squared')
axes[1, 1].set_ylabel('Velocity (mm/year)')
axes[1, 1].set_title('Velocity vs Quality')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 6: Small Baseline Subset (SBAS) Analysis

### 6.1 Network Design and Processing

```python
# SBAS network design
class SBASProcessor:
    """Small Baseline Subset InSAR processor"""
    
    def __init__(self, max_temporal_baseline=90, max_spatial_baseline=200):
        self.max_temp_baseline = max_temporal_baseline  # days
        self.max_spat_baseline = max_spatial_baseline   # meters
        self.interferogram_network = []
        
    def design_network(self, acquisition_dates, spatial_baselines=None):
        """Design SBAS interferogram network"""
        n_images = len(acquisition_dates)
        
        # Generate all possible pairs
        for i in range(n_images):
            for j in range(i + 1, n_images):
                temp_baseline = (acquisition_dates[j] - acquisition_dates[i]).days
                
                # Check temporal baseline
                if temp_baseline <= self.max_temp_baseline:
                    # Check spatial baseline if provided
                    if spatial_baselines is not None:
                        spat_baseline = abs(spatial_baselines[j] - spatial_baselines[i])
                        if spat_baseline > self.max_spat_baseline:
                            continue
                    
                    self.interferogram_network.append({
                        'master_idx': i,
                        'slave_idx': j,
                        'temporal_baseline': temp_baseline,
                        'spatial_baseline': spat_baseline if spatial_baselines else None
                    })
        
        print(f"SBAS network designed with {len(self.interferogram_network)} interferograms")
        return self.interferogram_network
    
    def solve_displacement_timeseries(self, interferogram_phases, ps_indices):
        """Solve for displacement time series using SBAS approach"""
        n_ps = len(ps_indices[0])
        n_images = len(set([pair['master_idx'] for pair in self.interferogram_network] + 
                          [pair['slave_idx'] for pair in self.interferogram_network])) + 1
        n_ifgs = len(self.interferogram_network)
        
        # Build design matrix A
        A = np.zeros((n_ifgs, n_images))
        for i, pair in enumerate(self.interferogram_network):
            A[i, pair['master_idx']] = -1
            A[i, pair['slave_idx']] = 1
        
        # Add temporal smoothness constraint
        L = np.eye(n_images - 1, n_images) - np.eye(n_images - 1, n_images, k=1)
        alpha = 0.01  # Smoothing parameter
        
        # Solve for each PS point
        displacement_timeseries = np.zeros((n_ps, n_images))
        
        for ps_idx in range(n_ps):
            # Extract interferogram phases for this PS point
            y, x = ps_indices[0][ps_idx], ps_indices[1][ps_idx]
            
            ifg_phases = []
            for i, pair in enumerate(self.interferogram_network):
                pair_key = f"master_{sar_stack['acquisition_dates'][pair['slave_idx']].strftime('%Y%m%d')}"
                if pair_key in interferogram_phases:
                    ifg_phases.append(interferogram_phases[pair_key][y, x])
            
            if len(ifg_phases) != n_ifgs:
                continue
                
            ifg_phases = np.array(ifg_phases)
            
            # Regularized least squares solution
            ATA = A.T @ A + alpha * (L.T @ L)
            ATb = A.T @ ifg_phases
            
            try:
                displacement_ts = np.linalg.solve(ATA, ATb)
                displacement_timeseries[ps_idx, :] = displacement_ts
            except np.linalg.LinAlgError:
                displacement_timeseries[ps_idx, :] = np.nan
        
        return displacement_timeseries

# Initialize SBAS processor
sbas_processor = SBASProcessor(max_temporal_baseline=48, max_spatial_baseline=150)

# Design network
sbas_network = sbas_processor.design_network(sar_stack['acquisition_dates'])

# Solve for displacement time series
print("Solving SBAS displacement time series...")
sbas_displacement = sbas_processor.solve_displacement_timeseries(
    corrected_phases, ps_indices
)

# Convert phase to displacement
sbas_displacement_m = sbas_displacement * wavelength / (4 * np.pi)

print(f"SBAS time series shape: {sbas_displacement_m.shape}")
```

### 6.2 SBAS Results Visualization

```python
# Compare PSI and SBAS results
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Time series comparison for selected PS points
selected_points = [0, num_ps//4, num_ps//2]

for i, ps_idx in enumerate(selected_points):
    # PSI results
    axes[i, 0].plot(time_axis, corrected_displacement_matrix[ps_idx, :] * 1000, 
                   'o-', label='PSI', linewidth=2, markersize=6)
    
    # SBAS results
    axes[i, 0].plot(sar_stack['acquisition_dates'], sbas_displacement_m[ps_idx, :] * 1000, 
                   's-', label='SBAS', linewidth=2, markersize=6, alpha=0.8)
    
    axes[i, 0].set_ylabel('LOS Displacement (mm)')
    axes[i, 0].set_title(f'PS Point {ps_idx+1}: PSI vs SBAS')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].tick_params(axis='x', rotation=45)
    
    # Difference plot
    # Interpolate SBAS to PSI time points for comparison
    from scipy.interpolate import interp1d
    
    sbas_dates_num = np.array([t.timestamp() for t in sar_stack['acquisition_dates']])
    psi_dates_num = np.array([t.timestamp() for t in time_axis])
    
    if not np.any(np.isnan(sbas_displacement_m[ps_idx, :])):
        f_interp = interp1d(sbas_dates_num, sbas_displacement_m[ps_idx, :], 
                           kind='linear', fill_value='extrapolate')
        sbas_interp = f_interp(psi_dates_num)
        
        difference = (corrected_displacement_matrix[ps_idx, :] - sbas_interp) * 1000
        axes[i, 1].plot(time_axis, difference, 'o-', color='red', linewidth=2)
        axes[i, 1].set_ylabel('PSI - SBAS (mm)')
        axes[i, 1].set_title(f'Difference: PS Point {ps_idx+1}')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Network visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Temporal-spatial baseline plot
temp_baselines = [pair['temporal_baseline'] for pair in sbas_network]
spat_baselines = [pair['spatial_baseline'] for pair in sbas_network if pair['spatial_baseline']]

if spat_baselines:
    axes[0].scatter(temp_baselines[:len(spat_baselines)], spat_baselines, alpha=0.6)
    axes[0].set_xlabel('Temporal Baseline (days)')
    axes[0].set_ylabel('Spatial Baseline (m)')
    axes[0].set_title('SBAS Network: Baseline Distribution')
    axes[0].grid(True, alpha=0.3)

# Time series connectivity
time_indices = list(range(len(sar_stack['acquisition_dates'])))
axes[1].scatter(time_indices, sar_stack['acquisition_dates'], s=100, c='red', zorder=5)

for pair in sbas_network:
    x_coords = [pair['master_idx'], pair['slave_idx']]
    y_coords = [sar_stack['acquisition_dates'][pair['master_idx']], 
                sar_stack['acquisition_dates'][pair['slave_idx']]]
    axes[1].plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=1)

axes[1].set_xlabel('Image Index')
axes[1].set_ylabel('Acquisition Date')
axes[1].set_title('SBAS Network Connectivity')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 7: Advanced Applications

### 7.1 Nonlinear Deformation Detection

```python
# Detect nonlinear deformation patterns
def detect_nonlinear_deformation(displacement_ts, time_axis, significance_level=0.05):
    """Detect significant nonlinear deformation patterns"""
    from scipy import stats
    
    # Convert time to numerical format
    time_numeric = np.array([(t - time_axis[0]).days for t in time_axis])
    
    # Remove NaN values
    valid_idx = ~np.isnan(displacement_ts)
    if np.sum(valid_idx) < 4:
        return False, np.nan, np.nan
    
    time_valid = time_numeric[valid_idx]
    disp_valid = displacement_ts[valid_idx]
    
    # Linear fit
    linear_coef = np.polyfit(time_valid, disp_valid, 1)
    linear_pred = np.polyval(linear_coef, time_valid)
    linear_residuals = disp_valid - linear_pred
    
    # Quadratic fit
    quad_coef = np.polyfit(time_valid, disp_valid, 2)
    quad_pred = np.polyval(quad_coef, time_valid)
    quad_residuals = disp_valid - quad_pred
    
    # F-test for model comparison
    rss_linear = np.sum(linear_residuals**2)
    rss_quad = np.sum(quad_residuals**2)
    
    n = len(time_valid)
    f_stat = ((rss_linear - rss_quad) / 1) / (rss_quad / (n - 3))
    p_value = 1 - stats.f.cdf(f_stat, 1, n - 3)
    
    is_nonlinear = p_value < significance_level
    acceleration = quad_coef[0] * 2 * 365.25 * 1000  # mm/year²
    
    return is_nonlinear, acceleration, p_value

# Apply nonlinear detection to all PS points
nonlinear_flags = []
accelerations = []
p_values = []

print("Detecting nonlinear deformation patterns...")

for i in range(num_ps):
    is_nonlinear, acceleration, p_value = detect_nonlinear_deformation(
        sbas_displacement_m[i, :], sar_stack['acquisition_dates']
    )
    nonlinear_flags.append(is_nonlinear)
    accelerations.append(acceleration)
    p_values.append(p_value)

nonlinear_flags = np.array(nonlinear_flags)
accelerations = np.array(accelerations)
p_values = np.array(p_values)

print(f"Nonlinear points detected: {np.sum(nonlinear_flags)} / {num_ps}")
print(f"Percentage of nonlinear points: {np.sum(nonlinear_flags) / num_ps * 100:.1f}%")

# Visualize nonlinear deformation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Acceleration map
acceleration_map = np.full(master_data.shape, np.nan)
acceleration_map[ps_indices] = accelerations

acc_range = np.nanpercentile(accelerations[~np.isnan(accelerations)], [5, 95])
im1 = axes[0, 0].imshow(acceleration_map, cmap='RdBu_r', 
                       vmin=acc_range[0], vmax=acc_range[1])
axes[0, 0].set_title('Deformation Acceleration (mm/year²)')
plt.colorbar(im1, ax=axes[0, 0])

# Nonlinear points overlay
nonlinear_map = np.full(master_data.shape, np.nan)
nonlinear_map[ps_indices] = nonlinear_flags.astype(float)

axes[0, 1].imshow(np.abs(master_data), cmap='gray', alpha=0.7)
nonlinear_overlay = np.ma.masked_where(np.isnan(nonlinear_map), nonlinear_map)
axes[0, 1].imshow(nonlinear_overlay, cmap='Reds', alpha=0.8)
axes[0, 1].set_title('Nonlinear Deformation Points')

# P-value histogram
valid_p = p_values[~np.isnan(p_values)]
axes[1, 0].hist(valid_p, bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
axes[1, 0].set_xlabel('P-value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Nonlinearity Test P-values')
axes[1, 0].legend()

# Acceleration histogram
valid_acc = accelerations[~np.isnan(accelerations)]
axes[1, 1].hist(valid_acc, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Acceleration (mm/year²)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Acceleration Distribution')
axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

### 7.2 Export Results and Create Reports

```python
# Export results for further analysis
import json
from datetime import datetime

def export_insar_results(output_dir='insar_results'):
    """Export InSAR analysis results"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Export PS coordinates and results
    ps_results = {
        'metadata': {
            'processing_date': datetime.now().isoformat(),
            'num_ps_points': int(num_ps),
            'num_images': len(sar_stack['acquisition_dates']),
            'wavelength_m': wavelength,
            'reference_ps_index': int(reference_ps_idx)
        },
        'ps_coordinates': {
            'row': ps_indices[0].tolist(),
            'col': ps_indices[1].tolist()
        },
        'velocities': {
            'values_mm_year': velocities_mm_year.tolist(),
            'quality_r_squared': r_squared_values.tolist()
        },
        'nonlinear_analysis': {
            'is_nonlinear': nonlinear_flags.tolist(),
            'acceleration_mm_year2': accelerations.tolist(),
            'p_values': p_values.tolist()
        },
        'acquisition_dates': [date.isoformat() for date in sar_stack['acquisition_dates']]
    }
    
    # Save as JSON
    with open(f'{output_dir}/ps_analysis_results.json', 'w') as f:
        json.dump(ps_results, f, indent=2)
    
    # Save displacement time series
    np.save(f'{output_dir}/psi_displacement_timeseries.npy', corrected_displacement_matrix)
    np.save(f'{output_dir}/sbas_displacement_timeseries.npy', sbas_displacement_m)
    
    # Save velocity and quality maps
    np.save(f'{output_dir}/velocity_map_mm_year.npy', velocity_map)
    np.save(f'{output_dir}/quality_map_r_squared.npy', quality_map)
    np.save(f'{output_dir}/acceleration_map_mm_year2.npy', acceleration_map)
    
    print(f"Results exported to {output_dir}/")
    
    return output_dir

# Generate processing report
def generate_processing_report():
    """Generate a comprehensive processing report"""
    
    report = f"""
# InSAR Processing Report

## Processing Summary
- **Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Number of SAR Images**: {len(sar_stack['acquisition_dates'])}
- **Time Span**: {sar_stack['acquisition_dates'][0].strftime('%Y-%m-%d')} to {sar_stack['acquisition_dates'][-1].strftime('%Y-%m-%d')}
- **Total Interferograms**: {len(interferograms)}
- **SBAS Network Pairs**: {len(sbas_network)}

## Persistent Scatterer Analysis
- **PS Candidates Identified**: {num_ps:,}
- **PS Density**: {np.sum(ps_candidates) / ps_candidates.size * 100:.2f}%
- **Selection Criteria**:
  - ADI Threshold: {ps_criteria['adi_threshold']}
  - Coherence Threshold: {ps_criteria['coherence_threshold']}

## Velocity Analysis
- **Mean LOS Velocity**: {np.nanmean(velocities_mm_year):.2f} ± {np.nanstd(velocities_mm_year):.2f} mm/year
- **Velocity Range**: {np.nanmin(velocities_mm_year):.2f} to {np.nanmax(velocities_mm_year):.2f} mm/year
- **Stable Points** (|v| < 2 mm/year): {np.sum(np.abs(velocities_mm_year) < 2)}/{num_ps} ({np.sum(np.abs(velocities_mm_year) < 2)/num_ps*100:.1f}%)

## Nonlinear Deformation Analysis
- **Nonlinear Points Detected**: {np.sum(nonlinear_flags)}/{num_ps} ({np.sum(nonlinear_flags)/num_ps*100:.1f}%)
- **Acceleration Range**: {np.nanmin(accelerations):.3f} to {np.nanmax(accelerations):.3f} mm/year²

## Quality Assessment
- **Mean Coherence**: {np.nanmean(mean_coherence):.3f}
- **Mean Velocity R²**: {np.nanmean(r_squared_values):.3f}
- **Atmospheric Correction Applied**: Yes

## Processing Configuration
- **Wavelength**: {wavelength} m
- **Reference PS Index**: {reference_ps_idx}
- **Atmospheric Correction**: Linear regression with elevation dependence
- **Phase Unwrapping**: Minimum cost flow algorithm
"""
    
    return report

# Export results and generate report
export_dir = export_insar_results()
processing_report = generate_processing_report()

print(processing_report)

# Save report
with open(f'{export_dir}/processing_report.md', 'w') as f:
    f.write(processing_report)

print(f"\nProcessing complete! Results saved to {export_dir}/")
```

## Key Takeaways

1. **Comprehensive Workflow**: This tutorial covers the complete InSAR processing chain from coregistration to advanced time series analysis

2. **Multiple Techniques**: Both PSI and SBAS approaches are demonstrated, showing their complementary strengths

3. **Quality Control**: Extensive quality assessment and validation procedures ensure reliable results

4. **Advanced Analysis**: Nonlinear deformation detection and atmospheric correction improve measurement accuracy

5. **Operational Ready**: The workflow is designed for operational use with batch processing and automated reporting

## Next Steps

1. **Validate results** with independent ground truth data (GPS, leveling)
2. **Optimize parameters** for your specific study area and application
3. **Integrate with GIS systems** for spatial analysis and interpretation
4. **Develop automated monitoring systems** for continuous deformation monitoring

## Additional Resources

- InSAR theory and processing principles
- Atmospheric correction techniques
- Phase unwrapping algorithms
- Time series analysis methods for Earth observation

---

*This tutorial provides a comprehensive framework for advanced interferometric analysis using SARPYX. The techniques demonstrated here enable precise deformation monitoring and surface change detection for a wide range of applications including volcano monitoring, landslide analysis, subsidence studies, and infrastructure monitoring.*
