# Tutorial 4: Multi-temporal Analysis

Learn to process time series of SAR data for change detection and temporal pattern analysis using SARPyX.

## Overview

This tutorial covers:
- Loading and organizing multi-temporal SAR datasets
- Temporal SLA processing workflows
- Change detection techniques
- Time series analysis and trend detection
- Multi-temporal coherence analysis
- Automated processing pipelines

**Duration**: 30 minutes  
**Prerequisites**: Tutorials 1-3 completed  
**Data**: Multi-temporal Sentinel-1 dataset (3+ acquisitions)

## 1. Dataset Preparation and Organization

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sarpyx.sla import SubLookAnalysis
from sarpyx.utils.viz import show_image, show_histogram
from sarpyx.utils.io import save_matlab_mat
import glob
import os

# Set up directories
data_dir = "path/to/multitemporal/data"
output_dir = "tutorial4_outputs"
os.makedirs(output_dir, exist_ok=True)

# Define temporal dataset
temporal_data = {
    'acquisition_1': {
        'path': f"{data_dir}/S1A_20230101_*.dim",
        'date': '2023-01-01',
        'orbit': 'ascending'
    },
    'acquisition_2': {
        'path': f"{data_dir}/S1A_20230201_*.dim", 
        'date': '2023-02-01',
        'orbit': 'ascending'
    },
    'acquisition_3': {
        'path': f"{data_dir}/S1A_20230301_*.dim",
        'date': '2023-03-01', 
        'orbit': 'ascending'
    },
    'acquisition_4': {
        'path': f"{data_dir}/S1A_20230401_*.dim",
        'date': '2023-04-01',
        'orbit': 'ascending'
    }
}

print("Multi-temporal dataset configuration:")
for acq_id, info in temporal_data.items():
    print(f"  {acq_id}: {info['date']} ({info['orbit']})")
```

## 2. Multi-temporal SLA Processing

### 2.1 Batch Processing Function

```python
def process_temporal_sla(data_info, band_name='Sigma0_VV_slc', 
                        num_looks=4, overlap_factor=0.5):
    """Process multiple acquisitions with consistent SLA parameters"""
    
    processed_data = {}
    
    for acq_id, info in data_info.items():
        print(f"Processing {acq_id} ({info['date']})...")
        
        # Find actual file path
        file_paths = glob.glob(info['path'])
        if not file_paths:
            print(f"  Warning: No files found for {acq_id}")
            continue
        
        file_path = file_paths[0]  # Take first match
        
        try:
            # Initialize SLA processor
            sla = SubLookAnalysis()
            
            # Load and process data
            sla.load_data(file_path, band_name=band_name)
            sla.set_sublook_parameters(num_looks=num_looks, 
                                     overlap_factor=overlap_factor)
            sla.decompose()
            
            # Store results
            processed_data[acq_id] = {
                'sla_processor': sla,
                'sublooks': sla.get_sublooks(),
                'master_image': sla.get_master_image(),
                'coherence': sla.get_coherence_matrix(),
                'date': info['date'],
                'orbit': info['orbit'],
                'file_path': file_path
            }
            
            print(f"  Successfully processed {acq_id}")
            
        except Exception as e:
            print(f"  Error processing {acq_id}: {str(e)}")
            continue
    
    return processed_data

# Process all temporal acquisitions
print("Starting multi-temporal SLA processing...")
temporal_results = process_temporal_sla(temporal_data)

print(f"\nSuccessfully processed {len(temporal_results)} acquisitions")
```

### 2.2 Data Alignment and Registration

```python
def align_temporal_data(temporal_results, reference_key=None):
    """Align temporal data to common grid (basic implementation)"""
    
    if reference_key is None:
        reference_key = list(temporal_results.keys())[0]
    
    print(f"Using {reference_key} as reference for alignment")
    
    # Get reference dimensions
    ref_master = temporal_results[reference_key]['master_image']
    ref_shape = ref_master.shape
    
    aligned_data = {}
    
    for acq_id, data in temporal_results.items():
        master = data['master_image']
        
        # Simple alignment - crop/pad to reference size
        if master.shape == ref_shape:
            aligned_master = master
            aligned_sublooks = data['sublooks']
        else:
            # Crop to minimum common size (simplified approach)
            min_rows = min(master.shape[0], ref_shape[0])
            min_cols = min(master.shape[1], ref_shape[1])
            
            aligned_master = master[:min_rows, :min_cols]
            aligned_sublooks = [sl[:min_rows, :min_cols] for sl in data['sublooks']]
            
            print(f"  {acq_id}: resized from {master.shape} to {aligned_master.shape}")
        
        aligned_data[acq_id] = {
            **data,  # Copy all original data
            'aligned_master': aligned_master,
            'aligned_sublooks': aligned_sublooks
        }
    
    return aligned_data

# Align all temporal data
aligned_results = align_temporal_data(temporal_results)
print("Temporal data alignment completed")
```

## 3. Change Detection Analysis

### 3.1 Intensity-based Change Detection

```python
def detect_intensity_changes(aligned_data, threshold_db=3.0):
    """Detect changes based on intensity differences"""
    
    dates = sorted([data['date'] for data in aligned_data.values()])
    acq_ids = [k for k, v in aligned_data.items() if v['date'] in dates]
    
    change_results = {}
    
    # Calculate pairwise changes
    for i in range(len(acq_ids)-1):
        acq1_id = acq_ids[i]
        acq2_id = acq_ids[i+1]
        
        master1 = aligned_data[acq1_id]['aligned_master']
        master2 = aligned_data[acq2_id]['aligned_master']
        
        # Convert to dB
        master1_db = 10 * np.log10(np.abs(master1) + 1e-10)
        master2_db = 10 * np.log10(np.abs(master2) + 1e-10)
        
        # Calculate change
        intensity_change = master2_db - master1_db
        
        # Create change mask
        change_mask = np.abs(intensity_change) > threshold_db
        
        change_pair = f"{aligned_data[acq1_id]['date']}_to_{aligned_data[acq2_id]['date']}"
        
        change_results[change_pair] = {
            'intensity_change': intensity_change,
            'change_mask': change_mask,
            'change_percentage': np.sum(change_mask) / change_mask.size * 100,
            'mean_change': np.mean(intensity_change),
            'std_change': np.std(intensity_change),
            'acq1_date': aligned_data[acq1_id]['date'],
            'acq2_date': aligned_data[acq2_id]['date']
        }
        
        print(f"Change detection {change_pair}:")
        print(f"  Changed pixels: {change_results[change_pair]['change_percentage']:.2f}%")
        print(f"  Mean change: {change_results[change_pair]['mean_change']:.2f} dB")
    
    return change_results

# Perform change detection
print("Performing intensity-based change detection...")
change_results = detect_intensity_changes(aligned_results, threshold_db=2.5)
```

### 3.2 Visualize Change Detection Results

```python
# Visualize change detection results
num_changes = len(change_results)
fig, axes = plt.subplots(2, num_changes, figsize=(5*num_changes, 10))

if num_changes == 1:
    axes = axes.reshape(2, 1)

for i, (change_pair, results) in enumerate(change_results.items()):
    intensity_change = results['intensity_change']
    change_mask = results['change_mask']
    
    # Plot intensity change
    im1 = show_image(intensity_change,
                    title=f'Intensity Change (dB)\n{change_pair}',
                    ax=axes[0, i],
                    cmap='RdBu_r',
                    vmin=-10, vmax=10)
    plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
    
    # Plot change mask
    im2 = show_image(change_mask.astype(float),
                    title=f'Change Mask\n{results["change_percentage"]:.1f}% changed',
                    ax=axes[1, i],
                    cmap='hot')
    plt.colorbar(im2, ax=axes[1, i], shrink=0.8)

plt.tight_layout()
plt.savefig(f'{output_dir}/change_detection_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.3 Coherence-based Change Detection

```python
def detect_coherence_changes(aligned_data, coherence_threshold=0.3):
    """Detect changes based on coherence analysis"""
    
    coherence_results = {}
    acq_ids = list(aligned_data.keys())
    
    for i in range(len(acq_ids)-1):
        acq1_id = acq_ids[i]
        acq2_id = acq_ids[i+1]
        
        sublooks1 = aligned_data[acq1_id]['aligned_sublooks']
        sublooks2 = aligned_data[acq2_id]['aligned_sublooks']
        
        # Calculate inter-acquisition coherence
        inter_coherence = []
        for sl1, sl2 in zip(sublooks1[:4], sublooks2[:4]):  # Use first 4 sub-looks
            # Calculate normalized cross-correlation
            numerator = np.mean(sl1 * np.conj(sl2))
            denominator = np.sqrt(np.mean(np.abs(sl1)**2) * np.mean(np.abs(sl2)**2))
            coherence = np.abs(numerator / (denominator + 1e-10))
            inter_coherence.append(coherence)
        
        mean_inter_coherence = np.mean(inter_coherence)
        
        # Detect decorrelation (low coherence indicates change)
        change_indicator = mean_inter_coherence < coherence_threshold
        
        change_pair = f"{aligned_data[acq1_id]['date']}_to_{aligned_data[acq2_id]['date']}"
        
        coherence_results[change_pair] = {
            'inter_coherence': inter_coherence,
            'mean_coherence': mean_inter_coherence,
            'change_detected': change_indicator,
            'decorrelation_level': 1 - mean_inter_coherence
        }
        
        print(f"Coherence analysis {change_pair}:")
        print(f"  Mean inter-acquisition coherence: {mean_inter_coherence:.3f}")
        print(f"  Change detected: {change_indicator}")
    
    return coherence_results

# Perform coherence-based change detection
print("\nPerforming coherence-based change detection...")
coherence_changes = detect_coherence_changes(aligned_results)
```

## 4. Time Series Analysis

### 4.1 Temporal Intensity Evolution

```python
def analyze_temporal_evolution(aligned_data, sample_points=None):
    """Analyze temporal evolution at specific points or regions"""
    
    dates = sorted([data['date'] for data in aligned_data.values()])
    acq_ids = [k for k, v in aligned_data.items() if v['date'] in dates]
    
    # Convert dates to datetime objects
    datetime_dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    if sample_points is None:
        # Define default sample points (y, x coordinates)
        height, width = aligned_data[acq_ids[0]]['aligned_master'].shape
        sample_points = {
            'point_1': (height//4, width//4),
            'point_2': (height//2, width//2), 
            'point_3': (3*height//4, 3*width//4)
        }
    
    temporal_evolution = {}
    
    for point_name, (y, x) in sample_points.items():
        intensities = []
        intensities_db = []
        
        for acq_id in acq_ids:
            master = aligned_data[acq_id]['aligned_master']
            intensity = np.abs(master[y, x])
            intensity_db = 10 * np.log10(intensity + 1e-10)
            
            intensities.append(intensity)
            intensities_db.append(intensity_db)
        
        temporal_evolution[point_name] = {
            'coordinates': (y, x),
            'dates': dates,
            'datetime_dates': datetime_dates,
            'intensities': intensities,
            'intensities_db': intensities_db,
            'mean_intensity_db': np.mean(intensities_db),
            'std_intensity_db': np.std(intensities_db),
            'trend': np.polyfit(range(len(intensities_db)), intensities_db, 1)[0]  # Linear trend
        }
    
    return temporal_evolution

# Analyze temporal evolution
print("Analyzing temporal intensity evolution...")
temporal_evolution = analyze_temporal_evolution(aligned_results)

# Plot temporal evolution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time series plot
for point_name, data in temporal_evolution.items():
    axes[0, 0].plot(data['datetime_dates'], data['intensities_db'], 
                   'o-', label=f"{point_name} (trend: {data['trend']:.3f} dB/month)")

axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Intensity (dB)')
axes[0, 0].set_title('Temporal Intensity Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Statistical summary
point_names = list(temporal_evolution.keys())
means = [temporal_evolution[name]['mean_intensity_db'] for name in point_names]
stds = [temporal_evolution[name]['std_intensity_db'] for name in point_names]
trends = [temporal_evolution[name]['trend'] for name in point_names]

x_pos = np.arange(len(point_names))
axes[0, 1].bar(x_pos - 0.2, means, 0.4, label='Mean Intensity (dB)', alpha=0.7)
axes[0, 1].bar(x_pos + 0.2, stds, 0.4, label='Std Intensity (dB)', alpha=0.7)
axes[0, 1].set_xlabel('Sample Points')
axes[0, 1].set_ylabel('Intensity (dB)')
axes[0, 1].set_title('Statistical Summary')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(point_names)
axes[0, 1].legend()

# Trend analysis
axes[1, 0].bar(range(len(trends)), trends, alpha=0.7, 
              color=['green' if t > 0 else 'red' for t in trends])
axes[1, 0].set_xlabel('Sample Points')
axes[1, 0].set_ylabel('Trend (dB/month)')
axes[1, 0].set_title('Intensity Trends')
axes[1, 0].set_xticks(range(len(point_names)))
axes[1, 0].set_xticklabels(point_names)
axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)

# Show sample point locations
reference_master = aligned_results[list(aligned_results.keys())[0]]['aligned_master']
master_db = 10 * np.log10(np.abs(reference_master) + 1e-10)
im = axes[1, 1].imshow(master_db, cmap='gray',
                      vmin=np.percentile(master_db, 5),
                      vmax=np.percentile(master_db, 95))

colors = ['red', 'green', 'blue']
for i, (point_name, data) in enumerate(temporal_evolution.items()):
    y, x = data['coordinates']
    axes[1, 1].plot(x, y, 'o', color=colors[i], markersize=10, label=point_name)

axes[1, 1].set_title('Sample Point Locations')
axes[1, 1].legend()
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(f'{output_dir}/temporal_evolution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print temporal evolution summary
print("\nTemporal Evolution Summary:")
for point_name, data in temporal_evolution.items():
    print(f"{point_name}:")
    print(f"  Coordinates: {data['coordinates']}")
    print(f"  Mean intensity: {data['mean_intensity_db']:.2f} dB")
    print(f"  Intensity variability: {data['std_intensity_db']:.2f} dB")
    print(f"  Trend: {data['trend']:.3f} dB/month")
```

### 4.2 Regional Change Analysis

```python
def analyze_regional_changes(aligned_data, regions=None):
    """Analyze changes in specific regions over time"""
    
    if regions is None:
        # Define default regions
        height, width = aligned_data[list(aligned_data.keys())[0]]['aligned_master'].shape
        regions = {
            'northwest': [0, width//2, 0, height//2],
            'northeast': [width//2, width, 0, height//2],
            'southwest': [0, width//2, height//2, height],
            'southeast': [width//2, width, height//2, height]
        }
    
    dates = sorted([data['date'] for data in aligned_data.values()])
    acq_ids = [k for k, v in aligned_data.items() if v['date'] in dates]
    
    regional_stats = {}
    
    for region_name, coords in regions.items():
        x1, x2, y1, y2 = coords
        
        regional_data = {
            'dates': dates,
            'mean_intensities': [],
            'std_intensities': [],
            'area_size': (x2-x1) * (y2-y1)
        }
        
        for acq_id in acq_ids:
            master = aligned_data[acq_id]['aligned_master']
            region_data = master[y1:y2, x1:x2]
            
            mean_intensity = np.mean(np.abs(region_data))
            std_intensity = np.std(np.abs(region_data))
            
            regional_data['mean_intensities'].append(mean_intensity)
            regional_data['std_intensities'].append(std_intensity)
        
        # Convert to dB
        regional_data['mean_intensities_db'] = [10 * np.log10(i + 1e-10) 
                                              for i in regional_data['mean_intensities']]
        
        # Calculate regional statistics
        regional_data['overall_mean'] = np.mean(regional_data['mean_intensities_db'])
        regional_data['temporal_variability'] = np.std(regional_data['mean_intensities_db'])
        regional_data['trend'] = np.polyfit(range(len(regional_data['mean_intensities_db'])), 
                                          regional_data['mean_intensities_db'], 1)[0]
        
        regional_stats[region_name] = regional_data
    
    return regional_stats

# Perform regional analysis
print("Analyzing regional changes...")
regional_analysis = analyze_regional_changes(aligned_results)

# Visualize regional analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Regional time series
for region_name, data in regional_analysis.items():
    datetime_dates = [datetime.strptime(date, '%Y-%m-%d') for date in data['dates']]
    axes[0, 0].plot(datetime_dates, data['mean_intensities_db'], 'o-', label=region_name)

axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Mean Intensity (dB)')
axes[0, 0].set_title('Regional Intensity Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Regional variability comparison
region_names = list(regional_analysis.keys())
variabilities = [regional_analysis[name]['temporal_variability'] for name in region_names]
overall_means = [regional_analysis[name]['overall_mean'] for name in region_names]

axes[0, 1].scatter(overall_means, variabilities, s=100, alpha=0.7)
for i, name in enumerate(region_names):
    axes[0, 1].annotate(name, (overall_means[i], variabilities[i]), 
                       xytext=(5, 5), textcoords='offset points')

axes[0, 1].set_xlabel('Overall Mean Intensity (dB)')
axes[0, 1].set_ylabel('Temporal Variability (dB)')
axes[0, 1].set_title('Regional Characteristics')
axes[0, 1].grid(True, alpha=0.3)

# Regional trends
trends = [regional_analysis[name]['trend'] for name in region_names]
axes[1, 0].bar(range(len(trends)), trends, alpha=0.7,
              color=['green' if t > 0 else 'red' for t in trends])
axes[1, 0].set_xlabel('Regions')
axes[1, 0].set_ylabel('Trend (dB/month)')
axes[1, 0].set_title('Regional Intensity Trends')
axes[1, 0].set_xticks(range(len(region_names)))
axes[1, 0].set_xticklabels(region_names, rotation=45)
axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)

# Regional map
reference_master = aligned_results[list(aligned_results.keys())[0]]['aligned_master']
master_db = 10 * np.log10(np.abs(reference_master) + 1e-10)
im = axes[1, 1].imshow(master_db, cmap='gray',
                      vmin=np.percentile(master_db, 5),
                      vmax=np.percentile(master_db, 95))

# Overlay region boundaries
height, width = master_db.shape
regions_default = {
    'northwest': [0, width//2, 0, height//2],
    'northeast': [width//2, width, 0, height//2],
    'southwest': [0, width//2, height//2, height],
    'southeast': [width//2, width, height//2, height]
}

colors = ['red', 'green', 'blue', 'orange']
for i, (region_name, coords) in enumerate(regions_default.items()):
    x1, x2, y1, y2 = coords
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, color=colors[i], linewidth=2)
    axes[1, 1].add_patch(rect)
    axes[1, 1].text(x1+5, y1+10, region_name, color=colors[i], fontweight='bold')

axes[1, 1].set_title('Regional Boundaries')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(f'{output_dir}/regional_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print regional analysis summary
print("\nRegional Analysis Summary:")
for region_name, data in regional_analysis.items():
    print(f"{region_name}:")
    print(f"  Overall mean: {data['overall_mean']:.2f} dB")
    print(f"  Temporal variability: {data['temporal_variability']:.2f} dB")
    print(f"  Trend: {data['trend']:.3f} dB/month")
```

## 5. Automated Processing Pipeline

### 5.1 Complete Multi-temporal Workflow

```python
class MultiTemporalSLA:
    """Automated multi-temporal SLA processing pipeline"""
    
    def __init__(self, output_dir, band_name='Sigma0_VV_slc'):
        self.output_dir = output_dir
        self.band_name = band_name
        self.temporal_data = {}
        self.processed_results = {}
        self.analysis_results = {}
        
    def add_acquisition(self, acq_id, file_path, date, orbit='ascending'):
        """Add acquisition to processing queue"""
        self.temporal_data[acq_id] = {
            'path': file_path,
            'date': date,
            'orbit': orbit
        }
        
    def process_all_acquisitions(self, num_looks=4, overlap_factor=0.5):
        """Process all acquisitions with consistent parameters"""
        print("Processing all acquisitions...")
        self.processed_results = process_temporal_sla(
            self.temporal_data, self.band_name, num_looks, overlap_factor
        )
        
        # Align data
        self.aligned_results = align_temporal_data(self.processed_results)
        
    def analyze_changes(self, intensity_threshold=2.5, coherence_threshold=0.3):
        """Perform comprehensive change analysis"""
        print("Analyzing temporal changes...")
        
        # Intensity-based change detection
        self.analysis_results['intensity_changes'] = detect_intensity_changes(
            self.aligned_results, intensity_threshold
        )
        
        # Coherence-based change detection
        self.analysis_results['coherence_changes'] = detect_coherence_changes(
            self.aligned_results, coherence_threshold
        )
        
        # Temporal evolution analysis
        self.analysis_results['temporal_evolution'] = analyze_temporal_evolution(
            self.aligned_results
        )
        
        # Regional analysis
        self.analysis_results['regional_analysis'] = analyze_regional_changes(
            self.aligned_results
        )
        
    def generate_report(self):
        """Generate comprehensive multi-temporal analysis report"""
        
        report = {
            'processing_info': {
                'num_acquisitions': len(self.processed_results),
                'date_range': self._get_date_range(),
                'band_processed': self.band_name
            },
            'change_detection': {},
            'temporal_trends': {},
            'summary_statistics': {}
        }
        
        # Change detection summary
        if 'intensity_changes' in self.analysis_results:
            intensity_changes = self.analysis_results['intensity_changes']
            report['change_detection']['intensity_based'] = {
                'num_change_pairs': len(intensity_changes),
                'mean_change_percentage': np.mean([r['change_percentage'] 
                                                 for r in intensity_changes.values()]),
                'max_change_percentage': np.max([r['change_percentage'] 
                                               for r in intensity_changes.values()])
            }
        
        # Temporal trends summary
        if 'temporal_evolution' in self.analysis_results:
            evolution = self.analysis_results['temporal_evolution']
            trends = [data['trend'] for data in evolution.values()]
            report['temporal_trends'] = {
                'mean_trend': np.mean(trends),
                'trend_variability': np.std(trends),
                'increasing_points': sum(1 for t in trends if t > 0),
                'decreasing_points': sum(1 for t in trends if t < 0)
            }
        
        # Save report
        import json
        with open(f"{self.output_dir}/multitemporal_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _get_date_range(self):
        """Get date range of processed acquisitions"""
        dates = [data['date'] for data in self.processed_results.values()]
        return {'start': min(dates), 'end': max(dates)}
    
    def save_results(self):
        """Save all processing results"""
        print("Saving results...")
        
        # Save aligned master images
        for acq_id, data in self.aligned_results.items():
            np.save(f"{self.output_dir}/master_{acq_id}.npy", 
                   data['aligned_master'])
        
        # Save change detection results
        if 'intensity_changes' in self.analysis_results:
            for change_pair, results in self.analysis_results['intensity_changes'].items():
                np.save(f"{self.output_dir}/change_{change_pair}.npy",
                       results['intensity_change'])
                np.save(f"{self.output_dir}/mask_{change_pair}.npy",
                       results['change_mask'])
        
        print(f"Results saved to {self.output_dir}")

# Example usage of automated pipeline
print("\n=== AUTOMATED MULTI-TEMPORAL PROCESSING ===")

# Initialize pipeline
mt_pipeline = MultiTemporalSLA(output_dir)

# Add acquisitions (update paths as needed)
for acq_id, info in temporal_data.items():
    mt_pipeline.add_acquisition(acq_id, info['path'], info['date'], info['orbit'])

# Process all acquisitions
mt_pipeline.process_all_acquisitions(num_looks=4, overlap_factor=0.5)

# Analyze changes
mt_pipeline.analyze_changes(intensity_threshold=2.5, coherence_threshold=0.3)

# Generate and display report
report = mt_pipeline.generate_report()
print("\nMulti-temporal Processing Report:")
print(f"Processed acquisitions: {report['processing_info']['num_acquisitions']}")
print(f"Date range: {report['processing_info']['date_range']['start']} to {report['processing_info']['date_range']['end']}")

if 'intensity_based' in report['change_detection']:
    intensity_stats = report['change_detection']['intensity_based']
    print(f"Mean change percentage: {intensity_stats['mean_change_percentage']:.2f}%")
    print(f"Maximum change percentage: {intensity_stats['max_change_percentage']:.2f}%")

if 'temporal_trends' in report:
    trend_stats = report['temporal_trends']
    print(f"Mean temporal trend: {trend_stats['mean_trend']:.3f} dB/month")
    print(f"Points with increasing trend: {trend_stats['increasing_points']}")
    print(f"Points with decreasing trend: {trend_stats['decreasing_points']}")

# Save all results
mt_pipeline.save_results()
```

## 6. Export and Visualization Summary

```python
# Create final summary visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# 1. Temporal data overview
acq_ids = list(aligned_results.keys())
dates = [aligned_results[acq_id]['date'] for acq_id in acq_ids]
datetime_dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

for i, acq_id in enumerate(acq_ids):
    master = aligned_results[acq_id]['aligned_master']
    master_db = 10 * np.log10(np.abs(master) + 1e-10)
    
    # Show subset for clarity
    subset = master_db[::4, ::4]  # Downsample for overview
    
    if i < 2:  # Show first two acquisitions
        im = axes[0, i].imshow(subset, cmap='gray',
                              vmin=np.percentile(subset, 5),
                              vmax=np.percentile(subset, 95))
        axes[0, i].set_title(f'Acquisition {i+1}\n{dates[i]}')
        plt.colorbar(im, ax=axes[0, i])

# 2. Change detection summary
if change_results:
    change_pair = list(change_results.keys())[0]
    change_data = change_results[change_pair]
    
    # Show change map
    subset_change = change_data['intensity_change'][::4, ::4]
    im1 = axes[1, 0].imshow(subset_change, cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1, 0].set_title(f'Intensity Change\n{change_pair}')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Show change statistics
    change_percentages = [r['change_percentage'] for r in change_results.values()]
    change_pairs = list(change_results.keys())
    
    axes[1, 1].bar(range(len(change_percentages)), change_percentages, alpha=0.7)
    axes[1, 1].set_xlabel('Time Period')
    axes[1, 1].set_ylabel('Change Percentage (%)')
    axes[1, 1].set_title('Change Detection Summary')
    axes[1, 1].set_xticks(range(len(change_pairs)))
    axes[1, 1].set_xticklabels([cp.split('_to_')[0][-5:] + ' to ' + cp.split('_to_')[1][-5:] 
                               for cp in change_pairs], rotation=45)

# 3. Temporal evolution summary
if temporal_evolution:
    # Plot all point evolutions
    for point_name, data in temporal_evolution.items():
        axes[2, 0].plot(datetime_dates, data['intensities_db'], 'o-', label=point_name)
    
    axes[2, 0].set_xlabel('Date')
    axes[2, 0].set_ylabel('Intensity (dB)')
    axes[2, 0].set_title('Temporal Evolution Summary')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].tick_params(axis='x', rotation=45)
    
    # Summary statistics
    trends = [data['trend'] for data in temporal_evolution.values()]
    point_names = list(temporal_evolution.keys())
    
    axes[2, 1].bar(range(len(trends)), trends, alpha=0.7,
                  color=['green' if t > 0 else 'red' for t in trends])
    axes[2, 1].set_xlabel('Sample Points')
    axes[2, 1].set_ylabel('Trend (dB/month)')
    axes[2, 1].set_title('Temporal Trends')
    axes[2, 1].set_xticks(range(len(point_names)))
    axes[2, 1].set_xticklabels(point_names)
    axes[2, 1].axhline(0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{output_dir}/multitemporal_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nMulti-temporal analysis complete!")
print(f"Output files saved to: {output_dir}/")
print("Generated files:")
print("- change_detection_results.png")
print("- temporal_evolution_analysis.png") 
print("- regional_analysis.png")
print("- multitemporal_summary.png")
print("- multitemporal_report.json")
print("- master_*.npy (aligned master images)")
print("- change_*.npy (change maps)")
print("- mask_*.npy (change masks)")
```

## Summary

In this tutorial, you learned:

1. **Multi-temporal dataset organization** and batch processing
2. **Temporal alignment** techniques for consistent analysis
3. **Change detection** using intensity and coherence methods
4. **Time series analysis** for trend detection
5. **Regional analysis** for spatial pattern assessment
6. **Automated processing pipelines** for efficient workflows

## Next Steps

- **Tutorial 5**: Polarimetric analysis with dual-pol data
- **Tutorial 6**: Custom processing workflows for specific applications
- Experiment with different change detection thresholds
- Apply to longer time series datasets
- Integrate with external change validation data

## Troubleshooting

**Common Issues:**

1. **Memory limitations**: Process smaller spatial subsets or reduce temporal sampling
2. **Registration errors**: Use more sophisticated co-registration techniques
3. **Temporal decorrelation**: Adjust coherence thresholds based on temporal baseline
4. **Inconsistent illumination**: Apply radiometric normalization
5. **Seasonal effects**: Consider phenological cycles in trend analysis

For more help, see the [Troubleshooting Guide](../user_guide/troubleshooting.md).
