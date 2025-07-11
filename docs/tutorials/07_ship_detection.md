# Tutorial 7: Ship Detection with CFAR Algorithms

## Overview

This tutorial demonstrates how to use SARPYX for ship detection in SAR imagery using Constant False Alarm Rate (CFAR) algorithms. We'll explore different CFAR detectors and how to optimize their parameters for various sea conditions.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Implement CFAR-based ship detection algorithms
- Optimize detection parameters for different scenarios
- Validate detection results against ground truth
- Integrate ship detection into automated processing workflows

## Prerequisites

- Completion of Tutorials 1-3
- Understanding of SAR backscatter principles
- Basic knowledge of statistical detection theory

## Dataset Requirements

```python
# Required data for this tutorial
data_requirements = {
    'sar_data': 'Sentinel-1 GRD or SLC data over maritime areas',
    'resolution': 'High resolution (< 20m)',
    'polarization': 'VV or VH (VV preferred for ship detection)',
    'area': 'Maritime area with known ship traffic',
    'ground_truth': 'AIS data or manual annotations (optional)'
}
```

## Step 1: Data Preparation and Preprocessing

### 1.1 Load and Preprocess SAR Data

```python
import numpy as np
import matplotlib.pyplot as plt
from sarpyx import SLA, SNAPProcessor
from sarpyx.detection import CFARDetector
from sarpyx.utils import visualization, metrics
from sarpyx.science import radar_indices
import warnings
warnings.filterwarnings('ignore')

# Initialize processors
snap_processor = SNAPProcessor()
detector = CFARDetector()

# Load and preprocess SAR data
sar_file = 'path/to/sentinel1_maritime.zip'

# Standard preprocessing for ship detection
preprocessing_config = {
    'operations': [
        'apply_orbit_file',
        'thermal_noise_removal',
        'calibration',
        'speckle_filtering',  # Important for CFAR detection
        'terrain_correction'
    ],
    'speckle_filter': {
        'filter_type': 'Lee',
        'filter_size': '5x5'
    },
    'calibration': {
        'output_sigma': True,
        'output_beta': False,
        'output_gamma': False
    }
}

# Process with SNAP
processed_data = snap_processor.process_sar_data(
    input_file=sar_file,
    config=preprocessing_config
)

print(f"Processed data shape: {processed_data.shape}")
print(f"Data type: {processed_data.dtype}")
```

### 1.2 Extract Intensity Data

```python
# Extract intensity data for detection
if processed_data.ndim == 3:  # Multi-band data
    # Use VV polarization if available
    if 'VV' in processed_data.band_names:
        intensity = processed_data.get_band('VV')
    else:
        intensity = processed_data.get_band(0)
else:
    intensity = processed_data

# Convert to linear scale if in dB
if np.mean(intensity) < 0:  # Likely in dB
    intensity = 10 ** (intensity / 10)

# Display preprocessing results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Original intensity (log scale)
axes[0].imshow(10 * np.log10(intensity + 1e-10), cmap='gray', vmin=-25, vmax=5)
axes[0].set_title('SAR Intensity (dB)')
axes[0].set_xlabel('Range')
axes[0].set_ylabel('Azimuth')

# Histogram of intensity values
axes[1].hist(intensity.flatten(), bins=100, alpha=0.7, density=True)
axes[1].set_xlabel('Intensity (linear)')
axes[1].set_ylabel('Density')
axes[1].set_title('Intensity Distribution')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()
```

## Step 2: CFAR Detection Implementation

### 2.1 Configure CFAR Parameters

```python
# CFAR detector configuration
cfar_config = {
    'detector_type': 'cell_averaging',  # 'cell_averaging', 'greatest_of', 'smallest_of'
    'guard_cells': {
        'azimuth': 2,
        'range': 2
    },
    'training_cells': {
        'azimuth': 10,
        'range': 10
    },
    'threshold_factor': 3.0,  # Multiplication factor for threshold
    'false_alarm_rate': 1e-5,  # Target false alarm rate
    'min_target_size': 3,  # Minimum number of connected pixels
    'max_target_size': 1000  # Maximum number of connected pixels
}

# Initialize CFAR detector
detector.configure(cfar_config)
print("CFAR detector configured successfully")
print(f"Window size: {cfar_config['training_cells']}")
print(f"Guard cells: {cfar_config['guard_cells']}")
print(f"Threshold factor: {cfar_config['threshold_factor']}")
```

### 2.2 Apply CFAR Detection

```python
# Apply CFAR detection
detection_results = detector.detect_targets(
    intensity_data=intensity,
    return_threshold_map=True,
    return_statistics=True
)

# Extract results
detections = detection_results['detections']
threshold_map = detection_results['threshold_map']
statistics = detection_results['statistics']

print(f"Number of detections: {len(detections)}")
print(f"Detection statistics: {statistics}")

# Display detection results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Original image
axes[0, 0].imshow(10 * np.log10(intensity + 1e-10), cmap='gray', vmin=-25, vmax=5)
axes[0, 0].set_title('Original SAR Image (dB)')

# Threshold map
axes[0, 1].imshow(10 * np.log10(threshold_map + 1e-10), cmap='viridis')
axes[0, 1].set_title('CFAR Threshold Map (dB)')

# Detection mask
detection_mask = np.zeros_like(intensity)
for detection in detections:
    y, x = detection['centroid']
    detection_mask[y-2:y+3, x-2:x+3] = 1

axes[1, 0].imshow(detection_mask, cmap='Reds', alpha=0.7)
axes[1, 0].imshow(10 * np.log10(intensity + 1e-10), cmap='gray', alpha=0.3, vmin=-25, vmax=5)
axes[1, 0].set_title('Detections Overlay')

# Detection statistics
detection_sizes = [d['size'] for d in detections]
detection_intensities = [d['peak_intensity'] for d in detections]

axes[1, 1].scatter(detection_sizes, 10 * np.log10(np.array(detection_intensities)))
axes[1, 1].set_xlabel('Target Size (pixels)')
axes[1, 1].set_ylabel('Peak Intensity (dB)')
axes[1, 1].set_title('Detection Characteristics')

plt.tight_layout()
plt.show()
```

## Step 3: Advanced CFAR Algorithms

### 3.1 Implement Multiple CFAR Variants

```python
# Compare different CFAR algorithms
cfar_variants = {
    'cell_averaging': {
        'detector_type': 'cell_averaging',
        'threshold_factor': 3.0
    },
    'greatest_of': {
        'detector_type': 'greatest_of',
        'threshold_factor': 2.5
    },
    'smallest_of': {
        'detector_type': 'smallest_of',
        'threshold_factor': 4.0
    },
    'ordered_statistic': {
        'detector_type': 'ordered_statistic',
        'threshold_factor': 3.0,
        'order_parameter': 0.75
    }
}

# Apply each variant
variant_results = {}
for variant_name, config in cfar_variants.items():
    # Update base configuration
    variant_config = cfar_config.copy()
    variant_config.update(config)
    
    # Configure and run detector
    detector.configure(variant_config)
    results = detector.detect_targets(intensity_data=intensity)
    
    variant_results[variant_name] = {
        'detections': results['detections'],
        'count': len(results['detections']),
        'config': variant_config
    }
    
    print(f"{variant_name}: {len(results['detections'])} detections")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (variant_name, results) in enumerate(variant_results.items()):
    if i < 4:
        # Create detection overlay
        detection_mask = np.zeros_like(intensity)
        for detection in results['detections']:
            y, x = detection['centroid']
            detection_mask[y-1:y+2, x-1:x+2] = 1
        
        axes[i].imshow(10 * np.log10(intensity + 1e-10), cmap='gray', vmin=-25, vmax=5)
        axes[i].imshow(detection_mask, cmap='Reds', alpha=0.7)
        axes[i].set_title(f'{variant_name.replace("_", " ").title()}\n{results["count"]} detections')

plt.tight_layout()
plt.show()
```

### 3.2 Adaptive Threshold Optimization

```python
# Implement adaptive threshold optimization
class AdaptiveCFAR:
    def __init__(self, base_detector):
        self.detector = base_detector
        
    def optimize_threshold(self, intensity_data, target_far=1e-5, 
                          threshold_range=(1.0, 10.0), num_steps=20):
        """Optimize threshold factor for target false alarm rate"""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
        far_estimates = []
        detection_counts = []
        
        # Estimate false alarm rate for each threshold
        for threshold in thresholds:
            config = cfar_config.copy()
            config['threshold_factor'] = threshold
            
            self.detector.configure(config)
            results = self.detector.detect_targets(intensity_data)
            
            # Estimate FAR (simplified approach)
            total_pixels = intensity_data.size
            detection_count = len(results['detections'])
            estimated_far = detection_count / total_pixels
            
            far_estimates.append(estimated_far)
            detection_counts.append(detection_count)
        
        # Find optimal threshold
        far_errors = np.abs(np.array(far_estimates) - target_far)
        optimal_idx = np.argmin(far_errors)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'thresholds': thresholds,
            'far_estimates': far_estimates,
            'detection_counts': detection_counts
        }

# Apply adaptive optimization
adaptive_cfar = AdaptiveCFAR(detector)
optimization_results = adaptive_cfar.optimize_threshold(
    intensity_data=intensity,
    target_far=1e-5
)

print(f"Optimal threshold factor: {optimization_results['optimal_threshold']:.2f}")

# Plot optimization results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(optimization_results['thresholds'], optimization_results['far_estimates'], 'b-', label='Estimated FAR')
ax1.axhline(y=1e-5, color='r', linestyle='--', label='Target FAR')
ax1.axvline(x=optimization_results['optimal_threshold'], color='g', linestyle='--', label='Optimal Threshold')
ax1.set_xlabel('Threshold Factor')
ax1.set_ylabel('False Alarm Rate')
ax1.set_yscale('log')
ax1.set_title('FAR vs Threshold Factor')
ax1.legend()
ax1.grid(True)

ax2.plot(optimization_results['thresholds'], optimization_results['detection_counts'], 'b-')
ax2.axvline(x=optimization_results['optimal_threshold'], color='g', linestyle='--', label='Optimal Threshold')
ax2.set_xlabel('Threshold Factor')
ax2.set_ylabel('Number of Detections')
ax2.set_title('Detection Count vs Threshold Factor')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Step 4: Detection Validation and Performance Assessment

### 4.1 Load Ground Truth Data (if available)

```python
# Example ground truth loading (adapt to your data format)
def load_ground_truth(gt_file=None):
    """Load ground truth ship positions"""
    if gt_file is None:
        # Generate synthetic ground truth for demonstration
        np.random.seed(42)
        num_ships = 15
        height, width = intensity.shape
        
        ground_truth = []
        for i in range(num_ships):
            # Place ships in areas with higher backscatter
            candidates = np.where(intensity > np.percentile(intensity, 90))
            if len(candidates[0]) > 0:
                idx = np.random.randint(len(candidates[0]))
                y, x = candidates[0][idx], candidates[1][idx]
                
                ground_truth.append({
                    'id': i,
                    'position': (y, x),
                    'size': np.random.randint(5, 25)
                })
        
        return ground_truth
    else:
        # Load from AIS data or annotations
        # Implementation depends on data format
        pass

# Load ground truth
ground_truth = load_ground_truth()
print(f"Ground truth contains {len(ground_truth)} ships")
```

### 4.2 Performance Metrics Calculation

```python
def calculate_detection_metrics(detections, ground_truth, match_threshold=10):
    """Calculate detection performance metrics"""
    
    # Match detections to ground truth
    matched_detections = []
    matched_gt = []
    
    for detection in detections:
        det_pos = detection['centroid']
        
        for i, gt in enumerate(ground_truth):
            gt_pos = gt['position']
            distance = np.sqrt((det_pos[0] - gt_pos[0])**2 + 
                             (det_pos[1] - gt_pos[1])**2)
            
            if distance <= match_threshold and i not in matched_gt:
                matched_detections.append(detection)
                matched_gt.append(i)
                break
    
    # Calculate metrics
    true_positives = len(matched_detections)
    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matched_detections': matched_detections,
        'matched_gt_indices': matched_gt
    }

# Calculate metrics for optimal threshold
optimal_config = cfar_config.copy()
optimal_config['threshold_factor'] = optimization_results['optimal_threshold']
detector.configure(optimal_config)
optimal_detections = detector.detect_targets(intensity_data=intensity)

metrics_results = calculate_detection_metrics(
    optimal_detections['detections'], 
    ground_truth
)

print("\nDetection Performance Metrics:")
print(f"Precision: {metrics_results['precision']:.3f}")
print(f"Recall: {metrics_results['recall']:.3f}")
print(f"F1-Score: {metrics_results['f1_score']:.3f}")
print(f"True Positives: {metrics_results['true_positives']}")
print(f"False Positives: {metrics_results['false_positives']}")
print(f"False Negatives: {metrics_results['false_negatives']}")
```

### 4.3 Visualize Detection Performance

```python
# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Detection overlay with ground truth
axes[0, 0].imshow(10 * np.log10(intensity + 1e-10), cmap='gray', vmin=-25, vmax=5)

# Plot ground truth
for gt in ground_truth:
    y, x = gt['position']
    circle = plt.Circle((x, y), radius=8, color='blue', fill=False, linewidth=2, label='Ground Truth')
    axes[0, 0].add_patch(circle)

# Plot detections (color-coded by match status)
for detection in optimal_detections['detections']:
    y, x = detection['centroid']
    if detection in metrics_results['matched_detections']:
        color = 'green'  # True positive
        label = 'True Positive'
    else:
        color = 'red'  # False positive
        label = 'False Positive'
    
    marker = plt.scatter(x, y, c=color, s=50, marker='x', linewidth=3)

axes[0, 0].set_title('Detection Results vs Ground Truth')
axes[0, 0].legend(['Ground Truth', 'True Positive', 'False Positive'])

# ROC-like curve for different thresholds
thresholds = np.linspace(1.0, 8.0, 15)
precision_values = []
recall_values = []

for threshold in thresholds:
    config = cfar_config.copy()
    config['threshold_factor'] = threshold
    detector.configure(config)
    results = detector.detect_targets(intensity_data=intensity)
    
    metrics = calculate_detection_metrics(results['detections'], ground_truth)
    precision_values.append(metrics['precision'])
    recall_values.append(metrics['recall'])

axes[0, 1].plot(recall_values, precision_values, 'b-o', markersize=5)
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].grid(True)
axes[0, 1].set_xlim([0, 1])
axes[0, 1].set_ylim([0, 1])

# Detection statistics
detection_stats = {
    'sizes': [d['size'] for d in optimal_detections['detections']],
    'intensities': [d['peak_intensity'] for d in optimal_detections['detections']],
    'contrasts': [d.get('contrast', 0) for d in optimal_detections['detections']]
}

axes[1, 0].hist(detection_stats['sizes'], bins=20, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Detection Size (pixels)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Detection Sizes')

# Performance summary table
performance_text = f"""
Detection Performance Summary:

Threshold Factor: {optimization_results['optimal_threshold']:.2f}
Total Detections: {len(optimal_detections['detections'])}
True Positives: {metrics_results['true_positives']}
False Positives: {metrics_results['false_positives']}
False Negatives: {metrics_results['false_negatives']}

Precision: {metrics_results['precision']:.3f}
Recall: {metrics_results['recall']:.3f}
F1-Score: {metrics_results['f1_score']:.3f}
"""

axes[1, 1].text(0.1, 0.5, performance_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_ylim([0, 1])
axes[1, 1].axis('off')
axes[1, 1].set_title('Performance Summary')

plt.tight_layout()
plt.show()
```

## Step 5: Automated Ship Detection Pipeline

### 5.1 Create Processing Pipeline Class

```python
class ShipDetectionPipeline:
    """Complete ship detection pipeline with CFAR algorithms"""
    
    def __init__(self, snap_processor=None):
        self.snap_processor = snap_processor or SNAPProcessor()
        self.cfar_detector = CFARDetector()
        self.processing_history = []
        
    def process_scene(self, input_file, output_dir=None, 
                     preprocessing_config=None, detection_config=None):
        """Process a complete SAR scene for ship detection"""
        
        # Default configurations
        if preprocessing_config is None:
            preprocessing_config = {
                'operations': ['apply_orbit_file', 'thermal_noise_removal', 
                             'calibration', 'speckle_filtering', 'terrain_correction'],
                'speckle_filter': {'filter_type': 'Lee', 'filter_size': '5x5'},
                'calibration': {'output_sigma': True}
            }
        
        if detection_config is None:
            detection_config = cfar_config.copy()
        
        # Step 1: Preprocessing
        print("Step 1: Preprocessing SAR data...")
        processed_data = self.snap_processor.process_sar_data(
            input_file=input_file,
            config=preprocessing_config
        )
        
        # Step 2: Extract intensity
        print("Step 2: Extracting intensity data...")
        if processed_data.ndim == 3:
            intensity = processed_data.get_band('VV') if 'VV' in processed_data.band_names else processed_data.get_band(0)
        else:
            intensity = processed_data
            
        if np.mean(intensity) < 0:
            intensity = 10 ** (intensity / 10)
        
        # Step 3: CFAR detection
        print("Step 3: Applying CFAR detection...")
        self.cfar_detector.configure(detection_config)
        detection_results = self.cfar_detector.detect_targets(
            intensity_data=intensity,
            return_threshold_map=True,
            return_statistics=True
        )
        
        # Step 4: Post-processing and filtering
        print("Step 4: Post-processing detections...")
        filtered_detections = self._filter_detections(
            detection_results['detections'],
            intensity
        )
        
        # Step 5: Generate output products
        print("Step 5: Generating output products...")
        output_products = self._generate_outputs(
            intensity, filtered_detections, detection_results,
            output_dir, input_file
        )
        
        # Record processing
        processing_record = {
            'input_file': input_file,
            'timestamp': np.datetime64('now'),
            'preprocessing_config': preprocessing_config,
            'detection_config': detection_config,
            'detection_count': len(filtered_detections),
            'output_products': output_products
        }
        self.processing_history.append(processing_record)
        
        return {
            'detections': filtered_detections,
            'intensity': intensity,
            'threshold_map': detection_results['threshold_map'],
            'statistics': detection_results['statistics'],
            'output_products': output_products,
            'processing_record': processing_record
        }
    
    def _filter_detections(self, detections, intensity):
        """Apply additional filtering criteria to detections"""
        filtered = []
        
        for detection in detections:
            # Size filtering
            if detection['size'] < 3 or detection['size'] > 1000:
                continue
                
            # Intensity filtering
            if detection['peak_intensity'] < np.percentile(intensity, 95):
                continue
                
            # Shape filtering (aspect ratio)
            if 'bounding_box' in detection:
                bbox = detection['bounding_box']
                height = bbox[2] - bbox[0]
                width = bbox[3] - bbox[1]
                aspect_ratio = max(height, width) / min(height, width)
                if aspect_ratio > 5:  # Too elongated
                    continue
            
            filtered.append(detection)
        
        return filtered
    
    def _generate_outputs(self, intensity, detections, detection_results, 
                         output_dir, input_file):
        """Generate output products"""
        outputs = {}
        
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            
            # Save detection results as GeoTIFF
            detection_mask = np.zeros_like(intensity)
            for i, detection in enumerate(detections):
                y, x = detection['centroid']
                detection_mask[y-2:y+3, x-2:x+3] = i + 1
            
            # Save products (implementation depends on your I/O utilities)
            outputs['detection_mask'] = f"{output_dir}/{base_name}_detections.tif"
            outputs['threshold_map'] = f"{output_dir}/{base_name}_threshold.tif"
            outputs['detection_list'] = f"{output_dir}/{base_name}_detections.json"
            
            # Save detection list as JSON
            import json
            detection_data = {
                'metadata': {
                    'input_file': input_file,
                    'detection_count': len(detections),
                    'processing_timestamp': str(np.datetime64('now'))
                },
                'detections': [
                    {
                        'id': i,
                        'position': {'y': int(d['centroid'][0]), 'x': int(d['centroid'][1])},
                        'size': int(d['size']),
                        'peak_intensity_db': float(10 * np.log10(d['peak_intensity'])),
                        'confidence': float(d.get('confidence', 1.0))
                    }
                    for i, d in enumerate(detections)
                ]
            }
            
            with open(outputs['detection_list'], 'w') as f:
                json.dump(detection_data, f, indent=2)
        
        return outputs

# Initialize and test pipeline
pipeline = ShipDetectionPipeline(snap_processor)

# Process the scene
pipeline_results = pipeline.process_scene(
    input_file=sar_file,
    output_dir='ship_detection_output',
    detection_config=optimal_config
)

print(f"\nPipeline Results:")
print(f"Detections found: {len(pipeline_results['detections'])}")
print(f"Output products: {list(pipeline_results['output_products'].keys())}")
```

### 5.2 Batch Processing Capability

```python
def batch_ship_detection(input_files, pipeline, output_base_dir='batch_outputs'):
    """Process multiple SAR files for ship detection"""
    
    batch_results = []
    
    for i, input_file in enumerate(input_files):
        print(f"\nProcessing file {i+1}/{len(input_files)}: {input_file}")
        
        try:
            # Create individual output directory
            file_output_dir = f"{output_base_dir}/scene_{i:03d}"
            
            # Process scene
            results = pipeline.process_scene(
                input_file=input_file,
                output_dir=file_output_dir
            )
            
            batch_results.append({
                'input_file': input_file,
                'status': 'success',
                'detection_count': len(results['detections']),
                'output_dir': file_output_dir,
                'results': results
            })
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            batch_results.append({
                'input_file': input_file,
                'status': 'error',
                'error': str(e)
            })
    
    # Generate batch summary
    successful_files = [r for r in batch_results if r['status'] == 'success']
    total_detections = sum(r['detection_count'] for r in successful_files)
    
    print(f"\nBatch Processing Summary:")
    print(f"Files processed: {len(input_files)}")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed: {len(input_files) - len(successful_files)}")
    print(f"Total detections: {total_detections}")
    
    return batch_results

# Example batch processing
# input_files = ['file1.zip', 'file2.zip', 'file3.zip']  # Your SAR files
# batch_results = batch_ship_detection(input_files, pipeline)
```

## Step 6: Advanced Applications and Integration

### 6.1 Multi-temporal Ship Tracking

```python
class ShipTracker:
    """Track ships across multiple SAR acquisitions"""
    
    def __init__(self, max_distance=50, max_time_gap=24):
        self.max_distance = max_distance  # pixels
        self.max_time_gap = max_time_gap  # hours
        self.tracks = []
        
    def add_detections(self, detections, timestamp, scene_id):
        """Add new detections and update tracks"""
        
        # Convert detections to tracking format
        new_detections = [
            {
                'position': d['centroid'],
                'size': d['size'],
                'intensity': d['peak_intensity'],
                'timestamp': timestamp,
                'scene_id': scene_id,
                'detection_id': i
            }
            for i, d in enumerate(detections)
        ]
        
        if not self.tracks:
            # Initialize tracks with first detections
            for detection in new_detections:
                self.tracks.append({
                    'track_id': len(self.tracks),
                    'detections': [detection],
                    'start_time': timestamp,
                    'last_time': timestamp
                })
        else:
            # Match detections to existing tracks
            self._update_tracks(new_detections, timestamp)
    
    def _update_tracks(self, new_detections, timestamp):
        """Update existing tracks with new detections"""
        
        # Calculate distances between new detections and track endpoints
        unmatched_detections = new_detections.copy()
        
        for track in self.tracks:
            if not track['detections']:
                continue
                
            last_detection = track['detections'][-1]
            time_gap = (timestamp - last_detection['timestamp']).total_seconds() / 3600
            
            if time_gap > self.max_time_gap:
                continue
            
            # Find closest detection
            best_match = None
            best_distance = float('inf')
            
            for detection in unmatched_detections:
                distance = np.sqrt(
                    (detection['position'][0] - last_detection['position'][0])**2 +
                    (detection['position'][1] - last_detection['position'][1])**2
                )
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = detection
            
            # Update track if match found
            if best_match:
                track['detections'].append(best_match)
                track['last_time'] = timestamp
                unmatched_detections.remove(best_match)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            self.tracks.append({
                'track_id': len(self.tracks),
                'detections': [detection],
                'start_time': timestamp,
                'last_time': timestamp
            })
    
    def get_active_tracks(self, min_length=2):
        """Get tracks with minimum number of detections"""
        return [track for track in self.tracks 
                if len(track['detections']) >= min_length]

# Example multi-temporal tracking
# tracker = ShipTracker()
# 
# # Add detections from multiple scenes
# for scene_data in time_series_data:
#     tracker.add_detections(
#         detections=scene_data['detections'],
#         timestamp=scene_data['timestamp'],
#         scene_id=scene_data['scene_id']
#     )
# 
# active_tracks = tracker.get_active_tracks(min_length=3)
# print(f"Found {len(active_tracks)} ship tracks")
```

## Key Takeaways

1. **CFAR Algorithms**: Different CFAR variants (CA, GO, SO, OS) have different strengths for various clutter conditions

2. **Parameter Optimization**: Automatic threshold optimization helps achieve target false alarm rates

3. **Performance Validation**: Systematic validation against ground truth is essential for operational systems

4. **Pipeline Integration**: Automated pipelines enable operational ship detection workflows

5. **Multi-temporal Analysis**: Ship tracking across multiple acquisitions provides additional information

## Next Steps

1. **Experiment with different CFAR algorithms** for your specific data and conditions
2. **Validate results** with AIS data or other ground truth sources
3. **Optimize parameters** for your operational requirements
4. **Integrate with other processing workflows** for comprehensive maritime monitoring

## Additional Resources

- CFAR detection theory and implementation
- Ship detection in SAR imagery best practices
- Maritime surveillance applications
- AIS data integration techniques

---

*This tutorial demonstrates advanced ship detection capabilities in SARPYX. The techniques shown here form the foundation for operational maritime monitoring systems using SAR data.*
