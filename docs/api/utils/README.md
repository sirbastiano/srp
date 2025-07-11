# Utils Module API

The `sarpyx.utils` module provides essential utility functions for data visualization, input/output operations, and general helper functions that support the entire SARPyX ecosystem.

## Overview

The utils module includes:
- **Visualization functions** for displaying SAR data and analysis results
- **I/O utilities** for data format conversion and file management  
- **Helper functions** for common operations across SARPyX modules

## Module Structure

```
sarpyx.utils/
├── viz.py         # Visualization and plotting functions
└── io.py          # Input/output and file management utilities
```

## Quick Start

```python
from sarpyx.utils import show_image
import numpy as np

# Display SAR data
sar_image = np.random.random((512, 512))
show_image(sar_image, title='SAR Amplitude', cmap='gray')

# I/O operations
from sarpyx.utils.io import save_matlab_mat
save_matlab_mat(sar_image, 'sar_data.mat', '.')
```

## Visualization Functions

### show_image()

```python
show_image(image, title=None, cmap=None, vmin=None, vmax=None, 
           colorbar=True, ax=None)
```

Display an image using matplotlib with optional customization.

**Parameters:**
- `image` (np.ndarray): Input image to display
- `title` (str, optional): Title to display above the image
- `cmap` (str, optional): Matplotlib colormap name
- `vmin` (float, optional): Minimum value for colormap scaling
- `vmax` (float, optional): Maximum value for colormap scaling
- `colorbar` (bool): Whether to add a colorbar (default: True)
- `ax` (plt.Axes, optional): Matplotlib axes to plot on

**Returns:**
- `plt.Axes`: The matplotlib axes containing the image

**Examples:**

```python
import numpy as np
from sarpyx.utils import show_image

# Basic image display
sar_data = np.random.random((256, 256))
show_image(sar_data)

# Customized display
show_image(sar_data, 
          title='SAR Backscatter', 
          cmap='viridis',
          vmin=0, vmax=1,
          colorbar=True)

# Multiple subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
show_image(vv_data, 'VV Polarization', ax=axes[0])
show_image(vh_data, 'VH Polarization', ax=axes[1])
```

### show_histogram()

```python
show_histogram(image, title=None, ax=None)
```

Display a histogram of pixel values in an image.

**Parameters:**
- `image` (np.ndarray): Input image as a numpy array
- `title` (str, optional): Title for the histogram
- `ax` (plt.Axes, optional): Matplotlib axes to plot on

**Returns:**
- `plt.Axes`: The matplotlib axes containing the histogram

**Example:**

```python
# Display intensity distribution
sar_intensity = np.abs(sar_complex)**2
show_histogram(sar_intensity, title='SAR Intensity Distribution')

# Compare distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
show_histogram(before_filtering, 'Before Filtering', ax=axes[0])
show_histogram(after_filtering, 'After Filtering', ax=axes[1])
```

### image_histogram_equalization()

```python
image_histogram_equalization(image, number_bins=8)
```

Perform histogram equalization on an image to enhance contrast.

**Parameters:**
- `image` (np.ndarray): Input image
- `number_bins` (int): Number of bins for histogram computation

**Returns:**
- `tuple`: (equalized_image, cdf_normalized)

**Example:**

```python
# Enhance SAR image contrast
enhanced_image, cdf = image_histogram_equalization(sar_image, number_bins=16)

# Display comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
show_image(sar_image, 'Original', ax=axes[0])
show_image(enhanced_image, 'Enhanced', ax=axes[1])
```

### show_histogram_equalization()

```python
show_histogram_equalization(image, number_bins=8, title=None, ax=None)
```

Display the histogram of an equalized image.

**Parameters:**
- `image` (np.ndarray): Input image
- `number_bins` (int): Number of bins for histogram equalization
- `title` (str, optional): Title for the histogram
- `ax` (plt.Axes, optional): Matplotlib axes to plot on

**Returns:**
- `plt.Axes`: The matplotlib axes containing the histogram

## I/O Utilities

### save_matlab_mat()

```python
save_matlab_mat(data_object, filename, filepath)
```

Save data in MATLAB .mat format.

**Parameters:**
- `data_object` (Any): Data to save
- `filename` (str): Output filename (without extension)
- `filepath` (str | Path): Directory path for output file

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**

```python
from sarpyx.utils.io import save_matlab_mat

# Save processing results
results = {
    'amplitude': sar_amplitude,
    'phase': sar_phase,
    'metadata': processing_info
}

success = save_matlab_mat(results, 'sar_results', './output/')
if success:
    print("Data saved successfully")
```

### delete()

```python
delete(path_to_delete)
```

Safely delete files or directories.

**Parameters:**
- `path_to_delete` (str | Path): Path to file or directory to delete

**Example:**

```python
from sarpyx.utils.io import delete

# Clean up temporary files
delete('./temp_processing/')
delete('./intermediate_result.tif')
```

### unzip()

```python
unzip(path_to_zip_file)
```

Extract ZIP archives, commonly used for SAR data products.

**Parameters:**
- `path_to_zip_file` (str | Path): Path to ZIP file

**Example:**

```python
from sarpyx.utils.io import unzip

# Extract Sentinel-1 product
unzip('./S1A_IW_SLC__1SDV_20230101T120000.zip')
```

### delProd()

```python
delProd(prodToDelete)
```

Delete SNAP/SAR products and associated files.

**Parameters:**
- `prodToDelete` (str | Path): Path to product to delete

### command_line()

```python
command_line(cmd)
```

Execute command line operations safely.

**Parameters:**
- `cmd` (str): Command to execute

### mode_identifier()

```python
mode_identifier(product_name)
```

Identify SAR product type from filename.

**Parameters:**
- `product_name` (str | Path): Product filename or path

**Returns:**
- `str`: Product type identifier ('SEN', 'CSK', 'SAO', etc.)

**Example:**

```python
from sarpyx.utils.io import mode_identifier

product_type = mode_identifier('S1A_IW_SLC__1SDV_20230101.SAFE')
print(f"Product type: {product_type}")  # Output: SEN
```

## Usage Examples

### SAR Data Visualization Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from sarpyx.utils import show_image, show_histogram

# Load and display SAR data
sar_complex = load_sar_data()
amplitude = np.abs(sar_complex)
phase = np.angle(sar_complex)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Display amplitude with different visualizations
show_image(amplitude, 'Amplitude (Linear)', ax=axes[0, 0])
show_image(20*np.log10(amplitude), 'Amplitude (dB)', 
          cmap='gray', ax=axes[0, 1])
show_histogram(amplitude, 'Amplitude Distribution', ax=axes[0, 2])

# Display phase
show_image(phase, 'Phase', cmap='hsv', 
          vmin=-np.pi, vmax=np.pi, ax=axes[1, 0])

# Enhanced contrast version
enhanced_amp, _ = image_histogram_equalization(amplitude)
show_image(enhanced_amp, 'Enhanced Amplitude', ax=axes[1, 1])
show_histogram(enhanced_amp, 'Enhanced Distribution', ax=axes[1, 2])

plt.tight_layout()
plt.show()
```

### Multi-Temporal Data Analysis

```python
from sarpyx.utils import show_image
from sarpyx.science import indices

# Load time series
dates = ['2023-01', '2023-04', '2023-07', '2023-10']
rvi_series = [load_rvi_data(date) for date in dates]

# Visualize temporal changes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (rvi, date) in enumerate(zip(rvi_series, dates)):
    show_image(rvi, f'RVI {date}', 
              vmin=0, vmax=2, 
              cmap='RdYlGn', 
              ax=axes[i])

plt.tight_layout()
plt.show()

# Calculate and visualize temporal statistics
rvi_stack = np.stack(rvi_series, axis=0)
rvi_mean = np.mean(rvi_stack, axis=0)
rvi_std = np.std(rvi_stack, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
show_image(rvi_mean, 'Mean RVI', ax=axes[0])
show_image(rvi_std, 'RVI Variability', ax=axes[1])
```

### Quality Assessment Visualization

```python
def visualize_processing_quality(original, processed, title_prefix=""):
    """Visualize before/after processing comparison."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Original data
    show_image(original, f'{title_prefix}Original', ax=axes[0, 0])
    show_histogram(original, 'Original Distribution', ax=axes[0, 1])
    
    # Processed data
    show_image(processed, f'{title_prefix}Processed', ax=axes[1, 0])
    show_histogram(processed, 'Processed Distribution', ax=axes[1, 1])
    
    # Difference
    difference = processed - original
    show_image(difference, 'Difference', 
              cmap='RdBu_r', ax=axes[0, 2])
    
    # Statistics
    axes[1, 2].text(0.1, 0.8, f'Original Mean: {np.mean(original):.3f}')
    axes[1, 2].text(0.1, 0.6, f'Processed Mean: {np.mean(processed):.3f}')
    axes[1, 2].text(0.1, 0.4, f'Std Original: {np.std(original):.3f}')
    axes[1, 2].text(0.1, 0.2, f'Std Processed: {np.std(processed):.3f}')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Statistics')
    
    plt.tight_layout()
    return fig

# Use for speckle filtering assessment
original_sar = load_sar_data()
filtered_sar = apply_speckle_filter(original_sar)
visualize_processing_quality(original_sar, filtered_sar, "Speckle Filter ")
```

### Data Export and Documentation

```python
from sarpyx.utils.io import save_matlab_mat
import json

def export_analysis_results(results_dict, output_dir, session_name):
    """Export analysis results in multiple formats."""
    
    # Save numerical data as MATLAB file
    matlab_data = {
        'amplitude': results_dict['amplitude'],
        'phase': results_dict['phase'],
        'vegetation_indices': results_dict['indices'],
        'processing_parameters': results_dict['parameters']
    }
    
    save_matlab_mat(matlab_data, f'{session_name}_data', output_dir)
    
    # Save metadata as JSON
    metadata = {
        'session_name': session_name,
        'processing_date': str(datetime.now()),
        'input_files': results_dict['input_files'],
        'processing_steps': results_dict['processing_steps'],
        'quality_metrics': results_dict['quality_metrics']
    }
    
    with open(f'{output_dir}/{session_name}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate visualization report
    create_visualization_report(results_dict, output_dir, session_name)

def create_visualization_report(results, output_dir, session_name):
    """Create comprehensive visualization report."""
    
    # Create multi-page visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Page 1: Raw data
    show_image(results['amplitude'], 'Amplitude', ax=axes[0, 0])
    show_image(results['phase'], 'Phase', ax=axes[0, 1])
    show_histogram(results['amplitude'], 'Amplitude Hist', ax=axes[0, 2])
    
    # Page 2: Vegetation indices
    show_image(results['indices']['rvi'], 'RVI', ax=axes[1, 0])
    show_image(results['indices']['ndpoll'], 'NDPoll', ax=axes[1, 1])
    show_image(results['indices']['dprvi'], 'DpRVI', ax=axes[1, 2])
    
    # Page 3: Quality metrics
    show_image(results['quality']['coherence'], 'Coherence', ax=axes[2, 0])
    show_image(results['quality']['intensity_variance'], 'Intensity Var', ax=axes[2, 1])
    
    # Summary statistics
    axes[2, 2].text(0.1, 0.8, f"Mean RVI: {np.mean(results['indices']['rvi']):.3f}")
    axes[2, 2].text(0.1, 0.6, f"Vegetated area: {np.sum(results['indices']['rvi'] > 0.5)}")
    axes[2, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{session_name}_report.png', dpi=300, bbox_inches='tight')
    plt.close()
```

### Integration with Other Modules

```python
# Integrated workflow with SLA and visualization
from sarpyx.sla import SubLookAnalysis
from sarpyx.utils import show_image
from sarpyx.science import indices

def analyze_and_visualize_sublooks(product_path, output_dir):
    """Complete sub-look analysis with visualization."""
    
    # Perform SLA
    sla = SubLookAnalysis(product_path)
    sla.choice = 1
    sla.numberOfLooks = 3
    sla.SpectrumComputation()
    sla.Generation()
    
    # Extract sub-looks
    sublooks = [sla.Looks[i, :, :] for i in range(sla.numberOfLooks)]
    
    # Visualize sub-looks
    fig, axes = plt.subplots(1, len(sublooks), figsize=(15, 5))
    for i, (sublook, ax) in enumerate(zip(sublooks, axes)):
        show_image(np.abs(sublook), f'Sub-look {i+1}', ax=ax)
    
    plt.savefig(f'{output_dir}/sublooks_comparison.png', dpi=300)
    
    # Calculate vegetation indices if dual-pol
    if hasattr(sla, 'vh_data'):
        rvi_sublooks = []
        for i in range(len(sublooks)):
            if len(sublooks[i].shape) == 3:  # Multi-polarization
                rvi = indices.calculate_rvi(sublooks[i][0], sublooks[i][1])
                rvi_sublooks.append(rvi)
        
        # Visualize RVI differences
        if rvi_sublooks:
            fig, axes = plt.subplots(1, len(rvi_sublooks), figsize=(15, 5))
            for i, (rvi, ax) in enumerate(zip(rvi_sublooks, axes)):
                show_image(rvi, f'RVI Sub-look {i+1}', 
                          vmin=0, vmax=2, ax=ax)
            
            plt.savefig(f'{output_dir}/rvi_sublooks.png', dpi=300)
    
    return sublooks
```

## Performance and Best Practices

### Memory-Efficient Visualization

```python
def visualize_large_dataset(data_path, chunk_size=(1000, 1000)):
    """Visualize large datasets without loading everything into memory."""
    
    # Load only a representative subset
    with rasterio.open(data_path) as src:
        # Read a central chunk
        height, width = src.height, src.width
        row_start = height // 2 - chunk_size[0] // 2
        col_start = width // 2 - chunk_size[1] // 2
        
        window = rasterio.windows.Window(
            col_start, row_start, chunk_size[1], chunk_size[0]
        )
        
        data_chunk = src.read(1, window=window)
    
    # Visualize the subset
    show_image(data_chunk, f'Sample from {Path(data_path).name}')
    
    return data_chunk
```

### Automated Visualization Reports

```python
def create_automated_report(processing_results, output_path):
    """Create standardized visualization reports."""
    
    # Define standard visualizations
    visualizations = [
        ('amplitude', 'SAR Amplitude', 'gray'),
        ('phase', 'SAR Phase', 'hsv'),
        ('coherence', 'Coherence', 'viridis'),
        ('vegetation_index', 'Vegetation Index', 'RdYlGn')
    ]
    
    n_plots = len(visualizations)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (key, title, cmap) in enumerate(visualizations):
        if key in processing_results:
            show_image(processing_results[key], title, 
                      cmap=cmap, ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, f'{title}\nNot Available', 
                        ha='center', va='center')
            axes[i].set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

## See Also

- [API Reference](../README.md): Overview of all SARPyX modules
- [SLA Module](../sla/README.md): Sub-aperture analysis visualization
- [Science Module](../science/README.md): Scientific analysis and indices
- [Examples](../../examples/README.md): Ready-to-run visualization examples
