# Getting Started with SARPyX

This guide provides a quick introduction to SARPyX and walks you through your first SAR processing tasks.

## Your First SARPyX Program

Let's start with a simple example using SARPyX's visualization utilities:

```python
import numpy as np
from sarpyx.utils import show_image, image_histogram_equalization

# Create sample SAR-like data
data = np.random.rayleigh(1.0, (512, 512))

# Display the image
show_image(data, title="Sample SAR Data", cmap='gray')

# Apply histogram equalization
enhanced_data = image_histogram_equalization(data)
show_image(enhanced_data, title="Enhanced SAR Data", cmap='gray')
```

## Basic Workflow Overview

SARPyX follows a modular approach to SAR processing:

1. **Data Loading**: Import SAR data from various formats
2. **Preprocessing**: Apply calibration, filtering, and corrections
3. **Core Processing**: Execute focusing, autofocus, or analysis algorithms
4. **Analysis**: Perform sub-look analysis, interferometry, or scientific applications
5. **Visualization**: Display results and generate reports

## Working with Real SAR Data

### Sub-Look Analysis Example

Sub-Look Analysis (SLA) is one of SARPyX's core capabilities:

```python
from sarpyx.sla import SubLookAnalysis

# Initialize SLA processor with Sentinel-1 data
product_path = "path/to/your/S1A_IW_SLC_*.zip"
sla = SubLookAnalysis(product_path)

# Configure analysis parameters
sla.choice = 1  # Azimuth direction (0 for range)
sla.numberOfLooks = 3
sla.centroidSeparations = 700  # Hz
sla.subLookBandwidth = 700     # Hz

# Execute the processing chain
sla.frequencyComputation()
sla.SpectrumComputation(VERBOSE=True)
sla.AncillaryDeWe()
sla.Generation(VERBOSE=True)

# Access results
sublook_images = sla.Looks  # Shape: (numberOfLooks, rows, cols)
```

### SNAP Integration Example

SARPyX provides seamless integration with ESA SNAP:

```python
from sarpyx.snap import GPT
from pathlib import Path

# Initialize GPT processor
product_path = "S1A_IW_GRDH_*.zip"
output_dir = "processed_data"
gpt = GPT(product=product_path, outdir=output_dir)

# Apply thermal noise removal
gpt.ThermalNoiseRemoval()

# Apply radiometric calibration
calibrated = gpt.Calibration(outputSigmaBand=True)

# Apply terrain correction
terrain_corrected = gpt.TerrainCorrection(
    demName='SRTM 1Sec HGT',
    pixelSpacingInMeter=10.0
)

print(f"Processed product saved to: {terrain_corrected}")
```

### Science Applications

Calculate vegetation indices from dual-polarization SAR data:

```python
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll
import numpy as np

# Load dual-pol data (replace with actual data loading)
sigma_vv = np.random.gamma(2, 1, (1000, 1000))  # VV backscatter
sigma_vh = np.random.gamma(1, 1, (1000, 1000))  # VH backscatter

# Calculate Radar Vegetation Index
rvi = calculate_rvi(sigma_vv, sigma_vh)

# Calculate Normalized Difference Polarization Index  
ndpoll = calculate_ndpoll(sigma_vv, sigma_vh)

# Visualize results
from sarpyx.utils import show_image
show_image(rvi, title="Radar Vegetation Index", cmap='RdYlGn')
show_image(ndpoll, title="NDPOLL Index", cmap='RdBu')
```

## Understanding the Module Structure

SARPyX is organized into several main modules:

### `sarpyx.processor`
Core SAR processing algorithms:
- `core`: Focus algorithms, transforms, decoding
- `autofocus`: Quality metrics and autofocus methods
- `algorithms`: High-level algorithms (RDA, back-projection)
- `data`: Data I/O and format conversion
- `utils`: Processing utilities

### `sarpyx.sla`
Sub-Look Analysis for aperture decomposition:
- Frequency domain processing
- Sub-aperture extraction
- Motion analysis capabilities

### `sarpyx.snap`
SNAP integration:
- GPT workflow automation
- Graph processing chains
- Product manipulation

### `sarpyx.science`
Scientific analysis tools:
- Vegetation indices
- Polarimetric analysis
- Change detection methods

### `sarpyx.utils`
General utilities:
- Visualization functions
- Image processing helpers
- Data manipulation tools

## Configuration and Parameters

### Setting Processing Parameters

Many SARPyX functions accept configuration parameters:

```python
# Configure sub-look analysis
sla_config = {
    'choice': 1,              # Processing direction
    'numberOfLooks': 3,       # Number of sub-looks
    'centroidSeparations': 700,  # Frequency separation
    'subLookBandwidth': 700      # Sub-look bandwidth
}

# Apply configuration
sla = SubLookAnalysis(product_path)
for key, value in sla_config.items():
    setattr(sla, key, value)
```

### Working with Different SAR Sensors

SARPyX supports multiple SAR missions:

```python
# Sentinel-1 processing
from sarpyx.snap import GPT
gpt_s1 = GPT(product="S1A_*.zip", mode="MacOS")

# COSMO-SkyMed processing  
gpt_csk = GPT(product="CSK_*.h5", mode="Ubuntu")

# SAO-COM processing
gpt_sao = GPT(product="SAO_*.zip", mode=None)
```

## Error Handling and Debugging

SARPyX provides verbose options for debugging:

```python
# Enable verbose output
sla.SpectrumComputation(VERBOSE=True)
sla.Generation(VERBOSE=True)

# Error handling
try:
    result = sla.frequencyComputation()
except AssertionError as e:
    print(f"Parameter validation failed: {e}")
except Exception as e:
    print(f"Processing error: {e}")
```

## Performance Considerations

### Memory Management
For large datasets:
- Process data in chunks
- Use appropriate data types
- Monitor memory usage

```python
import psutil
import gc

# Monitor memory before processing
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Process data
result = your_processing_function()

# Clean up
gc.collect()
print(f"Memory usage after cleanup: {psutil.virtual_memory().percent}%")
```

### Parallel Processing
SARPyX can leverage multiple cores:
- GDAL operations are often parallelized
- NumPy operations use multiple threads
- Custom parallel processing can be implemented

## Next Steps

After completing this guide:

1. **Learn the fundamentals**: Read [Basic Concepts](basic_concepts.md)
2. **Explore data formats**: See [Data Formats](data_formats.md)  
3. **Try advanced workflows**: Check [Processing Workflows](processing_workflows.md)
4. **Run examples**: Browse the [Examples](../examples/README.md) directory
5. **API deep dive**: Explore the [API Reference](../api/README.md)

## Common Patterns

### Batch Processing

```python
from pathlib import Path

def process_sar_batch(input_dir, output_dir):
    """Process multiple SAR products in batch."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for product in input_path.glob("*.zip"):
        try:
            gpt = GPT(product=str(product), outdir=str(output_path))
            result = gpt.Calibration()
            print(f"Processed: {product.name}")
        except Exception as e:
            print(f"Failed to process {product.name}: {e}")

# Usage
process_sar_batch("input_products", "processed_products")
```

### Pipeline Chaining

```python
def sar_processing_pipeline(product_path):
    """Complete SAR processing pipeline."""
    
    # Step 1: SNAP preprocessing
    gpt = GPT(product=product_path, outdir="temp")
    calibrated = gpt.Calibration()
    corrected = gpt.TerrainCorrection()
    
    # Step 2: Sub-look analysis
    sla = SubLookAnalysis(corrected)
    sla.choice = 1
    sla.numberOfLooks = 3
    sla.frequencyComputation()
    sla.SpectrumComputation()
    sla.Generation()
    
    # Step 3: Analysis
    from sarpyx.science.indices import calculate_rvi
    # Additional processing...
    
    return sla.Looks

# Execute pipeline
results = sar_processing_pipeline("my_product.zip")
```

Ready to dive deeper? Continue with [Basic Concepts](basic_concepts.md) to understand the theoretical foundations of SAR processing with SARPyX.
