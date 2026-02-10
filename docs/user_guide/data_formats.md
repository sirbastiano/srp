# Data Formats and Compatibility

This guide covers the data formats supported by sarpyx for input and output operations, along with format-specific considerations and best practices.

## Supported Input Formats

### SAR Product Formats

sarpyx supports the following SAR mission data formats:

#### Sentinel-1 (Copernicus)
- **Format**: SAFE (Standard Archive Format for Europe)
- **File Extension**: `.zip` (compressed SAFE packages)
- **Product Types**: 
  - SLC (Single Look Complex)
  - GRD (Ground Range Detected)
- **Polarizations**: VV, VH, HH, HV (mission dependent)
- **Processing Levels**: Level-1 and Level-2

```python
# Example: Loading Sentinel-1 SLC data
from sarpyx.sla import SubLookAnalysis

# Input: ZIP file containing SAFE format
sla = SubLookAnalysis("S1A_IW_SLC_1SDV_20230101T120000_20230101T120030_046123_058A2B_1234.zip")
```

#### COSMO-SkyMed (Italian Space Agency)
- **Format**: HDF5-based proprietary format
- **File Extension**: Various (`.h5`, custom extensions)
- **Product Types**: SCS (Single Look Complex Slant Range)
- **Polarizations**: HH, HV, VH, VV
- **Processing Levels**: Level-1A, Level-1B

```python
# Example: COSMO-SkyMed processing
from sarpyx.snapflow.engine import GPT

# Initialize with COSMO-SkyMed product
gpt = GPT(product_path="CSK_product.h5", 
          outdir="output/", 
          mode="MacOS")  # or "Ubuntu"
```

#### ALOS PALSAR / ALOS-2 PALSAR-2 (JAXA)
- **Format**: CEOS (Committee on Earth Observation Satellites)
- **File Extension**: Various (directory structures)
- **Product Types**: Level 1.1, Level 1.5
- **Polarizations**: HH, HV, VH, VV
- **Band**: L-band

### Derived Data Formats

#### Complex SAR Data
- **Format**: GeoTIFF with complex data types
- **File Extension**: `.tif`, `.tiff`
- **Data Type**: Complex64 (32-bit real + 32-bit imaginary)
- **Usage**: Output from sub-look analysis, interferometric processing

```python
# Reading complex GeoTIFF
from osgeo import gdal
import numpy as np

# Open complex image (I/Q channels)
dataset = gdal.Open("complex_data.tif")
real_band = dataset.GetRasterBand(1).ReadAsArray()
imag_band = dataset.GetRasterBand(2).ReadAsArray()
complex_data = real_band + 1j * imag_band
```

#### Amplitude/Intensity Images
- **Format**: GeoTIFF (single-band)
- **File Extension**: `.tif`, `.tiff`
- **Data Type**: Float32, UInt16
- **Usage**: Calibrated backscatter, derived indices

## SNAP Integration Formats

sarpyx integrates with ESA's SNAP (Sentinel Application Platform) for advanced processing workflows.

### SNAP Native Formats
- **BEAM-DIMAP**: `.dim` + `.data` directory
- **NetCDF**: `.nc` files
- **GeoTIFF**: `.tif` for final products
- **HDF5**: `.h5` for complex datasets

### Processing Chain Output
```python
from sarpyx.snapflow.engine import GPT

# Example SNAP processing chain
gpt = GPT(product_path="input.zip", outdir="output/")

# Calibration (outputs BEAM-DIMAP by default)
cal_product = gpt.Calibration(Pols=['VV', 'VH'])

# Export to GeoTIFF for external use
gpt.format = 'GeoTIFF'
final_product = gpt._call(suffix='FINAL')
```

## Output Formats and Data Products

### Sub-Look Analysis Products

#### NumPy Arrays (.npz)
Compressed arrays containing sub-look decomposition results:

```python
# Save sub-look data
np.savez_compressed(
    'sublook_results.npz',
    looks=sla.Looks,                    # Complex sub-look images
    frequencies=sla.freqCentr,          # Center frequencies
    freq_min=sla.freqMin,              # Minimum frequencies  
    freq_max=sla.freqMax               # Maximum frequencies
)

# Load sub-look data
data = np.load('sublook_results.npz')
sublooks = data['looks']
```

#### Metadata (JSON)
Processing parameters and quality metrics:

```python
import json

metadata = {
    'processing_parameters': {
        'numberOfLooks': 3,
        'centroidSeparations': 700,
        'subLookBandwidth': 700
    },
    'quality_metrics': {
        'coherence': coherence_values,
        'snr': signal_to_noise_ratio
    }
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Scientific Products

#### Vegetation Indices
```python
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll

# Calculate indices (returns NumPy arrays)
rvi = calculate_rvi(sigma_vv, sigma_vh)
ndpoll = calculate_ndpoll(sigma_vv, sigma_vh)

# Save as GeoTIFF with spatial reference
from osgeo import gdal, osr

def save_geotiff(array, filename, geotransform, projection):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset = None
```

### Visualization Outputs

#### Matplotlib Figures (.png, .pdf)
```python
import matplotlib.pyplot as plt

# High-quality figure export
plt.figure(figsize=(12, 8), dpi=300)
plt.imshow(np.abs(sublook_data[0]), cmap='viridis')
plt.title('Sub-look 1 Amplitude')
plt.colorbar()
plt.savefig('sublook_amplitude.png', dpi=300, bbox_inches='tight')
plt.savefig('sublook_amplitude.pdf', bbox_inches='tight')  # Vector format
```

## Format Conversion Utilities

### MATLAB Compatibility
```python
from sarpyx.utils.io import save_matlab_mat

# Save data for MATLAB analysis
save_matlab_mat(sublook_data, "sublooks", "output/")
# Creates: output/sublooks.mat
```

### Data Type Conversions
```python
# Convert complex to amplitude/phase
amplitude = np.abs(complex_data)
phase = np.angle(complex_data)

# Convert to dB scale
amplitude_db = 20 * np.log10(amplitude + 1e-10)  # Avoid log(0)

# Normalize for visualization
normalized = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
```

## File Organization Best Practices

### Recommended Directory Structure
```
project/
├── data/
│   ├── raw/                    # Original SAR products
│   │   ├── S1A_IW_SLC_*.zip
│   │   └── CSK_*.h5
│   ├── processed/              # Intermediate products
│   │   ├── calibrated/
│   │   ├── coregistered/
│   │   └── filtered/
│   └── products/               # Final outputs
│       ├── sublooks/
│       ├── indices/
│       └── figures/
├── scripts/                    # Processing scripts
├── config/                     # Configuration files
└── docs/                       # Documentation
```

### Naming Conventions
```python
# Recommended file naming patterns
def generate_filename(product_type, date, polarization, processing_level):
    return f"{product_type}_{date}_{polarization}_{processing_level}"

# Examples:
# S1A_SLC_20230101_VV_L1.tif
# CSK_SCS_20230101_HH_sublook.npz
# indices_RVI_20230101_VV_VH.tif
```

## Performance Considerations

### Memory-Efficient Processing
```python
# For large datasets, process in chunks
def process_large_image(input_path, chunk_size=1024):
    dataset = gdal.Open(input_path)
    rows, cols = dataset.RasterYSize, dataset.RasterXSize
    
    for row in range(0, rows, chunk_size):
        for col in range(0, cols, chunk_size):
            # Read chunk
            window = dataset.ReadAsArray(col, row, 
                                       min(chunk_size, cols-col),
                                       min(chunk_size, rows-row))
            # Process chunk
            processed = process_chunk(window)
            # Save result
            save_chunk(processed, row, col)
```

### Compression Settings
```python
# Optimal compression for different data types
compression_settings = {
    'complex_data': 'LZW',      # Good for complex patterns
    'amplitude': 'DEFLATE',     # Better for smooth amplitude
    'binary_mask': 'CCITT_T6',  # Best for binary data
    'indices': 'LZW'            # Good for scientific data
}
```

## Common Format Issues and Solutions

### Issue 1: Large File Sizes
**Problem**: SAR products can be very large (>1GB)
**Solution**: Use chunked processing and compression

```python
# Enable chunked reading
gdal.SetConfigOption('GDAL_CACHEMAX', '512')  # 512MB cache
gdal.SetConfigOption('VSI_CACHE', 'TRUE')
```

### Issue 2: Coordinate Reference Systems
**Problem**: Inconsistent CRS between products
**Solution**: Standardize to common projection

```python
from osgeo import osr

# Define target CRS (e.g., UTM)
target_srs = osr.SpatialReference()
target_srs.ImportFromEPSG(32633)  # UTM Zone 33N

# Reproject if needed
gdal.Warp('output_reprojected.tif', 'input.tif', dstSRS=target_srs)
```

### Issue 3: Complex Data Visualization
**Problem**: Complex data cannot be directly displayed
**Solution**: Use amplitude/phase or RGB composite

```python
# Create RGB composite from complex data
def complex_to_rgb(complex_data):
    amplitude = np.abs(complex_data)
    phase = np.angle(complex_data)
    
    # Normalize
    amp_norm = (amplitude / amplitude.max() * 255).astype(np.uint8)
    phase_norm = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    
    # Create RGB (R=amplitude, G=phase, B=constant)
    rgb = np.stack([amp_norm, phase_norm, np.zeros_like(amp_norm)], axis=-1)
    return rgb
```

## Integration with External Tools

### QGIS Integration
- Export to GeoTIFF with proper CRS
- Use .qml files for consistent styling
- Create pyramids for large datasets

### Python Ecosystem
- **rasterio**: Modern raster I/O
- **xarray**: N-dimensional labeled arrays
- **dask**: Parallel processing for large datasets
- **geopandas**: Vector data integration

```python
# Example: Using rasterio for modern I/O
import rasterio
from rasterio.enums import Resampling

with rasterio.open('input.tif') as src:
    # Read with resampling
    data = src.read(out_shape=(src.count, src.height//2, src.width//2),
                   resampling=Resampling.bilinear)
```

For more information on specific format requirements or conversion utilities, see the [API Reference](../api/README.md) or check the [examples](../examples/README.md) for practical implementations.
