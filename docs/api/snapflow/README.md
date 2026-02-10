# Snapflow Module API

The `sarpyx.snapflow` module provides seamless integration with ESA's SNAP (Sentinel Application Platform) Graph Processing Tool (GPT), enabling automated SAR preprocessing, calibration, coregistration, and advanced processing workflows.

## Overview

The SNAP module serves as a Python wrapper for SNAP's command-line GPT tool, allowing users to:
- Execute SNAP processing graphs programmatically
- Chain multiple processing operations
- Handle different SAR missions (Sentinel-1, COSMO-SkyMed, etc.)
- Automate batch processing workflows
- Integrate SNAP preprocessing with sarpyx analysis

## Quick Start

```python
from sarpyx.snapflow.engine import GPT

# Initialize GPT processor
gpt = GPT(
    product='path/to/sentinel1.SAFE',
    outdir='output/',
    format='GEOTIFF'
)

# Chain processing operations
result = (gpt.Calibration(['VV', 'VH'])
             .ThermalNoiseRemoval()
             .Speckle_Filter()
             .TerrainCorrection())

print(f"Processed product: {result}")
```

## GPT Class

### Constructor

```python
GPT(product, outdir, format='BEAM-DIMAP', mode=None)
```

**Parameters:**
- `product` (str | Path): Path to input SAR product
- `outdir` (str | Path): Output directory for processed products
- `format` (str): Output format ('BEAM-DIMAP' or 'GEOTIFF')
- `mode` (str, optional): Platform mode ('Ubuntu', 'MacOS', 'Windows')

**Example:**
```python
gpt = GPT(
    product='/data/S1A_IW_SLC__1SDV_20230101T120000.SAFE',
    outdir='/output/',
    format='GEOTIFF',
    mode='MacOS'
)
```

### Core Processing Methods

#### Calibration
```python
gpt.Calibration(Pols=['VH'], output_complex=True)
```
Applies radiometric calibration to convert digital numbers to backscatter coefficients.

**Parameters:**
- `Pols` (list): Polarizations to process (e.g., ['VV', 'VH'])
- `output_complex` (bool): Whether to output complex data

**Example:**
```python
# Calibrate dual-pol data
calibrated = gpt.Calibration(['VV', 'VH'], output_complex=False)

# Calibrate maintaining complex format for interferometry
calibrated_complex = gpt.Calibration(['VV'], output_complex=True)
```

#### Thermal Noise Removal
```python
gpt.ThermalNoiseRemoval()
```
Removes thermal noise from Sentinel-1 data.

#### Deburst (TOPSAR)
```python
gpt.Deburst(Pols=['VH'])
```
Removes burst boundaries from TOPSAR data.

**Parameters:**
- `Pols` (list): Polarizations to deburst

#### Speckle Filtering
```python
gpt.Speckle_Filter(filterSizeX=5, filterSizeY=5, dampingFactor=2)
```
Applies speckle noise reduction.

**Parameters:**
- `filterSizeX` (int): Filter size in range direction
- `filterSizeY` (int): Filter size in azimuth direction
- `dampingFactor` (int): Damping factor for filter

#### Terrain Correction
```python
gpt.TerrainCorrection(demName='SRTM 3Sec', 
                     demResamplingMethod='BILINEAR_INTERPOLATION',
                     imgResamplingMethod='BILINEAR_INTERPOLATION',
                     pixelSpacingInMeter=10.0,
                     nodataValueAtSea=False)
```
Applies geometric terrain correction and geocoding.

**Parameters:**
- `demName` (str): DEM to use ('SRTM 3Sec', 'SRTM 1Sec', 'ASTER 1sec GDEM')
- `demResamplingMethod` (str): DEM resampling method
- `imgResamplingMethod` (str): Image resampling method  
- `pixelSpacingInMeter` (float): Output pixel spacing
- `nodataValueAtSea` (bool): Set nodata values over sea

#### Multilook
```python
gpt.Multilook(nRgLooks=1, nAzLooks=1)
```
Applies spatial averaging to reduce speckle.

**Parameters:**
- `nRgLooks` (int): Number of looks in range direction
- `nAzLooks` (int): Number of looks in azimuth direction

### Advanced Processing Methods

#### Land-Sea Masking
```python
gpt.LandMask(shoreline_extension=300, 
             geometry_name="Buff_750", 
             use_srtm=True, 
             invert_geometry=True, 
             land_mask=False)
```
Applies land-sea masking using coastline data.

**Parameters:**
- `shoreline_extension` (int): Shoreline buffer distance in meters
- `geometry_name` (str): Geometry buffer name
- `use_srtm` (bool): Use SRTM for elevation-based masking
- `invert_geometry` (bool): Invert the geometry mask
- `land_mask` (bool): Create land mask (True) or sea mask (False)

#### CFAR Ship Detection
```python
gpt.AdaptiveThresholding(background_window_m=800,
                        guard_window_m=500, 
                        target_window_m=50,
                        pfa=6.5)
```
Applies Constant False Alarm Rate (CFAR) detection for ships.

**Parameters:**
- `background_window_m` (float): Background window size in meters
- `guard_window_m` (float): Guard window size in meters
- `target_window_m` (float): Target window size in meters
- `pfa` (float): Probability of false alarm

#### Object Discrimination
```python
gpt.ObjectDiscrimination(min_target_m=50, max_target_m=2000)
```
Filters detected objects by size.

**Parameters:**
- `min_target_m` (float): Minimum target size in meters
- `max_target_m` (float): Maximum target size in meters

### Subsetting and ROI

#### Subset
```python
gpt.Subset(loc=[1000, 1000], 
           sourceBands=['Intensity_VV', 'Intensity_VH'],
           idx='01',
           winSize=128, 
           GeoCoords=False,
           copy_metadata=True)
```
Creates spatial subsets of the data.

**Parameters:**
- `loc` (list): Center location [x, y] or [lon, lat]
- `sourceBands` (list): Bands to include in subset
- `idx` (str): Subset identifier for output naming
- `winSize` (int): Window size in pixels or degrees
- `GeoCoords` (bool): Whether loc is in geographic coordinates
- `copy_metadata` (bool): Whether to copy metadata

#### Vector Import
```python
gpt.ImportVector(vector_data='path/to/shapefile.shp')
```
Imports vector data for masking or analysis.

**Parameters:**
- `vector_data` (str | Path): Path to vector file (shapefile, KML, etc.)

## Usage Examples

### Basic Sentinel-1 Processing

```python
from sarpyx.snapflow.engine import GPT

# Standard Sentinel-1 preprocessing workflow
gpt = GPT(product='S1A_IW_SLC__1SDV.SAFE', outdir='output/')

result = (gpt.Calibration(['VV', 'VH'], output_complex=False)
             .ThermalNoiseRemoval()
             .Deburst(['VV', 'VH'])
             .Speckle_Filter(filterSizeX=7, filterSizeY=7)
             .TerrainCorrection(pixelSpacingInMeter=20))

print(f"Processed GRD product: {result}")
```

### COSMO-SkyMed Processing

```python
# COSMO-SkyMed spotlight mode processing
gpt = GPT(product='CSKS4_SCS_B_S2_08_HH.h5', 
          outdir='output/', 
          mode='Ubuntu')

result = (gpt.Calibration(['HH'], output_complex=False)
             .Speckle_Filter(filterSizeX=5, filterSizeY=5)
             .TerrainCorrection(demName='SRTM 1Sec', 
                              pixelSpacingInMeter=5))
```

### Ship Detection Workflow

```python
# Complete ship detection pipeline
gpt = GPT(product='sentinel1_product.SAFE', outdir='ship_detection/')

# Preprocessing
preprocessed = (gpt.Calibration(['VV'])
                   .ThermalNoiseRemoval()
                   .Deburst(['VV'])
                   .Calibration(['VV']))

# Apply land mask to focus on ocean areas
land_masked = gpt.LandMask(land_mask=False,  # Create sea mask
                          shoreline_extension=500,
                          use_srtm=True)

# CFAR detection
cfar_detected = gpt.AdaptiveThresholding(background_window_m=1000,
                                        guard_window_m=600,
                                        target_window_m=80,
                                        pfa=6.5)

# Filter by ship size
ships = gpt.ObjectDiscrimination(min_target_m=30, max_target_m=1000)

print(f"Ship detection results: {ships}")
```

### Multi-Temporal Processing

```python
# Process time series of SAR data
product_dates = [
    'S1A_IW_SLC__20230101.SAFE',
    'S1A_IW_SLC__20230113.SAFE', 
    'S1A_IW_SLC__20230125.SAFE'
]

processed_products = []

for product in product_dates:
    gpt = GPT(product=product, outdir=f'output/{Path(product).stem}/')
    
    result = (gpt.Calibration(['VV', 'VH'])
                 .Deburst(['VV', 'VH'])
                 .TerrainCorrection(pixelSpacingInMeter=10))
    
    processed_products.append(result)

print(f"Processed {len(processed_products)} products")
```

### Region of Interest Processing

```python
# Process specific geographic regions
areas_of_interest = [
    {'name': 'Area1', 'lon': -74.0, 'lat': 40.7, 'size': 1000},  # 1km window
    {'name': 'Area2', 'lon': -73.9, 'lat': 40.8, 'size': 2000},  # 2km window
]

gpt = GPT(product='large_scene.SAFE', outdir='roi_processing/')

for roi in areas_of_interest:
    # Create subset for each ROI
    subset_result = gpt.Subset(
        loc=[roi['lon'], roi['lat']], 
        sourceBands=['Intensity_VV', 'Intensity_VH'],
        idx=roi['name'],
        winSize=roi['size'],
        GeoCoords=True
    )
    
    print(f"Processed {roi['name']}: {subset_result}")
```

## Batch Processing Functions

### CFAR Batch Processing
```python
from sarpyx.snapflow.engine import CFAR

# Process single product with multiple thresholds
result = CFAR(
    prod='sentinel1_product.SAFE',
    mask_shp_path='coastline.shp', 
    mode='MacOS',
    Thresh=[6.5, 8.0, 10.0],  # Multiple PFA thresholds
    DELETE=True  # Clean up intermediate files
)
```

### Batch Processing Multiple Products
```python
# Process multiple products automatically
products = ['product1.SAFE', 'product2.SAFE', 'product3.SAFE']

for product in products:
    try:
        gpt = GPT(product=product, outdir=f'batch_output/{Path(product).stem}/')
        result = (gpt.Calibration(['VV', 'VH'])
                     .ThermalNoiseRemoval()
                     .TerrainCorrection())
        print(f"✓ Processed: {product}")
    except Exception as e:
        print(f"✗ Failed: {product} - {e}")
```

## Configuration and Platform Support

### Platform-Specific Configuration

```python
# GPT executable paths for different platforms
GPT.GPT_PATHS = {
    'Ubuntu': '/home/user/ESA-STEP/snap/bin/gpt',
    'MacOS': '/Applications/snap/bin/gpt',
    'Windows': 'gpt.exe'
}

# Processing parallelism settings
GPT.DEFAULT_PARALLELISM = {
    'Ubuntu': 24,
    'MacOS': 8, 
    'Windows': 8
}
```

### Custom GPT Configuration

```python
# Initialize with custom settings
gpt = GPT(product=product_path, outdir=output_dir, mode='custom')

# Override executable path
gpt.gpt_executable = '/custom/path/to/gpt'

# Override parallelism
gpt.parallelism = 16
```

## Error Handling and Debugging

### Command Execution Monitoring

```python
gpt = GPT(product=product_path, outdir=output_dir)

# GPT automatically prints executed commands
result = gpt.Calibration(['VV'])
# Output: Executing: /path/to/gpt -q 8 -x -e -Ssource=... Calibration ...
```

### Error Recovery

```python
try:
    result = gpt.TerrainCorrection()
    if result is None:
        print("Terrain correction failed")
        # Implement fallback processing
except Exception as e:
    print(f"Processing error: {e}")
    # Log error and continue with next product
```

### Debugging Processing Issues

```python
# Enable verbose output
import subprocess

# Check SNAP installation
try:
    subprocess.run(['gpt', '-h'], check=True, capture_output=True)
    print("SNAP GPT is properly installed")
except FileNotFoundError:
    print("ERROR: SNAP GPT not found in PATH")
except subprocess.CalledProcessError as e:
    print(f"SNAP GPT error: {e}")
```

## Integration with Other sarpyx Modules

### With Sub-Look Analysis

```python
from sarpyx.snapflow.engine import GPT
from sarpyx.sla import SubLookAnalysis

# Preprocess with SNAP
gpt = GPT(product='sentinel1.SAFE', outdir='preprocessed/')
calibrated = gpt.Calibration(['VV'], output_complex=True)

# Apply SLA to preprocessed data
sla = SubLookAnalysis(calibrated)
sla.choice = 1  # Azimuth processing
sla.numberOfLooks = 3
sla.SpectrumComputation()
sla.Generation()
```

### With Science Applications

```python
from sarpyx.snapflow.engine import GPT
from sarpyx.science import indices

# Preprocess dual-pol data
gpt = GPT(product='sentinel1.SAFE', outdir='science_ready/')
result = (gpt.Calibration(['VV', 'VH'])
             .TerrainCorrection(pixelSpacingInMeter=10))

# Load processed data and calculate vegetation indices
vv_data = load_geotiff(result.replace('.dim', '_VV.tif'))
vh_data = load_geotiff(result.replace('.dim', '_VH.tif'))

rvi = indices.calculate_rvi(vv_data, vh_data)
```

## Performance Optimization

### Parallel Processing

```python
# Maximize parallelism for large products
gpt = GPT(product=large_product, outdir=output_dir)
gpt.parallelism = 32  # Use all available cores

# Process in memory-efficient chunks
result = gpt.TerrainCorrection(pixelSpacingInMeter=20)  # Lower resolution for speed
```

### Intermediate File Management

```python
# Clean up intermediate files automatically
gpt = GPT(product=product_path, outdir=temp_dir)

try:
    result = (gpt.Calibration(['VV'])
                 .TerrainCorrection())
    
    # Copy final result to permanent location
    shutil.copy(result, final_output_dir)
    
finally:
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
```

## See Also

- [User Guide: SNAP Integration](../../user_guide/snap_integration.md): Detailed SNAP setup and workflows
- [User Guide: Processing Workflows](../../user_guide/processing_workflows.md): Complete processing examples
- [Processor Module](../processor/README.md): Core SAR processing algorithms
- [SLA Module](../sla/README.md): Sub-aperture analysis capabilities
