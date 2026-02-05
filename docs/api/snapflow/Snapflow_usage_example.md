# GPT Class Usage Guide

## Overview

The `GPT` class is a Python wrapper for executing SNAP Graph Processing Tool (GPT) commands. It provides a streamlined interface for SAR data processing operations.

## Table of Contents

1. [Basic Initialization](#basic-initialization)
2. [Single Operator Examples](#single-operator-examples)
3. [Chaining Operations](#chaining-operations)
4. [Complete Processing Workflows](#complete-processing-workflows)
5. [Advanced Examples](#advanced-examples)

---

## Basic Initialization

### Minimum Required Parameters

```python
from pathlib import Path
from sarpyx.snap.engine import GPT

# Basic initialization
product_path = Path('/path/to/your/S1A_IW_GRDH_product.SAFE')
output_dir = Path('/path/to/output/')

op = GPT(
    product=product_path,
    outdir=output_dir
)
```

### Full Initialization with All Options

```python
op = GPT(
    product='/path/to/product.SAFE',
    outdir='/path/to/output/',
    format='BEAM-DIMAP',  # or 'GeoTIFF', 'ENVI', etc.
    gpt_path='/usr/local/snap/bin/gpt',
    mode='Ubuntu'  # or 'MacOS', 'Windows'
)
```

### Supported Output Formats

The class supports 40+ output formats including:
- `BEAM-DIMAP` (default) - SNAP native format
- `GeoTIFF` - Standard GeoTIFF
- `ENVI` - ENVI format
- `HDF5` - Hierarchical Data Format
- `NetCDF4-CF` - NetCDF Climate conventions
- And many more...

---

## Single Operator Examples

### 1. Calibration

```python
from sarpyx.snap.engine import GPT

op = GPT(
    product='S1A_IW_GRDH_product.SAFE',
    outdir='./output/'
)

# Calibrate VV polarization
calibrated_product = op.Calibration(
    Pols=['VV'],
    output_complex=True
)

print(f"Calibrated product: {calibrated_product}")
```

### 2. Debursting (TOPSAR-Deburst)

```python
# Deburst for VH polarization
debursted_product = op.Deburst(Pols=['VH'])
```

### 3. Multilooking

```python
# Apply 2x2 multilooking
multilooked_product = op.Multilook(
    nRgLooks=2,
    nAzLooks=2
)
```

### 4. Land-Sea Masking

```python
# Apply land mask using SRTM
masked_product = op.LandMask(
    shoreline_extension=300,
    geometry_name='Buff_750',
    use_srtm=True,
    invert_geometry=True,
    land_mask=False
)
```

### 5. Adaptive Thresholding (CFAR)

```python
# Apply CFAR detection
cfar_product = op.AdaptiveThresholding(
    background_window_m=800,
    guard_window_m=500,
    target_window_m=50,
    pfa=6.5
)
```

### 6. Object Discrimination

```python
# Filter detections by size
filtered_product = op.ObjectDiscrimination(
    min_target_m=35,
    max_target_m=500
)
```

### 7. Subset Extraction

```python
# Subset by pixel coordinates
subset_product = op.Subset(
    loc=[1000, 2000],  # x, y pixel coordinates
    sourceBands=['Intensity_VV', 'Intensity_VH'],
    idx='001',
    winSize=128,
    GeoCoords=False
)

# Subset by geographic coordinates
geo_subset = op.Subset(
    loc=[10.5, 45.3],  # longitude, latitude
    sourceBands=['Intensity_VV'],
    idx='002',
    winSize=256,
    GeoCoords=True
)
```

### 8. Import Vector Data

```python
# Import shapefile for masking
vector_product = op.ImportVector(
    vector_data='./shapefiles/study_area.shp'
)
```

### 9. Sea Surface Temperature (SST)

```python
# Process ATSR data for SST
sst_product = op.AatsrSST(
    dual=True,
    nadir=True,
    invalid_sst_value=-999.0
)
```

---

## Chaining Operations

The key to chaining is that each operation **updates** `op.prod_path` internally, so subsequent calls work on the output of the previous operation.

### Example 1: Basic Chain

```python
from sarpyx.snap.engine import GPT

# Initialize with raw product
op = GPT(
    product='S1A_IW_SLC_product.SAFE',
    outdir='./output/'
)

# Step 1: Deburst
op.Deburst(Pols=['VV', 'VH'])

# Step 2: Calibrate (now works on debursted product)
op.Calibration(Pols=['VV', 'VH'], output_complex=False)

# Step 3: Multilook (now works on calibrated product)
op.Multilook(nRgLooks=4, nAzLooks=1)

print(f"Final product: {op.prod_path}")
```

### Example 2: Sentinel-1 Preprocessing Chain

```python
from pathlib import Path
from sarpyx.snap.engine import GPT

input_product = Path('./data/S1A_IW_SLC_20240503.SAFE')
output_dir = Path('./processed/')
shapefile = Path('./masks/land_mask.shp')

# Initialize
op = GPT(product=input_product, outdir=output_dir)

# Processing chain
print("Step 1: Debursting...")
op.Deburst(Pols=['VH'])

print("Step 2: Calibration...")
op.Calibration(Pols=['VH'], output_complex=False)

print("Step 3: Import vector mask...")
op.ImportVector(vector_data=shapefile)

print("Step 4: Apply land mask...")
op.LandMask(
    shoreline_extension=300,
    use_srtm=True,
    invert_geometry=True
)

print(f"✓ Processing complete: {op.prod_path}")
```

### Example 3: Error Handling in Chains

```python
from sarpyx.snap.engine import GPT

op = GPT(product='input.SAFE', outdir='./output/')

# Each method returns the output path or None on failure
deburst_result = op.Deburst(Pols=['VV'])
if not deburst_result:
    print("Deburst failed!")
    exit(1)

cal_result = op.Calibration(Pols=['VV'])
if not cal_result:
    print("Calibration failed!")
    exit(1)

ml_result = op.Multilook(nRgLooks=2, nAzLooks=2)
if not ml_result:
    print("Multilook failed!")
    exit(1)

print("✓ All operations successful!")
```

---

## Complete Processing Workflows

### Workflow 1: Ship Detection for Sentinel-1

```python
from pathlib import Path
from sarpyx.snap.engine import GPT, CFAR
import pandas as pd

# Input data
product = Path('./data/S1A_IW_GRDH_20240503.SAFE')
land_mask = Path('./masks/coastline.shp')
output_dir = Path('./ship_detection/')
output_dir.mkdir(exist_ok=True)

# Initialize
op = GPT(product=product, outdir=output_dir)

# Preprocessing
print("Preprocessing...")
op.Deburst(Pols=['VH'])
op.Calibration(Pols=['VH'], output_complex=False)
op.ImportVector(vector_data=land_mask)
op.LandMask(use_srtm=True, invert_geometry=True)

# Ship detection
print("Detecting ships...")
op.AdaptiveThresholding(
    background_window_m=800,
    guard_window_m=500,
    target_window_m=50,
    pfa=6.5
)

# Filter detections
final_product = op.ObjectDiscrimination(
    min_target_m=35,
    max_target_m=500
)

# Extract results
product_path = Path(final_product)
data_dir = product_path.with_suffix('.data')
csv_files = list(data_dir.glob('*ship*.csv'))

if csv_files:
    detections = pd.read_csv(csv_files[0], header=1, sep='\t')
    excel_output = output_dir / 'ship_detections.xlsx'
    detections.to_excel(excel_output, index=False)
    print(f"✓ Detections saved: {excel_output}")
    print(f"✓ Found {len(detections)} ships")
```

### Workflow 2: Using the High-Level CFAR Function

```python
from pathlib import Path
from sarpyx.snap.engine import CFAR

# Single PFA threshold
first_product, excel_file = CFAR(
    prod='./data/S1A_IW_GRDH_product.SAFE',
    mask_shp_path='./masks/land_mask.shp',
    mode='Ubuntu',
    Thresh=6.5,
    DELETE=True  # Delete intermediate products
)

print(f"Processed product: {first_product}")
print(f"Detection results: {excel_file}")
```

### Workflow 3: Multiple PFA Thresholds

```python
from sarpyx.snap.engine import CFAR

# Test multiple PFA values
pfa_values = [4.5, 6.5, 8.5, 10.5, 12.5]

first_product, last_excel = CFAR(
    prod='./data/COSMO_SkyMed_product.h5',
    mask_shp_path='./masks/mediterranean.shp',
    mode='Ubuntu',
    Thresh=pfa_values,
    DELETE=False  # Keep all intermediate products
)

# This creates 5 Excel files with different detection thresholds:
# - COSMO_SkyMed_product_pfa_4.5.xlsx
# - COSMO_SkyMed_product_pfa_6.5.xlsx
# - ...
```

---

## Advanced Examples

### Example 1: Processing Multiple Products

```python
from pathlib import Path
from sarpyx.snap.engine import GPT

products = [
    Path('./data/S1A_20240501.SAFE'),
    Path('./data/S1A_20240502.SAFE'),
    Path('./data/S1A_20240503.SAFE'),
]

output_dir = Path('./batch_output/')
land_mask = Path('./masks/study_area.shp')

for product in products:
    print(f"\nProcessing: {product.name}")
    
    op = GPT(product=product, outdir=output_dir)
    
    try:
        op.Deburst(Pols=['VH'])
        op.Calibration(Pols=['VH'])
        op.ImportVector(vector_data=land_mask)
        result = op.LandMask()
        
        print(f"✓ Success: {result}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        continue
```

### Example 2: Custom Format Output

```python
from sarpyx.snap.engine import GPT

# Output as GeoTIFF
op = GPT(
    product='input.SAFE',
    outdir='./output/',
    format='GeoTIFF'
)

op.Calibration(Pols=['VV'])
op.Multilook(nRgLooks=2, nAzLooks=2)

# Output will be .tif files instead of .dim
```

### Example 3: Region of Interest Processing

```python
from sarpyx.snap.engine import GPT

op = GPT(product='large_scene.SAFE', outdir='./rois/')

# Define multiple ROIs
rois = [
    {'loc': [1000, 2000], 'idx': 'harbor'},
    {'loc': [3000, 1500], 'idx': 'bay'},
    {'loc': [2000, 3000], 'idx': 'strait'},
]

# Extract each ROI
for roi in rois:
    subset = op.Subset(
        loc=roi['loc'],
        sourceBands=['Intensity_VV', 'Intensity_VH'],
        idx=roi['idx'],
        winSize=256,
        GeoCoords=False
    )
    print(f"Extracted ROI: {subset}")
```

### Example 4: COSMO-SkyMed Processing

```python
from sarpyx.snap.engine import GPT
from pathlib import Path

# COSMO-SkyMed requires different processing
product = Path('./data/COSMO_SkyMed_HDF5.h5')
output_dir = Path('./cosmo_output/')

op = GPT(product=product, outdir=output_dir)
op.prod_type = 'COSMO-SkyMed'

# COSMO workflow
print("Multilooking...")
op.Multilook(nRgLooks=2, nAzLooks=2)

print("Calibration...")
op.Calibration(Pols=['HH'], output_complex=False)

print("Land masking...")
op.ImportVector(vector_data='./masks/land.shp')
op.LandMask()

print("CFAR detection...")
op.AdaptiveThresholding(
    background_window_m=650,
    guard_window_m=400,
    target_window_m=25,
    pfa=6.5
)

op.ObjectDiscrimination(min_target_m=35, max_target_m=500)

print(f"✓ Complete: {op.prod_path}")
```

---

## Tips and Best Practices

### 1. **Check Operation Success**

```python
result = op.Calibration(Pols=['VV'])
if result:
    print(f"Success: {result}")
else:
    print("Operation failed!")
```

### 2. **Clean Up Intermediate Files**

```python
from sarpyx.utils.io import delProd

# Save intermediate paths
deb_path = op.Deburst(Pols=['VH'])
cal_path = op.Calibration(Pols=['VH'])
final_path = op.Multilook(nRgLooks=2, nAzLooks=2)

# Delete intermediate products
if deb_path:
    delProd(deb_path)
if cal_path:
    delProd(cal_path)
```

### 3. **Use Absolute Paths**

```python
from pathlib import Path

# Good: absolute paths
product = Path('/full/path/to/product.SAFE').resolve()
output = Path('./output/').resolve()

op = GPT(product=product, outdir=output)
```

### 4. **Specify Product Type for Mixed Workflows**

```python
op = GPT(product='product.SAFE', outdir='./output/')
op.prod_type = 'Sentinel-1'  # or 'COSMO-SkyMed', 'SAOCOM'

# Now operations can adapt to product type
op.LandMask()
```

---

## Common Issues and Solutions

### Issue 1: GPT Not Found

**Error:** `GPT executable not found`

**Solution:**
```python
# Explicitly specify GPT path
op = GPT(
    product='input.SAFE',
    outdir='./output/',
    gpt_path='/home/username/snap/bin/gpt'
)
```

### Issue 2: Out of Memory

**Solution:** Adjust GPT parallelism or use XML graphs for complex operations

```python
# The parallelism is set automatically based on OS
# For manual control, modify after initialization
op = GPT(product='input.SAFE', outdir='./output/')
# Lower parallelism reduces memory usage
```

### Issue 3: Operation Timeout

**Solution:** The default timeout is 1 hour. For large products, operations may need more time. Consider processing smaller subsets or adjusting system resources.

---

## Summary

The `GPT` class provides a clean Python interface to SNAP processing:

1. **Initialize** with product and output directory
2. **Chain operations** by calling methods sequentially
3. **Each method** updates the internal product path automatically
4. **Check results** - methods return output path or `None` on failure
5. **Use high-level functions** like `CFAR()` for complete workflows

For more information, see the [SNAP documentation](https://step.esa.int/main/doc/) and the source code.
