# Tutorial 2: SNAP Integration Basics

**Duration**: 20 minutes  
**Prerequisites**: SNAP installation, Tutorial 1 completed  
**Data**: Sentinel-1 GRD product

This tutorial introduces you to sarpyx's SNAP integration capabilities, showing how to automate common preprocessing workflows using the Graph Processing Tool (GPT).

## Learning Objectives

By the end of this tutorial, you will:
- Understand sarpyx's SNAP integration architecture
- Configure SNAP GPT for use with sarpyx
- Execute basic preprocessing workflows
- Chain multiple processing operations
- Handle different output formats
- Troubleshoot common SNAP integration issues

## Prerequisites Check

Before starting, ensure you have:

```python
# Check SNAP installation
import subprocess
from pathlib import Path

try:
    result = subprocess.run(['gpt', '-h'], capture_output=True, text=True)
    print("✓ SNAP GPT is accessible")
    print(f"SNAP version info: {result.stdout.split('\\n')[0]}")
except FileNotFoundError:
    print("✗ SNAP GPT not found. Please install SNAP and add to PATH")
    # See installation guide: docs/user_guide/snap_integration.md

# Check sarpyx SNAP module
try:
    from sarpyx.snapflow.engine import GPT
    print("✓ sarpyx SNAP module imported successfully")
except ImportError as e:
    print(f"✗ sarpyx import error: {e}")
```

## Step 1: Setting Up Your Data

For this tutorial, we'll use a Sentinel-1 GRD (Ground Range Detected) product. If you don't have one, you can:

1. Download from [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
2. Use the sample data from Tutorial 1
3. Use any Sentinel-1 IW GRD product

```python
from pathlib import Path
from sarpyx.snapflow.engine import GPT

# Define data paths
data_dir = Path('./data')
input_product = data_dir / 'S1A_IW_GRDH_1SDV_20240621T052251.SAFE'
output_dir = Path('./tutorial2_outputs')

# Create output directory
output_dir.mkdir(exist_ok=True)

print(f"Input product: {input_product}")
print(f"Output directory: {output_dir}")
print(f"Product exists: {input_product.exists()}")
```

## Step 2: Basic GPT Initialization

The `GPT` class is your interface to SNAP's processing capabilities:

```python
# Initialize GPT processor
gpt = GPT(
    product=input_product,
    outdir=output_dir,
    format='GEOTIFF',  # Output as GeoTIFF for easy visualization
    mode='MacOS'       # Adjust for your platform: 'Ubuntu', 'Windows', or None
)

print(f"GPT executable: {gpt.gpt_executable}")
print(f"Parallelism: {gpt.parallelism}")
print(f"Output format: {gpt.format}")
```

**Platform Configuration:**
- **MacOS**: Uses `/Applications/snap/bin/gpt`
- **Ubuntu**: Uses `/home/<username>/ESA-STEP/snap/bin/gpt`  
- **Windows**: Uses `gpt.exe` (assumes in PATH)
- **None**: Auto-detect or use default `gpt`

## Step 3: Single Processing Operations

Let's start with individual processing steps:

### Calibration

```python
# Apply radiometric calibration
print("Applying radiometric calibration...")
calibrated_path = gpt.Calibration(
    Pols=['VV', 'VH'],        # Process both polarizations
    output_complex=False       # Output intensity values
)

print(f"Calibrated product saved: {calibrated_path}")

# Examine the command that was executed
print("\\nGPT command executed:")
print(" ".join(gpt.current_cmd))
```

**What this does:**
- Converts digital numbers to backscatter coefficients (σ°)
- Applies antenna pattern correction
- Applies range spreading loss correction
- Outputs calibrated intensity values

### Speckle Filtering

```python
# Reset GPT to work with calibrated product
gpt_filter = GPT(
    product=calibrated_path,
    outdir=output_dir,
    format='GEOTIFF'
)

# Apply speckle filtering
print("\\nApplying speckle filtering...")
filtered_path = gpt_filter.Speckle_Filter(
    filterSizeX=7,            # 7x7 filter window
    filterSizeY=7,
    dampingFactor=2           # Controls filter strength
)

print(f"Filtered product saved: {filtered_path}")
```

### Terrain Correction

```python
# Apply geometric terrain correction
gpt_terrain = GPT(
    product=filtered_path,
    outdir=output_dir,
    format='GEOTIFF'
)

print("\\nApplying terrain correction...")
terrain_corrected_path = gpt_terrain.TerrainCorrection(
    demName='SRTM 3Sec',                    # Digital Elevation Model
    pixelSpacingInMeter=10.0,               # Output resolution: 10m
    demResamplingMethod='BILINEAR_INTERPOLATION',
    imgResamplingMethod='BILINEAR_INTERPOLATION'
)

print(f"Terrain corrected product saved: {terrain_corrected_path}")
```

## Step 4: Chained Processing Operations

The real power of sarpyx SNAP integration comes from chaining operations:

```python
# Create new GPT instance for chained processing
gpt_chain = GPT(
    product=input_product,
    outdir=output_dir / 'chained',
    format='GEOTIFF'
)

print("\\nExecuting chained processing workflow...")

# Chain multiple operations
final_result = (gpt_chain
               .Calibration(['VV', 'VH'], output_complex=False)
               .Speckle_Filter(filterSizeX=5, filterSizeY=5)
               .TerrainCorrection(pixelSpacingInMeter=20.0))

print(f"Final chained result: {final_result}")
```

**Benefits of chaining:**
- Fewer intermediate files
- More efficient processing
- Cleaner workflow code
- Automatic file path management

## Step 5: Working with Different Output Formats

### BEAM-DIMAP Format

```python
# Process with BEAM-DIMAP format (SNAP native)
gpt_dimap = GPT(
    product=input_product,
    outdir=output_dir / 'dimap_output',
    format='BEAM-DIMAP'  # Native SNAP format
)

dimap_result = gpt_dimap.Calibration(['VV'])
print(f"BEAM-DIMAP result: {dimap_result}")
print(f"Associated files: {list(Path(dimap_result).parent.glob('*'))}")
```

### Handling Multi-band GeoTIFF

```python
# The GeoTIFF output contains both polarizations
import rasterio
import numpy as np

# Read the multi-band GeoTIFF
with rasterio.open(final_result) as src:
    print(f"Number of bands: {src.count}")
    print(f"Band descriptions: {src.descriptions}")
    print(f"Data type: {src.dtypes[0]}")
    print(f"Dimensions: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    
    # Read specific bands
    vv_band = src.read(1)  # Usually VV is band 1
    vh_band = src.read(2)  # Usually VH is band 2

print(f"VV data shape: {vv_band.shape}")
print(f"VH data shape: {vh_band.shape}")
print(f"VV data range: {np.min(vv_band):.3f} to {np.max(vv_band):.3f}")
```

## Step 6: Visualization and Quality Check

Let's visualize our processed results:

```python
import matplotlib.pyplot as plt
from sarpyx.utils import show_image

# Convert to dB scale for visualization
vv_db = 10 * np.log10(vv_band + 1e-10)  # Add small value to avoid log(0)
vh_db = 10 * np.log10(vh_band + 1e-10)

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# VV polarization
show_image(vv_db, 'VV Backscatter (dB)', 
          vmin=-25, vmax=0, cmap='gray', ax=axes[0])

# VH polarization  
show_image(vh_db, 'VH Backscatter (dB)',
          vmin=-30, vmax=-5, cmap='gray', ax=axes[1])

# RGB composite (VV-red, VH-green, VV/VH-blue)
rgb_composite = np.stack([
    np.clip((vv_db + 25) / 25, 0, 1),  # VV: -25 to 0 dB
    np.clip((vh_db + 30) / 25, 0, 1),  # VH: -30 to -5 dB  
    np.clip(vv_db - vh_db, 0, 20) / 20  # VV-VH difference
], axis=-1)

axes[2].imshow(rgb_composite)
axes[2].set_title('RGB Composite')
axes[2].set_xlabel('Range')
axes[2].set_ylabel('Azimuth')

plt.tight_layout()
plt.savefig(output_dir / 'snap_processing_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'snap_processing_results.png'")
```

## Step 7: Advanced Processing - Subsetting

For focused analysis, you often need to work with specific regions:

```python
# Define region of interest (ROI)
# These coordinates are in image coordinates (pixel, line)
roi_center = [2500, 1500]  # Center pixel coordinates
roi_size = 512             # Size in pixels

# Create subset
gpt_subset = GPT(
    product=input_product,
    outdir=output_dir / 'subset',
    format='GEOTIFF'
)

print(f"\\nCreating subset centered at {roi_center}...")

subset_result = (gpt_subset
                .Calibration(['VV', 'VH'])
                .Subset(
                    loc=roi_center,
                    sourceBands=['Intensity_VV', 'Intensity_VH'],
                    idx='ROI_01',
                    winSize=roi_size,
                    GeoCoords=False  # Using pixel coordinates
                ))

print(f"Subset result: {subset_result}")
```

### Geographic Subsetting

```python
# Subset using geographic coordinates
gpt_geo_subset = GPT(
    product=calibrated_path,  # Start from calibrated data
    outdir=output_dir / 'geo_subset',
    format='GEOTIFF'
)

# Define geographic ROI (longitude, latitude)
geo_center = [12.4924, 41.8902]  # Rome, Italy (example)
geo_size = 0.01  # Approximately 1 km

geographic_subset = gpt_geo_subset.Subset(
    loc=geo_center,
    sourceBands=['Intensity_VV', 'Intensity_VH'],
    idx='Rome',
    winSize=geo_size,
    GeoCoords=True  # Using geographic coordinates
)

print(f"Geographic subset result: {geographic_subset}")
```

## Step 8: Error Handling and Troubleshooting

Robust processing requires proper error handling:

```python
def robust_snap_processing(input_path, output_dir, workflow_name):
    """Robust SNAP processing with error handling."""
    
    try:
        # Initialize GPT
        gpt = GPT(
            product=input_path,
            outdir=output_dir / workflow_name,
            format='GEOTIFF'
        )
        
        print(f"Starting {workflow_name} processing...")
        
        # Execute processing with error checking
        result = gpt.Calibration(['VV', 'VH'])
        if result is None:
            raise RuntimeError("Calibration failed")
            
        result = gpt.TerrainCorrection()
        if result is None:
            raise RuntimeError("Terrain correction failed")
            
        print(f"✓ {workflow_name} completed successfully: {result}")
        return result
        
    except FileNotFoundError:
        print("✗ SNAP GPT not found. Check SNAP installation.")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"✗ SNAP processing error: {e}")
        print("Check SNAP logs for details")
        return None
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

# Test robust processing
robust_result = robust_snap_processing(
    input_product, 
    output_dir, 
    'robust_test'
)
```

## Step 9: Integration with sarpyx Analysis

Now let's integrate SNAP preprocessing with sarpyx analysis:

```python
from sarpyx.science import indices

# Load the SNAP-processed data
with rasterio.open(final_result) as src:
    vv_linear = src.read(1)  # Linear scale backscatter
    vh_linear = src.read(2)

# Calculate vegetation indices
print("\\nCalculating vegetation indices from SNAP-processed data...")

rvi = indices.calculate_rvi(vv_linear, vh_linear)
ndpoll = indices.calculate_ndpoll(vv_linear, vh_linear)
dprvi = indices.calculate_dprvi_vv(vv_linear, vh_linear)

# Visualize integrated results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

show_image(10*np.log10(vv_linear), 'VV (dB)', ax=axes[0,0])
show_image(rvi, 'RVI', vmin=0, vmax=2, cmap='RdYlGn', ax=axes[0,1])
show_image(ndpoll, 'NDPoll', vmin=-1, vmax=1, cmap='RdBu_r', ax=axes[1,0])
show_image(dprvi, 'DpRVI', vmin=0, vmax=2, cmap='viridis', ax=axes[1,1])

plt.tight_layout()
plt.savefig(output_dir / 'snap_sarpyx_integration.png', dpi=300)
plt.show()

print("Integrated analysis complete!")
```

## Step 10: Batch Processing

For operational workflows, you'll often need to process multiple products:

```python
def batch_snap_processing(product_list, base_output_dir):
    """Process multiple SAR products with SNAP."""
    
    results = []
    
    for i, product_path in enumerate(product_list):
        product_name = Path(product_path).stem
        product_output_dir = base_output_dir / f"product_{i:03d}_{product_name}"
        
        print(f"\\nProcessing {i+1}/{len(product_list)}: {product_name}")
        
        try:
            gpt = GPT(
                product=product_path,
                outdir=product_output_dir,
                format='GEOTIFF'
            )
            
            result = (gpt.Calibration(['VV', 'VH'])
                        .Speckle_Filter()
                        .TerrainCorrection(pixelSpacingInMeter=20))
            
            results.append({
                'input': product_path,
                'output': result,
                'status': 'success'
            })
            
            print(f"✓ Success: {result}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'input': product_path,
                'output': None,
                'status': 'failed',
                'error': str(e)
            })
    
    return results

# Example batch processing (uncomment if you have multiple products)
# product_list = [
#     './data/product1.SAFE',
#     './data/product2.SAFE',
#     './data/product3.SAFE'
# ]
# 
# batch_results = batch_snap_processing(product_list, output_dir / 'batch')
# 
# # Summary
# successful = sum(1 for r in batch_results if r['status'] == 'success')
# print(f"\\nBatch processing complete: {successful}/{len(batch_results)} successful")
```

## Summary and Next Steps

In this tutorial, you learned:

✅ **SNAP Integration Basics**
- Configuring GPT for different platforms
- Basic processing operations (calibration, filtering, terrain correction)
- Working with different output formats

✅ **Workflow Management**
- Chaining processing operations
- Error handling and troubleshooting
- Robust processing patterns

✅ **Data Integration**
- Loading and visualizing SNAP outputs
- Integrating with sarpyx science modules
- Quality assessment techniques

✅ **Advanced Features**
- Spatial and geographic subsetting
- Batch processing workflows
- Performance optimization

### What You've Accomplished

You now have processed SAR data that is:
- Radiometrically calibrated
- Speckle filtered
- Geometrically corrected
- Ready for scientific analysis

### Next Steps

**Recommended next tutorials:**
1. **[Tutorial 3: Visualization and Quality Assessment](03_visualization_quality.md)** - Learn advanced visualization techniques
2. **[Tutorial 4: Multi-temporal Analysis](04_multitemporal_analysis.md)** - Work with time series data
3. **[Tutorial 5: Polarimetric Analysis](05_polarimetric_analysis.md)** - Dive deeper into dual-pol analysis

**Advanced topics to explore:**
- Interferometric processing with SNAP
- CFAR ship detection workflows
- Custom XML graph processing
- Integration with cloud processing platforms

### Files Created

This tutorial created several output files:
```
tutorial2_outputs/
├── S1A_IW_GRDH_CAL.tif           # Calibrated data
├── S1A_IW_GRDH_CAL_SF.tif        # Speckle filtered
├── S1A_IW_GRDH_CAL_SF_TC.tif     # Terrain corrected
├── chained/                       # Chained processing results
├── snap_processing_results.png    # Visualization
└── snap_sarpyx_integration.png    # Integrated analysis
```

### Troubleshooting Tips

**Common Issues:**
1. **SNAP not found**: Check SNAP installation and PATH configuration
2. **Memory errors**: Reduce processing resolution or increase available memory
3. **DEM download failures**: Check internet connection and SNAP configuration
4. **Coordinate reference system errors**: Ensure input data has proper CRS information

**Performance Tips:**
1. Use appropriate output resolution for your analysis needs
2. Consider processing subsets for initial testing
3. Use BEAM-DIMAP format for intermediate processing steps
4. Monitor system memory usage during large product processing

Keep experimenting with different processing parameters and workflows to optimize for your specific applications!
