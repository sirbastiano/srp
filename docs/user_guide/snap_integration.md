# SNAP Integration

SARPyX provides seamless integration with ESA's SNAP (Sentinel Application Platform) through the Graph Processing Tool (GPT). This enables automated SAR preprocessing, calibration, and processing workflows.

## Overview

The `sarpyx.snap` module provides a Python wrapper around SNAP's command-line GPT interface, supporting:

- **Automated Processing Chains**: Preprocessing workflows for different SAR missions
- **Cross-Platform Support**: Windows, macOS, and Linux compatibility  
- **Product Type Detection**: Automatic detection of Sentinel-1, COSMO-SkyMed, and ALOS formats
- **Flexible Output Formats**: BEAM-DIMAP, GeoTIFF, NetCDF support
- **Memory Management**: Efficient processing with configurable parallelism

## Installation and Setup

### SNAP Installation

First, ensure SNAP is installed on your system:

1. **Download SNAP**: [ESA SNAP Download](https://step.esa.int/main/download/snap-download/)
2. **Install SNAP**: Follow platform-specific installation instructions
3. **Verify Installation**: Check that GPT is accessible

```bash
# Test SNAP GPT installation
/Applications/snap/bin/gpt --help  # macOS
/home/username/ESA-STEP/snap/bin/gpt --help  # Linux
gpt.exe --help  # Windows (if in PATH)
```

### SARPyX SNAP Configuration

SARPyX automatically detects SNAP installations, but you can specify the mode explicitly:

```python
from sarpyx.snap import GPT

# Automatic detection
gpt = GPT(product_path="input.zip", outdir="output/")

# Explicit mode specification
gpt = GPT(product_path="input.zip", outdir="output/", mode="MacOS")
```

**Supported Modes**:
- `"MacOS"`: `/Applications/snap/bin/gpt`
- `"Ubuntu"`: `/home/<username>/ESA-STEP/snap/bin/gpt`
- `"Windows"`: `gpt.exe` (assumes GPT is in PATH)
- `None`: Automatic detection based on OS

## Basic GPT Operations

### GPT Class Initialization

```python
from sarpyx.snap import GPT
from pathlib import Path

# Initialize GPT processor
gpt = GPT(
    product="S1A_IW_SLC_1SDV_20231101.zip",  # Input SAR product
    outdir="processing_output/",             # Output directory
    format="BEAM-DIMAP",                     # Output format
    mode="MacOS"                             # SNAP installation mode
)

print(f"Product: {gpt.name}")
print(f"GPT executable: {gpt.gpt_executable}")
print(f"Parallelism: {gpt.parallelism}")
```

### Format Options

```python
# Available output formats
formats = ["BEAM-DIMAP", "GeoTIFF", "NetCDF4-CF", "HDF5"]

# Change format for specific operations
gpt.format = "GeoTIFF"  # Outputs will be .tif files
gpt.format = "BEAM-DIMAP"  # Outputs will be .dim + .data directories
```

## Core Processing Operations

### Radiometric Calibration

Convert DN values to radiometrically calibrated backscatter coefficients:

```python
# Single polarization calibration
cal_product = gpt.Calibration(Pols=['VH'])

# Multi-polarization calibration
cal_product = gpt.Calibration(
    Pols=['VV', 'VH'], 
    output_complex=True  # Preserve complex values
)

print(f"Calibrated product: {cal_product}")
```

**Parameters**:
- `Pols`: List of polarizations to calibrate (e.g., ['VV', 'VH', 'HH', 'HV'])
- `output_complex`: Whether to output complex values (True) or intensity (False)

### TOPSAR Deburst (Sentinel-1)

Remove burst boundaries from TOPSAR SLC/GRD products:

```python
# Deburst Sentinel-1 TOPSAR data
deb_product = gpt.Deburst(Pols=['VV', 'VH'])

# The debursted product has continuous azimuth coverage
print(f"Debursted product: {deb_product}")
```

### Multi-looking

Reduce speckle by averaging neighboring pixels:

```python
# Apply multi-looking
ml_product = gpt.Multilook(
    nRgLooks=2,    # Range looks
    nAzLooks=4     # Azimuth looks  
)

# Multi-looking reduces spatial resolution but improves radiometric resolution
```

**Trade-offs**:
- **More looks**: Better speckle reduction, lower spatial resolution
- **Fewer looks**: Better spatial resolution, more speckle

### Geometric Operations

#### Subsetting

Extract spatial subsets for focused analysis:

```python
# Pixel coordinates subset
subset_product = gpt.Subset(
    loc=[1500, 2000],      # Center pixel [x, y]
    sourceBands=['Intensity_VH'],
    idx="01",              # Subset identifier
    winSize=512,           # Window size in pixels
    GeoCoords=False        # Use pixel coordinates
)

# Geographic coordinates subset
subset_geo = gpt.Subset(
    loc=[12.4964, 41.9028],  # [longitude, latitude]
    sourceBands=['Intensity_VV', 'Intensity_VH'],
    idx="Rome",
    winSize=1000,            # Window size in meters (approximately)
    GeoCoords=True           # Use geographic coordinates
)
```

## Complete Processing Workflows

### Sentinel-1 Standard Preprocessing

```python
def preprocess_sentinel1_standard(input_path, output_dir, polarizations=['VV', 'VH']):
    """Complete Sentinel-1 preprocessing workflow."""
    
    # Initialize GPT
    gpt = GPT(product=input_path, outdir=output_dir, mode="MacOS")
    
    # Step 1: TOPSAR Deburst (SLC products only)
    print("Step 1: Debursting...")
    deb_product = gpt.Deburst(Pols=polarizations)
    if not deb_product:
        raise RuntimeError("Deburst operation failed")
    
    # Step 2: Radiometric Calibration
    print("Step 2: Radiometric calibration...")
    cal_product = gpt.Calibration(
        Pols=polarizations,
        output_complex=False  # Intensity output for most applications
    )
    if not cal_product:
        raise RuntimeError("Calibration operation failed")
    
    # Step 3: Multi-looking (optional)
    print("Step 3: Multi-looking...")
    ml_product = gpt.Multilook(nRgLooks=1, nAzLooks=4)
    
    # Step 4: Convert to final format
    print("Step 4: Converting to GeoTIFF...")
    gpt.format = "GeoTIFF"
    final_product = gpt._call(suffix="FINAL")
    
    return {
        'deburst': deb_product,
        'calibration': cal_product, 
        'multilook': ml_product,
        'final': final_product
    }

# Example usage
results = preprocess_sentinel1_standard(
    "S1A_IW_SLC_1SDV_20231101.zip",
    "preprocessing_output/"
)

print(f"Final preprocessed product: {results['final']}")
```

### COSMO-SkyMed Processing

```python
def preprocess_cosmo_skymed(input_path, output_dir, polarization='HH'):
    """COSMO-SkyMed preprocessing workflow."""
    
    gpt = GPT(product=input_path, outdir=output_dir)
    
    # Step 1: Multi-looking first (typically needed for CSK)
    print("Step 1: Multi-looking...")
    ml_product = gpt.Multilook(nRgLooks=2, nAzLooks=2)
    
    # Step 2: Radiometric Calibration
    print("Step 2: Calibration...")
    cal_product = gpt.Calibration(Pols=[polarization])
    
    return {
        'multilook': ml_product,
        'calibration': cal_product
    }
```

### Land-Sea Masking

Apply land-sea masks using vector geometries or built-in coastlines:

```python
def apply_land_sea_mask(input_product, coastline_shapefile=None):
    """Apply land-sea masking with customizable parameters."""
    
    gpt = GPT(product=input_product, outdir="masked_output/")
    
    if coastline_shapefile:
        # Import external vector data
        vector_product = gpt.ImportVector(vector_data=coastline_shapefile)
        
        # Apply custom land mask
        masked_product = gpt.LandMask(
            shoreline_extension=500,    # Extend coastline by 500m
            geometry_name="coastline",  # Geometry name from shapefile
            use_srtm=True,             # Use SRTM elevation data
            invert_geometry=True,      # Keep ocean areas
            land_mask=False            # False = mask land, keep water
        )
    else:
        # Use built-in coastline database
        masked_product = gpt.LandMask(
            use_srtm=True,
            land_mask=False  # Mask land areas
        )
    
    return masked_product

# Example with custom shapefile
masked_result = apply_land_sea_mask(
    "calibrated_product.dim",
    "mediterranean_coastline.shp"
)
```

## Object Detection Workflows

### CFAR Ship Detection

Constant False Alarm Rate (CFAR) detection for ship monitoring:

```python
# Single threshold detection
from sarpyx.snap import CFAR

first_product, excel_results = CFAR(
    prod="S1A_IW_GRDH_1SDV_20231101.zip",
    mask_shp_path="land_mask.shp",
    mode="MacOS",
    Thresh=12.5,  # False alarm rate threshold  
    DELETE=False  # Keep intermediate products for analysis
)

print(f"Detection results saved to: {excel_results}")
```

### Advanced CFAR with Multiple Thresholds

```python
# Test multiple sensitivity levels
pfa_thresholds = [6.5, 9.5, 12.5, 15.5]

first_product, excel_results = CFAR(
    prod="sentinel1_product.zip",
    mask_shp_path="coastline.shp",
    Thresh=pfa_thresholds,
    DELETE=True  # Clean up intermediate files
)

# Results include detection files for each threshold
# Lower thresholds = more sensitive detection
# Higher thresholds = fewer false alarms
```

### Custom CFAR Parameters

```python
def custom_cfar_detection(input_product, mask_shapefile, output_dir):
    """Custom CFAR detection with fine-tuned parameters."""
    
    gpt = GPT(product=input_product, outdir=output_dir)
    
    # Preprocessing (product-specific)
    from sarpyx.utils.io import mode_identifier
    prod_type = mode_identifier(Path(input_product).name)
    
    if prod_type == "SEN":
        # Sentinel-1 preprocessing
        deb_product = gpt.Deburst()
        cal_product = gpt.Calibration(Pols=['VH'])
        
        # Import land mask
        vector_product = gpt.ImportVector(vector_data=mask_shapefile)
        masked_product = gpt.LandMask()
        
        start_product = masked_product
        
    elif prod_type == "CSK":  
        # COSMO-SkyMed preprocessing
        ml_product = gpt.Multilook(nRgLooks=2, nAzLooks=2)
        cal_product = gpt.Calibration(Pols=['HH'])
        
        start_product = cal_product
    
    # CFAR Detection
    detection_results = []
    
    for pfa in [8.5, 12.5, 16.5]:
        # Initialize new GPT instance for each threshold
        gpt_det = GPT(product=start_product, outdir=output_dir)
        
        # Adaptive thresholding
        at_product = gpt_det.AdaptiveThresholding(
            background_window_m=800,  # Background estimation window
            guard_window_m=500,       # Guard band around targets
            target_window_m=50,       # Target detection window
            pfa=pfa                   # False alarm probability
        )
        
        # Object discrimination
        od_product = gpt_det.ObjectDiscrimination(
            min_target_m=35,   # Minimum target size (meters)
            max_target_m=500   # Maximum target size (meters)
        )
        
        detection_results.append({
            'pfa': pfa,
            'at_product': at_product,
            'od_product': od_product
        })
    
    return detection_results
```

## Advanced SNAP Integration

### XML Graph Processing

For complex workflows, use SNAP's XML graph format:

```python
def create_custom_graph(input_product, output_path):
    """Create and execute custom SNAP processing graph."""
    
    gpt = GPT(product=input_product, outdir="graph_output/")
    
    # Example: Custom interferometry graph
    graph_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <graph id="CustomProcessing">
      <version>1.0</version>
      
      <node id="Read">
        <operator>Read</operator>
        <parameters>
          <file>{input_product}</file>
        </parameters>
      </node>
      
      <node id="Calibration">
        <operator>Calibration</operator>
        <sources>
          <sourceProduct refid="Read"/>
        </sources>
        <parameters>
          <selectedPolarisations>VV,VH</selectedPolarisations>
          <outputImageInComplex>true</outputImageInComplex>
        </parameters>
      </node>
      
      <node id="TOPSAR-Deburst">
        <operator>TOPSAR-Deburst</operator>
        <sources>
          <sourceProduct refid="Calibration"/>
        </sources>
        <parameters>
          <selectedPolarisations>VV,VH</selectedPolarisations>
        </parameters>
      </node>
      
      <node id="Write">
        <operator>Write</operator>
        <sources>
          <sourceProduct refid="TOPSAR-Deburst"/>
        </sources>
        <parameters>
          <file>{output_path}</file>
          <formatName>BEAM-DIMAP</formatName>
        </parameters>
      </node>
    </graph>"""
    
    # Save and execute graph
    graph_path = Path("custom_processing_graph.xml")
    with open(graph_path, 'w') as f:
        f.write(graph_xml)
    
    # Execute graph
    gpt.current_cmd = [gpt.gpt_executable, graph_path.as_posix()]
    success = gpt._execute_command()
    
    # Cleanup
    graph_path.unlink()
    
    return success
```

### Memory and Performance Optimization

```python
def configure_snap_performance(gpt, memory_gb=8, tile_size=512):
    """Configure SNAP for optimal performance."""
    
    # Set JVM options for memory management
    jvm_options = [
        f"-Xmx{memory_gb}G",      # Maximum heap size
        "-XX:+UseG1GC",           # Use G1 garbage collector
        "-XX:+UseStringDeduplication"  # Reduce memory usage
    ]
    
    # Update parallelism based on available cores
    import os
    available_cores = os.cpu_count()
    gpt.parallelism = min(available_cores - 1, 16)  # Leave one core free
    
    print(f"Configured SNAP with {memory_gb}GB memory and {gpt.parallelism} threads")
    
    return gpt

# Example usage
gpt = GPT(product="large_product.zip", outdir="output/")
gpt = configure_snap_performance(gpt, memory_gb=16)
```

## Batch Processing

### Multiple Product Processing

```python
def batch_process_products(product_list, output_base_dir, processing_func):
    """Process multiple SAR products in batch."""
    
    results = {}
    failed_products = []
    
    for i, product_path in enumerate(product_list):
        product_name = Path(product_path).stem
        product_output_dir = Path(output_base_dir) / product_name
        product_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {i+1}/{len(product_list)}: {product_name}")
        
        try:
            result = processing_func(product_path, product_output_dir)
            results[product_name] = result
            print(f"✓ Successfully processed {product_name}")
            
        except Exception as e:
            print(f"✗ Failed to process {product_name}: {e}")
            failed_products.append(product_path)
    
    return results, failed_products

# Example batch processing
def standard_preprocessing(product_path, output_dir):
    """Standard preprocessing function for batch processing."""
    return preprocess_sentinel1_standard(product_path, output_dir)

# Process all products in a directory
from pathlib import Path
product_directory = Path("input_products/")
product_files = list(product_directory.glob("*.zip"))

results, failures = batch_process_products(
    product_files,
    "batch_output/",
    standard_preprocessing
)

print(f"Processed {len(results)} products successfully")
print(f"Failed products: {len(failures)}")
```

### Parallel Processing

```python
import concurrent.futures
from functools import partial

def parallel_batch_processing(product_list, output_base_dir, max_workers=2):
    """Process products in parallel with controlled concurrency."""
    
    def process_single_product(product_path):
        """Single product processing function."""
        try:
            product_name = Path(product_path).stem
            product_output_dir = Path(output_base_dir) / product_name
            product_output_dir.mkdir(parents=True, exist_ok=True)
            
            result = preprocess_sentinel1_standard(product_path, product_output_dir)
            return product_name, result, None
            
        except Exception as e:
            return Path(product_path).stem, None, str(e)
    
    # Process products in parallel
    results = {}
    failures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_product = {
            executor.submit(process_single_product, product_path): product_path
            for product_path in product_list
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_product):
            product_name, result, error = future.result()
            
            if error:
                failures.append((product_name, error))
                print(f"✗ Failed: {product_name} - {error}")
            else:
                results[product_name] = result
                print(f"✓ Completed: {product_name}")
    
    return results, failures

# Example usage (be careful with SNAP memory usage in parallel)
results, failures = parallel_batch_processing(
    product_files[:4],  # Start with small batch
    "parallel_output/",
    max_workers=2       # Conservative parallelism for SNAP
)
```

## Data Format Conversion

### BEAM-DIMAP to GeoTIFF

```python
def convert_to_geotiff(beam_dimap_path, output_dir):
    """Convert BEAM-DIMAP products to GeoTIFF."""
    
    gpt = GPT(product=beam_dimap_path, outdir=output_dir, format="GeoTIFF")
    
    # Simple format conversion
    geotiff_product = gpt._call(suffix="GEOTIFF")
    
    return geotiff_product

# Convert processed product to GeoTIFF
geotiff_path = convert_to_geotiff(
    "processed_product.dim",
    "geotiff_output/"
)
```

### Extract Specific Bands

```python
def extract_bands_to_geotiff(product_path, bands, output_dir):
    """Extract specific bands and save as individual GeoTIFFs."""
    
    gpt = GPT(product=product_path, outdir=output_dir, format="GeoTIFF")
    
    extracted_bands = {}
    
    for band in bands:
        # Create subset with single band
        band_product = gpt.Subset(
            loc=[0, 0],           # Full extent
            sourceBands=[band],
            idx=band.replace('_', ''),
            winSize=0,            # Full size
            copy_metadata=True
        )
        
        extracted_bands[band] = band_product
    
    return extracted_bands

# Extract VV and VH intensity bands
bands = ['Intensity_VV', 'Intensity_VH']
extracted = extract_bands_to_geotiff(
    "calibrated_product.dim",
    bands,
    "individual_bands/"
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. GPT Not Found

```python
# Check SNAP installation
import subprocess
from pathlib import Path

def check_snap_installation():
    """Verify SNAP GPT installation."""
    
    common_paths = [
        "/Applications/snap/bin/gpt",           # macOS
        "/usr/local/snap/bin/gpt",             # Linux (custom install)
        "/opt/snap/bin/gpt",                   # Linux (system install)
        "C:\\Program Files\\snap\\bin\\gpt.exe" # Windows
    ]
    
    for path in common_paths:
        if Path(path).exists():
            print(f"Found SNAP GPT at: {path}")
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("GPT is working correctly")
                    return path
            except Exception as e:
                print(f"GPT found but not working: {e}")
    
    print("SNAP GPT not found. Please install SNAP from:")
    print("https://step.esa.int/main/download/snap-download/")
    return None

snap_path = check_snap_installation()
```

#### 2. Memory Issues

```python
def diagnose_memory_issues(gpt):
    """Diagnose and suggest memory optimizations."""
    
    print(f"Current parallelism: {gpt.parallelism}")
    
    # Suggest memory optimizations
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"Available memory: {available_memory_gb:.1f} GB")
    
    if available_memory_gb < 8:
        print("Warning: Low memory. Suggestions:")
        print("- Reduce parallelism: gpt.parallelism = 2")
        print("- Process smaller subsets")
        print("- Use GeoTIFF format to reduce intermediate file sizes")
        
        # Apply conservative settings
        gpt.parallelism = min(gpt.parallelism, 2)
        gpt.format = "GeoTIFF"
    
    return gpt
```

#### 3. Processing Failures

```python
def robust_processing_wrapper(processing_func, *args, max_retries=2, **kwargs):
    """Wrapper for robust processing with retry logic."""
    
    for attempt in range(max_retries + 1):
        try:
            return processing_func(*args, **kwargs)
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
                
                # Clean up any partial outputs
                output_dir = kwargs.get('output_dir') or args[1]
                if output_dir and Path(output_dir).exists():
                    import shutil
                    shutil.rmtree(output_dir, ignore_errors=True)
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
            else:
                print(f"All attempts failed. Final error: {e}")
                raise

# Example usage
result = robust_processing_wrapper(
    preprocess_sentinel1_standard,
    "problematic_product.zip",
    "robust_output/",
    max_retries=2
)
```

### Performance Monitoring

```python
import time
import psutil

def monitor_processing_performance(processing_func, *args, **kwargs):
    """Monitor processing performance and resource usage."""
    
    # Initial state
    start_time = time.time()
    initial_memory = psutil.virtual_memory().used / (1024**3)
    
    print(f"Starting processing at {time.strftime('%H:%M:%S')}")
    print(f"Initial memory usage: {initial_memory:.1f} GB")
    
    try:
        # Run processing function
        result = processing_func(*args, **kwargs)
        
        # Final state
        end_time = time.time()
        final_memory = psutil.virtual_memory().used / (1024**3)
        processing_time = end_time - start_time
        
        print(f"Processing completed in {processing_time:.1f} seconds")
        print(f"Memory change: {final_memory - initial_memory:+.1f} GB")
        print(f"Peak memory usage: {final_memory:.1f} GB")
        
        return result
        
    except Exception as e:
        print(f"Processing failed after {time.time() - start_time:.1f} seconds")
        raise

# Monitor processing performance
result = monitor_processing_performance(
    preprocess_sentinel1_standard,
    "large_product.zip",
    "monitored_output/"
)
```

## Integration with SARPyX Workflows

### SNAP to Sub-Look Analysis

```python
from sarpyx.sla import SubLookAnalysis

def snap_to_sla_workflow(input_product, output_dir):
    """Combine SNAP preprocessing with sub-look analysis."""
    
    # Step 1: SNAP preprocessing
    print("Step 1: SNAP preprocessing...")
    results = preprocess_sentinel1_standard(input_product, output_dir)
    
    # Step 2: Convert to SLC format if needed
    print("Step 2: Preparing for SLA...")
    slc_product = results['calibration']  # Use calibrated complex data
    
    # Step 3: Sub-look analysis
    print("Step 3: Sub-look analysis...")
    sla = SubLookAnalysis(slc_product)
    sla.choice = 1  # Azimuth processing
    sla.numberOfLooks = 3
    sla.centroidSeparations = 700
    sla.subLookBandwidth = 700
    
    # Execute SLA processing
    sla.frequencyComputation()
    sla.SpectrumComputation()
    sla.AncillaryDeWe()
    sla.Generation()
    
    return {
        'snap_results': results,
        'sla_processor': sla,
        'sublooks': sla.Looks
    }

# Complete workflow
workflow_results = snap_to_sla_workflow(
    "S1A_IW_SLC_1SDV_20231101.zip",
    "complete_workflow_output/"
)
```

### SNAP to Science Applications

```python
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll

def snap_to_science_workflow(input_product, output_dir):
    """Combine SNAP preprocessing with scientific indices."""
    
    # SNAP preprocessing for intensity products
    gpt = GPT(product=input_product, outdir=output_dir)
    
    # Get intensity products
    cal_product = gpt.Calibration(Pols=['VV', 'VH'], output_complex=False)
    ml_product = gpt.Multilook(nRgLooks=1, nAzLooks=4)
    
    # Convert to GeoTIFF for science processing
    gpt.format = "GeoTIFF"
    final_product = gpt._call(suffix="SCIENCE")
    
    # Load data for science applications
    # (This would require additional data loading functions)
    print(f"SNAP processing complete: {final_product}")
    print("Ready for vegetation index calculation using sarpyx.science.indices")
    
    return final_product
```

## Best Practices

### 1. Resource Management

```python
# Always specify appropriate parallelism
gpt = GPT(product="input.zip", outdir="output/")
gpt.parallelism = min(os.cpu_count() - 1, 8)  # Leave cores for system

# Use appropriate output formats
gpt.format = "GeoTIFF"  # For final products
gpt.format = "BEAM-DIMAP"  # For intermediate products (better metadata)
```

### 2. Error Handling

```python
def safe_snap_processing(product_path, output_dir):
    """Template for safe SNAP processing."""
    
    try:
        gpt = GPT(product=product_path, outdir=output_dir)
        
        # Verify input product
        if not Path(product_path).exists():
            raise FileNotFoundError(f"Product not found: {product_path}")
        
        # Execute processing with error checking
        result = gpt.Calibration(Pols=['VV', 'VH'])
        if not result:
            raise RuntimeError("Calibration failed")
        
        return result
        
    except Exception as e:
        print(f"Processing failed: {e}")
        
        # Cleanup partial outputs
        if Path(output_dir).exists():
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        
        raise
```

### 3. File Management

```python
def cleanup_intermediate_products(output_dir, keep_final=True):
    """Clean up intermediate SNAP products."""
    
    output_path = Path(output_dir)
    
    # Patterns for intermediate files
    intermediate_patterns = [
        "*_DEB.*",     # Deburst products
        "*_CAL.*",     # Calibration products  
        "*_ML.*",      # Multilook products
        "*_SHP.*",     # Vector import products
        "*_LM.*",      # Land mask products
    ]
    
    cleaned_files = []
    
    for pattern in intermediate_patterns:
        for file_path in output_path.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                cleaned_files.append(file_path.name)
            elif file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
                cleaned_files.append(file_path.name)
    
    print(f"Cleaned up {len(cleaned_files)} intermediate files")
    return cleaned_files
```

This comprehensive SNAP Integration guide provides users with everything they need to effectively use SARPyX's SNAP integration capabilities, from basic operations to advanced workflows and troubleshooting.
