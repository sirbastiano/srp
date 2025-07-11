# Processing Workflows

This guide covers the various processing workflows available in SARPyX, from basic sub-look analysis to advanced SNAP-integrated processing chains for ship detection and scientific applications.

## Overview

SARPyX provides several processing pathways depending on your objectives:

1. **Sub-Look Analysis (SLA)**: Core functionality for azimuthal/range frequency decomposition
2. **SNAP Integration**: SAR preprocessing using SNAP GPT (Graph Processing Tool)
3. **Scientific Applications**: Vegetation indices, polarimetric analysis
4. **Object Detection**: CFAR-based ship detection workflows

## Sub-Look Analysis Workflow

The Sub-Look Analysis is SARPyX's core functionality for performing azimuthal sub-band partitioning.

### Basic SLA Pipeline

```python
from sarpyx.sla import SubLookAnalysis

# Initialize with SAR product
sla = SubLookAnalysis("S1A_IW_SLC_product.zip")

# Configure parameters
sla.choice = 1                    # 0=Range, 1=Azimuth
sla.numberOfLooks = 3             # Number of sub-looks
sla.centroidSeparations = 700     # Frequency separation (Hz)
sla.subLookBandwidth = 700        # Bandwidth per sub-look (Hz)

# Execute processing chain
sla.frequencyComputation()        # Compute frequency bins
sla.SpectrumComputation()         # Compute spectrum
sla.AncillaryDeWe()              # Apply de-weighting
sla.Generation()                  # Generate sub-look images

# Access results
sublook_images = sla.Looks        # Complex sub-look images [n_looks, height, width]
frequencies = sla.freqCentr       # Center frequencies for each look
```

### Step-by-Step SLA Process

#### 1. Initialization and Parameter Setup
```python
# Initialize SubLookAnalysis
sla = SubLookAnalysis(product_path)

# Key parameters
sla.choice = 1                    # Processing direction (azimuth recommended)
sla.numberOfLooks = 3             # Typically 3-5 for good balance
sla.centroidSeparations = 700     # Depends on platform and application
sla.subLookBandwidth = 700        # Should match PRF characteristics
sla.choiceDeWe = 0               # De-weighting method (0=ancillary, 1=average)
```

#### 2. Frequency Computation
```python
# Compute frequency bins and validate parameters
try:
    sla.frequencyComputation()
    print(f"Generated {len(sla.freqCentr)} frequency bins:")
    for i, freq in enumerate(sla.freqCentr):
        print(f"  Look {i+1}: {freq:.1f} Hz")
        print(f"    Range: [{sla.freqMin[i]:.1f}, {sla.freqMax[i]:.1f}] Hz")
except AssertionError as e:
    print(f"Frequency computation failed: {e}")
    # Adjust parameters and retry
```

#### 3. Spectrum Computation
```python
# Compute the azimuth/range spectrum
sla.SpectrumComputation(VERBOSE=True)

# Access spectrum data
spectrum_2d = sla.SpectrumOneDim      # 2D spectrum
spectrum_norm = sla.SpectrumOneDimNorm # Normalized spectrum

print(f"Spectrum shape: {spectrum_2d.shape}")
```

#### 4. De-weighting
```python
# Apply de-weighting to compensate for antenna patterns
sla.AncillaryDeWe(VERBOSE=True)

# De-weighted spectrum available as
deweighted = sla.SpectrumOneDimNormDeWe
```

#### 5. Sub-look Generation
```python
# Generate final sub-look images
sla.Generation(VERBOSE=True)

# Results
looks = sla.Looks  # Shape: [numberOfLooks, height, width]

# Quality assessment
for i in range(sla.numberOfLooks):
    amplitude = np.abs(looks[i])
    print(f"Look {i+1}: mean={amplitude.mean():.3f}, std={amplitude.std():.3f}")
```

### Advanced SLA Configuration

#### Multi-temporal Processing
```python
def process_time_series(product_list, output_dir):
    """Process multiple products for temporal analysis."""
    results = {}
    
    for product_path in product_list:
        print(f"Processing {product_path}")
        
        # Extract date from filename
        date = extract_date(product_path)
        
        # Process sub-looks
        sla = SubLookAnalysis(product_path)
        configure_and_process(sla)
        
        # Store results
        results[date] = {
            'looks': sla.Looks.copy(),
            'frequencies': sla.freqCentr.copy(),
            'metadata': extract_metadata(sla)
        }
        
        # Save individual results
        save_sublook_results(sla, output_dir / date)
    
    return results
```

#### Quality Control
```python
def validate_sublook_quality(sla, min_correlation=0.3, max_correlation=0.8):
    """Validate sub-look quality metrics."""
    correlations = []
    
    for i in range(sla.numberOfLooks):
        for j in range(i+1, sla.numberOfLooks):
            corr = np.abs(np.corrcoef(
                sla.Looks[i].flatten(),
                sla.Looks[j].flatten()
            )[0, 1])
            correlations.append((i, j, corr))
            
            if corr < min_correlation:
                print(f"Warning: Low correlation between looks {i+1}-{j+1}: {corr:.3f}")
            elif corr > max_correlation:
                print(f"Warning: High correlation between looks {i+1}-{j+1}: {corr:.3f}")
    
    return correlations
```

## SNAP Integration Workflows

SARPyX integrates with ESA's SNAP (Sentinel Application Platform) for comprehensive SAR preprocessing.

### Basic SNAP Processing Chain

```python
from sarpyx.snap import GPT

# Initialize GPT processor
gpt = GPT(product_path="input.zip", 
          outdir="output/", 
          mode="MacOS")  # or "Ubuntu", None

# Basic preprocessing chain
cal_product = gpt.Calibration(Pols=['VV', 'VH'])
deb_product = gpt.Deburst(Pols=['VV', 'VH'])
ml_product = gpt.Multilook(nRgLooks=1, nAzLooks=4)

print(f"Final product: {ml_product}")
```

### Sentinel-1 Preprocessing Workflow

#### Standard Preprocessing
```python
def preprocess_sentinel1(input_path, output_dir, polarizations=['VV', 'VH']):
    """Standard Sentinel-1 preprocessing workflow."""
    
    gpt = GPT(product_path=input_path, outdir=output_dir)
    
    # Step 1: TOPSAR Deburst
    print("1. Debursting...")
    deb_product = gpt.Deburst(Pols=polarizations)
    if not deb_product:
        raise RuntimeError("Deburst failed")
    
    # Step 2: Radiometric Calibration
    print("2. Calibrating...")
    cal_product = gpt.Calibration(Pols=polarizations, output_complex=True)
    if not cal_product:
        raise RuntimeError("Calibration failed")
    
    # Step 3: Multi-looking (optional)
    print("3. Multi-looking...")
    ml_product = gpt.Multilook(nRgLooks=1, nAzLooks=4)
    
    return ml_product
```

#### Land Masking
```python
def apply_land_mask(product_path, shapefile_path, output_dir):
    """Apply land masking using shapefile."""
    
    gpt = GPT(product_path=product_path, outdir=output_dir)
    
    # Import vector mask
    vector_product = gpt.ImportVector(vector_data=shapefile_path)
    
    # Apply land mask
    masked_product = gpt.LandMask(
        shoreline_extension=300,
        geometry_name="Buff_750",
        use_srtm=True,
        invert_geometry=True,
        land_mask=False
    )
    
    return masked_product
```

### COSMO-SkyMed Processing

```python
def preprocess_cosmo_skymed(input_path, output_dir):
    """COSMO-SkyMed specific preprocessing."""
    
    gpt = GPT(product_path=input_path, outdir=output_dir)
    
    # Multi-looking first (CSK is already focused)
    ml_product = gpt.Multilook(nRgLooks=2, nAzLooks=2)
    
    # Calibration
    cal_product = gpt.Calibration(Pols=['HH'], output_complex=True)
    
    return cal_product
```

## Ship Detection Workflow (CFAR)

SARPyX provides automated ship detection using Constant False Alarm Rate (CFAR) algorithms.

### Single Product CFAR

```python
from sarpyx.snap import CFAR

# Basic ship detection
first_product, excel_results = CFAR(
    prod="S1A_IW_GRDH_product.zip",
    mask_shp_path="land_mask.shp",
    mode="MacOS",
    Thresh=12.5,  # PFA threshold
    DELETE=False  # Keep intermediate products
)

print(f"Processed product: {first_product}")
print(f"Detection results: {excel_results}")
```

### Multi-threshold CFAR

```python
# Test multiple PFA thresholds
pfa_thresholds = [6.5, 9.5, 12.5, 15.5]

first_product, excel_results = CFAR(
    prod="input_product.zip",
    mask_shp_path="coastline.shp",
    Thresh=pfa_thresholds,
    DELETE=True  # Clean up intermediate files
)

# Results will include detection files for each threshold
```

### Custom CFAR Workflow

```python
def advanced_ship_detection(product_path, mask_path, output_dir):
    """Advanced ship detection with custom parameters."""
    
    # Initialize GPT
    gpt = GPT(product_path=product_path, outdir=output_dir)
    
    # Preprocessing based on product type
    prod_type = mode_identifier(Path(product_path).name)
    
    if prod_type == "SEN":
        # Sentinel-1 preprocessing
        deb_product = gpt.Deburst()
        cal_product = gpt.Calibration(Pols=['VH'])
        
        # Import and apply mask
        vector_product = gpt.ImportVector(vector_data=mask_path)
        masked_product = gpt.LandMask()
        
        start_product = masked_product
        
    elif prod_type == "CSK":
        # COSMO-SkyMed preprocessing
        ml_product = gpt.Multilook(nRgLooks=2, nAzLooks=2)
        cal_product = gpt.Calibration(Pols=['HH'])
        start_product = cal_product
    
    # Adaptive thresholding
    detection_products = []
    pfa_values = [6.5, 9.5, 12.5, 15.5]
    
    for pfa in pfa_values:
        gpt_det = GPT(product=start_product, outdir=output_dir)
        
        at_product = gpt_det.AdaptiveThresholding(
            background_window_m=800,
            guard_window_m=500,
            target_window_m=50,
            pfa=pfa
        )
        
        od_product = gpt_det.ObjectDiscrimination(
            min_target_m=35,
            max_target_m=500
        )
        
        detection_products.append(od_product)
    
    return detection_products
```

## Scientific Applications Workflow

### Vegetation Index Calculation

```python
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll, calculate_dpdd

def compute_vegetation_indices(vv_path, vh_path, output_dir):
    """Compute vegetation indices from dual-pol SAR data."""
    
    # Load backscatter data (linear scale)
    sigma_vv = load_backscatter(vv_path)  # Your loading function
    sigma_vh = load_backscatter(vh_path)
    
    # Calculate indices
    rvi = calculate_rvi(sigma_vv, sigma_vh)
    ndpoll = calculate_ndpoll(sigma_vv, sigma_vh)
    dpdd = calculate_dpdd(sigma_vv, sigma_vh)
    
    # Save results
    save_geotiff(rvi, output_dir / "RVI.tif", geotransform, projection)
    save_geotiff(ndpoll, output_dir / "NDPOLL.tif", geotransform, projection)
    save_geotiff(dpdd, output_dir / "DPDD.tif", geotransform, projection)
    
    return {"rvi": rvi, "ndpoll": ndpoll, "dpdd": dpdd}
```

### Polarimetric Analysis

```python
def polarimetric_analysis_workflow(slc_product, output_dir):
    """Complete polarimetric analysis workflow."""
    
    # Step 1: Sub-look decomposition
    sla = SubLookAnalysis(slc_product)
    sla.choice = 1  # Azimuth processing
    sla.numberOfLooks = 3
    
    # Process sub-looks
    sla.frequencyComputation()
    sla.SpectrumComputation()
    sla.AncillaryDeWe()
    sla.Generation()
    
    # Step 2: Extract polarimetric features
    features = {}
    for i, look in enumerate(sla.Looks):
        # Coherency matrix elements
        features[f'look_{i+1}'] = {
            'amplitude': np.abs(look),
            'phase': np.angle(look),
            'intensity': np.abs(look)**2
        }
    
    # Step 3: Multi-temporal coherence (if time series)
    if has_time_series:
        coherence = calculate_temporal_coherence(sla.Looks)
        features['temporal_coherence'] = coherence
    
    # Step 4: Save results
    save_polarimetric_results(features, output_dir)
    
    return features
```

## Batch Processing Workflows

### Large Dataset Processing

```python
def batch_process_products(product_list, output_dir, workflow_type="sla"):
    """Process multiple products in batch mode."""
    
    results = []
    failed_products = []
    
    for i, product_path in enumerate(product_list):
        print(f"Processing {i+1}/{len(product_list)}: {product_path}")
        
        try:
            if workflow_type == "sla":
                result = process_single_sla(product_path, output_dir)
            elif workflow_type == "cfar":
                result = process_single_cfar(product_path, output_dir)
            elif workflow_type == "indices":
                result = process_single_indices(product_path, output_dir)
            
            results.append(result)
            
        except Exception as e:
            print(f"Failed to process {product_path}: {e}")
            failed_products.append(product_path)
            continue
    
    # Generate summary report
    generate_batch_report(results, failed_products, output_dir)
    
    return results, failed_products
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def parallel_sla_processing(product_list, output_dir, n_workers=4):
    """Process products in parallel."""
    
    # Create worker function
    worker_func = partial(process_single_sla, output_dir=output_dir)
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(worker_func, product_list)
    
    return results

def process_single_sla(product_path, output_dir):
    """Single product SLA processing (for parallel execution)."""
    try:
        sla = SubLookAnalysis(product_path)
        configure_and_process(sla)
        
        # Save results
        product_output = output_dir / Path(product_path).stem
        save_sublook_results(sla, product_output)
        
        return {"status": "success", "product": product_path}
        
    except Exception as e:
        return {"status": "failed", "product": product_path, "error": str(e)}
```

## Workflow Automation

### Configuration-Based Processing

```python
import yaml

def load_processing_config(config_path):
    """Load processing configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def automated_workflow(config_path):
    """Execute workflow based on configuration file."""
    
    config = load_processing_config(config_path)
    
    # Example config structure:
    # workflow:
    #   type: "sla"
    #   parameters:
    #     numberOfLooks: 3
    #     choice: 1
    #   inputs:
    #     - "product1.zip"
    #     - "product2.zip"
    #   output_dir: "results/"
    
    workflow_type = config['workflow']['type']
    parameters = config['workflow']['parameters']
    inputs = config['workflow']['inputs']
    output_dir = Path(config['workflow']['output_dir'])
    
    results = []
    
    for input_path in inputs:
        if workflow_type == "sla":
            result = execute_sla_workflow(input_path, output_dir, parameters)
        elif workflow_type == "cfar":
            result = execute_cfar_workflow(input_path, output_dir, parameters)
        # Add more workflow types as needed
        
        results.append(result)
    
    return results
```

### Example Configuration File

```yaml
# processing_config.yaml
workflow:
  type: "sla"
  parameters:
    numberOfLooks: 3
    choice: 1
    centroidSeparations: 700
    subLookBandwidth: 700
  
  inputs:
    - "data/S1A_IW_SLC_product1.zip"
    - "data/S1A_IW_SLC_product2.zip"
  
  output_dir: "results/"
  
  options:
    verbose: true
    save_intermediate: false
    quality_check: true

visualization:
  enabled: true
  formats: ["png", "pdf"]
  dpi: 300
```

## Error Handling and Recovery

### Robust Processing

```python
def robust_processing_pipeline(product_path, output_dir, max_retries=3):
    """Robust processing with error recovery."""
    
    for attempt in range(max_retries):
        try:
            # Main processing
            result = main_processing_function(product_path, output_dir)
            return result
            
        except MemoryError:
            print(f"Memory error on attempt {attempt+1}, reducing processing parameters")
            # Reduce parameters and retry
            if attempt < max_retries - 1:
                reduce_processing_parameters()
                continue
            else:
                raise
                
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            if check_alternative_paths(product_path):
                product_path = get_alternative_path(product_path)
                continue
            else:
                raise
                
        except Exception as e:
            print(f"Unexpected error on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Performance Optimization

### Memory Management

```python
def memory_efficient_processing(large_product_list, output_dir):
    """Process large datasets with memory constraints."""
    
    # Process in chunks
    chunk_size = 10  # Adjust based on available memory
    
    for i in range(0, len(large_product_list), chunk_size):
        chunk = large_product_list[i:i+chunk_size]
        
        print(f"Processing chunk {i//chunk_size + 1}")
        
        # Process chunk
        chunk_results = process_product_chunk(chunk, output_dir)
        
        # Clear memory
        gc.collect()
        
        # Save intermediate results
        save_chunk_results(chunk_results, output_dir, i//chunk_size)
```

For more detailed information on specific workflows, see the [Tutorials](../tutorials/README.md) section or check the [API Reference](../api/README.md) for detailed parameter descriptions.
