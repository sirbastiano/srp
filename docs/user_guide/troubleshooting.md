# Troubleshooting

This guide helps you diagnose and resolve common issues when using sarpyx. Issues are organized by module and include practical solutions and prevention strategies.

## General Troubleshooting

### Installation Issues

#### Problem: Package Import Errors

```python
# Error: ModuleNotFoundError: No module named 'sarpyx'
import sarpyx  # Fails
```

**Solutions:**

1. **Verify Installation**:
   ```bash
   pip show sarpyx
   # or
   pdm list | grep sarpyx
   ```

2. **Check Python Environment**:
   ```bash
   which python
   python -c "import sys; print(sys.path)"
   ```

3. **Reinstall Package**:
   ```bash
   pip uninstall sarpyx
   pip install sarpyx
   # or for development
   pip install -e .
   ```

#### Problem: Dependency Conflicts

```bash
# Error: Package conflicts detected
ERROR: pip's dependency resolver does not currently have a workaround for this error.
```

**Solutions:**

1. **Use Virtual Environment**:
   ```bash
   python -m venv sarpyx_env
   source sarpyx_env/bin/activate  # Linux/macOS
   # sarpyx_env\Scripts\activate  # Windows
   pip install sarpyx
   ```

2. **Use PDM for Better Dependency Management**:
   ```bash
   pdm init
   pdm add sarpyx
   pdm install
   ```

3. **Check Conflicting Packages**:
   ```bash
   pip check
   ```

### Memory Issues

#### Problem: Out of Memory Errors

```python
# Error: MemoryError or system freezing during processing
```

**Diagnosis:**

```python
import psutil

def check_memory_usage():
    """Check current memory usage."""
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.1f} GB")
    print(f"Available memory: {memory.available / (1024**3):.1f} GB")
    print(f"Memory usage: {memory.percent}%")
    
    if memory.percent > 80:
        print("Warning: High memory usage detected!")
    
    return memory

check_memory_usage()
```

**Solutions:**

1. **Process Smaller Chunks**:
   ```python
   def process_large_dataset_chunked(large_array, chunk_size=1024):
       """Process large arrays in chunks."""
       result = np.zeros_like(large_array)
       
       for i in range(0, large_array.shape[0], chunk_size):
           for j in range(0, large_array.shape[1], chunk_size):
               chunk = large_array[i:i+chunk_size, j:j+chunk_size]
               # Process chunk
               result[i:i+chunk_size, j:j+chunk_size] = process_chunk(chunk)
       
       return result
   ```

2. **Use Memory Mapping**:
   ```python
   import numpy as np
   
   # For large files, use memory mapping
   large_array = np.memmap('large_file.dat', dtype='float32', mode='r', shape=(10000, 10000))
   ```

3. **Clear Variables**:
   ```python
   import gc
   
   # Explicitly delete large variables
   del large_array
   gc.collect()  # Force garbage collection
   ```

## Sub-Look Analysis (SLA) Issues

### Common SLA Problems

#### Problem: Frequency Computation Fails

```python
# Error during sla.frequencyComputation()
```

**Check Input Data:**

```python
def diagnose_sla_input(sla):
    """Diagnose SLA input data issues."""
    
    print(f"Product path: {sla.filepath}")
    print(f"Choice (0=Range, 1=Azimuth): {sla.choice}")
    print(f"Number of looks: {sla.numberOfLooks}")
    print(f"Centroid separations: {sla.centroidSeparations}")
    print(f"Sub-look bandwidth: {sla.subLookBandwidth}")
    
    # Check if file exists
    from pathlib import Path
    if not Path(sla.filepath).exists():
        print(f"Error: Product file not found: {sla.filepath}")
        return False
    
    # Check parameters
    if sla.numberOfLooks <= 0:
        print("Error: numberOfLooks must be positive")
        return False
    
    if sla.centroidSeparations <= 0:
        print("Error: centroidSeparations must be positive")
        return False
    
    return True

# Usage
sla = SubLookAnalysis("product.zip")
if diagnose_sla_input(sla):
    print("Input parameters look good")
```

**Solutions:**

1. **Verify Product Format**:
   ```python
   # Ensure product is in correct format
   supported_formats = ['.zip', '.SAFE', '.dim']
   product_path = Path("your_product")
   
   if not any(str(product_path).endswith(fmt) for fmt in supported_formats):
       print(f"Warning: Unsupported format. Supported: {supported_formats}")
   ```

2. **Adjust Processing Parameters**:
   ```python
   # Conservative parameters for problematic data
   sla = SubLookAnalysis("product.zip")
   sla.choice = 1  # Azimuth processing (more stable)
   sla.numberOfLooks = 3  # Start with fewer looks
   sla.centroidSeparations = 500  # Reduce separation
   sla.subLookBandwidth = 500  # Reduce bandwidth
   ```

#### Problem: Spectrum Computation Issues

```python
# Error: Invalid spectrum or NaN values
```

**Diagnosis:**

```python
def check_spectrum_quality(sla):
    """Check spectrum computation quality."""
    
    if not hasattr(sla, 'spectrum') or sla.spectrum is None:
        print("Error: No spectrum computed")
        return False
    
    # Check for NaN values
    nan_count = np.sum(np.isnan(sla.spectrum))
    total_count = sla.spectrum.size
    
    print(f"Spectrum shape: {sla.spectrum.shape}")
    print(f"NaN values: {nan_count}/{total_count} ({100*nan_count/total_count:.1f}%)")
    
    if nan_count > total_count * 0.1:  # More than 10% NaN
        print("Warning: High percentage of NaN values in spectrum")
        return False
    
    return True
```

**Solutions:**

1. **Check Input Data Quality**:
   ```python
   # Verify input SLC data
   def check_slc_quality(product_path):
       """Basic SLC data quality check."""
       try:
           # This would depend on your SLC reading implementation
           # Example for GDAL-readable products
           from osgeo import gdal
           dataset = gdal.Open(str(product_path))
           
           if dataset is None:
               print("Error: Cannot open product")
               return False
           
           print(f"Bands: {dataset.RasterCount}")
           print(f"Size: {dataset.RasterXSize} x {dataset.RasterYSize}")
           
           # Check first band
           band = dataset.GetRasterBand(1)
           data_sample = band.ReadAsArray(0, 0, 100, 100)
           
           if np.all(data_sample == 0):
               print("Warning: Data appears to be all zeros")
               return False
           
           return True
           
       except Exception as e:
           print(f"Error checking SLC quality: {e}")
           return False
   ```

2. **Use Alternative Processing Parameters**:
   ```python
   # Try different de-weighting options
   sla.choiceDeWe = 1  # Use average spectrum instead of ancillary data
   ```

### Performance Issues

#### Problem: Slow SLA Processing

**Optimization Strategies:**

```python
def optimize_sla_performance(sla):
    """Optimize SLA processing parameters for speed."""
    
    # Reduce number of looks for faster processing
    if sla.numberOfLooks > 5:
        print("Reducing number of looks for faster processing")
        sla.numberOfLooks = 3
    
    # Use smaller bandwidth for initial tests
    if sla.subLookBandwidth > 1000:
        print("Reducing bandwidth for faster processing")
        sla.subLookBandwidth = 700
    
    # Process smaller image sections first
    print("Consider processing subsets for testing")
    
    return sla

# Usage
sla = SubLookAnalysis("large_product.zip")
sla = optimize_sla_performance(sla)
```

## SNAP Integration Issues

### SNAP Installation Problems

#### Problem: GPT Not Found

```bash
# Error: GPT command 'gpt' not found
```

**Diagnosis:**

```python
def diagnose_snap_installation():
    """Comprehensive SNAP installation diagnosis."""
    
    import subprocess
    from pathlib import Path
    
    # Common SNAP installation paths
    snap_paths = {
        'macOS': '/Applications/snap/bin/gpt',
        'Linux': [
            '/usr/local/snap/bin/gpt',
            '/opt/snap/bin/gpt', 
            '/home/*/ESA-STEP/snap/bin/gpt'
        ],
        'Windows': [
            'C:\\Program Files\\snap\\bin\\gpt.exe',
            'C:\\ESA\\snap\\bin\\gpt.exe'
        ]
    }
    
    print("Checking SNAP installation...")
    
    # Check common paths
    found_paths = []
    for os_name, paths in snap_paths.items():
        if isinstance(paths, str):
            paths = [paths]
        
        for path in paths:
            if '*' in path:
                # Handle wildcard paths
                import glob
                expanded_paths = glob.glob(path)
                for exp_path in expanded_paths:
                    if Path(exp_path).exists():
                        found_paths.append(exp_path)
            else:
                if Path(path).exists():
                    found_paths.append(path)
    
    if found_paths:
        print(f"Found SNAP installations: {found_paths}")
        
        # Test each installation
        for path in found_paths:
            try:
                result = subprocess.run([path, '--help'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✓ Working SNAP GPT found at: {path}")
                    return path
                else:
                    print(f"✗ SNAP GPT not responding at: {path}")
            except Exception as e:
                print(f"✗ Error testing {path}: {e}")
    else:
        print("✗ No SNAP installation found")
        print("Download SNAP from: https://step.esa.int/main/download/snap-download/")
    
    return None

snap_path = diagnose_snap_installation()
```

**Solutions:**

1. **Install SNAP**:
   - Download from [ESA STEP](https://step.esa.int/main/download/snap-download/)
   - Follow installation instructions for your OS

2. **Manual Path Configuration**:
   ```python
   from sarpyx.snapflow.engine import GPT
   
   # Specify SNAP path manually
   gpt = GPT(product="input.zip", outdir="output/")
   gpt.gpt_executable = "/path/to/your/snap/bin/gpt"
   ```

3. **Environment Variables**:
   ```bash
   # Add SNAP to PATH
   export PATH="/Applications/snap/bin:$PATH"  # macOS
   export PATH="/usr/local/snap/bin:$PATH"     # Linux
   ```

#### Problem: SNAP Processing Failures

```python
# Error: GPT command execution failed
```

**Debugging SNAP Commands:**

```python
def debug_snap_command(gpt):
    """Debug SNAP GPT command execution."""
    
    print("=== SNAP Debug Information ===")
    print(f"GPT Executable: {gpt.gpt_executable}")
    print(f"Product Path: {gpt.prod_path}")
    print(f"Output Directory: {gpt.outdir}")
    print(f"Format: {gpt.format}")
    print(f"Parallelism: {gpt.parallelism}")
    
    # Check if input exists
    if not gpt.prod_path.exists():
        print(f"ERROR: Input product not found: {gpt.prod_path}")
        return False
    
    # Check if output directory is writable
    try:
        gpt.outdir.mkdir(parents=True, exist_ok=True)
        test_file = gpt.outdir / "test_write.tmp"
        test_file.touch()
        test_file.unlink()
        print("✓ Output directory is writable")
    except Exception as e:
        print(f"ERROR: Cannot write to output directory: {e}")
        return False
    
    # Test basic GPT command
    try:
        import subprocess
        result = subprocess.run([gpt.gpt_executable, '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ GPT executable is working")
        else:
            print(f"ERROR: GPT executable failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: Cannot execute GPT: {e}")
        return False
    
    return True

# Usage
gpt = GPT(product="input.zip", outdir="output/")
if debug_snap_command(gpt):
    print("SNAP setup looks good")
```

### SNAP Memory Issues

#### Problem: SNAP Out of Memory

```bash
# Error: Java heap space / OutOfMemoryError
```

**Solutions:**

1. **Reduce Parallelism**:
   ```python
   gpt = GPT(product="input.zip", outdir="output/")
   gpt.parallelism = 2  # Reduce from default
   ```

2. **Process Smaller Subsets**:
   ```python
   # Process geographic subsets
   subset_product = gpt.Subset(
       loc=[lon, lat],
       sourceBands=['Intensity_VV'],
       idx="small",
       winSize=2000,  # Smaller window
       GeoCoords=True
   )
   ```

3. **Use Different Output Format**:
   ```python
   # GeoTIFF typically uses less memory than BEAM-DIMAP
   gpt.format = "GeoTIFF"
   ```

## Science Module Issues

### Vegetation Index Problems

#### Problem: NaN Values in Indices

```python
# Error: calculate_rvi returns all NaN values
from sarpyx.science.indices import calculate_rvi

rvi = calculate_rvi(sigma_vv, sigma_vh)
print(f"NaN percentage: {100 * np.sum(np.isnan(rvi)) / rvi.size:.1f}%")
```

**Diagnosis:**

```python
def diagnose_vegetation_index_inputs(sigma_vv, sigma_vh):
    """Diagnose vegetation index input data."""
    
    print("=== Input Data Diagnosis ===")
    print(f"VV shape: {sigma_vv.shape}")
    print(f"VH shape: {sigma_vh.shape}")
    
    # Check data scale
    vv_range = (np.nanmin(sigma_vv), np.nanmax(sigma_vv))
    vh_range = (np.nanmin(sigma_vh), np.nanmax(sigma_vh))
    
    print(f"VV range: {vv_range}")
    print(f"VH range: {vh_range}")
    
    # Check for common issues
    issues = []
    
    # Issue 1: Wrong data scale (dB instead of linear)
    if vv_range[1] < 1.0:
        issues.append("Data might be in dB scale (should be linear)")
    
    # Issue 2: Zero or negative values
    if np.any(sigma_vv <= 0) or np.any(sigma_vh <= 0):
        issues.append("Zero or negative values detected")
        zero_vv = np.sum(sigma_vv <= 0)
        zero_vh = np.sum(sigma_vh <= 0)
        print(f"Zero/negative VV pixels: {zero_vv}")
        print(f"Zero/negative VH pixels: {zero_vh}")
    
    # Issue 3: Extreme values
    if vv_range[1] > 10 or vh_range[1] > 10:
        issues.append("Very high backscatter values detected")
    
    # Issue 4: Shape mismatch
    if sigma_vv.shape != sigma_vh.shape:
        issues.append("VV and VH arrays have different shapes")
    
    if issues:
        print("ISSUES DETECTED:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("✓ Input data looks good")
    
    return len(issues) == 0

# Usage
data_ok = diagnose_vegetation_index_inputs(sigma_vv, sigma_vh)
```

**Solutions:**

1. **Convert dB to Linear**:
   ```python
   # If data is in dB, convert to linear
   def db_to_linear(db_values):
       """Convert dB values to linear scale."""
       return 10**(db_values / 10.0)
   
   # Check if conversion is needed
   if np.max(sigma_vv) < 1.0:
       print("Converting from dB to linear scale")
       sigma_vv = db_to_linear(sigma_vv)
       sigma_vh = db_to_linear(sigma_vh)
   ```

2. **Handle Zero/Negative Values**:
   ```python
   def clean_backscatter_data(sigma_vv, sigma_vh, min_value=1e-6):
       """Clean backscatter data for vegetation index calculation."""
       
       # Replace zeros and negatives with small positive value
       sigma_vv_clean = np.where(sigma_vv <= 0, min_value, sigma_vv)
       sigma_vh_clean = np.where(sigma_vh <= 0, min_value, sigma_vh)
       
       # Log cleaning actions
       zeros_vv = np.sum(sigma_vv <= 0)
       zeros_vh = np.sum(sigma_vh <= 0)
       
       if zeros_vv > 0 or zeros_vh > 0:
           print(f"Cleaned {zeros_vv} zero/negative VV pixels")
           print(f"Cleaned {zeros_vh} zero/negative VH pixels")
       
       return sigma_vv_clean, sigma_vh_clean
   
   # Clean data before index calculation
   sigma_vv_clean, sigma_vh_clean = clean_backscatter_data(sigma_vv, sigma_vh)
   rvi = calculate_rvi(sigma_vv_clean, sigma_vh_clean)
   ```

3. **Quality Control Results**:
   ```python
   def quality_control_rvi(rvi):
       """Apply quality control to RVI results."""
       
       # RVI should be between 0 and 1
       valid_mask = (rvi >= 0) & (rvi <= 1) & np.isfinite(rvi)
       
       print(f"Valid RVI pixels: {np.sum(valid_mask)}/{rvi.size}")
       print(f"RVI range: {np.nanmin(rvi):.3f} to {np.nanmax(rvi):.3f}")
       
       # Flag outliers
       if np.nanmax(rvi) > 1.0:
           print("Warning: RVI values > 1.0 detected (check input data scale)")
       
       if np.nanmin(rvi) < 0.0:
           print("Warning: Negative RVI values detected")
       
       # Apply mask
       rvi_qc = np.where(valid_mask, rvi, np.nan)
       
       return rvi_qc, valid_mask
   
   rvi_clean, valid_pixels = quality_control_rvi(rvi)
   ```

## Data I/O Issues

### File Format Problems

#### Problem: Cannot Read SAR Products

```python
# Error: Unable to load product data
```

**Check File Integrity:**

```python
def check_sar_product_integrity(product_path):
    """Check SAR product file integrity."""
    
    from pathlib import Path
    import zipfile
    
    product_path = Path(product_path)
    
    print(f"Checking: {product_path}")
    
    # Basic existence check
    if not product_path.exists():
        print("ERROR: File does not exist")
        return False
    
    # Size check
    file_size_mb = product_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    if file_size_mb < 10:  # Typical SAR products are much larger
        print("WARNING: File size seems small for SAR product")
    
    # Format-specific checks
    if product_path.suffix == '.zip':
        try:
            with zipfile.ZipFile(product_path, 'r') as zf:
                file_list = zf.namelist()
                print(f"Archive contains {len(file_list)} files")
                
                # Check for essential files
                essential_patterns = ['manifest.safe', '.xml', '.tiff']
                found_patterns = []
                
                for pattern in essential_patterns:
                    matching_files = [f for f in file_list if pattern in f.lower()]
                    if matching_files:
                        found_patterns.append(pattern)
                        print(f"✓ Found {pattern} files: {len(matching_files)}")
                    else:
                        print(f"✗ No {pattern} files found")
                
                return len(found_patterns) > 0
                
        except zipfile.BadZipFile:
            print("ERROR: Corrupted ZIP file")
            return False
    
    elif product_path.suffix == '.SAFE' or product_path.is_dir():
        # Check SAFE directory structure
        required_dirs = ['annotation', 'measurement']
        found_dirs = []
        
        for req_dir in required_dirs:
            dir_path = product_path / req_dir
            if dir_path.exists():
                found_dirs.append(req_dir)
                print(f"✓ Found {req_dir} directory")
            else:
                print(f"✗ Missing {req_dir} directory")
        
        return len(found_dirs) == len(required_dirs)
    
    return True

# Usage
is_valid = check_sar_product_integrity("S1A_product.zip")
```

### Memory Mapping Issues

#### Problem: Large Files Cannot Be Loaded

**Use Memory-Efficient Loading:**

```python
def memory_efficient_loading(file_path, chunk_size=1024):
    """Load large files using memory mapping or chunked reading."""
    
    from pathlib import Path
    import numpy as np
    
    file_path = Path(file_path)
    file_size_gb = file_path.stat().st_size / (1024**3)
    
    print(f"File size: {file_size_gb:.2f} GB")
    
    if file_size_gb > 2.0:  # Large file
        print("Using memory mapping for large file")
        
        # Example for binary files
        if file_path.suffix == '.dat':
            # Assume float32 data, adjust as needed
            total_elements = file_path.stat().st_size // 4
            shape = int(np.sqrt(total_elements))  # Assume square
            
            data = np.memmap(file_path, dtype='float32', mode='r', 
                           shape=(shape, shape))
            return data
        else:
            print("Consider converting to memory-mappable format")
    
    else:
        print("Loading normally")
        # Normal loading for smaller files
        # Implementation depends on file format
        pass

# Usage
data = memory_efficient_loading("large_sar_data.dat")
```

## Performance Optimization

### General Performance Issues

#### Problem: Slow Processing

**Performance Profiling:**

```python
import time
import psutil
from functools import wraps

def profile_performance(func):
    """Decorator to profile function performance."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initial measurements
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        print(f"Starting {func.__name__}...")
        
        try:
            result = func(*args, **kwargs)
            
            # Final measurements
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**3)
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            print(f"✓ {func.__name__} completed:")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Memory change: {memory_delta:+.2f} GB")
            
            return result
            
        except Exception as e:
            print(f"✗ {func.__name__} failed: {e}")
            raise
    
    return wrapper

# Usage
@profile_performance
def process_sar_data(product_path):
    # Your processing function
    pass
```

**Optimization Strategies:**

```python
def optimize_processing_environment():
    """Optimize system for SAR processing."""
    
    import os
    import multiprocessing
    
    # Check system resources
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"System: {cpu_count} CPUs, {memory_gb:.1f} GB RAM")
    
    # Optimization recommendations
    recommendations = []
    
    if memory_gb < 8:
        recommendations.append("Consider upgrading RAM (minimum 8GB recommended)")
    
    if cpu_count < 4:
        recommendations.append("Consider parallel processing optimization")
    
    # Set environment variables for optimization
    os.environ['OMP_NUM_THREADS'] = str(min(cpu_count, 8))
    os.environ['NUMBA_NUM_THREADS'] = str(min(cpu_count, 8))
    
    # Disable GDAL warnings for performance
    os.environ['CPL_LOG'] = '/dev/null'
    
    print("Applied performance optimizations")
    
    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")

optimize_processing_environment()
```

## Error Recovery

### Robust Processing Patterns

```python
import shutil
from pathlib import Path

def robust_processing_with_recovery(processing_func, input_path, output_dir, 
                                  max_retries=2, cleanup_on_failure=True):
    """Robust processing with automatic recovery."""
    
    output_path = Path(output_dir)
    
    for attempt in range(max_retries + 1):
        try:
            print(f"Processing attempt {attempt + 1}/{max_retries + 1}")
            
            # Ensure clean output directory
            if output_path.exists() and cleanup_on_failure:
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Run processing
            result = processing_func(input_path, output_dir)
            
            print(f"✓ Processing successful on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
                
                # Cleanup for retry
                if output_path.exists() and cleanup_on_failure:
                    shutil.rmtree(output_path, ignore_errors=True)
            else:
                print("All attempts failed")
                raise RuntimeError(f"Processing failed after {max_retries + 1} attempts")

# Usage
def my_processing_function(input_path, output_dir):
    # Your processing logic here
    pass

result = robust_processing_with_recovery(
    my_processing_function,
    "input_product.zip",
    "output/",
    max_retries=2
)
```

### Partial Recovery

```python
def checkpoint_processing(processing_steps, input_data, output_dir, 
                        resume_from=None):
    """Processing with checkpoints for partial recovery."""
    
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Find resume point
    start_step = 0
    if resume_from:
        for i, (step_name, _) in enumerate(processing_steps):
            if step_name == resume_from:
                start_step = i
                print(f"Resuming from step: {resume_from}")
                break
    
    # Process steps with checkpointing
    current_data = input_data
    
    for i, (step_name, step_func) in enumerate(processing_steps[start_step:], start_step):
        checkpoint_file = checkpoint_dir / f"checkpoint_{i:02d}_{step_name}.pkl"
        
        try:
            if checkpoint_file.exists():
                print(f"Loading checkpoint for {step_name}")
                import pickle
                with open(checkpoint_file, 'rb') as f:
                    current_data = pickle.load(f)
            else:
                print(f"Processing step {i+1}: {step_name}")
                current_data = step_func(current_data)
                
                # Save checkpoint
                print(f"Saving checkpoint for {step_name}")
                import pickle
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(current_data, f)
            
            results[step_name] = current_data
            
        except Exception as e:
            print(f"Step {step_name} failed: {e}")
            print(f"Can resume from this point using resume_from='{step_name}'")
            raise
    
    return results

# Usage
processing_steps = [
    ("calibration", lambda data: calibrate_data(data)),
    ("multilook", lambda data: multilook_data(data)),
    ("subset", lambda data: subset_data(data)),
    ("vegetation_indices", lambda data: calculate_indices(data))
]

results = checkpoint_processing(
    processing_steps,
    input_data,
    "processing_output/",
    resume_from="multilook"  # Resume from failed step
)
```

## Getting Help

### Diagnostic Information Collection

```python
def collect_diagnostic_info():
    """Collect comprehensive diagnostic information."""
    
    import sys
    import platform
    import numpy as np
    from pathlib import Path
    
    print("=== sarpyx Diagnostic Information ===\n")
    
    # System information
    print("System Information:")
    print(f"- OS: {platform.system()} {platform.release()}")
    print(f"- Python: {sys.version}")
    print(f"- Architecture: {platform.machine()}")
    
    # Package versions
    print("\nPackage Versions:")
    try:
        import sarpyx
        print(f"- sarpyx: {sarpyx.__version__ if hasattr(sarpyx, '__version__') else 'unknown'}")
    except ImportError:
        print("- sarpyx: NOT INSTALLED")
    
    try:
        print(f"- NumPy: {np.__version__}")
    except ImportError:
        print("- NumPy: NOT INSTALLED")
    
    try:
        from osgeo import gdal
        print(f"- GDAL: {gdal.VersionInfo()}")
    except ImportError:
        print("- GDAL: NOT INSTALLED")
    
    # Memory information
    print("\nMemory Information:")
    memory = psutil.virtual_memory()
    print(f"- Total: {memory.total / (1024**3):.1f} GB")
    print(f"- Available: {memory.available / (1024**3):.1f} GB")
    print(f"- Used: {memory.percent}%")
    
    # SNAP information
    print("\nSNAP Information:")
    try:
        from sarpyx.snapflow.engine import GPT
        gpt = GPT(product="dummy", outdir="dummy")
        print(f"- GPT Path: {gpt.gpt_executable}")
        print(f"- Parallelism: {gpt.parallelism}")
    except Exception as e:
        print(f"- SNAP: Error - {e}")
    
    print("\n" + "="*50)

# Run diagnostics
collect_diagnostic_info()
```

### Community Support

1. **GitHub Issues**: [sarpyx Issues](https://github.com/your-repo/sarpyx/issues)
   - Include diagnostic information
   - Provide minimal reproducible examples
   - Specify your use case clearly

2. **Documentation**: Check the complete documentation for detailed examples

3. **Examples Repository**: Look for similar use cases in the examples

### Creating Minimal Reproducible Examples

```python
def create_minimal_example():
    """Template for creating minimal reproducible examples."""
    
    import numpy as np
    from sarpyx.science.indices import calculate_rvi
    
    print("=== Minimal Reproducible Example ===")
    print("Problem: RVI calculation returns unexpected results")
    print()
    
    # Create minimal test data
    print("Creating test data...")
    sigma_vv = np.array([[0.1, 0.2], [0.3, 0.4]])
    sigma_vh = np.array([[0.05, 0.1], [0.15, 0.2]])
    
    print(f"VV data: {sigma_vv}")
    print(f"VH data: {sigma_vh}")
    
    # Show the problem
    print("\nCalculating RVI...")
    try:
        rvi = calculate_rvi(sigma_vv, sigma_vh)
        print(f"RVI result: {rvi}")
        print(f"Expected: values between 0 and 1")
        
        # Show what's wrong
        if np.any(np.isnan(rvi)):
            print("PROBLEM: NaN values in result")
        if np.any(rvi > 1):
            print("PROBLEM: RVI values > 1")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\nSystem info:")
    collect_diagnostic_info()

# Use this template when reporting issues
create_minimal_example()
```

This comprehensive troubleshooting guide should help users diagnose and resolve most common issues they encounter when using sarpyx, with practical solutions and prevention strategies for each problem area.
