# Processor Module API

The `sarpyx.processor` module provides comprehensive SAR processing capabilities including focusing algorithms, autofocus methods, data I/O, and various processing utilities.

## Module Structure

```
sarpyx.processor/
├── core/          # Core processing algorithms (focus, decode, transforms)
├── autofocus/     # Autofocus algorithms and quality metrics  
├── algorithms/    # High-level processing algorithms (RDA, back-projection)
├── data/          # Data readers, writers, and format converters
├── utils/         # Processing utilities and helper functions
└── AutoFocusNet/  # Legacy AutoFocusNet implementation (maintained for compatibility)
```

## Quick Start

```python
from sarpyx import processor

# Import specific submodules
from sarpyx.processor import core, autofocus, algorithms, data, utils

# Import specific functions
from sarpyx.processor.autofocus.metrics import calculate_entropy
from sarpyx.processor.core.focus import range_compression
from sarpyx.processor.utils.io import save_matlab_mat
```

## Core Submodules

### [core](core/README.md)
Core SAR processing algorithms including:
- **focus**: Range and azimuth compression algorithms
- **decode**: Signal decoding and preprocessing
- **transforms**: Fourier and other mathematical transforms  
- **constants**: Physical and processing constants
- **rda_simple**: Simplified Range-Doppler Algorithm implementation

**Key Functions:**
- `range_compression()`: Applies range compression to SAR data
- `azimuth_compression()`: Applies azimuth compression  
- `fft_shift()`: Optimized FFT operations for SAR processing

### [autofocus](autofocus/README.md)
Autofocus algorithms and image quality metrics including:
- **metrics**: Focus quality assessment functions
- **compressor**: Autofocus algorithm implementations
- **constants**: Autofocus-specific parameters

**Key Functions:**
- `calculate_entropy()`: Computes image entropy for focus assessment
- `calculate_contrast()`: Measures image contrast
- `phase_gradient_autofocus()`: PGA autofocus implementation

### [algorithms](algorithms/README.md)
High-level SAR processing algorithms including:
- **rda**: Range-Doppler Algorithm implementation
- **backprojection**: Time-domain back-projection algorithms

**Key Classes:**
- `RDA`: Range-Doppler Algorithm processor
- `BackProjection`: Back-projection algorithm implementation

### [data](data/README.md)
Data I/O and format conversion utilities including:
- **readers**: SAR data format readers
- **writers**: Output format writers
- **formatters**: Data format conversion utilities

**Key Functions:**
- `read_sar_data()`: Generic SAR data reader
- `write_geotiff()`: GeoTIFF output writer
- `format_converter()`: Convert between SAR data formats

### [utils](utils/README.md)
Processing utilities and helper functions including:
- **io**: Input/output utilities
- **printsummary**: Data summary and diagnostic functions

**Key Functions:**
- `save_matlab_mat()`: Save data in MATLAB format
- `print_sar_summary()`: Display SAR data information
- `validate_sar_data()`: Data validation utilities

## Usage Examples

### Basic SAR Processing Workflow

```python
from sarpyx.processor import core, autofocus, data

# Read SAR data
sar_data = data.readers.read_sar_data('path/to/sar_file.slc')

# Apply range compression
range_compressed = core.focus.range_compression(sar_data)

# Apply azimuth compression  
focused_data = core.focus.azimuth_compression(range_compressed)

# Assess focus quality
entropy = autofocus.metrics.calculate_entropy(focused_data)
print(f"Focus quality (entropy): {entropy}")

# Save results
data.writers.write_geotiff(focused_data, 'output.tif')
```

### Autofocus Processing

```python
from sarpyx.processor.autofocus import metrics, compressor

# Load unfocused SAR data
unfocused_data = load_sar_data()

# Calculate initial focus metrics
initial_entropy = metrics.calculate_entropy(unfocused_data)
initial_contrast = metrics.calculate_contrast(unfocused_data)

# Apply Phase Gradient Autofocus
focused_data = compressor.phase_gradient_autofocus(unfocused_data)

# Compare focus improvement
final_entropy = metrics.calculate_entropy(focused_data)
print(f"Focus improvement: {final_entropy - initial_entropy}")
```

### Range-Doppler Algorithm Processing

```python
from sarpyx.processor.algorithms import rda

# Initialize RDA processor
processor = rda.RDA(
    range_bandwidth=40e6,  # 40 MHz
    prf=1000,              # 1 kHz PRF
    platform_velocity=7000 # 7 km/s
)

# Process raw SAR data
focused_image = processor.process(raw_data)
```

## Configuration

### Processing Parameters

Many processor functions accept configuration parameters:

```python
# Range compression with custom parameters
compressed = core.focus.range_compression(
    data=sar_data,
    bandwidth=40e6,
    window_function='hamming',
    zero_padding=2
)

# Autofocus with custom settings
focused = compressor.phase_gradient_autofocus(
    data=unfocused_data,
    iterations=10,
    window_size=64,
    overlap=0.5
)
```

### Output Formats

The processor module supports multiple output formats:

```python
# Save as MATLAB format
data.writers.save_matlab_mat(processed_data, 'output.mat')

# Save as GeoTIFF
data.writers.write_geotiff(processed_data, 'output.tif', 
                          projection='EPSG:4326')

# Save as HDF5
data.writers.write_hdf5(processed_data, 'output.h5')
```

## Performance Considerations

### Memory Management

For large SAR datasets:

```python
# Process data in chunks to manage memory
chunk_size = 1024  # Process 1024 lines at a time

for chunk in data.utils.chunk_processor(sar_data, chunk_size):
    processed_chunk = core.focus.range_compression(chunk)
    # Save or accumulate results
```

### Parallel Processing

Use built-in parallelization for improved performance:

```python
# Enable parallel processing
import multiprocessing
processor.set_num_threads(multiprocessing.cpu_count())

# Process with parallel algorithms
result = algorithms.rda.process_parallel(data, num_workers=8)
```

## Error Handling

The processor module includes comprehensive error handling:

```python
try:
    result = core.focus.range_compression(data)
except processor.ProcessingError as e:
    print(f"Processing error: {e}")
    # Handle specific processing errors
except processor.DataFormatError as e:
    print(f"Data format error: {e}")
    # Handle data format issues
```

## Legacy Support

### AutoFocusNet

The original AutoFocusNet implementation is maintained for backward compatibility:

```python
from sarpyx.processor import AutoFocusNet

# Use legacy AutoFocusNet
legacy_processor = AutoFocusNet.AutoFocus()
result = legacy_processor.process(data)
```

Note: New projects should use the updated `autofocus` submodule for better performance and features.

## See Also

- [SLA Module](../sla/README.md): Sub-Look Analysis capabilities
- [SNAP Module](../snap/README.md): SNAP integration for preprocessing
- [Science Module](../science/README.md): Scientific analysis tools
- [Utils Module](../utils/README.md): General utilities and visualization
