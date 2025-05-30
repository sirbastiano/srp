# API Reference

Comprehensive documentation for all SARPyX modules, classes, and functions.

## Overview

The SARPyX API is organized into several main modules, each serving specific aspects of SAR processing:

- **[processor](processor/README.md)**: Core SAR processing algorithms and utilities
- **[sla](sla/README.md)**: Sub-Look Analysis for aperture decomposition  
- **[snap](snap/README.md)**: Integration with ESA's SNAP platform
- **[science](science/README.md)**: Scientific analysis tools and indices
- **[utils](utils/README.md)**: General utilities and visualization tools

## Quick Reference

### Most Common Classes

| Class | Module | Description |
|-------|--------|-------------|
| `SubLookAnalysis` | `sarpyx.sla` | Main class for sub-aperture analysis |
| `GPT` | `sarpyx.snap` | SNAP Graph Processing Tool wrapper |
| `Handler` | `sarpyx.sla.core` | Metadata extraction and handling |

### Most Common Functions

| Function | Module | Description |
|----------|--------|-------------|
| `show_image()` | `sarpyx.utils` | Display SAR images with customization |
| `calculate_rvi()` | `sarpyx.science.indices` | Radar Vegetation Index calculation |
| `calculate_entropy()` | `sarpyx.processor.autofocus.metrics` | Image focus quality metric |
| `calculate_ndpoll()` | `sarpyx.science.indices` | Normalized Difference Polarization Index |
| `save_matlab_mat()` | `sarpyx.utils.io` | Save data in MATLAB format |
| `Calibration()` | `sarpyx.snap.GPT` | SNAP radiometric calibration |
| `TerrainCorrection()` | `sarpyx.snap.GPT` | SNAP geometric terrain correction |

## Module Structure

```
sarpyx/
├── processor/          # Core processing algorithms
│   ├── core/          # Focus, decode, transforms
│   ├── autofocus/     # Autofocus algorithms and metrics
│   ├── algorithms/    # High-level processing algorithms
│   ├── data/          # Data I/O and format conversion
│   └── utils/         # Processing utilities
├── sla/               # Sub-Look Analysis
│   ├── core/          # Main SLA implementation
│   └── utils/         # SLA-specific utilities
├── snap/              # SNAP integration
├── science/           # Scientific analysis tools
└── utils/             # General utilities
```

## Usage Patterns

### Importing Modules

```python
# Import main modules
from sarpyx import sla, snap, utils, science

# Import specific classes
from sarpyx.sla import SubLookAnalysis
from sarpyx.snap import GPT
from sarpyx.utils import show_image

# Import specific functions
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll
from sarpyx.processor.autofocus.metrics import calculate_entropy
```

### Common API Patterns

#### Configuration Objects
Many SARPyX classes use attribute-based configuration:

```python
# Configure SubLookAnalysis
sla = SubLookAnalysis(product_path)
sla.choice = 1                    # Azimuth processing
sla.numberOfLooks = 3             # Number of sub-looks
sla.centroidSeparations = 700     # Frequency separation
sla.subLookBandwidth = 700        # Sub-look bandwidth
```

#### Method Chaining
Some operations can be chained:

```python
# SNAP processing chain
gpt = GPT(product=path, outdir=output)
result = (gpt.Calibration()
             .ThermalNoiseRemoval()
             .TerrainCorrection())
```

#### Verbose Output
Many functions support verbose output for debugging:

```python
# Enable detailed output
sla.SpectrumComputation(VERBOSE=True)
sla.Generation(VERBOSE=True)
```

## Error Handling

### Common Exceptions

| Exception | Description | Common Causes |
|-----------|-------------|---------------|
| `AssertionError` | Parameter validation failed | Invalid parameter values |
| `FileNotFoundError` | Input file not found | Incorrect file paths |
| `MemoryError` | Insufficient memory | Large datasets, limited RAM |
| `ValueError` | Invalid data values | Corrupted data, wrong data types |

### Exception Handling Example

```python
try:
    sla = SubLookAnalysis(product_path)
    sla.frequencyComputation()
    sla.SpectrumComputation()
    result = sla.Generation()
except AssertionError as e:
    print(f"Parameter validation error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Data Types

### Input Data Types

| Type | Description | Example |
|------|-------------|---------|
| `str` | File paths | `"/path/to/product.zip"` |
| `Path` | Pathlib objects | `Path("product.zip")` |
| `ndarray` | NumPy arrays | `np.array([[1, 2], [3, 4]])` |
| `complex` | Complex numbers | `1+2j` |

### Output Data Types

| Type | Description | Shape |
|------|-------------|-------|
| `ndarray` | SAR images | `(rows, cols)` |
| `ndarray` | Sub-look data | `(looks, rows, cols)` |
| `dict` | Metadata | Variable |
| `str` | File paths | N/A |

## Performance Guidelines

### Memory Optimization

```python
# For large datasets, monitor memory usage
import psutil

def process_with_memory_check(data):
    memory_before = psutil.virtual_memory().percent
    result = your_processing_function(data)
    memory_after = psutil.virtual_memory().percent
    print(f"Memory usage: {memory_before}% → {memory_after}%")
    return result
```

### Computational Efficiency

- Use vectorized NumPy operations when possible
- Consider processing data in chunks for large datasets
- Enable multi-threading for supported operations

## Version Compatibility

### API Stability

- **Stable API**: Core functions in stable modules (marked in documentation)
- **Experimental API**: New features may change between versions
- **Deprecated API**: Marked with deprecation warnings

### Version Information

```python
import sarpyx
print(f"SARPyX version: {sarpyx.__version__}")

# Check module versions
from sarpyx import processor
print(f"Processor module version: {processor.__version__}")
```

## Contributing to API Documentation

### Documentation Standards

- All public functions must have docstrings
- Use NumPy docstring format
- Include parameter types and descriptions
- Provide usage examples
- Document exceptions that may be raised

### Example Docstring

```python
def calculate_metric(data, method='entropy', normalize=True):
    """
    Calculate focus quality metric for SAR data.
    
    Parameters
    ----------
    data : ndarray
        Input SAR data array, shape (rows, cols)
    method : str, optional
        Metric calculation method. Options: 'entropy', 'contrast', 'gradient'
        Default is 'entropy'
    normalize : bool, optional
        Whether to normalize the result. Default is True
        
    Returns
    -------
    float
        Calculated metric value
        
    Raises
    ------
    ValueError
        If method is not recognized
    AssertionError
        If data dimensions are invalid
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.random((100, 100))
    >>> metric = calculate_metric(data, method='entropy')
    >>> print(f"Focus quality: {metric:.3f}")
    """
```

## Related Documentation

- **[User Guide](../user_guide/README.md)**: High-level usage instructions
- **[Tutorials](../tutorials/README.md)**: Step-by-step learning materials
- **[Examples](../examples/README.md)**: Ready-to-run code examples
- **[Developer Guide](../developer_guide/README.md)**: Contributing and development information

## Search and Navigation

Use the module-specific API documentation pages for detailed information:

- **[Processor API](processor/README.md)**: Core processing algorithms
- **[SLA API](sla/README.md)**: Sub-Look Analysis
- **[SNAP API](snap/README.md)**: SNAP integration  
- **[Science API](science/README.md)**: Scientific analysis
- **[Utils API](utils/README.md)**: Utilities and visualization
