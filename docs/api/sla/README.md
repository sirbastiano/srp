# SLA (Sub-Look Analysis) API

Complete API reference for the Sub-Look Analysis module.

## Overview

The `sarpyx.sla` module provides functionality for decomposing SAR apertures into sub-looks, enabling advanced analysis techniques such as motion detection, interferometry, and resolution enhancement.

## Classes

### `SubLookAnalysis`

Main class for performing sub-aperture decomposition analysis.

```python
from sarpyx.sla import SubLookAnalysis

sla = SubLookAnalysis(productPath)
```

#### Constructor

```python
SubLookAnalysis(productPath: str)
```

**Parameters:**
- `productPath` (str): Path to the SAR product file (ZIP format for Sentinel-1)

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `choice` | int | 1 | Processing direction (0=Range, 1=Azimuth) |
| `numberOfLooks` | int | 3 | Number of sub-looks to generate |
| `centroidSeparations` | float | 700 | Frequency separation between sub-looks (Hz) |
| `subLookBandwidth` | float | 700 | Bandwidth of each sub-look (Hz) |
| `choiceDeWe` | int | 0 | De-weighting method (0=Ancillary, 1=Average) |

**Computed Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `Box` | ndarray(complex) | Input SAR data subset |
| `SpectrumOneDim` | ndarray(complex) | 1D spectrum representation |
| `SpectrumOneDimNorm` | ndarray(complex) | Normalized spectrum |
| `Looks` | ndarray(complex) | Generated sub-look images, shape (looks, rows, cols) |
| `freqVect` | ndarray(float) | Frequency vector |
| `freqCentr` | ndarray(float) | Center frequencies of sub-looks |
| `freqMin` | ndarray(float) | Minimum frequencies of sub-looks |
| `freqMax` | ndarray(float) | Maximum frequencies of sub-looks |

#### Methods

##### `frequencyComputation()`

Computes the frequency bins and center frequencies for sub-look decomposition.

```python
sla.frequencyComputation()
```

**Raises:**
- `AssertionError`: If sub-look frequencies exceed available bandwidth

**Example:**
```python
sla = SubLookAnalysis(product_path)
sla.numberOfLooks = 5
sla.centroidSeparations = 500
sla.frequencyComputation()
print(f"Sub-look frequencies: {sla.freqCentr}")
```

##### `SpectrumComputation(VERBOSE=False)`

Transforms SAR data to frequency domain and computes the spectrum.

```python
sla.SpectrumComputation(VERBOSE=False)
```

**Parameters:**
- `VERBOSE` (bool, optional): Enable detailed output and plots. Default is False.

**Raises:**
- `AssertionError`: If spectrum computation fails

**Example:**
```python
sla.SpectrumComputation(VERBOSE=True)  # Shows spectrum plots
print(f"Spectrum shape: {sla.SpectrumOneDim.shape}")
```

##### `AncillaryDeWe(VERBOSE=False)`

Performs de-weighting using ancillary data and theoretical weighting functions.

```python
sla.AncillaryDeWe(VERBOSE=False)
```

**Parameters:**
- `VERBOSE` (bool, optional): Enable detailed output. Default is False.

**Note:** This method is used when `choiceDeWe = 0`.

##### `Generation(VERBOSE=False)`

Generates the final sub-look images by extracting sub-bands and applying inverse transforms.

```python
sla.Generation(VERBOSE=False)
```

**Parameters:**
- `VERBOSE` (bool, optional): Enable detailed output and visualization. Default is False.

**Example:**
```python
sla.Generation(VERBOSE=True)
# Access results
sublook_images = sla.Looks  # Shape: (numberOfLooks, rows, cols)

# Display first sub-look
import matplotlib.pyplot as plt
plt.imshow(np.abs(sla.Looks[0]), cmap='gray')
plt.title('Sub-look 0')
plt.show()
```

#### Complete Processing Example

```python
from sarpyx.sla import SubLookAnalysis
import numpy as np
import matplotlib.pyplot as plt

# Initialize and configure
sla = SubLookAnalysis("/path/to/sentinel1_product.zip")
sla.choice = 1                    # Azimuth processing
sla.numberOfLooks = 3             # Generate 3 sub-looks
sla.centroidSeparations = 700     # 700 Hz separation
sla.subLookBandwidth = 700        # 700 Hz bandwidth per sub-look

# Execute processing chain
try:
    # Step 1: Compute frequency bins
    sla.frequencyComputation()
    print(f"Sub-look center frequencies: {sla.freqCentr}")
    
    # Step 2: Transform to frequency domain
    sla.SpectrumComputation(VERBOSE=True)
    
    # Step 3: Apply de-weighting
    sla.AncillaryDeWe()
    
    # Step 4: Generate sub-look images
    sla.Generation(VERBOSE=True)
    
    # Access results
    print(f"Generated {sla.numberOfLooks} sub-looks")
    print(f"Sub-look shape: {sla.Looks[0].shape}")
    
    # Calculate coherence between sub-looks
    coherence = np.abs(np.corrcoef(
        sla.Looks[0].flatten(), 
        sla.Looks[1].flatten()
    )[0, 1])
    print(f"Coherence between sub-looks 0 and 1: {coherence:.3f}")
    
except AssertionError as e:
    print(f"Parameter validation error: {e}")
except Exception as e:
    print(f"Processing error: {e}")
```

### `Handler`

Metadata extraction and processing class for SAR products.

```python
from sarpyx.sla.core import Handler

handler = Handler(filepath)
```

#### Constructor

```python
Handler(filepath: str)
```

**Parameters:**
- `filepath` (str): Path to the SAR product ZIP file

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `PRF` | float | Pulse Repetition Frequency |
| `AzimBand` | float | Total azimuth bandwidth |
| `RangeBand` | float | Total range bandwidth |
| `ChirpBand` | float | Chirp bandwidth |
| `AzimRes` | int | Azimuth resolution (default: 5) |
| `RangeRes` | int | Range resolution (default: 20) |
| `AzimSpacing` | float | Azimuth pixel spacing |
| `RangeSpacing` | float | Range pixel spacing |
| `WeightFunctRange` | str | Range weighting function type |
| `WeightFunctRangeParams` | float | Range window coefficient |
| `WeightFunctAzim` | str | Azimuth weighting function type |
| `WeightFunctAzimParams` | float | Azimuth window coefficient |
| `CentralFreqRange` | int | Central frequency for range processing |
| `CentralFreqAzim` | int | Central frequency for azimuth processing |

#### Methods

##### `Chain()`

Processes metadata from the ZIP file and extracts relevant parameters.

```python
handler.Chain()
```

**Example:**
```python
from sarpyx.sla.core import Handler

# Extract metadata
handler = Handler("/path/to/sentinel1_product.zip")
handler.Chain()

print(f"PRF: {handler.PRF} Hz")
print(f"Azimuth bandwidth: {handler.AzimBand} Hz")
print(f"Range bandwidth: {handler.RangeBand} Hz")
print(f"Azimuth spacing: {handler.AzimSpacing} m")
print(f"Range spacing: {handler.RangeSpacing} m")
```

## Utility Functions

### Processing Direction

```python
# Choose processing direction
RANGE_PROCESSING = 0
AZIMUTH_PROCESSING = 1

sla.choice = AZIMUTH_PROCESSING  # Most common for motion analysis
```

### De-weighting Methods

```python
# Choose de-weighting method
ANCILLARY_DEWEIGHTING = 0  # Use theoretical weighting functions
AVERAGE_DEWEIGHTING = 1    # Use average spectrum

sla.choiceDeWe = ANCILLARY_DEWEIGHTING
```

## Advanced Usage

### Custom Parameter Optimization

```python
def optimize_sublook_parameters(sla, target_coherence=0.3):
    """Optimize sub-look parameters for desired coherence."""
    
    best_params = None
    best_coherence = float('inf')
    
    # Test different parameter combinations
    for n_looks in [3, 5, 7]:
        for bandwidth in [500, 700, 900]:
            for separation in [400, 600, 800]:
                try:
                    sla.numberOfLooks = n_looks
                    sla.subLookBandwidth = bandwidth
                    sla.centroidSeparations = separation
                    
                    sla.frequencyComputation()
                    sla.SpectrumComputation()
                    sla.Generation()
                    
                    # Calculate coherence
                    coherence = calculate_coherence(sla.Looks)
                    
                    if abs(coherence - target_coherence) < best_coherence:
                        best_coherence = abs(coherence - target_coherence)
                        best_params = (n_looks, bandwidth, separation)
                        
                except AssertionError:
                    continue  # Skip invalid parameter combinations
    
    return best_params

# Usage
best_n_looks, best_bw, best_sep = optimize_sublook_parameters(sla)
print(f"Optimal parameters: {best_n_looks} looks, {best_bw} Hz BW, {best_sep} Hz sep")
```

### Batch Processing

```python
def process_multiple_products(product_list, config):
    """Process multiple SAR products with same configuration."""
    
    results = {}
    
    for product_path in product_list:
        try:
            sla = SubLookAnalysis(product_path)
            
            # Apply configuration
            for key, value in config.items():
                setattr(sla, key, value)
            
            # Process
            sla.frequencyComputation()
            sla.SpectrumComputation()
            sla.Generation()
            
            # Store results
            results[product_path] = {
                'looks': sla.Looks.copy(),
                'frequencies': sla.freqCentr.copy(),
                'shape': sla.Looks.shape
            }
            
        except Exception as e:
            print(f"Failed to process {product_path}: {e}")
            results[product_path] = None
    
    return results

# Usage
config = {
    'choice': 1,
    'numberOfLooks': 3,
    'centroidSeparations': 700,
    'subLookBandwidth': 700
}

products = [
    "/path/to/product1.zip",
    "/path/to/product2.zip",
    "/path/to/product3.zip"
]

results = process_multiple_products(products, config)
```

## Error Handling

### Common Error Scenarios

1. **Invalid frequency parameters**:
```python
try:
    sla.frequencyComputation()
except AssertionError as e:
    if "sub-look spectrum outside" in str(e):
        print("Reduce sub-look bandwidth or adjust center frequencies")
```

2. **Memory limitations**:
```python
try:
    sla.SpectrumComputation()
except MemoryError:
    print("Reduce data size or process in chunks")
```

3. **File access issues**:
```python
try:
    sla = SubLookAnalysis(product_path)
except FileNotFoundError:
    print(f"Product file not found: {product_path}")
```

## Performance Considerations

### Memory Usage
- Large datasets require significant memory for spectrum computation
- Consider processing smaller spatial subsets for initial testing
- Monitor memory usage with `psutil` for production workflows

### Computational Complexity
- FFT operations scale as O(N log N)
- Processing time increases with number of looks and data size
- Azimuth processing typically faster than range processing

### Optimization Tips

```python
# For better performance:
# 1. Use appropriate data types
sla.Box = sla.Box.astype(np.complex64)  # If precision allows

# 2. Process in chunks for very large data
def process_in_chunks(sla, chunk_size=1024):
    """Process large datasets in chunks."""
    # Implementation depends on specific use case
    pass

# 3. Disable verbose output in production
sla.Generation(VERBOSE=False)  # Faster execution
```

## Related Functions

See also:
- [`sarpyx.utils`](../utils/README.md) for visualization functions
- [`sarpyx.processor.autofocus.metrics`](../processor/README.md) for quality assessment
- [`sarpyx.science.indices`](../science/README.md) for analysis applications
