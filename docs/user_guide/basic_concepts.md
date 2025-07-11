# Basic Concepts

This section introduces the fundamental concepts underlying SAR processing and how they are implemented in SARPyX.

## Synthetic Aperture Radar (SAR) Fundamentals

### What is SAR?

Synthetic Aperture Radar is a form of radar that creates high-resolution images by using the motion of the radar antenna over a target region. Unlike optical sensors, SAR can operate day and night and penetrate clouds, making it invaluable for Earth observation.

### Key SAR Parameters

#### Range and Azimuth
- **Range**: The line-of-sight distance from radar to target
- **Azimuth**: The along-track direction of platform motion

```python
# In SARPyX, these are often referenced as:
sla.choice = 0  # Range processing
sla.choice = 1  # Azimuth processing
```

#### Resolution
- **Range Resolution**: Ability to distinguish targets at different ranges
- **Azimuth Resolution**: Ability to distinguish targets in the flight direction

#### Polarization
SAR can transmit and receive in different polarizations:
- **HH**: Horizontal transmit, Horizontal receive
- **VV**: Vertical transmit, Vertical receive  
- **HV/VH**: Cross-polarizations

## Sub-Look Analysis (SLA)

### Concept

Sub-Look Analysis decomposes the full SAR aperture into smaller sub-apertures, each providing:
- Different viewing angles (squint angles)
- Enhanced motion sensitivity
- Improved resolution in specific directions

### Mathematical Foundation

The sub-look decomposition is performed in the frequency domain:

```
S_sub(f) = S_full(f) · W(f - f_center)
```

Where:
- `S_sub(f)`: Sub-look spectrum
- `S_full(f)`: Full-aperture spectrum  
- `W(f)`: Weighting window
- `f_center`: Sub-look center frequency

### Implementation in SARPyX

```python
from sarpyx.sla import SubLookAnalysis

# Initialize SLA processor
sla = SubLookAnalysis(product_path)

# Configure sub-look parameters
sla.numberOfLooks = 3           # Number of sub-looks
sla.centroidSeparations = 700   # Frequency separation (Hz)
sla.subLookBandwidth = 700      # Bandwidth per sub-look (Hz)

# Execute processing
sla.frequencyComputation()      # Compute frequency bins
sla.SpectrumComputation()       # Transform to frequency domain
sla.Generation()                # Generate sub-look images
```

### Applications

Sub-look analysis enables:
- **Motion detection**: Different sub-looks see moving targets differently
- **Interferometry**: Phase differences between sub-looks
- **Speckle reduction**: Incoherent averaging of sub-looks
- **Resolution enhancement**: Combining sub-looks optimally

## SAR Processing Pipeline

### Level 0 to Level 1 Processing

1. **Raw Data (Level 0)**: Compressed, unfocused SAR data
2. **Single Look Complex (SLC)**: Focused, complex-valued SAR images
3. **Ground Range Detected (GRD)**: Multi-looked, geocoded products

### SARPyX Processing Stages

#### 1. Data Loading and Preprocessing
```python
# Load SAR data
from sarpyx.processor.data import readers
sar_data = readers.load_sar_product(product_path)

# Apply preprocessing
from sarpyx.processor.core import decode
processed_data = decode.preprocess(sar_data)
```

#### 2. Focus Processing
```python
# Apply focusing algorithm
from sarpyx.processor.core import focus
focused_data = focus.range_compression(processed_data)
focused_data = focus.azimuth_compression(focused_data)
```

#### 3. Autofocus (if needed)
```python
# Check focus quality
from sarpyx.processor.autofocus import metrics
focus_quality = metrics.calculate_entropy(focused_data)

# Apply autofocus if necessary
if focus_quality < threshold:
    from sarpyx.processor.autofocus import compressor
    autofocused = compressor.autofocus(focused_data)
```

## Frequency Domain Processing

### Discrete Fourier Transform (DFT)

SARPyX extensively uses frequency domain processing:

```python
import numpy as np

# Transform to frequency domain
spectrum = np.fft.fft2(sar_image)
spectrum_shifted = np.fft.fftshift(spectrum)

# Process in frequency domain
filtered_spectrum = apply_frequency_filter(spectrum_shifted)

# Transform back to spatial domain
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))
```

### Window Functions

For spectral analysis, SARPyX supports various window functions:

```python
# Available windows in SARPyX
window_types = ['HAMMING', 'HANNING', 'BLACKMAN', 'KAISER']

# Applied in sub-look analysis
sla.WeightFunctRange = 'HAMMING'
sla.WeightFunctAzim = 'HAMMING'
```

## Data Types and Formats

### Complex SAR Data

SAR data is inherently complex, representing both amplitude and phase:

```python
# Complex SAR data structure
complex_data = real_part + 1j * imaginary_part

# Extract amplitude and phase
amplitude = np.abs(complex_data)
phase = np.angle(complex_data)

# Intensity (power)
intensity = amplitude ** 2
```

### Multi-dimensional Arrays

SARPyX works with multi-dimensional arrays:

```python
# Typical data shapes
sar_image.shape       # (rows, cols) - Single image
sublook_data.shape    # (looks, rows, cols) - Multiple sub-looks
timeseries.shape      # (time, rows, cols) - Time series
```

## Coordinate Systems

### Sensor Coordinates
- **Range**: Perpendicular to flight direction
- **Azimuth**: Along flight direction

### Geographic Coordinates
- **Latitude/Longitude**: WGS84 coordinate system
- **UTM**: Universal Transverse Mercator projection

### Coordinate Transformations

```python
# SARPyX handles coordinate transformations through SNAP
from sarpyx.snap import GPT

gpt = GPT(product=product_path)
# Terrain correction converts from sensor to geographic coordinates
geocoded = gpt.TerrainCorrection(
    demName='SRTM 1Sec HGT',
    pixelSpacingInMeter=10.0
)
```

## Quality Metrics

### Focus Quality Assessment

```python
from sarpyx.processor.autofocus.metrics import (
    calculate_entropy,
    calculate_contrast,
    calculate_sharpness
)

# Assess image focus quality
entropy = calculate_entropy(sar_image)
contrast = calculate_contrast(sar_image)
sharpness = calculate_sharpness(sar_image)

print(f"Focus Quality - Entropy: {entropy:.3f}, Contrast: {contrast:.3f}")
```

### Statistical Measures

```python
# Common SAR statistics
mean_intensity = np.mean(np.abs(sar_data)**2)
std_intensity = np.std(np.abs(sar_data)**2)

# Equivalent Number of Looks (ENL)
enl = mean_intensity**2 / std_intensity**2
print(f"Equivalent Number of Looks: {enl:.2f}")
```

## Error Handling and Validation

### Parameter Validation

SARPyX includes built-in parameter validation:

```python
# Example validation in sub-look analysis
def validate_parameters(self):
    assert self.numberOfLooks > 0, "Number of looks must be positive"
    assert self.subLookBandwidth > 0, "Bandwidth must be positive"
    
    # Check frequency bounds
    max_freq = self.AzimBand / 2
    min_freq = -max_freq
    
    for i, freq in enumerate(self.freqMin):
        assert freq > min_freq, f"Sub-look {i} frequency too low"
        assert freq < max_freq, f"Sub-look {i} frequency too high"
```

### Debugging Tips

```python
# Enable verbose output for debugging
sla.SpectrumComputation(VERBOSE=True)
sla.Generation(VERBOSE=True)

# Check intermediate results
print(f"Spectrum shape: {sla.SpectrumOneDim.shape}")
print(f"Frequency vector: {sla.freqVect[:10]}...")  # First 10 elements
print(f"Sub-look frequencies: {sla.freqCentr}")
```

## Performance Considerations

### Memory Usage

Large SAR datasets require careful memory management:

```python
# Monitor memory usage
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}% ({memory.available / 1e9:.1f} GB available)")

# Process in chunks if needed
chunk_size = 1024
for i in range(0, data.shape[0], chunk_size):
    chunk = data[i:i+chunk_size]
    process_chunk(chunk)
```

### Computational Complexity

- **FFT operations**: O(N log N) complexity
- **Sub-look generation**: Linear in number of looks
- **Focus processing**: Depends on algorithm choice

## Common Workflows

### 1. Quality Assessment Workflow

```python
def assess_sar_quality(product_path):
    """Assess SAR data quality."""
    
    # Load data
    sla = SubLookAnalysis(product_path)
    
    # Compute spectrum
    sla.SpectrumComputation()
    
    # Calculate quality metrics
    entropy = calculate_entropy(sla.Box)
    
    # Generate report
    return {
        'entropy': entropy,
        'mean_intensity': np.mean(np.abs(sla.Box)**2),
        'data_shape': sla.Box.shape
    }
```

### 2. Multi-temporal Analysis

```python
def compare_temporal_data(path1, path2):
    """Compare two SAR acquisitions."""
    
    # Process both datasets
    sla1 = SubLookAnalysis(path1)
    sla2 = SubLookAnalysis(path2)
    
    # Configure identical parameters
    for sla in [sla1, sla2]:
        sla.numberOfLooks = 3
        sla.choice = 1
        sla.frequencyComputation()
        sla.SpectrumComputation()
        sla.Generation()
    
    # Calculate temporal coherence
    coherence = calculate_coherence(sla1.Looks, sla2.Looks)
    
    return coherence
```

## Next Steps

Now that you understand the basic concepts:

1. **Explore data formats**: [Data Formats](data_formats.md)
2. **Learn processing workflows**: [Processing Workflows](processing_workflows.md)
3. **Try scientific applications**: [Science Applications](science_applications.md)
4. **Integrate with SNAP**: [SNAP Integration](snap_integration.md)

## References

- [ESA SAR Fundamentals](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1)
- [SNAP Documentation](https://step.esa.int/main/doc/desktop/tutorials/)
- [IEEE Transactions on Geoscience and Remote Sensing](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=36)

Understanding these fundamentals will help you make the most of SARPyX's capabilities and choose the right processing approaches for your applications.
