# SarPyx Processor Module

A comprehensive Python refactoring of the sentinel1_decode C++ library for processing Sentinel-1 Level-0 SAR data.

## Overview

This module provides a complete pipeline for processing Sentinel-1 Level-0 data, including:

- **Packet Decoding**: Parse and decode Level-0 packets with all header information
- **Signal Processing**: FFT operations, filtering, windowing, and other DSP functions
- **Image Formation**: Range and azimuth compression for SAR focusing
- **State Vector Processing**: Satellite orbit determination and interpolation
- **Doppler Estimation**: Various methods for Doppler centroid and rate estimation

## Architecture

The module is organized into several key components:

### Core Classes

- **`L0Packet`**: Individual Level-0 packet with decoding capabilities
- **`S1Decoder`**: Main processing pipeline orchestrator
- **`StateVectors`**: Satellite orbit state vector management
- **`DopplerEstimator`**: Doppler frequency estimation algorithms

### Processing Modules

- **`constants.py`**: All constants, lookup tables, and parameters
- **`decoding_utils.py`**: Low-level bit manipulation and decoding functions
- **`signal_processing.py`**: Digital signal processing functions
- **`image_formation.py`**: SAR image formation algorithms
- **`state_vectors.py`**: Orbit processing and interpolation
- **`doppler.py`**: Doppler estimation methods

## Installation

The processor module is part of the SarPyx package. Ensure you have the required dependencies:

```bash
pip install numpy scipy
```

## Usage Examples

### Basic Packet Information

```python
from processor import L0Packet

# Load packets from file
packets = L0Packet.get_packets('data.dat', max_packets=100)

# Print information about first packet
packet = packets[0]
packet.print_primary_header()
packet.print_secondary_header()
packet.print_modes()
packet.print_pulse_info()
```

### Complete SAR Processing

```python
from processor import S1Decoder

# Initialize decoder with data file
decoder = S1Decoder('sentinel1_data.dat')

# Get available swaths
swaths = decoder.get_swath_names()
print(f"Available swaths: {swaths}")

# Process specific swath
swath_name = 'IW1'
range_compressed = decoder.get_range_compressed_swath(swath_name)
azimuth_compressed = decoder.get_azimuth_compressed_swath(swath_name)

# Create intensity image
intensity = np.abs(azimuth_compressed)**2
```

### State Vector Processing

```python
from processor import S1Decoder

decoder = S1Decoder('sentinel1_data.dat')
state_vectors = decoder.get_state_vectors()

# Print state vector information
state_vectors.print()

# Get satellite position at specific time
position = state_vectors.get_satellite_position(time=1000.0)
velocity = state_vectors.get_satellite_velocity(time=1000.0)
```

### Doppler Estimation

```python
from processor import DopplerEstimator
import numpy as np

# Create estimator
prf = 1000.0  # Hz
estimator = DopplerEstimator(prf)

# Estimate Doppler centroid from signal
doppler_freq = estimator.estimate_doppler_centroid_fft(signal_data)
doppler_rate = estimator.estimate_doppler_rate(image_data, dc_estimates)
```

## Command Line Tools

### Print Packet Information

```bash
python examples/print_packets.py data.dat --summary
python examples/print_packets.py data.dat --packet 5
```

### Process SAR Images

```bash
python examples/process_image.py data.dat --swath IW1 --output ./results
python examples/process_image.py data.dat --all-swaths --output ./results
python examples/process_image.py data.dat --state-vectors
```

## Key Features

### Packet Decoding

- Complete parsing of primary and secondary headers
- Support for all BAQ compression modes (BYPASS, FBAQ, SMFBAQ)
- Automatic packet organization by swath and signal type
- Comprehensive error handling and validation

### Signal Processing

- Optimized FFT operations using SciPy
- Various window functions (Hamming, Hanning, Blackman, Kaiser)
- Filtering operations (bandpass, lowpass, highpass)
- Cross-correlation and autocorrelation
- Hilbert transform and envelope detection

### Image Formation

- Range compression using matched filtering
- Azimuth compression for stripmap and TOPS modes
- Range-Doppler correction
- Multilook processing for speckle reduction
- Support for various processing bandwidths

### State Vector Processing

- Polynomial interpolation of orbit data
- Doppler frequency calculation for arbitrary targets
- Range rate computation
- Orbital period estimation

### Doppler Estimation

- Multiple estimation methods (FFT, correlation, energy balance)
- Multilook Doppler estimation with confidence metrics
- Doppler rate estimation using short-time analysis
- Quality assessment and validation

## Processing Modes

### Stripmap (SM)

Standard stripmap processing with continuous azimuth illumination:

```python
# SM mode processing
decoder = S1Decoder('sm_data.dat')
compressed_data = decoder.get_azimuth_compressed_swath('S1')
```

### Interferometric Wide Swath (IW)

TOPS mode processing with burst-based acquisition:

```python
# IW mode processing
decoder = S1Decoder('iw_data.dat')
burst_data = decoder.get_azimuth_compressed_burst('IW1', burst=0)
```

### Extra Wide Swath (EW)

Similar to IW but with wider coverage:

```python
# EW mode processing
decoder = S1Decoder('ew_data.dat')
swath_data = decoder.get_azimuth_compressed_swath('EW1')
```

## Error Handling

The module includes comprehensive error handling:

- File validation and existence checking
- Packet format validation
- Processing parameter validation
- Graceful degradation for missing data
- Detailed logging and error messages

## Performance Considerations

- Uses NumPy for efficient array operations
- SciPy FFT for optimized frequency domain processing
- Memory-efficient processing of large datasets
- Parallel processing support where applicable

## Testing

Run the test suite to verify installation:

```bash
python test_processor.py
```

This will test all major components and verify functionality.

## Differences from C++ Implementation

### Improvements

- More Pythonic API with clear method names
- Comprehensive error handling and validation
- Better documentation and examples
- Integrated logging system
- Modular design for easier extension

### Simplifications

- Some low-level optimizations replaced with NumPy operations
- Complex bit manipulation simplified using Python libraries
- Huffman decoding uses simplified implementation

### Extensions

- Additional signal processing functions
- Enhanced state vector interpolation
- Multiple Doppler estimation methods
- Quality assessment metrics

## Dependencies

- **NumPy**: Efficient array operations and mathematical functions
- **SciPy**: Advanced signal processing and scientific computing
- **Python 3.7+**: Modern Python features and type hints

## Future Enhancements

- GPU acceleration using CuPy
- Distributed processing support
- Additional processing algorithms
- Integration with other SAR processing tools
- Export to standard formats (GeoTIFF, NetCDF)

## Contributing

When contributing to this module:

1. Follow the established coding style (Google docstrings, type hints)
2. Add comprehensive tests for new functionality  
3. Update documentation and examples
4. Ensure compatibility with existing interfaces

## References

- SAR Space Packet Protocol Data Unit specification
- Sentinel-1 Level-0 Data Decoding Package documentation
- Original C++ implementation by Andrew Player

## License

This module follows the same license terms as the original C++ implementation.
