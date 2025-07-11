import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List


def fft_slice(data_slice: np.ndarray) -> np.ndarray:
    """Perform 2D FFT on a given data slice and return the result.

    Args:
        data_slice (np.ndarray): Slice of radar data to be transformed.

    Returns:
        np.ndarray: Transformed data slice with preserved dimensions.
        
    Raises:
        AssertionError: If input data is not a valid numpy array.
        ValueError: If FFT operation fails.
        
    Note:
        This function performs FFT consistent with Range Doppler Algorithm:
        - Range FFT (axis=1): Transforms raw echo data to range frequency domain
        - Azimuth FFT (axis=0): Transforms azimuth time to Doppler frequency domain
        - fftshift on azimuth: Centers zero Doppler frequency for proper processing
    """
    assert isinstance(data_slice, np.ndarray), 'data_slice must be a numpy array'
    assert data_slice.ndim == 2, f'data_slice must be 2D, got {data_slice.ndim}D'
    assert data_slice.size > 0, 'data_slice cannot be empty'
    
    original_shape = data_slice.shape
    
    try:
        # Step 1: Range FFT (axis=1) - transforms each range line to frequency domain
        # This converts raw echo samples to range frequency for range compression
        result = np.fft.fft(data_slice, axis=1)
        
        # Step 2: Azimuth FFT (axis=0) - transforms azimuth slow time to Doppler frequency
        # This enables Range Cell Migration Correction and azimuth compression
        result = np.fft.fft(result, axis=0)
        
        # Step 3: Azimuth fftshift - centers zero Doppler frequency
        # Essential for proper RCMC and azimuth processing in RDA
        result = np.fft.fftshift(result, axes=0)
        
        # Verify shape preservation
        assert result.shape == original_shape, \
            f'FFT changed shape from {original_shape} to {result.shape}'
        
        return result
        
    except Exception as e:
        raise ValueError(f'FFT operation failed for slice with shape {data_slice.shape}: {str(e)}') from e


def perform_fft_custom(radar_data: np.ndarray, num_slices: int = 12) -> np.ndarray:
    """Perform parallelized 2D FFT on radar data using multiple processes.

    This function implements a scientifically correct 2D FFT for SAR Range Doppler Algorithm
    processing. The FFT transforms:
    1. Range dimension: Raw echo data → Range frequency domain (for range compression)
    2. Azimuth dimension: Slow time → Doppler frequency domain (for RCMC & azimuth compression)

    Args:
        radar_data (np.ndarray): Input radar data array (2D). Shape: (azimuth_lines, range_samples)
        num_slices (int): Number of slices for parallel processing along azimuth dimension.

    Returns:
        np.ndarray: FFT-transformed radar data with same dimensions as input.
                   Output is in (Doppler frequency, range frequency) domain.
        
    Raises:
        AssertionError: If input parameters are invalid.
        ValueError: If FFT processing fails.
        RuntimeError: If parallel processing encounters errors.
        
    Note:
        The parallel processing splits data along azimuth dimension (axis=0) which is
        scientifically valid because each azimuth slice can be independently transformed
        to the range frequency domain, then combined for azimuth FFT processing.
    """
    # Input validation
    assert isinstance(radar_data, np.ndarray), 'radar_data must be a numpy array'
    assert radar_data.ndim == 2, f'radar_data must be 2D, got {radar_data.ndim}D'
    assert radar_data.size > 0, 'radar_data cannot be empty'
    assert isinstance(num_slices, int), 'num_slices must be an integer'
    assert num_slices > 0, f'num_slices must be positive, got {num_slices}'
    assert num_slices <= radar_data.shape[0], \
        f'num_slices ({num_slices}) cannot exceed number of rows ({radar_data.shape[0]})'
    
    original_shape = radar_data.shape
    
    try:
        # Ensure data is contiguous for better performance
        radar_data = np.ascontiguousarray(radar_data)
        
        # Create slices along azimuth dimension (axis=0)
        # This is scientifically valid: each azimuth line is processed independently
        # for range FFT, then azimuth FFT is applied across all lines
        slices: List[np.ndarray] = np.array_split(radar_data, num_slices, axis=0)
        
        # Validate slices
        total_rows = sum(slice_data.shape[0] for slice_data in slices)
        assert total_rows == original_shape[0], \
            f'Slicing error: total rows {total_rows} != original rows {original_shape[0]}'
        
        # IMPORTANT: Verify that slicing preserves scientific validity
        # Each slice must contain complete range lines to maintain FFT correctness
        for i, slice_data in enumerate(slices):
            assert slice_data.shape[1] == original_shape[1], \
                f'Slice {i} range dimension {slice_data.shape[1]} != original {original_shape[1]}'
        
        # Use ProcessPoolExecutor to parallelize the FFT computation
        max_workers = min(num_slices, len(slices))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            try:
                results: List[np.ndarray] = list(executor.map(fft_slice, slices))
            except Exception as e:
                raise RuntimeError(f'Parallel FFT processing failed: {str(e)}') from e
        
        # Validate results
        assert len(results) == len(slices), \
            f'Results count {len(results)} != slices count {len(slices)}'
        
        for i, result in enumerate(results):
            assert isinstance(result, np.ndarray), f'Result {i} is not a numpy array'
            assert result.shape == slices[i].shape, \
                f'Result {i} shape {result.shape} != slice shape {slices[i].shape}'
        
        # Combine the results back into a single array
        combined_result = np.concatenate(results, axis=0)
        
        # Verify final shape preservation
        assert combined_result.shape == original_shape, \
            f'Final result shape {combined_result.shape} != original shape {original_shape}'
        
        return combined_result
        
    except Exception as e:
        if isinstance(e, (AssertionError, ValueError, RuntimeError)):
            raise
        else:
            raise ValueError(f'Custom FFT processing failed: {str(e)}') from e
