import numpy as np
from concurrent.futures import ProcessPoolExecutor

def fft_slice(data_slice):
    """
    Perform 2D FFT on a given data slice and return the result.

    Parameters:
    data_slice (np.ndarray): Slice of radar data to be transformed.

    Returns:
    np.ndarray: Transformed data slice.
    """
    return np.fft.fftshift(np.fft.fft2(data_slice), axes=(0, 1))

def perform_fft_custom(radar_data, num_slices=42):
        radar_data = np.ascontiguousarray(radar_data)
        
        # Determine the number of slices and create slices
        num_slices = num_slices  # This can be adjusted based on your requirements
        slices = np.array_split(radar_data, num_slices, axis=0)
        
        # Use ProcessPoolExecutor to parallelize the FFT computation
        with ProcessPoolExecutor(max_workers=num_slices) as executor:
            results = list(executor.map(fft_slice, slices))
        
        # Combine the results back into a single array
        return np.concatenate(results, axis=0)
