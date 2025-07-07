from typing import Dict, Any, Optional, Union, Tuple, Callable, List
import numpy as np
from scipy import signal

try:
    import torch
except ImportError:
    print('Unable to import torch module')
    torch = None
import scipy
from .aux import *


# ------------- ifft methods ------------- #
@flush_mem
@timing_decorator
def ifft2d(radar_data: Union[np.ndarray, torch.Tensor], backend: str = 'numpy', verbose: bool = False, n: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
    """Perform memory-efficient 2D inverse FFT on radar data.
    
    Args:
        radar_data: Input radar data array.
        backend: Backend to use ('numpy', 'torch', or 'scipy').
        verbose: Whether to print verbose output.
        n: Number of points for the transform. If None, uses input size.
        
    Returns:
        Processed radar data after 2D inverse FFT.
        
    Raises:
        ValueError: If backend is not supported.
    """
    if verbose:
        print('Performing 2D inverse FFT...')
    
    # Inverse FFT along azimuth dimension first
    if backend == 'numpy':
        radar_data = np.fft.ifft(radar_data, n=n, axis=0)
    elif backend == 'torch':
        radar_data = torch.fft.ifft(radar_data, n=n, dim=0)
    elif backend == 'scipy':
        radar_data = scipy.fft.ifft(radar_data, n=n, axis=0, workers=-1)
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    
    # Then inverse FFT along range dimension
    if backend == 'numpy':
        radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, n=n, axis=1), axes=1)
    elif backend == 'torch':
        radar_data = torch.fft.ifft(radar_data, n=n, dim=1)
        radar_data = torch.fft.ifftshift(radar_data, dim=1)
    elif backend == 'scipy':
        radar_data = scipy.fft.ifft(radar_data, n=n, axis=1, workers=-1)
        radar_data = scipy.fft.ifftshift(radar_data, axes=1)
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    
    if verbose:
        print(f'2D inverse FFT completed, data shape: {radar_data.shape}')
        print_memory()
    
    return radar_data

@flush_mem
@timing_decorator
def ifft_azimuth(
    radar_data: Union[np.ndarray, torch.Tensor], 
    backend: str = 'numpy', 
    verbose: bool = False,
    n: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Perform memory-efficient inverse FFT along azimuth dimension.
    
    Args:
        radar_data: Input radar data array.
        backend: Backend to use ('numpy', 'torch', or 'scipy').
        verbose: Whether to print verbose output.
        n: Number of points for the transform. If None, uses input size.
        
    Returns:
        Processed radar data after inverse FFT along azimuth dimension.
        
    Raises:
        ValueError: If backend is not supported.
    """
    if verbose:
        print('Performing inverse FFT along azimuth dimension...')
    
    if backend == 'numpy':
        radar_data = np.fft.ifft(radar_data, n=n, axis=0)
    elif backend == 'torch':
        radar_data = torch.fft.ifft(radar_data, n=n, dim=0)
    elif backend == 'scipy':
        radar_data = scipy.fft.ifft(radar_data, n=n, axis=0, workers=-1)
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    
    if verbose:
        print(f'Inverse FFT along azimuth completed, data shape: {radar_data.shape}')
        print_memory()
    
    return radar_data

@flush_mem
@timing_decorator
def ifft_range(
    radar_data: Union[np.ndarray, torch.Tensor], 
    backend: str = 'numpy', 
    verbose: bool = False,
    n: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Perform memory-efficient inverse FFT along range dimension.
    
    Args:
        radar_data: Input radar data array.
        backend: Backend to use ('numpy', 'torch', or 'scipy').
        verbose: Whether to print verbose output.
        n: Number of points for the transform. If None, uses input size.
        
    Returns:
        Processed radar data after inverse FFT along range dimension.
        
    Raises:
        ValueError: If backend is not supported.
    """
    if verbose:
        print('Performing inverse FFT along range dimension...')
    
    if backend == 'numpy':
        radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, n=n, axis=1), axes=1)
    elif backend == 'torch':
        radar_data = torch.fft.ifft(radar_data, n=n, dim=1)
        radar_data = torch.fft.ifftshift(radar_data, dim=1)
    elif backend == 'scipy':
        radar_data = scipy.fft.ifft(radar_data, n=n, axis=1, workers=-1)
        radar_data = scipy.fft.ifftshift(radar_data, axes=1)
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    
    if verbose:
        print(f'Inverse FFT along range completed, data shape: {radar_data.shape}')
        print_memory()
    
    return radar_data

# ------------- fft methods ------------- #
@flush_mem
@timing_decorator
def fft_range(
    radar_data: Union[np.ndarray, torch.Tensor], 
    backend: str = 'numpy', 
    verbose: bool = False,
    n: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Perform memory-efficient FFT along range dimension.
    
    Args:
        radar_data: Input radar data array.
        backend: Backend to use ('numpy', 'torch', or 'scipy').
        verbose: Whether to print verbose output.
        n: Number of points for the transform. If None, uses input size.
        
    Returns:
        Processed radar data after FFT along range dimension.
        
    Raises:
        ValueError: If backend is not supported.
    """
    if verbose:
        print('Performing FFT along range dimension...')
    
    if backend == 'numpy':
        radar_data = np.fft.fftshift(np.fft.fft(radar_data, n=n, axis=1), axes=1)
    elif backend == 'torch':
        radar_data = torch.fft.fft(radar_data, n=n, dim=1)
        radar_data = torch.fft.fftshift(radar_data, dim=1)
    elif backend == 'scipy':
        radar_data = scipy.fft.fftshift(scipy.fft.fft(radar_data, n=n, axis=1, workers=-1), axes=1)
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    
    if verbose:
        print(f'FFT along range completed, data shape: {radar_data.shape}')
        print_memory()
    
    return radar_data

@flush_mem
@timing_decorator
def fft_azimuth(
    radar_data: Union[np.ndarray, torch.Tensor], 
    backend: str = 'numpy', 
    verbose: bool = False,
    n: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Perform memory-efficient FFT along azimuth dimension.
    
    Args:
        radar_data: Input radar data array.
        backend: Backend to use ('numpy', 'torch', or 'scipy').
        verbose: Whether to print verbose output.
        n: Number of points for the transform. If None, uses input size.
        
    Returns:
        Processed radar data after FFT along azimuth dimension.
        
    Raises:
        ValueError: If backend is not supported.
    """
    if verbose:
        print('Performing FFT along azimuth dimension...')
    
    if backend == 'numpy':
        radar_data = np.fft.fft(radar_data, n=n, axis=0)
    elif backend == 'torch':
        radar_data = torch.fft.fft(radar_data, n=n, dim=0)
    elif backend == 'scipy':
        radar_data = scipy.fft.fft(radar_data, n=n, axis=0, workers=-1)
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    
    if verbose:
        print(f'FFT along azimuth completed, data shape: {radar_data.shape}')
        print_memory()
    
    return radar_data

# ------------- 2d fft methods ------------- #
@flush_mem
@timing_decorator
def fft2d(radar_data: Union[np.ndarray, torch.Tensor], backend: str = 'numpy', verbose: bool = False, n: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
    """Perform memory-efficient 2D FFT on radar data in range and azimuth dimensions.

    Args:
        radar_data: Input radar data array.
        backend: Backend to use ('numpy', 'torch', or 'scipy').
        verbose: Whether to print verbose output.
        n: Number of points for the transform. If None, uses input size.
        
    Returns:
        Processed radar data after 2D FFT.
        
    Raises:
        ValueError: If backend is not supported.
    """
    if verbose:
        print(f'FFT input data shape: {radar_data.shape}')
        print_memory()
    
    if backend == 'numpy':
        radar_data = _fft2d_numpy_efficient(radar_data, verbose, n)
    elif backend == 'torch':
        if torch is None:
            raise ValueError('PyTorch is not available')
        radar_data = _fft2d_torch_efficient(radar_data, verbose, n)
    elif backend == 'scipy':
        radar_data = _fft2d_scipy_efficient(radar_data, verbose, n)
    else:
        raise ValueError(f'Backend {backend} not supported')
    
    if verbose:
        print(f'FFT output data shape: {radar_data.shape}')
        print('- FFT performed successfully!')
        print_memory()
    
    return radar_data

def _fft2d_numpy_efficient(radar_data: np.ndarray, verbose: bool = False, n: Optional[int] = None) -> np.ndarray:
    """Perform memory-efficient 2D FFT using NumPy backend preserving original dimensions.
    
    Args:
        radar_data: Input radar data array.
        verbose: Whether to print verbose output.
        n: Number of points for the transform.
        
    Returns:
        Processed radar data after 2D FFT.
    """
    # Store original shape for verification
    original_shape = radar_data.shape
    if verbose:
        print(f'Original radar data shape: {original_shape}')
    
    # Ensure data is contiguous and maintain original precision
    if not radar_data.flags.c_contiguous:
        if verbose:
            print('Making data contiguous...')
        radar_data = np.ascontiguousarray(radar_data)
    
    # FFT each range line (axis=1)
    if verbose:
        print('Performing FFT along range dimension (axis=1)...')
    
    radar_data = np.fft.fft(radar_data, n=n, axis=1)
    
    if verbose:
        print(f'First FFT along range dimension completed, shape: {radar_data.shape}')
        print_memory()
    
    # FFT each azimuth line (axis=0) with fftshift
    if verbose:
        print('Performing FFT along azimuth dimension (axis=0) with fftshift...')
    
    radar_data = np.fft.fftshift(np.fft.fft(radar_data, n=n, axis=0), axes=0)
    
    if verbose:
        print(f'Second FFT along azimuth dimension completed, shape: {radar_data.shape}')
        print_memory()
    
    return radar_data

def _fft2d_torch_efficient(radar_data: torch.Tensor, verbose: bool = False, n: Optional[int] = None) -> torch.Tensor:
    """Perform memory-efficient 2D FFT using PyTorch backend preserving dimensions.
    
    Args:
        radar_data: Input radar data tensor.
        verbose: Whether to print verbose output.
        n: Number of points for the transform.
        
    Returns:
        Processed radar data after 2D FFT.
    """
    if torch is None:
        raise ValueError('PyTorch is not available')
        
    original_shape = radar_data.shape
    
    if verbose:
        print('Performing memory-efficient PyTorch FFT...')
        print_memory()
    
    # FFT each range line (axis=1)
    radar_data = torch.fft.fft(radar_data, n=n, dim=1)
    
    # FFT each azimuth line (axis=0) with fftshift
    radar_data = torch.fft.fftshift(torch.fft.fft(radar_data, n=n, dim=0), dim=0)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return radar_data

def _fft2d_scipy_efficient(radar_data: np.ndarray, verbose: bool = False, n: Optional[int] = None) -> np.ndarray:
    """Perform memory-efficient 2D FFT using SciPy backend preserving original dimensions.
    
    Args:
        radar_data: Input radar data array.
        verbose: Whether to print verbose output.
        n: Number of points for the transform.
        
    Returns:
        Processed radar data after 2D FFT.
        
    Raises:
        ImportError: If SciPy is not available.
    """

    # Store original shape for verification
    original_shape = radar_data.shape
    if verbose:
        print(f'SciPy FFT - Original radar data shape: {original_shape}')
    
    # Ensure data is contiguous and maintain original precision
    if not radar_data.flags.c_contiguous:
        if verbose:
            print('Making data contiguous...')
        radar_data = np.ascontiguousarray(radar_data)
    
    # FFT each range line (axis=1)
    if verbose:
        print('Performing SciPy FFT along range dimension (axis=1)...')

    radar_data = scipy.fft.fft(radar_data, n=n, axis=1, workers=-1)

    if verbose:
        print(f'First FFT along range dimension completed, shape: {radar_data.shape}')
        print_memory()
    
    # FFT each azimuth line (axis=0) with fftshift
    if verbose:
        print('Performing SciPy FFT along azimuth dimension (axis=0) with fftshift...')

    radar_data = scipy.fft.fftshift(scipy.fft.fft(radar_data, n=n, axis=0, workers=-1), axes=0)

    if verbose:
        print(f'Second FFT along azimuth dimension completed, shape: {radar_data.shape}')
        print_memory()
    
    return radar_data


# ------------- 2d Convolution methods ------------- #
@flush_mem
@timing_decorator
def linear_convolution_2d(
    signal1: Union[np.ndarray, torch.Tensor],
    signal2: Union[np.ndarray, torch.Tensor],
    backend: str = 'numpy',
    verbose: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Perform 2D linear convolution of two signals using FFT with proper padding.
    
    Args:
        signal1: First input signal array.
        signal2: Second input signal (kernel) array.
        backend: Backend to use ('numpy', 'torch', or 'scipy').
        verbose: Whether to print verbose output.
        
    Returns:
        Result of linear convolution with size (M+N-1, P+Q-1) where
        signal1 is MxP and signal2 is NxQ.
        
    Raises:
        ValueError: If backend is not supported or arrays have different backends.
        AssertionError: If inputs are not 2D arrays.
    """
    assert signal1.ndim == 2, f'signal1 must be 2D, got {signal1.ndim}D'
    assert signal2.ndim == 2, f'signal2 must be 2D, got {signal2.ndim}D'
    
    if verbose:
        print(f'Linear convolution: signal1 shape {signal1.shape}, signal2 shape {signal2.shape}')
    
    # Get dimensions
    m1, p1 = signal1.shape
    m2, p2 = signal2.shape
    
    # Output size for linear convolution
    output_rows = m1 + m2 - 1
    output_cols = p1 + p2 - 1
    
    if verbose:
        print(f'Output size will be: ({output_rows}, {output_cols})')
    
    if backend == 'numpy':
        return _linear_conv_numpy(signal1, signal2, output_rows, output_cols, verbose)
    elif backend == 'torch':
        if torch is None:
            raise ValueError('PyTorch is not available')
        return _linear_conv_torch(signal1, signal2, output_rows, output_cols, verbose)
    elif backend == 'scipy':
        return _linear_conv_scipy(signal1, signal2, output_rows, output_cols, verbose)
    else:
        raise ValueError(f'Unsupported backend: {backend}')

def _linear_conv_numpy(
    signal1: np.ndarray,
    signal2: np.ndarray, 
    output_rows: int,
    output_cols: int,
    verbose: bool = False
) -> np.ndarray:
    """Perform linear convolution using NumPy with zero-padding.
    
    Args:
        signal1: First input signal.
        signal2: Second input signal.
        output_rows: Number of rows in output.
        output_cols: Number of columns in output.
        verbose: Whether to print verbose output.
        
    Returns:
        Convolution result.
    """
    if verbose:
        print('Padding signals for linear convolution...')
    
    # Zero-pad both signals to output size
    padded_signal1 = np.zeros((output_rows, output_cols), dtype=signal1.dtype)
    padded_signal2 = np.zeros((output_rows, output_cols), dtype=signal2.dtype)
    
    # Place original signals at the beginning
    padded_signal1[:signal1.shape[0], :signal1.shape[1]] = signal1
    padded_signal2[:signal2.shape[0], :signal2.shape[1]] = signal2
    
    if verbose:
        print('Performing FFT-based convolution...')
        print_memory()
    
    # FFT both signals
    fft_signal1 = np.fft.fft2(padded_signal1)
    fft_signal2 = np.fft.fft2(padded_signal2)
    
    # Multiply in frequency domain
    fft_result = fft_signal1 * fft_signal2
    
    # IFFT to get convolution result
    result = np.fft.ifft2(fft_result)
    
    # Take real part (imaginary part should be negligible for real inputs)
    if np.iscomplexobj(signal1) or np.iscomplexobj(signal2):
        result = result
    else:
        result = np.real(result)
    
    if verbose:
        print(f'Linear convolution completed, result shape: {result.shape}')
        print_memory()
    
    return result

def _linear_conv_torch(
    signal1: torch.Tensor,
    signal2: torch.Tensor,
    output_rows: int, 
    output_cols: int,
    verbose: bool = False
) -> torch.Tensor:
    """Perform linear convolution using PyTorch with zero-padding.
    
    Args:
        signal1: First input signal tensor.
        signal2: Second input signal tensor.
        output_rows: Number of rows in output.
        output_cols: Number of columns in output.
        verbose: Whether to print verbose output.
        
    Returns:
        Convolution result tensor.
    """
    if torch is None:
        raise ValueError('PyTorch is not available')
    
    if verbose:
        print('Padding signals for linear convolution...')
    
    # Ensure tensors are on the same device
    device = signal1.device
    signal2 = signal2.to(device)
    
    # Zero-pad both signals to output size
    padded_signal1 = torch.zeros((output_rows, output_cols), 
                                dtype=signal1.dtype, device=device)
    padded_signal2 = torch.zeros((output_rows, output_cols), 
                                dtype=signal2.dtype, device=device)
    
    # Place original signals at the beginning
    padded_signal1[:signal1.shape[0], :signal1.shape[1]] = signal1
    padded_signal2[:signal2.shape[0], :signal2.shape[1]] = signal2
    
    if verbose:
        print('Performing FFT-based convolution...')
        print_memory()
    
    # FFT both signals
    fft_signal1 = torch.fft.fft2(padded_signal1)
    fft_signal2 = torch.fft.fft2(padded_signal2)
    
    # Multiply in frequency domain
    fft_result = fft_signal1 * fft_signal2
    
    # IFFT to get convolution result
    result = torch.fft.ifft2(fft_result)
    
    # Take real part for real inputs
    if not torch.is_complex(signal1) and not torch.is_complex(signal2):
        result = torch.real(result)
    
    if verbose:
        print(f'Linear convolution completed, result shape: {result.shape}')
        print_memory()
    
    # Clean up GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result

def _linear_conv_scipy(
    signal1: np.ndarray,
    signal2: np.ndarray, 
    output_rows: int,
    output_cols: int,
    verbose: bool = False
) -> np.ndarray:
    """Perform linear convolution using SciPy with zero-padding.
    
    Args:
        signal1: First input signal.
        signal2: Second input signal.
        output_rows: Number of rows in output.
        output_cols: Number of columns in output.
        verbose: Whether to print verbose output.
        
    Returns:
        Convolution result.
    """
    if verbose:
        print('Padding signals for linear convolution with SciPy...')
    
    # Zero-pad both signals to output size
    padded_signal1 = np.zeros((output_rows, output_cols), dtype=signal1.dtype)
    padded_signal2 = np.zeros((output_rows, output_cols), dtype=signal2.dtype)
    
    # Place original signals at the beginning
    padded_signal1[:signal1.shape[0], :signal1.shape[1]] = signal1
    padded_signal2[:signal2.shape[0], :signal2.shape[1]] = signal2
    
    if verbose:
        print('Performing FFT-based convolution with SciPy...')
        print_memory()
    
    # FFT both signals
    fft_signal1 = scipy.fft.fft2(padded_signal1, workers=-1)
    fft_signal2 = scipy.fft.fft2(padded_signal2, workers=-1)
    
    # Multiply in frequency domain
    fft_result = fft_signal1 * fft_signal2
    
    # IFFT to get convolution result
    result = scipy.fft.ifft2(fft_result, workers=-1)
    
    # Take real part (imaginary part should be negligible for real inputs)
    if np.iscomplexobj(signal1) or np.iscomplexobj(signal2):
        result = result
    else:
        result = np.real(result)
    
    if verbose:
        print(f'Linear convolution with SciPy completed, result shape: {result.shape}')
        print_memory()
    
    return result




# ------------- 1d Convolution methods ------------- #


@flush_mem
@timing_decorator
def correlation(
    x1: np.ndarray, 
    nomchip: np.ndarray, 
    backend: str = 'numpy',
    verbose: bool = False
) -> np.ndarray:
    """Perform cross-correlation between radar data and nominal chip.
    
    Args:
        x1: Input radar data array of shape (azimuth, range).
        nomchip: Nominal chip array for correlation.
        backend: Backend to use ('numpy', 'torch', or 'scipy').
        verbose: Whether to print verbose output.
        
    Returns:
        Cross-correlation result with shape (range, azimuth).
        
    Raises:
        ValueError: If backend is not supported.
        AssertionError: If input validation fails.
    """
    assert isinstance(x1, np.ndarray), f'x1 must be numpy array, got {type(x1)}'
    assert isinstance(nomchip, np.ndarray), f'nomchip must be numpy array, got {type(nomchip)}'
    assert x1.ndim == 2, f'x1 must be 2D array, got {x1.ndim}D'
    assert nomchip.ndim == 1, f'nomchip must be 1D array, got {nomchip.ndim}D'
    
    if verbose:
        print(f'Performing correlation: x1 shape {x1.shape}, nomchip shape {nomchip.shape}')
        print_memory()
    
    if backend == 'numpy':
        return _correlation_numpy(x1, nomchip, verbose)
    elif backend == 'torch':
        if torch is None:
            raise ValueError('PyTorch is not available')
        return _correlation_torch(x1, nomchip, verbose)
    elif backend == 'scipy':
        return _correlation_scipy(x1, nomchip, verbose)
    else:
        raise ValueError(f'Unsupported backend: {backend}')

def _correlation_numpy(x1: np.ndarray, nomchip: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Perform correlation using NumPy backend.
    
    Args:
        x1: Input radar data array of shape (azimuth, range).
        nomchip: Nominal chip array.
        verbose: Whether to print verbose output.
        
    Returns:
        Cross-correlation result with shape (azimuth, correlation_length).
    """
    # Calculate correct correlation result length
    corr_length = x1.shape[1] + len(nomchip) - 1
    
    # Initialize output array with correct dimensions - shape should be (azimuth, range)
    sol = np.zeros((x1.shape[0], corr_length), dtype=np.complex128)
    
    if verbose:
        print(f'Correlating {x1.shape[0]} azimuth lines...')
        print(f'Input x1 shape: {x1.shape}, nomchip length: {len(nomchip)}')
        print(f'Expected correlation length: {corr_length}')
    
    # Perform correlation for each azimuth line
    for k in range(x1.shape[0]):
        corr_result = np.correlate(x1[k, :], nomchip, mode='full')
        
        # Handle potential size mismatch
        actual_length = len(corr_result)
        if actual_length != corr_length:
            if verbose:
                print(f'Warning: Expected length {corr_length}, got {actual_length}')
            # Resize sol if needed
            if actual_length > corr_length:
                sol = np.zeros((x1.shape[0], actual_length), dtype=np.complex128)
                corr_length = actual_length
        
        sol[k, :len(corr_result)] = corr_result
    
    # Extract valid part (equivalent to MATLAB's behavior)
    start_idx = x1.shape[1]
    sol = sol[:, :start_idx]


    if verbose:
        print(f'Correlation completed, output shape: {sol.shape}')
        print_memory()
    
    return sol

def _correlation_torch(x1: np.ndarray, nomchip: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Perform correlation using PyTorch backend.
    
    Args:
        x1: Input radar data array.
        nomchip: Nominal chip array.
        verbose: Whether to print verbose output.
        
    Returns:
        Cross-correlation result as numpy array.
    """
    if torch is None:
        raise ValueError('PyTorch is not available')
    
    # Convert to torch tensors
    x1_torch = torch.from_numpy(x1)
    nomchip_torch = torch.from_numpy(nomchip)
    
    # Calculate correct correlation result length
    expected_length = x1.shape[1] + len(nomchip) - 1
    
    if verbose:
        print(f'Correlating {x1.shape[0]} azimuth lines with PyTorch...')
        print(f'Input x1 shape: {x1.shape}, nomchip length: {len(nomchip)}')
        print(f'Expected correlation length: {expected_length}')
    
    # Store correlation results in a list first to handle variable lengths
    correlation_results = []
    
    # Perform correlation for each azimuth line
    for k in range(x1.shape[0]):
        # Use torch.nn.functional.conv1d for correlation
        x_padded = torch.nn.functional.pad(x1_torch[k, :].unsqueeze(0).unsqueeze(0), 
                                         (len(nomchip) - 1, len(nomchip) - 1))
        nomchip_flipped = torch.flip(nomchip_torch, [0]).unsqueeze(0).unsqueeze(0)
        corr_result = torch.nn.functional.conv1d(x_padded, nomchip_flipped)
        corr_result_squeezed = corr_result.squeeze()
        correlation_results.append(corr_result_squeezed)
    
    # Find maximum length and create output array
    max_length = max(len(result) for result in correlation_results)
    sol = torch.zeros((x1.shape[0], max_length), dtype=torch.complex128)
    
    # Fill the output array
    for k, result in enumerate(correlation_results):
        sol[k, :len(result)] = result
    
    # Extract valid part
    start_idx = min(x1.shape[1] - 1, sol.shape[1] - 1)
    if start_idx < sol.shape[1]:
        sol = sol[:, start_idx:]
    
    if verbose:
        print(f'PyTorch correlation completed, output shape: {sol.shape}')
        print_memory()
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return sol.numpy()

def _correlation_scipy(x1: np.ndarray, nomchip: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Perform correlation using SciPy backend.
    
    Args:
        x1: Input radar data array.
        nomchip: Nominal chip array.
        verbose: Whether to print verbose output.
        
    Returns:
        Cross-correlation result.
    """
    # Calculate expected correlation result length
    expected_length = x1.shape[1] + len(nomchip) - 1
    
    if verbose:
        print(f'Correlating {x1.shape[0]} azimuth lines with SciPy...')
        print(f'Input x1 shape: {x1.shape}, nomchip length: {len(nomchip)}')
        print(f'Expected correlation length: {expected_length}')
    
    # Store correlation results in a list first to handle variable lengths
    correlation_results = []
    
    # Perform correlation for each azimuth line using scipy
    for k in range(x1.shape[0]):
        corr_result = signal.correlate(x1[k, :], nomchip, mode='full')
        correlation_results.append(corr_result)
    
    # Find maximum length and create output array
    max_length = max(len(result) for result in correlation_results)
    sol = np.zeros((x1.shape[0], max_length), dtype=np.complex128)
    
    # Fill the output array
    for k, result in enumerate(correlation_results):
        sol[k, :len(result)] = result
    
    # Extract valid part
    start_idx = min(x1.shape[1] - 1, sol.shape[1] - 1)
    if start_idx < sol.shape[1]:
        sol = sol[:, start_idx:]
    
    if verbose:
        print(f'SciPy correlation completed, output shape: {sol.shape}')
        print_memory()
    
    return sol









def linear_convolution(
    x: Union[np.ndarray, list], 
    h: Union[np.ndarray, list], 
    mode: str = 'same',
    axis: Optional[int] = None,
    zero_pad_length: Optional[int] = None,
    method: str = 'fft'
) -> np.ndarray:
    """Perform linear convolution of two signals using scipy with enhanced control.
    
    Args:
        x: Input signal or 2D array.
        h: Filter/impulse response (1D signal).
        mode: Convolution mode ('full', 'valid', 'same').
        axis: Axis along which to perform convolution for 2D arrays.
        zero_pad_length: Additional zero-padding length to prevent circular convolution artifacts.
        method: Convolution method ('fft' for FFT-based, 'direct' for direct convolution).
    
    Returns:
        Convolved signal with proper phase preservation.
        
    Raises:
        AssertionError: If input validation fails.
        ValueError: If parameters are invalid.
    """
    valid_modes = ['full', 'valid', 'same']
    assert mode in valid_modes, f'Mode must be one of {valid_modes}, got {mode}'
    
    # Convert inputs to numpy arrays and preserve complex dtype
    x = np.asarray(x)
    h = np.asarray(h)
    
    # Enhanced input validation
    assert x.size > 0, f'Input signal cannot be empty, got size: {x.size}'
    assert h.size > 0, f'Filter signal cannot be empty, got size: {h.size}'
    assert h.ndim == 1, f'Filter must be 1D array, got {h.ndim}D'
    
    # Validate zero_pad_length
    if zero_pad_length is not None:
        assert isinstance(zero_pad_length, int), f'zero_pad_length must be int, got {type(zero_pad_length)}'
        assert zero_pad_length > 0, f'zero_pad_length must be positive, got {zero_pad_length}'
    
    # Determine output dtype to preserve complex data
    output_dtype = np.result_type(x.dtype, h.dtype)
    
    # Handle zero padding for aliasing prevention
    if zero_pad_length is None:
        zero_pad_length = len(h) - 1
    
    # Additional safety check
    assert zero_pad_length >= 0, f'Invalid zero_pad_length: {zero_pad_length}'
    
    # For 2D arrays with axis specified
    if x.ndim > 1 and axis is not None:
        assert axis < x.ndim, f'Axis {axis} is out of bounds for array with {x.ndim} dimensions'
        assert axis >= 0, f'Axis must be non-negative, got {axis}'
        
        if method == 'fft':
            # Use FFT-based convolution for better phase preservation
            return _fft_convolution_axis(x, h, axis, mode, zero_pad_length, output_dtype)
        else:
            # Add zero padding along specified axis
            if zero_pad_length > 0:
                pad_width = [(0, 0)] * x.ndim
                pad_width[axis] = (zero_pad_length, zero_pad_length)
                x_padded = np.pad(x, pad_width, mode='constant', constant_values=0)
            else:
                x_padded = x
            
            # Apply convolution along specified axis
            result = np.apply_along_axis(
                lambda arr: signal.convolve(arr, h, mode=mode), 
                axis=axis, 
                arr=x_padded
            )
            
            # Remove padding effects if needed for 'same' or 'valid' modes
            if zero_pad_length > 0 and mode in ['same', 'valid']:
                slice_obj = [slice(None)] * result.ndim
                if mode == 'same':
                    start_idx = zero_pad_length
                    end_idx = start_idx + x.shape[axis]
                    slice_obj[axis] = slice(start_idx, end_idx)
                elif mode == 'valid':
                    start_idx = zero_pad_length + len(h) - 1
                    end_idx = start_idx + max(0, x.shape[axis] - len(h) + 1)
                    slice_obj[axis] = slice(start_idx, end_idx)
                result = result[tuple(slice_obj)]
            
            return result.astype(output_dtype)
    
    else:
        # Standard 1D convolution
        if zero_pad_length > 0:
            x_padded = np.pad(x.flatten(), (zero_pad_length, zero_pad_length), 
                            mode='constant', constant_values=0)
        else:
            x_padded = x.flatten()
        
        result = signal.convolve(x_padded, h, mode=mode)
        
        # Remove padding effects for 'same' and 'valid' modes
        if zero_pad_length > 0 and mode in ['same', 'valid']:
            if mode == 'same':
                start_idx = zero_pad_length
                end_idx = start_idx + len(x.flatten())
                result = result[start_idx:end_idx]
            elif mode == 'valid':
                start_idx = zero_pad_length + len(h) - 1
                end_idx = start_idx + max(0, len(x.flatten()) - len(h) + 1)
                result = result[start_idx:end_idx]
        
        return result.astype(output_dtype)


def _fft_convolution_axis(
    x: np.ndarray, 
    h: np.ndarray, 
    axis: int, 
    mode: str, 
    zero_pad_length: int, 
    output_dtype: np.dtype
) -> np.ndarray:
    """Perform FFT-based convolution along a specific axis for better phase preservation.
    
    Args:
        x: Input array.
        h: Filter array (1D).
        axis: Axis along which to convolve.
        mode: Convolution mode.
        zero_pad_length: Zero padding length.
        output_dtype: Output data type.
    
    Returns:
        Convolved array with preserved phase information.
        
    Raises:
        AssertionError: If validation fails.
    """
    # Get the size along the specified axis
    n_signal = x.shape[axis]
    n_filter = len(h)
    
    # Enhanced validation
    assert n_signal > 0, f'Signal size along axis {axis} must be positive, got {n_signal}'
    assert n_filter > 0, f'Filter size must be positive, got {n_filter}'
    
    # Calculate optimal FFT size
    if zero_pad_length is None or zero_pad_length <= 0:
        conv_size = n_signal + n_filter - 1
    else:
        conv_size = n_signal + 2 * zero_pad_length + n_filter - 1
    
    assert conv_size > 0, f'Convolution size must be positive, got {conv_size}'
    
    # Use next fast FFT size
    fft_size = scipy.fft.next_fast_len(conv_size)
    assert fft_size > 0, f'FFT size must be positive, got {fft_size}'
    
    # Create padded versions
    x_padded = x
    if zero_pad_length > 0:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (zero_pad_length, zero_pad_length)
        x_padded = np.pad(x, pad_width, mode='constant', constant_values=0)
    
    # Pad filter to match FFT size
    h_padded = np.zeros(fft_size, dtype=h.dtype)
    h_padded[:len(h)] = h
    
    # Take FFT of filter
    H = scipy.fft.fft(h_padded)
    
    # Apply convolution along axis using broadcasting
    def convolve_1d(signal_1d: np.ndarray) -> np.ndarray:
        """Apply 1D convolution to a single array."""
        assert len(signal_1d) > 0, 'Signal cannot be empty'
        
        # Pad signal to FFT size
        signal_padded = np.zeros(fft_size, dtype=signal_1d.dtype)
        signal_padded[:len(signal_1d)] = signal_1d
        
        # FFT, multiply, IFFT
        X = scipy.fft.fft(signal_padded)
        Y = X * H
        y = scipy.fft.ifft(Y)
        
        # Extract valid part based on mode
        if mode == 'full':
            return y[:n_signal + n_filter - 1]
        elif mode == 'valid':
            start = n_filter - 1
            end = start + max(0, n_signal - n_filter + 1)
            return y[start:end]
        else:  # mode == 'same'
            start = n_filter // 2
            end = start + n_signal
            return y[start:end]
    
    # Apply along axis
    result = np.apply_along_axis(convolve_1d, axis, x_padded)
    
    # Remove zero padding effects if needed
    if zero_pad_length > 0 and mode == 'same':
        slice_obj = [slice(None)] * result.ndim
        slice_obj[axis] = slice(zero_pad_length, zero_pad_length + x.shape[axis])
        result = result[tuple(slice_obj)]
    
    return result.astype(output_dtype)
    




# ============= Example usage ============= #

if __name__ == '__main__':
    # Example usage
    data = np.random.rand(100, 100)  # Example radar data
    result = ifft2d(data, backend='numpy', verbose=True)
    print(result.shape)
    
    # If torch is available, you can also test with torch tensors
    if torch:
        torch_data = torch.rand(100, 100)
        torch_result = ifft2d(torch_data, backend='torch', verbose=True)
        print(torch_result.shape)