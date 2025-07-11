from typing import Dict, Any, Optional, Union, Tuple, Callable, List
try:
    import torch
except ImportError:
    print('Unable to import torch module')
    torch = None
import numpy as np
import gc
from functools import wraps
import psutil
import time


__VTIMING__ = False


# ---------- Padding/Trimming function ----------
def pad_array(
    array: Union[np.ndarray, torch.Tensor],
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
    mode: str = 'constant',
    constant_value: float = 0.0
) -> Union[np.ndarray, torch.Tensor]:
    """Pad array with specified padding on each side.
    
    Args:
        array: Input array to pad.
        left: Number of elements to pad on the left (axis=1 for 2D).
        right: Number of elements to pad on the right (axis=1 for 2D).
        top: Number of elements to pad on the top (axis=0 for 2D).
        bottom: Number of elements to pad on the bottom (axis=0 for 2D).
        mode: Padding mode ('constant', 'edge', 'reflect', 'wrap').
        constant_value: Value to use for constant padding.
        
    Returns:
        Padded array with same type as input.
        
    Raises:
        ValueError: If padding values are negative.
        TypeError: If array type is not supported.
    """
    assert all(pad >= 0 for pad in [left, right, top, bottom]), 'All padding values must be non-negative'
    
    if all(pad == 0 for pad in [left, right, top, bottom]):
        return array
    
    if isinstance(array, np.ndarray):
        if len(array.shape) == 1:
            # For 1D arrays, use left/right as start/end padding
            pad_width = [(left, right)]
        elif len(array.shape) == 2:
            # For 2D arrays: axis 0 is top/bottom, axis 1 is left/right
            pad_width = [(top, bottom), (left, right)]
        else:
            # For higher dimensions, pad only the last two dimensions
            pad_width = [(0, 0) for _ in range(len(array.shape) - 2)]
            pad_width.extend([(top, bottom), (left, right)])
        
        return np.pad(array, pad_width, mode=mode, constant_values=constant_value)
    
    elif torch is not None and isinstance(array, torch.Tensor):
        # PyTorch padding format: (pad_left, pad_right, pad_top, pad_bottom, ...)
        # Padding is specified from the last dimension to the first
        if len(array.shape) == 1:
            pad_spec = [left, right]
        elif len(array.shape) == 2:
            pad_spec = [left, right, top, bottom]
        else:
            # For higher dimensions, pad only the last two dimensions
            pad_spec = [0] * (2 * (len(array.shape) - 2))
            pad_spec.extend([left, right, top, bottom])
        
        if mode == 'constant':
            return torch.nn.functional.pad(array, pad_spec, mode='constant', value=constant_value)
        else:
            return torch.nn.functional.pad(array, pad_spec, mode=mode)
    
    else:
        raise TypeError(f'Unsupported array type: {type(array)}')

def trim_array(
    array: Union[np.ndarray, torch.Tensor],
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0
) -> Union[np.ndarray, torch.Tensor]:
    """Trim array by removing specified number of elements from each side.
    
    Args:
        array: Input array to trim.
        left: Number of elements to remove from the left (axis=1 for 2D).
        right: Number of elements to remove from the right (axis=1 for 2D).
        top: Number of elements to remove from the top (axis=0 for 2D).
        bottom: Number of elements to remove from the bottom (axis=0 for 2D).
        
    Returns:
        Trimmed array with same type as input.
        
    Raises:
        ValueError: If trim values are negative or exceed array dimensions.
    """
    assert all(trim >= 0 for trim in [left, right, top, bottom]), 'All trim values must be non-negative'
    
    if all(trim == 0 for trim in [left, right, top, bottom]):
        return array
    
    if len(array.shape) == 1:
        # For 1D arrays, use left/right as start/end trimming
        assert left + right < array.shape[0], f'Total trim ({left + right}) exceeds array length ({array.shape[0]})'
        end_idx = array.shape[0] - right if right > 0 else array.shape[0]
        return array[left:end_idx]
    
    elif len(array.shape) == 2:
        # For 2D arrays: axis 0 is top/bottom, axis 1 is left/right
        assert top + bottom < array.shape[0], f'Vertical trim ({top + bottom}) exceeds array height ({array.shape[0]})'
        assert left + right < array.shape[1], f'Horizontal trim ({left + right}) exceeds array width ({array.shape[1]})'
        
        row_end = array.shape[0] - bottom if bottom > 0 else array.shape[0]
        col_end = array.shape[1] - right if right > 0 else array.shape[1]
        
        return array[top:row_end, left:col_end]
    
    else:
        # For higher dimensions, trim only the last two dimensions
        assert top + bottom < array.shape[-2], f'Vertical trim ({top + bottom}) exceeds array height ({array.shape[-2]})'
        assert left + right < array.shape[-1], f'Horizontal trim ({left + right}) exceeds array width ({array.shape[-1]})'
        
        # Create slice objects for all dimensions
        slices = [slice(None)] * (len(array.shape) - 2)
        
        # Add slices for the last two dimensions
        row_end = array.shape[-2] - bottom if bottom > 0 else array.shape[-2]
        col_end = array.shape[-1] - right if right > 0 else array.shape[-1]
        
        slices.append(slice(top, row_end))
        slices.append(slice(left, col_end))
        
        return array[tuple(slices)]
    
    

# ---------- Decorators and utility functions ----------
def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure and print function execution time.
    
    Args:
        func: The function to measure.
        
    Returns:
        The wrapped function with timing measurement.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        if __VTIMING__:
            print(f'Elapsed time for {func.__name__}: {elapsed_time:.4f} seconds')
        else:
            # Only print if __VTIMING__ is enabled
            pass
        return result
    return wrapper

def print_memory() -> None:
    """Print current RAM memory usage percentage."""
    print(f'RAM memory usage: {psutil.virtual_memory().percent}%')

def flush_mem(func: Callable) -> Callable:
    """Decorator for memory-efficient operations with monitoring.
    
    Args:
        func: The function to wrap.
        
    Returns:
        The wrapped function with memory monitoring and cleanup.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Monitor memory before
        initial_memory = psutil.virtual_memory().percent
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Force garbage collection
        gc.collect()
        
        # Monitor memory after
        final_memory = psutil.virtual_memory().percent
        
        # Print memory change if verbose
        if hasattr(args[0], '_verbose') and args[0]._verbose:
            print(f'Memory usage: {initial_memory:.1f}% -> {final_memory:.1f}% '
                  f'(Δ{final_memory - initial_memory:+.1f}%)')
        
        return result
    return wrapper

def cleanup_variables(*variables: Any) -> None:
    """Explicitly delete variables and run garbage collection.
    
    Args:
        *variables: Variables to delete.
    """
    for var in variables:
        del var
    gc.collect()
     
def initialize_params(
    device: Optional[torch.device] = None,
    slant_range_vec: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
    c: Optional[float] = None,
    len_range_line: Optional[int] = None,
    range_sample_freq: Optional[float] = None,
    wavelength: Optional[float] = None
) -> Dict[str, Any]:
    """Initialize processing parameters dictionary.
    
    Args:
        device: PyTorch device for computation.
        slant_range_vec: Slant range vector.
        D: Cosine of instantaneous squint angle.
        c: Speed of light.
        len_range_line: Length of range line.
        range_sample_freq: Range sampling frequency.
        wavelength: Radar wavelength.
        
    Returns:
        Dictionary containing all parameters.
    """
    return {key: value for key, value in locals().items()}

def multiply_inplace(
    a: Union[np.ndarray, torch.Tensor], 
    b: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Multiply two arrays element-wise in-place with broadcasting support.
    
    Args:
        a: First array (modified in-place).
        b: Second array.
        
    Returns:
        Reference to modified first array.
        
    Raises:
        ValueError: If arrays have incompatible shapes for broadcasting.
    """
    if hasattr(a, 'shape') and hasattr(b, 'shape'):
        # Check if shapes are compatible for broadcasting
        if a.shape != b.shape and b.size != 1 and a.size != 1:
            # For 2D array * 1D array, the 1D array should match one of the 2D dimensions
            if len(a.shape) == 2 and len(b.shape) == 1:
                if b.shape[0] == a.shape[1]:
                    # Broadcasting along range dimension - use numpy broadcasting
                    pass  # NumPy will handle this automatically
                elif b.shape[0] == a.shape[0]:
                    # Need to reshape for azimuth dimension broadcasting
                    b = b.reshape(-1, 1)
                else:
                    raise ValueError(f'1D array length ({b.shape[0]}) does not match either dimension of 2D array {a.shape}')
    
    # Perform in-place multiplication
    try:
        if isinstance(a, np.ndarray):
            np.multiply(a, b, out=a)
        else:  # torch tensor
            a.mul_(b)
        return a
    except (ValueError, RuntimeError) as e:
        raise ValueError(f'Arrays have incompatible shapes for in-place broadcasting: {a.shape} and {b.shape}. '
                       f'Original error: {str(e)}') from e

def multiply(
    a: Union[np.ndarray, torch.Tensor], 
    b: Union[np.ndarray, torch.Tensor],
    debug: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Multiply two arrays element-wise with broadcasting support.
    
    Args:
        a: First array.
        b: Second array.
        
    Returns:
        Element-wise multiplication result.
        
    Raises:
        ValueError: If arrays have incompatible shapes for broadcasting.
    """
    if hasattr(a, 'shape') and hasattr(b, 'shape'):
        # Check if shapes are compatible for broadcasting
        if a.shape != b.shape and b.size != 1 and a.size != 1:
            # Try to understand the broadcasting scenario
            if debug:
                print(f'Debug: Attempting to multiply arrays with shapes {a.shape} and {b.shape}')
            
            # For 2D array * 1D array, the 1D array should match one of the 2D dimensions
            if len(a.shape) == 2 and len(b.shape) == 1:
                if debug:
                    if b.shape[0] == a.shape[1]:
                        print(f'Debug: Broadcasting 1D array along range dimension (axis=1)')
                    elif b.shape[0] == a.shape[0]:
                        print(f'Debug: Need to reshape 1D array for azimuth dimension (axis=0)')
                        b = b.reshape(-1, 1)  # Reshape for broadcasting along azimuth
                    else:
                        raise ValueError(f'1D array length ({b.shape[0]}) does not match either dimension of 2D array {a.shape}')
            
            # Allow broadcasting for compatible shapes
            try:
                result = a * b
                if debug:
                    print(f'Debug: Broadcasting successful, result shape: {result.shape}')
                return result
            except (ValueError, RuntimeError) as e:
                print(f'Debug: Broadcasting failed with error: {str(e)}')
                raise ValueError(f'Arrays have incompatible shapes for broadcasting: {a.shape} and {b.shape}. '
                               f'Original error: {str(e)}') from e
    
    return a * b