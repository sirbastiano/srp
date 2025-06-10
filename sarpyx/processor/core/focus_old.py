import argparse
from typing import Dict, Any, Optional, Union, Tuple, Callable, List
try:
    import torch
except ImportError:
    print('Unable to import torch module')
    torch = None
import pickle
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import math
from pathlib import Path 
import copy 
import gc
from functools import wraps
import psutil
import time
from os import environ

try:
    import zarr
    import numcodecs
    ZARR_AVAILABLE = True
except ImportError:
    print('Warning: zarr not available, falling back to pickle for saving')
    ZARR_AVAILABLE = False

# ---------- Import custom modules ----------
from .code2physical import range_dec_to_sample_rate
from .transforms import perform_fft_custom 
from . import constants as cnst
from ..utils.viz import dump

environ['OMP_NUM_THREADS'] = '12' # Set OpenMP threads for parallel processing

# ---------- Decorators and utility functions ----------
def auto_gc(func: Callable) -> Callable:
    """Decorator to automatically run garbage collection after function execution.
    
    Args:
        func: The function to wrap.
        
    Returns:
        The wrapped function with automatic garbage collection.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper

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
        print(f'Elapsed time for {func.__name__}: {elapsed_time:.4f} seconds')
        return result
    return wrapper

def print_memory() -> None:
    """Print current RAM memory usage percentage."""
    print(f'RAM memory usage: {psutil.virtual_memory().percent}%')

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

def multiply(
    a: Union[np.ndarray, torch.Tensor], 
    b: Union[np.ndarray, torch.Tensor]
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
            print(f'Debug: Attempting to multiply arrays with shapes {a.shape} and {b.shape}')
            
            # For 2D array * 1D array, the 1D array should match one of the 2D dimensions
            if len(a.shape) == 2 and len(b.shape) == 1:
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
                print(f'Debug: Broadcasting successful, result shape: {result.shape}')
                return result
            except (ValueError, RuntimeError) as e:
                print(f'Debug: Broadcasting failed with error: {str(e)}')
                raise ValueError(f'Arrays have incompatible shapes for broadcasting: {a.shape} and {b.shape}. '
                               f'Original error: {str(e)}') from e
    
    return a * b



# -------- Processing Class ----------
class CoarseRDA:
    """Coarse Range Doppler Algorithm processor for SAR data.
    
    This class implements a coarse Range Doppler Algorithm for processing
    synthetic aperture radar (SAR) data, specifically designed for Sentinel-1 data.
    
    The processing pipeline follows these main steps:
    1. Initialization and data loading
    2. Transmission replica generation
    3. 2D FFT transformation
    4. Range compression
    5. Range Cell Migration Correction (RCMC)
    6. Azimuth compression
    7. Final inverse transformations
    """

    # ==================== INITIALIZATION METHODS ====================
    
    def __init__(
        self, 
        raw_data: Dict[str, Any], 
        verbose: bool = False, 
        backend: str = 'numpy'
    ) -> None:
        """Initialize the CoarseRDA processor.
        
        Args:
            raw_data: Dictionary containing 'echo', 'ephemeris', and 'metadata'.
            verbose: Whether to print verbose output.
            backend: Backend to use ('numpy', 'torch', or 'custom').
            
        Raises:
            ValueError: If invalid backend is specified.
            AssertionError: If required data is missing.
        """
        self._validate_inputs(raw_data, backend)
        self._initialize_settings(verbose, backend)
        self._load_data(raw_data)
        self._setup_backend()
        self._initialize_transmission_replica()

    def _validate_inputs(self, raw_data: Dict[str, Any], backend: str) -> None:
        """Validate input parameters.
        
        Args:
            raw_data: Dictionary containing radar data.
            backend: Processing backend.
            
        Raises:
            AssertionError: If required data is missing.
            ValueError: If invalid backend is specified.
        """
        assert isinstance(raw_data, dict), 'raw_data must be a dictionary'
        assert 'echo' in raw_data, 'raw_data must contain "echo" key'
        assert 'ephemeris' in raw_data, 'raw_data must contain "ephemeris" key'
        assert 'metadata' in raw_data, 'raw_data must contain "metadata" key'
        
        valid_backends = {'numpy', 'torch', 'custom'}
        if backend not in valid_backends:
            raise ValueError(f'Backend must be one of {valid_backends}, got {backend}')

    def _initialize_settings(self, verbose: bool, backend: str) -> None:
        """Initialize processor settings.
        
        Args:
            verbose: Whether to print verbose output.
            backend: Processing backend.
        """
        self._backend = backend
        self._verbose = verbose

    def _load_data(self, raw_data: Dict[str, Any]) -> None:
        """Load and preprocess input data.
        
        Args:
            raw_data: Dictionary containing radar data.
        """
        self.radar_data = raw_data['echo']
        self.ephemeris = raw_data['ephemeris'].copy()
        self.ephemeris['time_stamp'] /= 2**24
        self.metadata = raw_data['metadata']
        
        # Initialize dimensions - these should remain constant throughout processing
        self.len_az_line, self.len_range_line = self.radar_data.shape
        
        if self._verbose:
            print(f'Loaded radar data with shape: {self.radar_data.shape}')
            print(f'Azimuth lines: {self.len_az_line}, Range lines: {self.len_range_line}')

    def _setup_backend(self) -> None:
        """Set up processing backend and device configuration."""
        if self._backend == 'torch':
            if torch is None:
                raise ImportError('PyTorch is required for torch backend but not available')
            self.device = getattr(
                self.radar_data, 
                'device', 
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            if self._verbose:
                print(f'Selected device: {self.device}')

    def _initialize_transmission_replica(self) -> None:
        """Initialize transmission replica during setup."""
        self._generate_tx_replica()

    # ==================== TRANSMISSION REPLICA METHODS ====================

    @timing_decorator
    @auto_gc
    def _generate_tx_replica(self) -> None:
        """Generate transmission replica based on metadata parameters.
        
        This method creates the transmission replica used for range compression
        based on the radar system parameters extracted from metadata.
        """
        if self._verbose:
            print('Generating transmission replica...')
        
        # Extract range decimation and calculate sample frequency
        rgdec = int(self.metadata['range_decimation'].unique()[0])
        if self._verbose:
            print(f'Range decimation code: {rgdec}')
        
        self.range_sample_freq = range_dec_to_sample_rate(rgdec)
        if self._verbose:
            print(f'Range sample frequency: {self.range_sample_freq:.2f} Hz')

        # Extract transmission parameters
        tx_params = self._extract_tx_parameters()
        
        # Generate replica signal
        self._create_replica_signal(tx_params)
        
        if self._verbose:
            print(f'Replica length: {self.replica_len}')
            print('Transmission replica generated successfully!')

    def _extract_tx_parameters(self) -> Dict[str, float]:
        """Extract transmission parameters from metadata.
        
        Returns:
            Dictionary containing transmission parameters.
        """
        txpsf = self.metadata['tx_pulse_start_freq'].unique()[0]
        txprr = self.metadata['tx_ramp_rate'].unique()[0]
        txpl = self.metadata['tx_pulse_length'].unique()[0]
        
        if self._verbose:
            print(f'TX pulse start frequency: {txpsf:.2f} Hz')
            print(f'TX ramp rate: {txprr:.2f} Hz/s')
            print(f'TX pulse length: {txpl:.6f} s')
        
        return {'start_freq': txpsf, 'ramp_rate': txprr, 'pulse_length': txpl}

    def _create_replica_signal(self, tx_params: Dict[str, float]) -> None:
        """Create the replica signal from transmission parameters.
        
        Args:
            tx_params: Dictionary containing transmission parameters.
        """
        txpsf = tx_params['start_freq']
        txprr = tx_params['ramp_rate'] 
        txpl = tx_params['pulse_length']
        
        # Generate replica
        self.num_tx_vals = int(txpl * self.range_sample_freq)
        if self._verbose:
            print(f'Number of TX values: {self.num_tx_vals}')
        
        tx_replica_time_vals = np.linspace(-txpl/2, txpl/2, num=self.num_tx_vals)
        phi1 = txpsf + txprr * txpl / 2
        phi2 = txprr / 2
        
        if self._verbose:
            print(f'Phase parameters - phi1: {phi1:.2f}, phi2: {phi2:.2e}')
        
        self.tx_replica = np.exp(
            2j * np.pi * (phi1 * tx_replica_time_vals + phi2 * tx_replica_time_vals**2)
        )
        self.replica_len = len(self.tx_replica)

    # ==================== TRANSFORM METHODS ====================

    @timing_decorator
    @auto_gc
    def fft2d(self, w_pad: Optional[int] = None, executors: int = 12) -> None:
        """Perform 2D FFT on radar data in range and azimuth dimensions.

        Args:
            w_pad: Width padding for range FFT (ignored for dimension preservation).
            executors: Number of executors for custom backend.
            
        Raises:
            ValueError: If backend is not supported.
        """
        if self._verbose:
            print(f'FFT input data shape: {self.radar_data.shape}')
        
        if self._backend == 'numpy':
            self._fft2d_numpy()
        elif self._backend == 'custom':
            self._fft2d_custom(executors)
        elif self._backend == 'torch':
            self._fft2d_torch()
        else:
            raise ValueError(f'Backend {self._backend} not supported')
        
        # Verify dimensions are preserved
        expected_shape = (self.len_az_line, self.len_range_line)
        if self.radar_data.shape != expected_shape:
            raise RuntimeError(f'FFT changed radar data shape from {expected_shape} to {self.radar_data.shape}')
        
        if self._verbose:
            print(f'FFT output data shape: {self.radar_data.shape}')
            print('- FFT performed successfully!')

    def _fft2d_numpy(self) -> None:
        """Perform 2D FFT using NumPy backend preserving original dimensions.
        
        Returns:
            None
        """
        # Store original shape for verification
        original_shape = self.radar_data.shape
        if self._verbose:
            print(f'Original radar data shape: {original_shape}')
        # Ensure data is contiguous for better performance
        # self.radar_data = np.ascontiguousarray(self.radar_data)
        
        # FFT each range line (axis=1) - no padding to preserve dimensions
        if self._verbose:
            print(f'Performing FFT along range dimension (axis=1)...')
        self.radar_data = np.fft.fft(self.radar_data, axis=1)
        if self._verbose:
            print(f'First FFT along range dimension completed, shape: {self.radar_data.shape}')
        # FFT each azimuth line (axis=0) with fftshift
        if self._verbose:
            print(f'Performing FFT along azimuth dimension (axis=0) with fftshift...')
        self.radar_data = np.fft.fftshift(np.fft.fft(self.radar_data, axis=0), axes=0)
        if self._verbose:
            print(f'Second FFT along azimuth dimension completed, shape: {self.radar_data.shape}')
        
        # Verify shape preservation
        assert self.radar_data.shape == original_shape, \
            f'FFT changed shape from {original_shape} to {self.radar_data.shape}'

    def _fft2d_custom(self, executors: int) -> None:
        """Perform 2D FFT using custom backend."""
        original_shape = self.radar_data.shape
        if self._verbose:
            print(f'Performing custom FFT with {executors} executors...')
        self.radar_data = perform_fft_custom(self.radar_data, num_slices=executors)
        
        # Verify shape preservation
        assert self.radar_data.shape == original_shape, \
            f'Custom FFT changed shape from {original_shape} to {self.radar_data.shape}'

    def _fft2d_torch(self) -> None:
        """Perform 2D FFT using PyTorch backend preserving dimensions.
        
        Returns:
            None
        """
        original_shape = self.radar_data.shape
        
        # FFT each range line (axis=1) - no padding
        self.radar_data = torch.fft.fft(self.radar_data, dim=1)
        
        # FFT each azimuth line (axis=0) with fftshift
        self.radar_data = torch.fft.fftshift(
            torch.fft.fft(self.radar_data, dim=0), 
            dim=0
        )
        
        # Verify shape preservation
        assert self.radar_data.shape == original_shape, \
            f'Torch FFT changed shape from {original_shape} to {self.radar_data.shape}'

    @timing_decorator
    @auto_gc
    def ifft_range(self) -> None:
        """Perform inverse FFT along range dimension."""
        if self._backend == 'numpy':
            self.radar_data = np.fft.ifftshift(np.fft.ifft(self.radar_data, axis=1), axes=1)
        elif self._backend == 'torch':
            self.radar_data = torch.fft.ifft(self.radar_data, dim=1)
            self.radar_data = torch.fft.ifftshift(self.radar_data, dim=1)
        else:
            raise ValueError(f'Unsupported backend: {self._backend}')

    @timing_decorator
    @auto_gc
    def ifft_azimuth(self) -> None:
        """Perform inverse FFT along azimuth dimension."""
        if self._backend == 'numpy':
            self.radar_data = np.fft.ifft(self.radar_data, axis=0)
        elif self._backend == 'torch':
            self.radar_data = torch.fft.ifft(self.radar_data, dim=0)
        else:
            raise ValueError(f'Unsupported backend: {self._backend}')

    # ==================== FILTER GENERATION METHODS ====================

    @timing_decorator
    @auto_gc
    def get_range_filter(self, pad_w: int = 0) -> np.ndarray:
        """Compute range filter for radar data compression.
    
        Args:
            pad_w: Width padding (ignored - filter always matches radar data dimensions).
            
        Returns:
            Range filter array exactly matching radar data range dimension.
            
        Raises:
            AssertionError: If filter dimensions are invalid.
        """
        # Use exact radar data dimensions - no padding considerations
        current_range_dim = self.radar_data.shape[1]
        
        if self._verbose:
            print(f'Creating range filter for radar data shape: {self.radar_data.shape}')
            print(f'Range dimension: {current_range_dim}')
            print(f'TX replica length: {self.num_tx_vals}')
        
        # Create range filter with exact radar data range dimension
        range_filter = np.zeros(current_range_dim, dtype=complex)
        
        # Place replica in center of filter
        if current_range_dim >= self.num_tx_vals:
            index_start = (current_range_dim - self.num_tx_vals) // 2
            index_end = index_start + self.num_tx_vals
            
            if self._verbose:
                print(f'Placing replica at indices [{index_start}:{index_end}] in filter of length {current_range_dim}')
            
            range_filter[index_start:index_end] = self.tx_replica
        else:
            # If range dimension is smaller than replica, truncate replica
            if self._verbose:
                print(f'⚠️  Range dimension ({current_range_dim}) < replica length ({self.num_tx_vals}), truncating replica')
            
            replica_start = (self.num_tx_vals - current_range_dim) // 2
            replica_end = replica_start + current_range_dim
            range_filter[:] = self.tx_replica[replica_start:replica_end]
        
        # Apply FFT and conjugate
        filter_result = np.conjugate(np.fft.fft(range_filter))
        
        if self._verbose:
            print(f'Range filter shape: {filter_result.shape}')
        
        # Ensure filter exactly matches radar data range dimension
        assert filter_result.shape[0] == current_range_dim, \
            f'Filter shape mismatch: expected {current_range_dim}, got {filter_result.shape[0]}'
        
        return filter_result

    @timing_decorator
    @auto_gc
    def get_rcmc(self) -> np.ndarray:
        """Calculate Range Cell Migration Correction filter.

        Returns:
            RCMC filter array matching radar data dimensions.
        """
        self._compute_effective_velocities()
        
        self.wavelength = cnst.TX_WAVELENGTH_M
        
        # Generate azimuth frequency values for the entire azimuth line length
        self.az_freq_vals = np.arange(
            -self.az_sample_freq/2, 
            self.az_sample_freq/2, 
            self.az_sample_freq/self.len_az_line
        )
        
        # Ensure we have exactly the right number of frequency values
        if len(self.az_freq_vals) != self.len_az_line:
            self.az_freq_vals = np.linspace(
                -self.az_sample_freq/2, 
                self.az_sample_freq/2, 
                self.len_az_line, 
                endpoint=False
            )
        
        if self._verbose:
            print(f'Azimuth frequency values shape: {self.az_freq_vals.shape}')
            print(f'Effective velocities shape: {self.effective_velocities.shape}')
        
        # Take mean effective velocity across range for each azimuth line
        # This reduces from (56130, 25724) to (56130,)
        mean_effective_velocities = np.mean(self.effective_velocities, axis=1)
        
        if self._verbose:
            print(f'Mean effective velocities shape: {mean_effective_velocities.shape}')
        
        # Cosine of instantaneous squint angle
        # Broadcasting: (56130,) with (56130,) -> (56130,)
        self.D = np.sqrt(
            1 - (self.wavelength**2 * self.az_freq_vals**2) / 
                (4 * mean_effective_velocities**2)
        )
        
        if self._verbose:
            print(f'D (cosine squint angle) shape: {self.D.shape}')
        
        # Create RCMC filter with CURRENT radar data dimensions (should be original dimensions)
        current_range_dim = self.radar_data.shape[1]
        
        range_freq_vals = np.linspace(
            -self.range_sample_freq/2, 
            self.range_sample_freq/2, 
            num=current_range_dim
        )
        
        if self._verbose:
            print(f'Range frequency values shape: {range_freq_vals.shape}')
            print(f'Current radar data range dimension: {current_range_dim}')
            print(f'Slant range vec shape: {self.slant_range_vec.shape}')
        
        # Calculate RCMC shift - use first slant range value for reference
        rcmc_shift = self.slant_range_vec[0] * (1/self.D - 1)
        
        if self._verbose:
            print(f'RCMC shift shape: {rcmc_shift.shape}')
        
        # Broadcasting for final filter calculation
        # range_freq_vals: (current_range_dim,), rcmc_shift: (56130,)
        # Need to reshape for proper broadcasting
        range_freq_2d = range_freq_vals[np.newaxis, :]  # (1, current_range_dim)
        rcmc_shift_2d = rcmc_shift[:, np.newaxis]       # (56130, 1)
        
        rcmc_filter = np.exp(4j * np.pi * range_freq_2d * rcmc_shift_2d / self.c)
        
        if self._verbose:
            print(f'Final RCMC filter shape: {rcmc_filter.shape}')
        
        return rcmc_filter

    @timing_decorator
    @auto_gc 
    def get_azimuth_filter(self) -> np.ndarray:
        """Calculate azimuth compression filter.
        
        Returns:
            Azimuth filter array matching radar data dimensions.
        """
        if self._verbose:
            print(f'Computing azimuth filter...')
            print(f'Slant range vec shape: {self.slant_range_vec.shape}')
            print(f'D shape: {self.D.shape}')
            print(f'Wavelength: {self.wavelength}')
        
        # Use current radar data dimensions (should match original slant range vector)
        current_range_dim = self.radar_data.shape[1]
        
        # Ensure slant range vector matches radar data dimensions
        if current_range_dim != len(self.slant_range_vec):
            if self._verbose:
                print(f'⚠️  Warning: Current range dim ({current_range_dim}) != slant range vec length ({len(self.slant_range_vec)})')
                print(f'This should not happen - using original slant range vector')
            # This should not happen if dimensions are preserved correctly
            current_slant_range_vec = self.slant_range_vec
        else:
            current_slant_range_vec = self.slant_range_vec
        
        # Broadcasting for azimuth filter calculation
        # current_slant_range_vec: (range_dim,), self.D: (56130,)
        # Need to create 2D arrays for proper broadcasting
        slant_range_2d = current_slant_range_vec[np.newaxis, :]  # (1, range_dim)
        D_2d = self.D[:, np.newaxis]                             # (56130, 1)
        
        azimuth_filter = np.exp(4j * np.pi * slant_range_2d * D_2d / self.wavelength)
        
        if self._verbose:
            print(f'Azimuth filter shape: {azimuth_filter.shape}')
        
        return azimuth_filter

    # ==================== VELOCITY COMPUTATION METHODS ====================

    @timing_decorator
    @auto_gc
    def _compute_effective_velocities(self) -> None:
        """Calculate effective spacecraft velocities for processing.
        
        This method computes the effective velocities needed for RCMC and
        azimuth compression by combining spacecraft and ground velocities.
        """
        # Initialize timing and geometry parameters
        self._initialize_timing_parameters()
        
        # Calculate spacecraft velocities and positions
        space_velocities, positions = self._calculate_spacecraft_dynamics()
        
        # Compute effective velocities using Earth model
        self._compute_ground_velocities(space_velocities, positions)

    def _initialize_timing_parameters(self) -> None:
        """Initialize timing and geometry parameters for velocity computation.
        
        Raises:
            KeyError: If required metadata columns are missing.
            ValueError: If metadata values are invalid.
        """
        self.c = cnst.SPEED_OF_LIGHT_MPS
        
        # Check for required columns with case-insensitive matching
        metadata_columns = {col.lower(): col for col in self.metadata.columns}
        
        required_mappings = {
            'pri': ['pri', 'pulse_repetition_interval'],
            'rank': ['rank'],
            'swst': ['swst', 'sampling_window_start_time', 'start_time']
        }
        
        column_map = {}
        for param, possible_names in required_mappings.items():
            found_column = None
            for name in possible_names:
                if name.lower() in metadata_columns:
                    found_column = metadata_columns[name.lower()]
                    break
            
            if found_column is None:
                available_cols = list(self.metadata.columns)
                raise KeyError(
                    f'Could not find column for {param}. Tried: {possible_names}. '
                    f'Available columns: {available_cols}'
                )
            column_map[param] = found_column
        
        if self._verbose:
            print('Column mapping:')
            for param, col in column_map.items():
                print(f'  {param} -> {col}')
        
        # Extract parameters with error handling
        try:
            self.pri = self.metadata[column_map['pri']].iloc[0]
            rank = self.metadata[column_map['rank']].iloc[0] 
            range_start_time_base = self.metadata[column_map['swst']].iloc[0]
        except (IndexError, TypeError) as e:
            raise ValueError(f'Error extracting metadata values: {str(e)}') from e
        
        # Validate values
        assert self.pri > 0, f'PRI must be positive, got {self.pri}'
        assert rank >= 0, f'Rank must be non-negative, got {rank}'
        
        if self._verbose:
            print(f'PRI: {self.pri:.6f} s')
            print(f'Rank: {rank}')
            print(f'Base range start time: {range_start_time_base:.6f} s')
        
        # Calculate derived parameters
        suppressed_data_time = 320 / (8 * cnst.F_REF)
        range_start_time = range_start_time_base + suppressed_data_time
        
        # Sample rates
        range_sample_period = 1 / self.range_sample_freq
        self.az_sample_freq = 1 / self.pri
        
        if self._verbose:
            print(f'Range start time: {range_start_time:.6f} s')
            print(f'Azimuth sample frequency: {self.az_sample_freq:.2f} Hz')
        
        # Fast time and slant range vectors
        sample_num_along_range_line = np.arange(0, self.len_range_line, 1)
        fast_time_vec = range_start_time + (range_sample_period * sample_num_along_range_line)
        self.slant_range_vec = ((rank * self.pri) + fast_time_vec) * self.c / 2
        
        if self._verbose:
            print(f'Slant range vector shape: {self.slant_range_vec.shape}')
            print(f'Slant range min/max: {self.slant_range_vec.min():.2f}/{self.slant_range_vec.max():.2f} m')

    def _calculate_spacecraft_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate spacecraft velocities and positions.
        
        Returns:
            Tuple of (space_velocities, positions) arrays.
            
        Raises:
            KeyError: If required columns are missing from ephemeris or metadata.
            ValueError: If interpolation fails.
        """
        if self._verbose:
            print('Calculating spacecraft dynamics...')
            print(f'Ephemeris shape: {self.ephemeris.shape}')
            print(f'Metadata shape: {self.metadata.shape}')
        
        # Spacecraft velocity calculations
        ecef_vels = self.ephemeris.apply(
            lambda x: math.sqrt(x['vx']**2 + x['vy']**2 + x['vz']**2), 
            axis=1
        )
        
        if self._verbose:
            print(f'ECEF velocities shape: {ecef_vels.shape}')
            print(f'ECEF velocities range: {ecef_vels.min():.2f} - {ecef_vels.max():.2f} m/s')
        
        # Extract arrays and ensure they are proper numpy arrays
        time_stamps = self.ephemeris['time_stamp'].values
        velocity_values = ecef_vels.values
        x_values = self.ephemeris['x'].values
        y_values = self.ephemeris['y'].values  
        z_values = self.ephemeris['z'].values
        
        if self._verbose:
            print(f'Time stamps shape: {time_stamps.shape}')
            print(f'Time stamps range: {time_stamps.min():.6f} - {time_stamps.max():.6f}')
            print(f' veocity values shape: {velocity_values.shape}')
            print(f'Position arrays shapes: x={x_values.shape}, y={y_values.shape}, z={z_values.shape}')
        
        # Ensure arrays are sorted by time for interpolation
        sort_indices = np.argsort(time_stamps)
        time_stamps = time_stamps[sort_indices]
        velocity_values = velocity_values[sort_indices]
        x_values = x_values[sort_indices]
        y_values = y_values[sort_indices]
        z_values = z_values[sort_indices]
        
        # Calculate metadata time stamps
        metadata_times = self.metadata.apply(
            lambda x: x['coarse_time'] + x['fine_time'], 
            axis=1
        ).values
        
        if self._verbose:
            print(f'Metadata times shape: {metadata_times.shape}')
            print(f'Metadata times range: {metadata_times.min():.6f} - {metadata_times.max():.6f}')
            print(f'Ephemeris time range: {time_stamps.min():.6f} - {time_stamps.max():.6f}')
        
        # Check if metadata times are within ephemeris time range
        time_within_range = (metadata_times >= time_stamps.min()) & (metadata_times <= time_stamps.max())
        if not np.all(time_within_range):
            out_of_range_count = np.sum(~time_within_range)
            if self._verbose:
                print(f'⚠️  Warning: {out_of_range_count} metadata times are outside ephemeris range')
                print(f'   Will use boundary values for extrapolation')
        
        # Create interpolation functions with bounds_error=False and fill_value for extrapolation
        try:
            velocity_interp = interp1d(
                time_stamps, velocity_values, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(velocity_values[0], velocity_values[-1])
            )
            x_interp = interp1d(
                time_stamps, x_values, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(x_values[0], x_values[-1])
            )
            y_interp = interp1d(
                time_stamps, y_values, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(y_values[0], y_values[-1])
            )
            z_interp = interp1d(
                time_stamps, z_values, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(z_values[0], z_values[-1])
            )
        except ValueError as e:
            raise ValueError(f'Failed to create interpolation functions: {str(e)}') from e
        
        # Interpolate at metadata time points
        try:
            space_velocities = velocity_interp(metadata_times)
            x_interp_vals = x_interp(metadata_times)
            y_interp_vals = y_interp(metadata_times)
            z_interp_vals = z_interp(metadata_times)
        except Exception as e:
            raise ValueError(f'Interpolation failed: {str(e)}') from e
        
        # Ensure interpolated values are arrays and handle any remaining NaN values
        space_velocities = np.asarray(space_velocities)
        x_interp_vals = np.asarray(x_interp_vals)
        y_interp_vals = np.asarray(y_interp_vals)
        z_interp_vals = np.asarray(z_interp_vals)
        
        # Check for and handle NaN values
        if np.any(np.isnan(space_velocities)):
            nan_count = np.sum(np.isnan(space_velocities))
            if self._verbose:
                print(f'⚠️  Found {nan_count} NaN values in space_velocities, filling with nearest valid values')
            
            # Fill NaN values with nearest valid values
            valid_mask = ~np.isnan(space_velocities)
            if np.any(valid_mask):
                # Forward fill
                space_velocities = pd.Series(space_velocities).fillna(method='ffill').fillna(method='bfill').values
            else:
                # If all NaN, use average ephemeris velocity
                space_velocities.fill(np.nanmean(velocity_values))
        
        # Handle NaN values in position components
        for vals, name in [(x_interp_vals, 'x'), (y_interp_vals, 'y'), (z_interp_vals, 'z')]:
            if np.any(np.isnan(vals)):
                nan_count = np.sum(np.isnan(vals))
                if self._verbose:
                    print(f'⚠️  Found {nan_count} NaN values in {name}_interp_vals, filling with nearest valid values')
                
                valid_mask = ~np.isnan(vals)
                if np.any(valid_mask):
                    vals_series = pd.Series(vals).fillna(method='ffill').fillna(method='bfill')
                    if name == 'x':
                        x_interp_vals = vals_series.values
                    elif name == 'y':
                        y_interp_vals = vals_series.values
                    else:
                        z_interp_vals = vals_series.values
        
        if self._verbose:
            print(f'Interpolated space_velocities shape: {space_velocities.shape}')
            print(f'Interpolated position component shapes: x={x_interp_vals.shape}, y={y_interp_vals.shape}, z={z_interp_vals.shape}')
        
        # Create position array
        positions = np.column_stack([x_interp_vals, y_interp_vals, z_interp_vals])
        
        if self._verbose:
            print(f'Final space_velocities shape: {space_velocities.shape}')
            print(f'Final positions shape: {positions.shape}')
            print(f'Space velocities range: {space_velocities.min():.2f} - {space_velocities.max():.2f} m/s')
            print(f'Position range - x: {positions[:, 0].min():.0f} to {positions[:, 0].max():.0f}')
            print(f'Position range - y: {positions[:, 1].min():.0f} to {positions[:, 1].max():.0f}')
            print(f'Position range - z: {positions[:, 2].min():.0f} to {positions[:, 2].max():.0f}')
        
        # Validate outputs
        assert isinstance(space_velocities, np.ndarray), 'space_velocities must be numpy array'
        assert isinstance(positions, np.ndarray), 'positions must be numpy array'
        assert len(space_velocities.shape) == 1, f'space_velocities must be 1D, got shape {space_velocities.shape}'
        assert len(positions.shape) == 2, f'positions must be 2D array, got shape {positions.shape}'
        assert positions.shape[1] == 3, f'positions must have 3 columns (x,y,z), got {positions.shape[1]}'
        assert space_velocities.shape[0] == positions.shape[0], f'velocity and position arrays must have same length'
        
        # Final check for NaN values after cleaning
        assert not np.any(np.isnan(space_velocities)), 'NaN values still present in space_velocities after cleaning'
        assert not np.any(np.isnan(positions)), 'NaN values still present in positions after cleaning'
        
        # Check for reasonable values
        assert np.all(space_velocities > 1000), f'Space velocities too low (min: {space_velocities.min():.2f} m/s)'
        assert np.all(space_velocities < 20000), f'Space velocities too high (max: {space_velocities.max():.2f} m/s)'
        
        position_magnitudes = np.linalg.norm(positions, axis=1)
        assert np.all(position_magnitudes > 6e6), f'Position magnitudes too small (min: {position_magnitudes.min():.0f} m)'
        assert np.all(position_magnitudes < 8e6), f'Position magnitudes too large (max: {position_magnitudes.max():.0f} m)'
        
        return space_velocities, positions

    def _compute_ground_velocities(self, space_velocities: np.ndarray, positions: np.ndarray) -> None:
        """Compute ground velocities and effective velocities.
        
        Args:
            space_velocities: Spacecraft velocity magnitudes (1D array).
            positions: Spacecraft position vectors (2D array, shape [N, 3]).
            
        Raises:
            AssertionError: If input arrays have incompatible shapes.
            ValueError: If calculations produce invalid results.
        """
        # Enhanced input validation
        assert isinstance(space_velocities, np.ndarray), f'space_velocities must be numpy array, got {type(space_velocities)}'
        assert isinstance(positions, np.ndarray), f'positions must be numpy array, got {type(positions)}'
        assert len(positions.shape) == 2, f'positions must be 2D, got shape {positions.shape}'
        assert positions.shape[1] == 3, f'positions must have 3 columns, got {positions.shape[1]}'
        assert space_velocities.shape[0] == positions.shape[0], f'Array lengths must match: velocities={space_velocities.shape[0]}, positions={positions.shape[0]}'
        
        # Ensure arrays are proper numpy arrays with correct dtypes
        space_velocities = np.asarray(space_velocities, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        
        # Check for NaN/inf values
        assert not np.any(np.isnan(space_velocities)), 'NaN values in space_velocities'
        assert not np.any(np.isnan(positions)), 'NaN values in positions'
        assert not np.any(np.isinf(space_velocities)), 'Infinite values in space_velocities'
        assert not np.any(np.isinf(positions)), 'Infinite values in positions'
        
        if self._verbose:
            print('Computing ground velocities...')
            print(f'Space velocities shape: {space_velocities.shape}')
            print(f'Positions shape: {positions.shape}')
            print(f'Slant range vec shape: {self.slant_range_vec.shape}')
            print(f'Input data ranges:')
            print(f'  Space velocities: {space_velocities.min():.2f} - {space_velocities.max():.2f} m/s')
            print(f'  Positions X: {positions[:, 0].min():.0f} - {positions[:, 0].max():.0f} m')
            print(f'  Positions Y: {positions[:, 1].min():.0f} - {positions[:, 1].max():.0f} m')
            print(f'  Positions Z: {positions[:, 2].min():.0f} - {positions[:, 2].max():.0f} m')
        
        # Earth model calculations
        a = float(cnst.WGS84_SEMI_MAJOR_AXIS_M)
        b = float(cnst.WGS84_SEMI_MINOR_AXIS_M)
        
        if self._verbose:
            print(f'Earth model parameters: a={a:.0f} m, b={b:.0f} m')
        
        # Calculate spacecraft heights (magnitudes of position vectors)
        H = np.linalg.norm(positions, axis=1)  # axis=1 for row-wise norm
        H = np.asarray(H, dtype=np.float64)
        
        # Validate H calculation
        assert H.shape == space_velocities.shape, f'H shape {H.shape} must match velocities shape {space_velocities.shape}'
        assert not np.any(np.isnan(H)), 'NaN values in H (spacecraft heights)'
        assert np.all(H > 0), 'All spacecraft heights must be positive'
        
        W = space_velocities / H
        W = np.asarray(W, dtype=np.float64)
        
        # Calculate latitude using more robust method
        xy_distance = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        xy_distance = np.asarray(xy_distance, dtype=np.float64)
        lat = np.arctan2(positions[:, 2], xy_distance)
        lat = np.asarray(lat, dtype=np.float64)
        
        if self._verbose:
            print(f'H (heights) shape: {H.shape}, range: {H.min():.0f} - {H.max():.0f} m')
            print(f'W (angular velocities) shape: {W.shape}, range: {W.min():.6f} - {W.max():.6f} rad/s')
            print(f'Latitudes range: {np.degrees(lat.min()):.2f} - {np.degrees(lat.max()):.2f} deg')
        
        # Validate intermediate calculations
        assert not np.any(np.isnan(W)), 'NaN values in W (angular velocities)'
        assert not np.any(np.isnan(lat)), 'NaN values in latitude'
        
        # Local Earth radius calculation with explicit numpy array operations
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        cos_lat = np.asarray(cos_lat, dtype=np.float64)
        sin_lat = np.asarray(sin_lat, dtype=np.float64)
        
        # Ensure all terms are numpy arrays before sqrt operation
        numerator = np.asarray(a**4 * cos_lat**2 + b**4 * sin_lat**2, dtype=np.float64)
        denominator = np.asarray(a**2 * cos_lat**2 + b**2 * sin_lat**2, dtype=np.float64)
        ratio = numerator / denominator
        ratio = np.asarray(ratio, dtype=np.float64)
        
        local_earth_rad = np.sqrt(ratio)
        local_earth_rad = np.asarray(local_earth_rad, dtype=np.float64)
        
        if self._verbose:
            print(f'Local Earth radius range: {local_earth_rad.min():.0f} - {local_earth_rad.max():.0f} m')
        
        # Validate local Earth radius
        assert not np.any(np.isnan(local_earth_rad)), 'NaN values in local_earth_rad'
        assert np.all(local_earth_rad > 0), 'All local Earth radii must be positive'
        
        # Ensure slant_range_vec is also a proper numpy array
        slant_range_vec = np.asarray(self.slant_range_vec, dtype=np.float64)
        
        # Broadcasting for slant range calculation
        slant_range_2d = slant_range_vec[np.newaxis, :]  # Shape: [1, range_samples]
        local_earth_rad_2d = local_earth_rad[:, np.newaxis]  # Shape: [azimuth_samples, 1]
        H_2d = H[:, np.newaxis]  # Shape: [azimuth_samples, 1]
        W_2d = W[:, np.newaxis]  # Shape: [azimuth_samples, 1]
        
        # Calculate cosine of look angle with explicit array operations
        term1 = np.asarray(local_earth_rad_2d**2, dtype=np.float64)
        term2 = np.asarray(H_2d**2, dtype=np.float64)
        term3 = np.asarray(slant_range_2d**2, dtype=np.float64)
        term4 = np.asarray(2 * local_earth_rad_2d * H_2d, dtype=np.float64)
        
        cos_beta = (term1 + term2 - term3) / term4
        cos_beta = np.asarray(cos_beta, dtype=np.float64)
        
        # Clip to valid range for cosine
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        
        if self._verbose:
            print(f'cos_beta shape: {cos_beta.shape}')
            print(f'cos_beta range: {cos_beta.min():.3f} - {cos_beta.max():.3f}')
        
        # Calculate ground velocities
        ground_velocities = local_earth_rad_2d * W_2d * cos_beta
        ground_velocities = np.asarray(ground_velocities, dtype=np.float64)
        
        if self._verbose:
            print(f'Ground velocities shape: {ground_velocities.shape}')
        
        # Calculate effective velocities
        space_velocities_2d = space_velocities[:, np.newaxis]  # Shape: [azimuth_samples, 1]
        effective_vel_product = space_velocities_2d * ground_velocities
        effective_vel_product = np.asarray(effective_vel_product, dtype=np.float64)
        
        # Ensure non-negative values before sqrt
        effective_vel_product = np.maximum(effective_vel_product, 0.0)
        
        self.effective_velocities = np.sqrt(effective_vel_product)
        self.effective_velocities = np.asarray(self.effective_velocities, dtype=np.float64)
        
        if self._verbose:
            print(f'Effective velocities shape: {self.effective_velocities.shape}')
            print(f'Effective velocities range: {self.effective_velocities.min():.2f} - {self.effective_velocities.max():.2f} m/s')
        
        # Final validation
        assert not np.any(np.isnan(self.effective_velocities)), 'NaN values in effective velocities'
        assert not np.any(np.isinf(self.effective_velocities)), 'Infinite values in effective velocities'
        assert np.all(self.effective_velocities >= 0), 'All effective velocities must be non-negative'

    # ==================== MAIN PROCESSING METHODS ====================

    @timing_decorator
    @auto_gc
    def data_focus(self) -> None:
        """Perform complete SAR data focusing using Range Doppler Algorithm.
        
        This method ensures radar data dimensions remain constant throughout processing.
        
        Raises:
            RuntimeError: If data dimensions change unexpectedly during processing.
        """
        if self._verbose:
            print('Starting SAR data focusing...')
            print(f'Initial radar data shape: {self.radar_data.shape}')
        
        # Store initial shape for verification
        initial_shape = self.radar_data.shape
        expected_shape = (self.len_az_line, self.len_range_line)
        
        assert initial_shape == expected_shape, \
            f'Initial data shape {initial_shape} does not match expected {expected_shape}'
        
        # Step 1: Get padding value (for legacy compatibility only)
        w_pad = getattr(self, 'replica_len', 0)
        w_pad = 0
        original_w = initial_shape[1]
        
        if self._verbose:
            print(f'Processing with w_pad={w_pad}, original_w={original_w}')
            print(f'Processing with original_w={original_w}')
        
        # Step 2: 2D FFT transformation (preserves dimensions)
        self.fft2d()
        assert self.radar_data.shape == initial_shape, \
            f'FFT changed data shape from {initial_shape} to {self.radar_data.shape}'
        
        # Step 3: Range compression (preserves dimensions)
        self._perform_range_compression(w_pad, original_w)
        assert self.radar_data.shape == initial_shape, \
            f'Range compression changed data shape from {initial_shape} to {self.radar_data.shape}'
        
        # Step 4: Range Cell Migration Correction (preserves dimensions)
        self._perform_rcmc()
        assert self.radar_data.shape == initial_shape, \
            f'RCMC changed data shape from {initial_shape} to {self.radar_data.shape}'
        
        # Step 5: Azimuth compression (preserves dimensions)
        self._perform_azimuth_compression()
        assert self.radar_data.shape == initial_shape, \
            f'Azimuth compression changed data shape from {initial_shape} to {self.radar_data.shape}'
        
        if self._verbose:
            print(f'SAR data focusing completed successfully!')
            print(f'Final radar data shape: {self.radar_data.shape}')
            print_memory()

    def _perform_range_compression(self, w_pad: int, original_w: int) -> None:
        """Perform range compression step while preserving data dimensions.
        
        Args:
            w_pad: Width padding (ignored - dimensions preserved).
            original_w: Original width (for verification).
            
        Raises:
            ValueError: If array shapes are incompatible.
            AssertionError: If dimensions change unexpectedly.
        """
        if self._verbose:
            print(f'Starting range compression...')
            print(f'Radar data shape: {self.radar_data.shape}')
        
        # Store original shape for verification
        original_shape = self.radar_data.shape
        expected_shape = (self.len_az_line, self.len_range_line)
        
        # Verify we still have expected dimensions
        assert original_shape == expected_shape, \
            f'Unexpected radar data shape: {original_shape}, expected: {expected_shape}'
        
        # Get range filter with matching dimensions
        range_filter = self.get_range_filter()
        
        if self._verbose:
            print(f'Range filter shape: {range_filter.shape}')
            print(f'Applying range compression filter...')
        
        # Apply range compression filter
        self.radar_data = multiply(self.radar_data, range_filter)
        
        # Verify dimensions are preserved
        assert self.radar_data.shape == original_shape, \
            f'Range compression changed data shape from {original_shape} to {self.radar_data.shape}'
        
        if self._verbose:
            print(f'Range compression completed. Data shape: {self.radar_data.shape}')

    def _perform_rcmc(self) -> None:
        """Perform Range Cell Migration Correction."""
        rcmc_filter = self.get_rcmc()
        self.radar_data = multiply(self.radar_data, rcmc_filter)
        
        # Inverse FFT in range
        self.ifft_range()

    def _perform_azimuth_compression(self) -> None:
        """Perform azimuth compression step.
        
        Raises:
            ValueError: If array shapes are incompatible.
        """
        if self._verbose:
            print('Starting azimuth compression...')
            print(f'Radar data shape before azimuth filter: {self.radar_data.shape}')
        
        # Get azimuth filter
        azimuth_filter = self.get_azimuth_filter()
        
        if self._verbose:
            print(f'Azimuth filter shape: {azimuth_filter.shape}')
        
        # Apply azimuth compression
        self.radar_data = multiply(self.radar_data, azimuth_filter)
        
        if self._verbose:
            print(f'Radar data shape after azimuth compression: {self.radar_data.shape}')
        
        # Inverse FFT in azimuth
        self.ifft_azimuth()
        
        if self._verbose:
            print(f'Final radar data shape: {self.radar_data.shape}')

    # ==================== UTILITY METHODS ====================

    @timing_decorator
    def save_file(self, save_path: Union[str, Path]) -> None:
        """Save processed radar data to file.
        
        Args:
            save_path: Path where to save the data.
        """
        dump(self.radar_data, save_path)
        if self._verbose:
            print(f'Data saved to {save_path}')

    # For backward compatibility - keep original method name as alias
    _prompt_tx_replica = _generate_tx_replica


class SteppedRDA(CoarseRDA):
    """Stepped Range Doppler Algorithm processor with intermediate step saving capability.
    
    This class extends CoarseRDA to allow saving of intermediate processing results
    at key stages as compressed Zarr files for optimal storage and performance.
    
    Attributes:
        save_intermediate: Whether to save intermediate steps.
        save_dir: Directory path for saving intermediate results.
        base_filename: Base filename for saved files.
        compression_level: Compression level for Zarr files (0-9).
        use_zarr: Whether to use Zarr format for saving.
    """
    
    def __init__(
        self, 
        raw_data: Dict[str, Any], 
        verbose: bool = False, 
        backend: str = 'numpy',
        save_intermediate: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
        base_filename: str = 'sar_processing',
        compression_level: int = 9,
        use_zarr: bool = True
    ) -> None:
        """Initialize the SteppedRDA processor.
        
        Args:
            raw_data: Dictionary containing 'echo', 'ephemeris', and 'metadata'.
            verbose: Whether to print verbose output.
            backend: Backend to use ('numpy', 'torch', or 'custom').
            save_intermediate: Whether to save intermediate processing steps.
            save_dir: Directory path for saving intermediate results.
            base_filename: Base filename for saved files.
            compression_level: Compression level for Zarr files (0-9, 9=max compression).
            use_zarr: Whether to use Zarr format (falls back to pickle if zarr unavailable).
            
        Raises:
            ValueError: If compression_level is not in valid range.
            ImportError: If zarr is requested but not available.
        """
        super().__init__(raw_data, verbose, backend)
        
        # Validate compression level
        if not 0 <= compression_level <= 9:
            raise ValueError(f'compression_level must be 0-9, got {compression_level}')
        
        self.save_intermediate = save_intermediate
        self.base_filename = base_filename
        self.compression_level = compression_level
        
        # Determine save format
        if use_zarr and not ZARR_AVAILABLE:
            if self._verbose:
                print('Warning: Zarr requested but not available, falling back to pickle')
            self.use_zarr = False
        else:
            self.use_zarr = use_zarr and ZARR_AVAILABLE
        
        if save_intermediate:
            if save_dir is None:
                save_dir = Path.cwd() / 'sar_intermediate_results'
            
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            if self._verbose:
                format_type = 'Zarr (compressed)' if self.use_zarr else 'Pickle'
                print(f'Intermediate results will be saved as {format_type} to: {self.save_dir}')
                if self.use_zarr:
                    print(f'Zarr compression level: {self.compression_level}')
        else:
            self.save_dir = None

    
    
    
    def main(self, save_all_steps: bool = False) -> None:

        """Run the SAR focusing pipeline, saving intermediate results after each step.

        This method performs SAR focusing and saves the radar data after each major processing step.
        
        Args:
            save_all_steps: Whether to save all intermediate steps including raw input and FFT.
        """
        if self._verbose:
            print('Starting SAR STEPPED data focusing...')
            print(f'Initial radar data shape: {self.radar_data.shape}')

        # Store initial shape for verification
        initial_shape = self.radar_data.shape
        expected_shape = (self.len_az_line, self.len_range_line)

        assert initial_shape == expected_shape, \
            f'Initial data shape {initial_shape} does not match expected {expected_shape}'

        # Step 1: Save raw input (optional)
        if save_all_steps:
            self._save_intermediate_step(
                radar_data=self.radar_data,
                save_intermediate=self.save_intermediate,
                save_dir=self.save_dir,
                use_zarr=self.use_zarr,
                step_name='raw_input',
                description='Raw input radar data',
                base_filename=self.base_filename,
                compression_level=self.compression_level,
                backend=self._backend,
                verbose=self._verbose
            )

        # Step 2: 2D FFT transformation (preserves dimensions)
        self.fft2d()
        if save_all_steps:
            # Save FFT result in frequency domain (as-is)
            self._save_intermediate_step(
                radar_data=self.radar_data,
                save_intermediate=self.save_intermediate,
                save_dir=self.save_dir,
                use_zarr=self.use_zarr,
                step_name='fft2d',
                description='After 2D FFT (frequency domain)',
                base_filename=self.base_filename,
                compression_level=self.compression_level,
                backend=self._backend,
                verbose=self._verbose
            )
        assert self.radar_data.shape == initial_shape, \
            f'FFT changed data shape from {initial_shape} to {self.radar_data.shape}'

        # Step 3: Range compression (preserves dimensions)
        w_pad = 0
        original_w = initial_shape[1]
        self._perform_range_compression(w_pad, original_w)
        
        # Save range compression result (converted back to time domain for visualization)
        self._save_intermediate_step(
            radar_data=self._ifft2d(self.radar_data),
            save_intermediate=self.save_intermediate,
            save_dir=self.save_dir,
            use_zarr=self.use_zarr,
            step_name='range_compression',
            description='After range compression (time domain)',
            base_filename=self.base_filename,
            compression_level=self.compression_level,
            backend=self._backend,
            verbose=self._verbose
        )
        
        # Verify dimensions are preserved
        assert self.radar_data.shape == initial_shape, \
            f'Range compression changed data shape from {initial_shape} to {self.radar_data.shape}'

        # Step 4: Range Cell Migration Correction (preserves dimensions)
        self._perform_rcmc()
        self._save_intermediate_step(
            radar_data=self._ifft2d(self.radar_data),
            save_intermediate=self.save_intermediate,
            save_dir=self.save_dir,
            use_zarr=self.use_zarr,
            step_name='rcmc',
            description='After Range Cell Migration Correction (time domain)',
            base_filename=self.base_filename,
            compression_level=self.compression_level,
            backend=self._backend,
            verbose=self._verbose
        )
        assert self.radar_data.shape == initial_shape, \
            f'RCMC changed data shape from {initial_shape} to {self.radar_data.shape}'

        # Step 5: Azimuth compression (preserves dimensions)
        self._perform_azimuth_compression()
        self._save_intermediate_step(
            radar_data=self._ifft2d(self.radar_data),
            save_intermediate=self.save_intermediate,
            save_dir=self.save_dir,
            use_zarr=self.use_zarr,
            step_name='azimuth_compression',
            description='After Azimuth Compression (final focused image)',
            base_filename=self.base_filename,
            compression_level=self.compression_level,
            backend=self._backend,
            verbose=self._verbose
        )
        assert self.radar_data.shape == initial_shape, \
            f'Azimuth compression changed data shape from {initial_shape} to {self.radar_data.shape}'

        if self._verbose:
            print('SAR data focusing completed successfully!')
            print(f'Final radar data shape: {self.radar_data.shape}')
            print_memory()
    
    
    
    
    @auto_gc
    @staticmethod
    def _ifft2d(radar_data: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Perform 2D inverse FFT using NumPy backend, preserving original dimensions.

        Args:
            radar_data: Input radar data array.
            verbose: Whether to print verbose output.

        Returns:
            Output array after 2D inverse FFT.

        Raises:
            AssertionError: If the shape of radar_data changes during the operation.
        """
        assert isinstance(radar_data, np.ndarray), 'radar_data must be a numpy array, got {type(radar_data)}'
        assert radar_data.ndim == 2, f'radar_data must be 2D array, got shape {radar_data.shape}'
        assert radar_data.dtype == np.complex64 or radar_data.dtype == np.complex128, \
            f'radar_data must be complex type, got {radar_data.dtype}'
        # Store original shape for verification
        original_shape = radar_data.shape
        
        if verbose:
            print(f'Original radar data shape: {original_shape}')
            print('Performing in-place inverse FFT along azimuth dimension (axis=0) with ifftshift...')
        
        # In-place operations to reduce memory usage - use out parameter where possible
        # First, ifftshift along azimuth (axis=0) - this creates a view when possible
        radar_data = np.fft.ifftshift(radar_data, axes=0)
        
        # Force garbage collection of any temporary arrays
        gc.collect()
        
        # Then inverse FFT along azimuth (axis=0) - overwrite input array
        radar_data[:] = np.fft.ifft(radar_data, axis=0)
        
        if verbose:
            print(f'First inverse FFT along azimuth completed, shape: {radar_data.shape}')
            print('Performing inverse FFT along range dimension (axis=1)...')
        
        # Force garbage collection before next operation
        gc.collect()
        
        # Inverse FFT along range (axis=1) - overwrite input array
        radar_data[:] = np.fft.ifft(radar_data, axis=1)
        
        if verbose:
            print(f'Second inverse FFT along range completed, shape: {radar_data.shape}')
        
        # Verify shape preservation
        assert radar_data.shape == original_shape, \
            f'IFFT changed shape from {original_shape} to {radar_data.shape}'
        
        return radar_data
    
    
    def _get_instance_zarr_compressor(self) -> Any:
        """Get the optimal Zarr compressor configuration for instance method.
        
        Returns:
            Configured compressor for maximum compression of complex SAR data.
        """
        # Use Blosc with LZ4HC algorithm for best compression on complex data
        return numcodecs.Blosc(
            cname='lz4hc',  # LZ4HC provides good compression for SAR data
            clevel=self.compression_level,
            shuffle=numcodecs.Blosc.BITSHUFFLE,  # Bit shuffle for better compression
            blocksize=0  # Auto-select optimal block size
        )
    
    @staticmethod
    def _save_intermediate_step(
        radar_data: np.ndarray,
        save_intermediate: bool,
        save_dir: Optional[Path],
        use_zarr: bool,
        step_name: str,
        description: str = '',
        base_filename: str = 'sar_processing',
        compression_level: int = 9,
        backend: str = 'numpy',
        verbose: bool = False
    ) -> None:
        """Save current radar data as an intermediate step.
        
        Args:
            radar_data: The radar data array to save.
            save_intermediate: Whether intermediate saving is enabled.
            save_dir: Directory path for saving intermediate results.
            use_zarr: Whether to use Zarr format.
            step_name: Name identifier for the processing step.
            description: Optional description of the step.
            base_filename: Base filename for saved files.
            compression_level: Compression level for Zarr files (0-9).
            backend: Processing backend used.
            verbose: Whether to print verbose output.
        """
        if not save_intermediate or save_dir is None:
            return
        
        if use_zarr:
            SteppedRDA._save_zarr_step(
                radar_data, save_dir, step_name, description, base_filename,
                compression_level, backend, verbose
            )
        else:
            SteppedRDA._save_pickle_step(
                radar_data, save_dir, step_name, description, base_filename, verbose
            )
    
    @staticmethod
    def _save_zarr_step(
        radar_data: np.ndarray,
        save_dir: Path,
        step_name: str,
        description: str = '',
        base_filename: str = 'sar_processing',
        compression_level: int = 9,
        backend: str = 'numpy',
        verbose: bool = False
    ) -> None:
        """Save step as compressed Zarr file with custom 5000x5000 chunking."""
        zarr_filename = f'{base_filename}_{step_name}.zarr'
        zarr_path = save_dir / zarr_filename
        
        try:
            # Remove existing zarr store if it exists
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path)
            
            # Get data as numpy array for consistent handling
            if hasattr(radar_data, 'cpu'):  # PyTorch tensor
                data_to_save = radar_data.cpu().numpy()
            else:
                data_to_save = np.asarray(radar_data)
            
            # Create Zarr array with maximum compression
            compressor = numcodecs.Blosc(
                cname='lz4hc',
                clevel=compression_level,
                shuffle=numcodecs.Blosc.BITSHUFFLE,
                blocksize=0
            )
            
            # Use 5000x5000 chunking as requested
            chunk_size = (
                min(5000, data_to_save.shape[0]),
                min(5000, data_to_save.shape[1])
            )
            
            # Create the zarr array
            z = zarr.open(
                str(zarr_path),
                mode='w',
                shape=data_to_save.shape,
                dtype=data_to_save.dtype,
                chunks=chunk_size,
                compressor=compressor
            )
            
            # Save the data
            z[:] = data_to_save
            
            # Save metadata as attributes
            z.attrs['step_name'] = step_name
            z.attrs['description'] = description
            z.attrs['data_shape'] = data_to_save.shape
            z.attrs['data_dtype'] = str(data_to_save.dtype)
            z.attrs['compression_level'] = compression_level
            z.attrs['processing_backend'] = backend
            z.attrs['chunk_size'] = chunk_size
            
            # Calculate compression ratio
            uncompressed_size = data_to_save.nbytes
            compressed_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
            compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0
            
            if verbose:
                print(f'Saved {step_name}: {zarr_path}')
                if description:
                    print(f'  Description: {description}')
                print(f'  Data shape: {data_to_save.shape}')
                print(f'  Data type: {data_to_save.dtype}')
                print(f'  Chunk size: {chunk_size}')
                print(f'  Uncompressed size: {uncompressed_size / 1024**3:.2f} GB')
                print(f'  Compressed size: {compressed_size / 1024**3:.2f} GB')
                print(f'  Compression ratio: {compression_ratio:.2f}x')
                
        except Exception as e:
            print(f'⚠️  Failed to save {step_name} as Zarr: {str(e)}')
            # Fallback to pickle
            if verbose:
                print('  Falling back to pickle format...')
            pickle_filename = f'{base_filename}_{step_name}.pkl'
            pickle_path = save_dir / pickle_filename
            joblib.dump(radar_data, pickle_path)
    
    @staticmethod
    def _save_pickle_step(
        radar_data: np.ndarray,
        save_dir: Path,
        step_name: str,
        description: str = '',
        base_filename: str = 'sar_processing',
        verbose: bool = False
    ) -> None:
        """Save step as pickle file (fallback method).
        
        Args:
            radar_data: The radar data array to save.
            save_dir: Directory path for saving.
            step_name: Name identifier for the processing step.
            description: Optional description of the step.
            base_filename: Base filename for saved files.
            verbose: Whether to print verbose output.
        """
        filename = f'{base_filename}_{step_name}.pkl'
        save_path = save_dir / filename
        
        try:
            dump(radar_data, save_path)
            if verbose:
                print(f'Saved {step_name}: {save_path}')
                if description:
                    print(f'  Description: {description}')
                print(f'  Data shape: {radar_data.shape}')
        except Exception as e:
            print(f'⚠️  Failed to save {step_name}: {str(e)}')
    @staticmethod
    def _get_zarr_compressor(compression_level: int) -> Any:
        """Get the optimal Zarr compressor configuration.
        
        Args:
            compression_level: Compression level (0-9).
            
        Returns:
            Configured compressor for maximum compression of complex SAR data.
        """
        # Use Blosc with LZ4HC algorithm for best compression on complex data
        # Ensure compatibility with different zarr formats
        return numcodecs.Blosc(
            cname='lz4hc',  # LZ4HC provides good compression for SAR data
            clevel=compression_level,
            shuffle=numcodecs.Blosc.BITSHUFFLE,  # Bit shuffle for better compression
            blocksize=0  # Auto-select optimal block size
        )

    
    @staticmethod
    def _get_zarr_size(zarr_path: Path) -> int:
        """Calculate total size of Zarr store on disk.
        
        Args:
            zarr_path: Path to Zarr store.
            
        Returns:
            Total size in bytes.
        """
        total_size = 0
        try:
            for file_path in zarr_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            # If we can't calculate size, return 0
            pass
        return total_size
    
    def load_step(self, step_name: str) -> np.ndarray:
        """Load a previously saved processing step.
        
        Args:
            step_name: Name of the step to load.
            
        Returns:
            Loaded radar data array.
            
        Raises:
            FileNotFoundError: If the step file doesn't exist.
            ValueError: If unable to load the file.
        """
        if self.save_dir is None:
            raise ValueError('No save directory configured')
        
        # Try Zarr first, then pickle
        zarr_path = self.save_dir / f'{self.base_filename}_{step_name}.zarr'
        pickle_path = self.save_dir / f'{self.base_filename}_{step_name}.pkl'
        
        if zarr_path.exists() and self.use_zarr:
            try:
                z = zarr.open(str(zarr_path), mode='r')
                data = np.array(z[:])  # Load into memory
                
                if self._verbose:
                    print(f'Loaded {step_name} from Zarr: {zarr_path}')
                    print(f'  Shape: {data.shape}')
                    print(f'  Type: {data.dtype}')
                    if 'description' in z.attrs:
                        print(f'  Description: {z.attrs["description"]}')
                
                return data
                
            except Exception as e:
                if self._verbose:
                    print(f'Failed to load Zarr file {zarr_path}: {str(e)}')
                    print('Trying pickle fallback...')
        
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                
                if self._verbose:
                    print(f'Loaded {step_name} from pickle: {pickle_path}')
                
                return np.asarray(data)
                
            except Exception as e:
                raise ValueError(f'Failed to load pickle file {pickle_path}: {str(e)}') from e
        
        raise FileNotFoundError(f'No saved file found for step "{step_name}"')
    
    def list_saved_files(self) -> List[Path]:
        """List all saved intermediate files.
        
        Returns:
            List of Path objects for saved intermediate files.
        """
        if not self.save_intermediate or self.save_dir is None:
            return []
        
        saved_files = []
        
        # Add Zarr files
        zarr_files = list(self.save_dir.glob(f'{self.base_filename}_*.zarr'))
        saved_files.extend(zarr_files)
        
        # Add pickle files
        pickle_files = list(self.save_dir.glob(f'{self.base_filename}_*.pkl'))
        saved_files.extend(pickle_files)
        
        return sorted(saved_files)
    
    def get_compression_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get compression statistics for all saved Zarr files.
        
        Returns:
            Dictionary with compression stats for each saved step.
        """
        if not self.use_zarr or self.save_dir is None:
            return {}
        
        stats = {}
        zarr_files = self.save_dir.glob(f'{self.base_filename}_*.zarr')
        
        for zarr_path in zarr_files:
            try:
                z = zarr.open(str(zarr_path), mode='r')
                compressed_size = self._get_zarr_size(zarr_path)
                uncompressed_size = z.size * z.dtype.itemsize
                
                step_name = zarr_path.stem.replace(f'{self.base_filename}_', '')
                
                stats[step_name] = {
                    'shape': z.shape,
                    'dtype': str(z.dtype),
                    'chunks': z.chunks,
                    'uncompressed_size_gb': uncompressed_size / 1024**3,
                    'compressed_size_gb': compressed_size / 1024**3,
                    'compression_ratio': uncompressed_size / compressed_size if compressed_size > 0 else 1.0,
                    'compressor': str(z.compressor) if hasattr(z, 'compressor') else 'unknown'
                }
                
            except Exception as e:
                if self._verbose:
                    print(f'Warning: Could not get stats for {zarr_path}: {str(e)}')
        
        return stats
    
    def print_compression_summary(self) -> None:
        """Print a summary of compression statistics for all saved files."""
        stats = self.get_compression_stats()
        
        if not stats:
            print('No compression statistics available')
            return
        
        print('\n=== Compression Summary ===')
        total_uncompressed = 0
        total_compressed = 0
        
        for step_name, step_stats in stats.items():
            uncompressed_gb = step_stats['uncompressed_size_gb']
            compressed_gb = step_stats['compressed_size_gb']
            ratio = step_stats['compression_ratio']
            
            total_uncompressed += uncompressed_gb
            total_compressed += compressed_gb
            
            print(f'{step_name}:')
            print(f'  Shape: {step_stats["shape"]}')
            print(f'  Uncompressed: {uncompressed_gb:.2f} GB')
            print(f'  Compressed: {compressed_gb:.2f} GB')
            print(f'  Ratio: {ratio:.2f}x')
            print()
        
        overall_ratio = total_uncompressed / total_compressed if total_compressed > 0 else 1.0
        print(f'Overall totals:')
        print(f'  Uncompressed: {total_uncompressed:.2f} GB')
        print(f'  Compressed: {total_compressed:.2f} GB')
        print(f'  Overall ratio: {overall_ratio:.2f}x')
        print(f'  Space saved: {total_uncompressed - total_compressed:.2f} GB')