import argparse
from typing import Dict, Any, Optional, Union, Tuple, Callable
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

# ---------- Import custom modules ----------
from .code2physical import range_dec_to_sample_rate
from .transforms import perform_fft_custom 
from . import constants as cnst
from ..utils.io import dump

environ['OMP_NUM_THREADS'] = '12' # Set OpenMP threads for parallel processing


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
        
        # Initialize dimensions
        self.len_az_line, self.len_range_line = self.radar_data.shape

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
            w_pad: Width padding for range FFT.
            executors: Number of executors for custom backend.
            
        Raises:
            ValueError: If backend is not supported.
        """
        if self._backend == 'numpy':
            self._fft2d_numpy(w_pad)
        elif self._backend == 'custom':
            self._fft2d_custom(executors)
        elif self._backend == 'torch':
            self._fft2d_torch(w_pad)
        else:
            raise ValueError(f'Backend {self._backend} not supported')
        
        if self._verbose:
            print('- FFT performed successfully!')

    def _fft2d_numpy(self, w_pad: Optional[int]) -> None:
        """Perform 2D FFT using NumPy backend."""
        self.radar_data = np.ascontiguousarray(self.radar_data)
        # FFT each range line
        self.radar_data = np.fft.fft(self.radar_data, axis=1, n=w_pad)
        # FFT each azimuth line
        self.radar_data = np.fft.fftshift(np.fft.fft(self.radar_data, axis=0), axes=0)

    def _fft2d_custom(self, executors: int) -> None:
        """Perform 2D FFT using custom backend."""
        self.radar_data = perform_fft_custom(self.radar_data, num_slices=executors)

    def _fft2d_torch(self, w_pad: Optional[int]) -> None:
        """Perform 2D FFT using PyTorch backend."""
        # FFT each range line
        if w_pad is not None:
            self.radar_data = torch.fft.fft(
                self.radar_data, 
                dim=1, 
                n=self.radar_data.shape[1] + w_pad
            )
        else:
            self.radar_data = torch.fft.fft(self.radar_data, dim=1)
            
        # FFT each azimuth line
        self.radar_data = torch.fft.fftshift(
            torch.fft.fft(self.radar_data, dim=0), 
            dim=0
        )

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
            pad_w: Width padding for the filter.
            
        Returns:
            Range filter array.
            
        Raises:
            AssertionError: If filter dimensions are invalid.
        """
        # Get current radar data dimensions after any transformations
        current_range_dim = self.radar_data.shape[1]
        
        if self._verbose:
            print(f'Current radar data shape: {self.radar_data.shape}')
            print(f'Range dimension: {current_range_dim}')
            print(f'Original range line length: {self.len_range_line}')
            print(f'Requested padding: {pad_w}')
        
        # Create range filter with correct dimensions
        # Use current range dimension instead of original + padding
        filter_length = current_range_dim
        range_filter = np.zeros(filter_length, dtype=complex)
        
        # Calculate correct index for replica placement
        if filter_length >= self.num_tx_vals:
            index_start = int(np.ceil((filter_length - self.num_tx_vals) / 2))
            # Ensure we don't exceed array bounds
            index_start = max(0, min(index_start, filter_length - self.num_tx_vals))
            
            if self._verbose:
                print(f'Filter length: {filter_length}')
                print(f'TX replica length: {self.num_tx_vals}')
                print(f'Range filter index start: {index_start}')
            
            range_filter[index_start:index_start + self.num_tx_vals] = self.tx_replica
        else:
            raise ValueError(f'Filter length ({filter_length}) is smaller than replica length ({self.num_tx_vals})')
        
        # Apply FFT and conjugate
        filter_result = np.conjugate(np.fft.fft(range_filter))
        
        if self._verbose:
            print(f'Range filter output shape: {filter_result.shape}')
        
        assert filter_result.shape[0] == current_range_dim, f'Filter shape mismatch: expected {current_range_dim}, got {filter_result.shape[0]}'
        
        return filter_result

    @timing_decorator
    @auto_gc
    def get_rcmc(self) -> np.ndarray:
        """Calculate Range Cell Migration Correction filter.

        Returns:
            RCMC filter array.
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
        
        # Create RCMC filter
        range_freq_vals = np.linspace(
            -self.range_sample_freq/2, 
            self.range_sample_freq/2, 
            num=self.len_range_line
        )
        
        if self._verbose:
            print(f'Range frequency values shape: {range_freq_vals.shape}')
            print(f'Slant range vec shape: {self.slant_range_vec.shape}')
        
        # Calculate RCMC shift - use first slant range value for reference
        rcmc_shift = self.slant_range_vec[0] * (1/self.D - 1)
        
        if self._verbose:
            print(f'RCMC shift shape: {rcmc_shift.shape}')
        
        # Broadcasting for final filter calculation
        # range_freq_vals: (25724,), rcmc_shift: (56130,)
        # Need to reshape for proper broadcasting
        range_freq_2d = range_freq_vals[np.newaxis, :]  # (1, 25724)
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
            Azimuth filter array.
        """
        if self._verbose:
            print(f'Computing azimuth filter...')
            print(f'Slant range vec shape: {self.slant_range_vec.shape}')
            print(f'D shape: {self.D.shape}')
            print(f'Wavelength: {self.wavelength}')
        
        # Broadcasting for azimuth filter calculation
        # self.slant_range_vec: (25724,), self.D: (56130,)
        # Need to create 2D arrays for proper broadcasting
        slant_range_2d = self.slant_range_vec[np.newaxis, :]  # (1, 25724)
        D_2d = self.D[:, np.newaxis]                          # (56130, 1)
        
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
            print(f'Velocity values shape: {velocity_values.shape}')
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
        
        This is the main processing method that orchestrates the entire
        Range Doppler Algorithm focusing pipeline.
        """
        if self._verbose:
            print('Starting SAR data focusing...')
        
        # Step 1: Initialize processing parameters
        w_pad = self.replica_len
        _, original_w = self.radar_data.shape
        
        # Step 2: 2D FFT transformation
        self.fft2d(w_pad=w_pad)
        
        # Step 3: Range compression
        self._perform_range_compression(w_pad, original_w)
        
        # Step 4: Range Cell Migration Correction
        self._perform_rcmc()
        
        # Step 5: Azimuth compression
        self._perform_azimuth_compression()
        
        if self._verbose:
            print('SAR data focusing completed!')
            print_memory()

    def _perform_range_compression(self, w_pad: int, original_w: int) -> None:
        """Perform range compression step.
        
        Args:
            w_pad: Width padding applied.
            original_w: Original width before padding.
            
        Raises:
            ValueError: If array shapes are incompatible.
        """
        if self._verbose:
            print(f'Starting range compression...')
            print(f'Radar data shape before range filter: {self.radar_data.shape}')
            print(f'Padding applied: {w_pad}')
            print(f'Original width: {original_w}')
        
        # Get range filter with correct dimensions
        range_filter = self.get_range_filter(pad_w=w_pad)
        
        if self._verbose:
            print(f'Range filter shape: {range_filter.shape}')
        
        # Ensure shapes are compatible for broadcasting
        if len(range_filter.shape) == 1 and len(self.radar_data.shape) == 2:
            # Range filter should be applied along range dimension (axis=1)
            assert range_filter.shape[0] == self.radar_data.shape[1], \
                f'Range filter length ({range_filter.shape[0]}) must match radar data range dimension ({self.radar_data.shape[1]})'
        
        # Apply range compression
        self.radar_data = multiply(self.radar_data, range_filter)
        
        if self._verbose:
            print(f'Radar data shape after range compression: {self.radar_data.shape}')
        
        # Remove padding - adjust indices based on actual dimensions
        if w_pad > 0 and self.radar_data.shape[1] > original_w:
            # Calculate padding removal indices
            total_width = self.radar_data.shape[1]
            start_index = (total_width - original_w) // 2
            end_index = start_index + original_w
            
            # Ensure indices are within bounds
            start_index = max(0, start_index)
            end_index = min(total_width, end_index)
            
            if self._verbose:
                print(f'Removing padding: [{start_index}:{end_index}] from total width {total_width}')
            
            self.radar_data = self.radar_data[:, start_index:end_index]
            
            if self._verbose:
                print(f'Radar data shape after padding removal: {self.radar_data.shape}')

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


if __name__ == '__main__':
    """Main execution block for running CoarseRDA as a standalone script.
    
    Raises:
        AssertionError: If required input files are missing or invalid.
    """
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Run CoarseRDA SAR processor.')
    parser.add_argument('--input', type=str, required=True, help='Path to input pickle file containing raw_data dictionary.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the processed radar data.')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch', 'custom'], help='Backend to use for processing.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    
    args = parser.parse_args()
    
    assert Path(args.input).exists(), f'Input file {args.input} does not exist.'
    
    with open(args.input, 'rb') as f:
        raw_data: Dict[str, Any] = pickle.load(f)
    
    assert isinstance(raw_data, dict), 'Input file must contain a dictionary.'
    
    processor = CoarseRDA(raw_data=raw_data, verbose=args.verbose, backend=args.backend)
    processor.data_focus()
    processor.save_file(args.output)