"""
SAR (Synthetic Aperture Radar) data processing module implementing the Coarse Range Doppler Algorithm.
This module provides functionality for processing raw SAR data, specifically designed for Sentinel-1 data,
using the Range Doppler Algorithm (RDA). The implementation supports multiple backends (NumPy, PyTorch, 
and custom) for computational flexibility and performance optimization.
Main Components:
    - CoarseRDA: Primary class for SAR data focusing using Range Doppler Algorithm
    - Utility functions for parameter initialization and data manipulation
    - Decorators for performance monitoring and memory management
    - Support for range and azimuth compression, range cell migration correction
Key Features:
    - Multi-backend support (NumPy, PyTorch, custom)
    - Memory-efficient processing with automatic garbage collection
    - Performance timing and memory usage monitoring
    - Comprehensive error handling and input validation
    - Command-line interface for batch processing
Typical Usage:
    # Load raw SAR data
    with open('raw_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    # Initialize processor
    processor = CoarseRDA(raw_data, verbose=True, backend='numpy')
    # Perform SAR focusing
    # Save processed data
    processor.save_file('focused_data.pkl')
Command Line Usage:
    python focus.py --input raw_data.pkl --output focused_data.pkl --backend numpy --verbose
Dependencies:
    - numpy: Core numerical computations
    - scipy: Interpolation functions
    - pandas: Data frame operations
    - torch (optional): GPU acceleration
    - psutil: Memory monitoring
Notes:
    - Designed specifically for Sentinel-1 SAR data format
    - Requires proper ephemeris data for accurate processing
    - Memory usage scales with input data size
    - PyTorch backend provides GPU acceleration when available
Author: SAR Processing Team
Version: 1.0
"""
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

from .transforms import perform_fft_custom 
from . import constants as cnst
from ..utils.io import dump

environ['OMP_NUM_THREADS'] = '8'


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


def range_dec_to_sample_rate(rgdec_code: int) -> float:
    """Convert range decimation code to sample rate.

    Args:
        rgdec_code: Range decimation code (0-11).

    Returns:
        Sample rate for this range decimation code.
        
    Raises:
        ValueError: If invalid range decimation code is provided.
    """
    assert isinstance(rgdec_code, int), f'Range decimation code must be integer, got {type(rgdec_code)}'
    
    decimation_map = {
        0: 3 * cnst.F_REF,
        1: (8/3) * cnst.F_REF,
        3: (20/9) * cnst.F_REF,
        4: (16/9) * cnst.F_REF,
        5: (3/2) * cnst.F_REF,
        6: (4/3) * cnst.F_REF,
        7: (2/3) * cnst.F_REF,
        8: (12/7) * cnst.F_REF,
        9: (5/4) * cnst.F_REF,
        10: (6/13) * cnst.F_REF,
        11: (16/11) * cnst.F_REF,
    }
    
    if rgdec_code not in decimation_map:
        raise ValueError(f'Invalid range decimation code {rgdec_code} - valid codes are {list(decimation_map.keys())}')
    
    return decimation_map[rgdec_code]


def multiply(
    a: Union[np.ndarray, torch.Tensor], 
    b: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Multiply two arrays element-wise.
    
    Args:
        a: First array.
        b: Second array.
        
    Returns:
        Element-wise multiplication result.
        
    Raises:
        ValueError: If arrays have incompatible shapes.
    """
    if hasattr(a, 'shape') and hasattr(b, 'shape'):
        if a.shape != b.shape and b.size != 1 and a.size != 1:
            # Allow broadcasting for compatible shapes
            try:
                return a * b
            except (ValueError, RuntimeError) as e:
                raise ValueError(f'Arrays have incompatible shapes: {a.shape} and {b.shape}') from e
    
    return a * b


class CoarseRDA:
    """Coarse Range Doppler Algorithm processor for SAR data.
    
    This class implements a coarse Range Doppler Algorithm for processing
    synthetic aperture radar (SAR) data, specifically designed for Sentinel-1 data.
    """

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
        # Validate inputs
        assert isinstance(raw_data, dict), 'raw_data must be a dictionary'
        assert 'echo' in raw_data, 'raw_data must contain "echo" key'
        assert 'ephemeris' in raw_data, 'raw_data must contain "ephemeris" key'
        assert 'metadata' in raw_data, 'raw_data must contain "metadata" key'
        
        valid_backends = {'numpy', 'torch', 'custom'}
        if backend not in valid_backends:
            raise ValueError(f'Backend must be one of {valid_backends}, got {backend}')
        
        # Initialize settings
        self._backend = backend
        self._verbose = verbose
        
        # Extract data
        self.radar_data = raw_data['echo']
        self.ephemeris = raw_data['ephemeris'].copy()
        self.ephemeris['time_stamp'] /= 2**24
        self.metadata = raw_data['metadata']
        
        # Initialize dimensions
        self.len_az_line, self.len_range_line = self.radar_data.shape
        
        # Set up device for torch backend
        if self._backend == 'torch':
            if torch is None:
                raise ImportError('PyTorch is required for torch backend but not available')
            self.device = getattr(self.radar_data, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if self._verbose:
                print(f'Selected device: {self.device}')
        
        # Initialize transmission replica
        self._prompt_tx_replica()

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
            self.radar_data = np.ascontiguousarray(self.radar_data)
            # FFT each range line
            self.radar_data = np.fft.fft(self.radar_data, axis=1, n=w_pad)
            # FFT each azimuth line
            self.radar_data = np.fft.fftshift(np.fft.fft(self.radar_data, axis=0), axes=0)
        
        elif self._backend == 'custom':
            self.radar_data = perform_fft_custom(self.radar_data, num_slices=executors)
            
        elif self._backend == 'torch':
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
        else:
            raise ValueError(f'Backend {self._backend} not supported')
        
        if self._verbose:
            print('- FFT performed successfully!')

    @timing_decorator
    @auto_gc
    def _prompt_tx_replica(self) -> None:
        """Generate transmission replica based on metadata parameters."""
        rgdec = self.metadata['Range Decimation'].unique()[0]
        self.range_sample_freq = range_dec_to_sample_rate(rgdec)

        # Extract nominal replica parameters
        txpsf = self.metadata['Tx Pulse Start Frequency'].unique()[0]
        txprr = self.metadata['Tx Ramp Rate'].unique()[0]
        txpl = self.metadata['Tx Pulse Length'].unique()[0]
        
        # Generate replica
        self.num_tx_vals = int(txpl * self.range_sample_freq)
        tx_replica_time_vals = np.linspace(-txpl/2, txpl/2, num=self.num_tx_vals)
        phi1 = txpsf + txprr * txpl / 2
        phi2 = txprr / 2
        
        self.tx_replica = np.exp(
            2j * np.pi * (phi1 * tx_replica_time_vals + phi2 * tx_replica_time_vals**2)
        )
        self.replica_len = len(self.tx_replica)

    @timing_decorator
    @auto_gc
    def get_range_filter(self, pad_w: int = 0) -> np.ndarray:
        """Compute range filter for radar data compression.

        Args:
            pad_w: Width padding for the filter.
            
        Returns:
            Range filter array.
        """
        # Create range filter from replica pulse
        range_filter = np.zeros(self.len_range_line + pad_w, dtype=complex)
        index_start = int(np.ceil((self.len_range_line - self.num_tx_vals) / 2) - 1)
        index_end = self.num_tx_vals + index_start
        
        range_filter[index_start:index_start + self.num_tx_vals] = self.tx_replica
        
        # Apply FFT and conjugate
        return np.conjugate(np.fft.fft(range_filter))

    @timing_decorator
    @auto_gc
    def _compute_effective_velocities(self) -> None:
        """Calculate effective spacecraft velocities for processing."""
        # Initialize constants
        self.c = cnst.SPEED_OF_LIGHT_MPS
        self.pri = self.metadata['PRI'].unique()[0]
        rank = self.metadata['Rank'].unique()[0]
        suppressed_data_time = 320 / (8 * cnst.F_REF)
        range_start_time = self.metadata['SWST'].unique()[0] + suppressed_data_time
        
        # Sample rates
        range_sample_period = 1 / self.range_sample_freq
        self.az_sample_freq = 1 / self.pri
        
        # Fast time and slant range vectors
        sample_num_along_range_line = np.arange(0, self.len_range_line, 1)
        fast_time_vec = range_start_time + (range_sample_period * sample_num_along_range_line)
        self.slant_range_vec = ((rank * self.pri) + fast_time_vec) * self.c / 2
        
        # Spacecraft velocity calculations
        ecef_vels = self.ephemeris.apply(
            lambda x: math.sqrt(x['vx']**2 + x['vy']**2 + x['vz']**2), 
            axis=1
        )
        
        # Create interpolation functions
        time_stamps = self.ephemeris['time_stamp'].unique()
        velocity_interp = interp1d(time_stamps, ecef_vels.unique(), fill_value='extrapolate')
        x_interp = interp1d(time_stamps, self.ephemeris['x'].unique(), fill_value='extrapolate')
        y_interp = interp1d(time_stamps, self.ephemeris['y'].unique(), fill_value='extrapolate')
        z_interp = interp1d(time_stamps, self.ephemeris['z'].unique(), fill_value='extrapolate')
        
        # Calculate positions and velocities
        time_data = self.metadata['Coarse Time'] + self.metadata['Fine Time']
        space_velocities = self.metadata.apply(
            lambda x: velocity_interp(x['Coarse Time'] + x['Fine Time']), 
            axis=1
        ).to_numpy().astype(float)
        
        positions = np.column_stack([
            self.metadata.apply(lambda x: x_interp(x['Coarse Time'] + x['Fine Time']), axis=1),
            self.metadata.apply(lambda x: y_interp(x['Coarse Time'] + x['Fine Time']), axis=1),
            self.metadata.apply(lambda x: z_interp(x['Coarse Time'] + x['Fine Time']), axis=1)
        ])
        
        # Earth model calculations
        a = cnst.WGS84_SEMI_MAJOR_AXIS_M
        b = cnst.WGS84_SEMI_MINOR_AXIS_M
        H = np.linalg.norm(positions, axis=1)
        W = space_velocities / H
        lat = np.arctan(positions[:, 2] / positions[:, 0])
        
        local_earth_rad = np.sqrt(
            (a**4 * np.cos(lat)**2 + b**4 * np.sin(lat)**2) /
            (a**2 * np.cos(lat)**2 + b**2 * np.sin(lat)**2)
        )
        
        cos_beta = (
            local_earth_rad**2 + H**2 - self.slant_range_vec[:, np.newaxis]**2
        ) / (2 * local_earth_rad * H)
        
        ground_velocities = local_earth_rad * W * cos_beta
        self.effective_velocities = np.sqrt(space_velocities * ground_velocities)

    @timing_decorator
    @auto_gc
    def get_rcmc(self) -> np.ndarray:
        """Calculate Range Cell Migration Correction filter.

        Returns:
            RCMC filter array.
        """
        self._compute_effective_velocities()
        
        self.wavelength = cnst.TX_WAVELENGTH_M
        self.az_freq_vals = np.arange(
            -self.az_sample_freq/2, 
            self.az_sample_freq/2, 
            1/(self.pri * self.len_az_line)
        )
        
        # Cosine of instantaneous squint angle
        self.D = np.sqrt(
            1 - (self.wavelength**2 * self.az_freq_vals**2) / 
                (4 * self.effective_velocities**2)
        ).T
        
        # Create RCMC filter
        range_freq_vals = np.linspace(
            -self.range_sample_freq/2, 
            self.range_sample_freq/2, 
            num=self.len_range_line
        )
        rcmc_shift = self.slant_range_vec[0] * (1/self.D - 1)
        
        return np.exp(4j * np.pi * range_freq_vals * rcmc_shift / self.c)
    
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
    
    @timing_decorator
    @auto_gc 
    def get_azimuth_filter(self) -> np.ndarray:
        """Calculate azimuth compression filter.
        
        Returns:
            Azimuth filter array.
        """
        return np.exp(4j * np.pi * self.slant_range_vec * self.D / self.wavelength)
    
    @timing_decorator
    @auto_gc
    def data_focus(self) -> None:
        """Perform complete SAR data focusing using Range Doppler Algorithm."""
        # Initialize padding
        w_pad = self.replica_len
        _, original_w = self.radar_data.shape
        
        if self._verbose:
            print('Starting SAR data focusing...')
        
        # 2D FFT
        self.fft2d(w_pad=w_pad)
        
        # Range compression
        range_filter = self.get_range_filter(pad_w=w_pad)
        self.radar_data = multiply(self.radar_data, range_filter)
        
        # Remove padding
        start_index = w_pad // 2
        end_index = start_index + original_w
        self.radar_data = self.radar_data[:, start_index:end_index]
        
        # Range Cell Migration Correction
        rcmc_filter = self.get_rcmc()
        self.radar_data = multiply(self.radar_data, rcmc_filter)
        
        # Inverse FFT in range
        self.ifft_range()
        
        # Azimuth compression
        azimuth_filter = self.get_azimuth_filter()
        self.radar_data = multiply(self.radar_data, azimuth_filter)
        
        # Inverse FFT in azimuth
        self.ifft_azimuth()
        
        if self._verbose:
            print('SAR data focusing completed!')
            print_memory()

    @timing_decorator
    def save_file(self, save_path: Union[str, Path]) -> None:
        """Save processed radar data to file.
        
        Args:
            save_path: Path where to save the data.
        """
        dump(self.radar_data, save_path)
        if self._verbose:
            print(f'Data saved to {save_path}')


if __name__ == '__main__':
    """
    Main entry point for running the CoarseRDA processor.

    Parses command-line arguments, loads input data, runs the focusing algorithm,
    and saves the processed output.

    Raises:
        AssertionError: If required input files are missing or invalid.
    """
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