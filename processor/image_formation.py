"""
Image formation functions for Sentinel-1 Level-0 processing.

This module provides functions for SAR image formation including:
- Pulse compression (range compression)
- Azimuth compression
- Unfocused SAR processing
- Reference function generation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift

from .signal_processing import (
    fft_1d, ifft_1d, apply_window, zero_pad,
    cross_correlate, hilbert_transform
)
from .constants import F_REF, SPEED_OF_LIGHT, CENTER_FREQ, WAVELENGTH

logger = logging.getLogger(__name__)


def pulse_compression(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Perform pulse compression using matched filtering.
    
    Args:
        signal: Input radar signal
        reference: Reference function (matched filter)
        
    Returns:
        Pulse-compressed signal
        
    Raises:
        ValueError: If input arrays are incompatible
    """
    if signal.ndim != 1 or reference.ndim != 1:
        raise ValueError('Input arrays must be 1-dimensional')
        
    if len(signal) == 0 or len(reference) == 0:
        raise ValueError('Input arrays cannot be empty')
        
    # Ensure signals have compatible lengths for FFT-based convolution
    output_length = len(signal) + len(reference) - 1
    fft_length = 2 ** int(np.ceil(np.log2(output_length)))
    
    # Zero-pad both signals
    signal_padded = zero_pad(signal, fft_length)
    reference_padded = zero_pad(reference, fft_length)
    
    # FFT-based convolution (correlation with conjugated reference)
    signal_fft = fft_1d(signal_padded)
    reference_fft = fft_1d(np.conj(reference_padded[::-1]))  # Time-reversed and conjugated
    
    # Multiply in frequency domain and transform back
    compressed_fft = signal_fft * reference_fft
    compressed_signal = ifft_1d(compressed_fft)
    
    # Extract valid portion and return real part for magnitude
    start_idx = len(reference) // 2
    end_idx = start_idx + len(signal)
    
    return compressed_signal[start_idx:end_idx]


def get_reference_function(replica_chirp: np.ndarray) -> np.ndarray:
    """
    Generate reference function for pulse compression from replica chirp.
    
    Args:
        replica_chirp: Replica chirp signal
        
    Returns:
        Reference function for matched filtering
    """
    if replica_chirp.ndim != 1:
        raise ValueError('Replica chirp must be 1-dimensional')
        
    # Conjugate and time-reverse the replica chirp
    reference = np.conj(replica_chirp[::-1])
    
    # Normalize to unit energy
    energy = np.sum(np.abs(reference)**2)
    if energy > 0:
        reference = reference / np.sqrt(energy)
        
    return reference


def generate_chirp(duration: float, bandwidth: float, sample_rate: float,
                  start_freq: float = 0.0, window_type: str = 'rectangular') -> np.ndarray:
    """
    Generate a linear frequency modulated (LFM) chirp signal.
    
    Args:
        duration: Chirp duration in seconds
        bandwidth: Chirp bandwidth in Hz
        sample_rate: Sampling rate in Hz
        start_freq: Starting frequency in Hz
        window_type: Window function to apply
        
    Returns:
        Generated chirp signal
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    
    # Linear frequency modulation
    chirp_rate = bandwidth / duration  # Hz/s
    phase = 2 * np.pi * (start_freq * t + 0.5 * chirp_rate * t**2)
    chirp = np.exp(1j * phase)
    
    # Apply window function
    if window_type.lower() != 'rectangular':
        chirp = apply_window(chirp, window_type)
        
    return chirp


def get_azimuth_blocks(packets: List) -> Tuple[List[List], int]:
    """
    Organize packets into azimuth blocks for processing.
    
    Args:
        packets: List of L0Packet objects
        
    Returns:
        Tuple of (azimuth_blocks, block_size)
    """
    if not packets:
        return [], 0
        
    # Group packets by azimuth position
    # This is a simplified implementation
    block_size = 100  # Typical block size
    blocks = []
    
    for i in range(0, len(packets), block_size):
        block = packets[i:i + block_size]
        if len(block) > 0:
            blocks.append(block)
            
    return blocks, block_size


def azimuth_frequency_ufr(range_compressed: np.ndarray,
                         dc_estimates: np.ndarray,
                         initial_packet,
                         dc_rate: float,
                         burst_duration: float,
                         prf: float,
                         processing_bandwidth: float) -> np.ndarray:
    """
    Perform azimuth compression using frequency domain unfocused processing.
    
    Args:
        range_compressed: Range-compressed data
        dc_estimates: Doppler centroid estimates
        initial_packet: Initial packet for parameters
        dc_rate: Doppler centroid rate
        burst_duration: Burst duration in seconds
        prf: Pulse repetition frequency
        processing_bandwidth: Processing bandwidth
        
    Returns:
        Azimuth-compressed data
    """
    if range_compressed.ndim != 2:
        raise ValueError('Range compressed data must be 2D')
        
    num_azimuth, num_range = range_compressed.shape
    
    # Create azimuth frequency vector
    az_freq = fftfreq(num_azimuth, 1/prf)
    az_freq = fftshift(az_freq)
    
    # Initialize output array
    azimuth_compressed = np.zeros_like(range_compressed)
    
    # Process each range bin
    for range_bin in range(num_range):
        # Extract azimuth signal for this range bin
        az_signal = range_compressed[:, range_bin]
        
        # Apply azimuth FFT
        az_spectrum = fftshift(fft(az_signal))
        
        # Apply azimuth matched filter in frequency domain
        # This is a simplified implementation
        dc_this_bin = dc_estimates[range_bin] if range_bin < len(dc_estimates) else 0.0
        
        # Create azimuth matched filter
        az_filter = _create_azimuth_filter(az_freq, dc_this_bin, dc_rate, 
                                         processing_bandwidth, burst_duration)
        
        # Apply filter
        filtered_spectrum = az_spectrum * np.conj(az_filter)
        
        # Transform back to time domain
        azimuth_compressed[:, range_bin] = ifft(ifftshift(filtered_spectrum))
        
    return azimuth_compressed


def azimuth_time_ufr(range_compressed: np.ndarray,
                    dc_estimates: np.ndarray,
                    az_fm_rate: np.ndarray,
                    initial_packet,
                    dc_rate: float,
                    burst_duration: float,
                    prf: float,
                    processing_bandwidth: float,
                    swath_number: int) -> np.ndarray:
    """
    Perform azimuth compression using time domain unfocused processing.
    
    Args:
        range_compressed: Range-compressed data
        dc_estimates: Doppler centroid estimates  
        az_fm_rate: Azimuth FM rate estimates
        initial_packet: Initial packet for parameters
        dc_rate: Doppler centroid rate
        burst_duration: Burst duration in seconds
        prf: Pulse repetition frequency
        processing_bandwidth: Processing bandwidth
        swath_number: Swath number for processing
        
    Returns:
        Azimuth-compressed data
    """
    if range_compressed.ndim != 2:
        raise ValueError('Range compressed data must be 2D')
        
    num_azimuth, num_range = range_compressed.shape
    
    # Create time vector
    t = np.arange(num_azimuth) / prf
    
    # Initialize output array
    azimuth_compressed = np.zeros_like(range_compressed)
    
    # Process each range bin
    for range_bin in range(num_range):
        # Extract azimuth signal for this range bin
        az_signal = range_compressed[:, range_bin]
        
        # Get parameters for this range bin
        dc_this_bin = dc_estimates[range_bin] if range_bin < len(dc_estimates) else 0.0
        fm_rate = az_fm_rate[range_bin, 0] if range_bin < az_fm_rate.shape[0] else 0.0
        
        # Create azimuth matched filter
        az_filter = _create_time_domain_azimuth_filter(t, dc_this_bin, fm_rate, 
                                                     burst_duration)
        
        # Apply matched filter using correlation
        compressed_signal = cross_correlate(az_signal, np.conj(az_filter[::-1]), 
                                          mode='same')
        
        azimuth_compressed[:, range_bin] = compressed_signal
        
    return azimuth_compressed


def _create_azimuth_filter(freq: np.ndarray, dc: float, dc_rate: float,
                          bandwidth: float, duration: float) -> np.ndarray:
    """
    Create azimuth matched filter in frequency domain.
    
    Args:
        freq: Frequency vector
        dc: Doppler centroid
        dc_rate: Doppler centroid rate
        bandwidth: Processing bandwidth
        duration: Burst duration
        
    Returns:
        Azimuth matched filter
    """
    # Shift frequency to be relative to Doppler centroid
    freq_shifted = freq - dc
    
    # Create filter based on quadratic phase
    # This is a simplified model
    phase = np.pi * dc_rate * duration * freq_shifted**2
    
    # Apply bandwidth limitation
    filter_response = np.exp(1j * phase)
    mask = np.abs(freq_shifted) <= bandwidth / 2
    filter_response[~mask] = 0
    
    return filter_response


def _create_time_domain_azimuth_filter(t: np.ndarray, dc: float, fm_rate: float,
                                     duration: float) -> np.ndarray:
    """
    Create azimuth matched filter in time domain.
    
    Args:
        t: Time vector
        dc: Doppler centroid
        fm_rate: FM rate
        duration: Filter duration
        
    Returns:
        Time domain azimuth matched filter
    """
    # Center time vector
    t_centered = t - t[len(t)//2]
    
    # Create quadratic phase function
    phase = 2 * np.pi * dc * t_centered + np.pi * fm_rate * t_centered**2
    
    # Apply time window
    window_samples = int(duration * len(t) / (t[-1] - t[0]))
    window_start = len(t)//2 - window_samples//2
    window_end = window_start + window_samples
    
    az_filter = np.zeros_like(t, dtype=complex)
    if window_start >= 0 and window_end <= len(t):
        az_filter[window_start:window_end] = np.exp(1j * phase[window_start:window_end])
        
    return az_filter


def range_migration_correction(data: np.ndarray, range_migration: np.ndarray) -> np.ndarray:
    """
    Correct for range cell migration in SAR data.
    
    Args:
        data: Input SAR data (azimuth x range)
        range_migration: Range migration correction values
        
    Returns:
        Range migration corrected data
    """
    if data.ndim != 2:
        raise ValueError('Input data must be 2D')
        
    num_azimuth, num_range = data.shape
    corrected_data = np.zeros_like(data)
    
    for az_idx in range(num_azimuth):
        migration = range_migration[az_idx] if az_idx < len(range_migration) else 0.0
        
        # Interpolate to correct for fractional pixel shifts
        shift_samples = migration
        
        if abs(shift_samples) < 0.01:  # No significant migration
            corrected_data[az_idx, :] = data[az_idx, :]
        else:
            # Use sinc interpolation for subpixel shifts
            corrected_data[az_idx, :] = _sinc_interpolate(data[az_idx, :], shift_samples)
            
    return corrected_data


def _sinc_interpolate(signal: np.ndarray, shift: float) -> np.ndarray:
    """
    Perform sinc interpolation for fractional pixel shifts.
    
    Args:
        signal: Input signal
        shift: Fractional shift in samples
        
    Returns:
        Interpolated signal
    """
    if abs(shift) < 1e-6:
        return signal.copy()
        
    # Use FFT-based fractional delay
    N = len(signal)
    freq = fftfreq(N)
    
    # Apply phase shift in frequency domain
    phase_shift = np.exp(-2j * np.pi * freq * shift)
    signal_fft = fft(signal)
    shifted_fft = signal_fft * phase_shift
    
    return np.real(ifft(shifted_fft))


def secondary_range_compression(data: np.ndarray, chirp_rate: float,
                              sample_rate: float) -> np.ndarray:
    """
    Apply secondary range compression for improved resolution.
    
    Args:
        data: Input range-compressed data
        chirp_rate: Chirp rate in Hz/s
        sample_rate: Sampling rate in Hz
        
    Returns:
        Secondary range compressed data
    """
    if data.ndim == 1:
        return _apply_secondary_compression_1d(data, chirp_rate, sample_rate)
    elif data.ndim == 2:
        # Apply to each azimuth line
        compressed_data = np.zeros_like(data)
        for az_idx in range(data.shape[0]):
            compressed_data[az_idx, :] = _apply_secondary_compression_1d(
                data[az_idx, :], chirp_rate, sample_rate)
        return compressed_data
    else:
        raise ValueError('Input data must be 1D or 2D')


def _apply_secondary_compression_1d(signal: np.ndarray, chirp_rate: float,
                                  sample_rate: float) -> np.ndarray:
    """
    Apply secondary range compression to 1D signal.
    
    Args:
        signal: Input 1D signal
        chirp_rate: Chirp rate in Hz/s
        sample_rate: Sampling rate in Hz
        
    Returns:
        Secondary compressed signal
    """
    N = len(signal)
    freq = fftfreq(N, 1/sample_rate)
    
    # Create secondary compression filter
    # This compensates for residual chirp
    phase_correction = -np.pi * freq**2 / chirp_rate
    correction_filter = np.exp(1j * phase_correction)
    
    # Apply in frequency domain
    signal_fft = fft(signal)
    compressed_fft = signal_fft * correction_filter
    
    return ifft(compressed_fft)


def multilook_processing(data: np.ndarray, range_looks: int = 1,
                        azimuth_looks: int = 1) -> np.ndarray:
    """
    Apply multilook processing to reduce speckle.
    
    Args:
        data: Input complex SAR data
        range_looks: Number of range looks
        azimuth_looks: Number of azimuth looks
        
    Returns:
        Multilooked intensity data
    """
    if data.ndim != 2:
        raise ValueError('Input data must be 2D')
        
    num_azimuth, num_range = data.shape
    
    # Calculate output dimensions
    out_azimuth = num_azimuth // azimuth_looks
    out_range = num_range // range_looks
    
    # Initialize output array
    multilooked = np.zeros((out_azimuth, out_range), dtype=np.float32)
    
    # Perform multilook averaging
    for az_out in range(out_azimuth):
        for rg_out in range(out_range):
            az_start = az_out * azimuth_looks
            az_end = az_start + azimuth_looks
            rg_start = rg_out * range_looks
            rg_end = rg_start + range_looks
            
            # Extract multilook window and compute intensity
            window = data[az_start:az_end, rg_start:rg_end]
            multilooked[az_out, rg_out] = np.mean(np.abs(window)**2)
            
    return multilooked
