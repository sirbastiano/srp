"""
Signal processing functions for Sentinel-1 Level-0 processing.

This module provides signal processing functions including FFT operations,
filtering, windowing, and other DSP utilities for SAR processing.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift

logger = logging.getLogger(__name__)


def fft_1d(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute 1D FFT of input data.
    
    Args:
        data: Input data array
        axis: Axis along which to compute FFT
        
    Returns:
        FFT of input data
    """
    return fft(data, axis=axis)


def ifft_1d(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute 1D inverse FFT of input data.
    
    Args:
        data: Input data array
        axis: Axis along which to compute IFFT
        
    Returns:
        IFFT of input data
    """
    return ifft(data, axis=axis)


def fft_2d(data: np.ndarray) -> np.ndarray:
    """
    Compute 2D FFT of input data.
    
    Args:
        data: Input 2D data array
        
    Returns:
        2D FFT of input data
    """
    return np.fft.fft2(data)


def ifft_2d(data: np.ndarray) -> np.ndarray:
    """
    Compute 2D inverse FFT of input data.
    
    Args:
        data: Input 2D data array
        
    Returns:
        2D IFFT of input data
    """
    return np.fft.ifft2(data)


def zero_pad(data: np.ndarray, new_length: int, axis: int = -1) -> np.ndarray:
    """
    Zero-pad data to specified length.
    
    Args:
        data: Input data array
        new_length: Target length after padding
        axis: Axis along which to pad
        
    Returns:
        Zero-padded data array
        
    Raises:
        ValueError: If new_length is smaller than current length
    """
    current_length = data.shape[axis]
    if new_length < current_length:
        raise ValueError(f'New length {new_length} smaller than current {current_length}')
        
    if new_length == current_length:
        return data.copy()
        
    # Calculate padding amounts
    pad_total = new_length - current_length
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    
    # Create padding specification for np.pad
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (pad_before, pad_after)
    
    return np.pad(data, pad_width, mode='constant', constant_values=0)


def apply_window(data: np.ndarray, window_type: str = 'hamming', 
                 axis: int = -1, beta: float = 8.6) -> np.ndarray:
    """
    Apply window function to data.
    
    Args:
        data: Input data array
        window_type: Type of window ('hamming', 'hanning', 'blackman', 'kaiser')
        axis: Axis along which to apply window
        beta: Beta parameter for Kaiser window
        
    Returns:
        Windowed data array
        
    Raises:
        ValueError: If window type is not supported
    """
    length = data.shape[axis]
    
    if window_type.lower() == 'hamming':
        window = np.hamming(length)
    elif window_type.lower() == 'hanning':
        window = np.hanning(length)
    elif window_type.lower() == 'blackman':
        window = np.blackman(length)
    elif window_type.lower() == 'kaiser':
        window = np.kaiser(length, beta)
    else:
        raise ValueError(f'Unsupported window type: {window_type}')
        
    # Reshape window to broadcast correctly
    window_shape = [1] * data.ndim
    window_shape[axis] = length
    window = window.reshape(window_shape)
    
    return data * window


def bandpass_filter(data: np.ndarray, low_freq: float, high_freq: float,
                   sample_rate: float, order: int = 5) -> np.ndarray:
    """
    Apply bandpass filter to data.
    
    Args:
        data: Input data array
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz)
        sample_rate: Sampling rate (Hz)
        order: Filter order
        
    Returns:
        Filtered data array
    """
    nyquist = 0.5 * sample_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        # Apply filter along last axis
        return signal.filtfilt(b, a, data, axis=-1)


def lowpass_filter(data: np.ndarray, cutoff_freq: float, 
                  sample_rate: float, order: int = 5) -> np.ndarray:
    """
    Apply lowpass filter to data.
    
    Args:
        data: Input data array
        cutoff_freq: Cutoff frequency (Hz)
        sample_rate: Sampling rate (Hz)
        order: Filter order
        
    Returns:
        Filtered data array
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    b, a = signal.butter(order, normal_cutoff, btype='low')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return signal.filtfilt(b, a, data, axis=-1)


def highpass_filter(data: np.ndarray, cutoff_freq: float,
                   sample_rate: float, order: int = 5) -> np.ndarray:
    """
    Apply highpass filter to data.
    
    Args:
        data: Input data array
        cutoff_freq: Cutoff frequency (Hz)
        sample_rate: Sampling rate (Hz)
        order: Filter order
        
    Returns:
        Filtered data array
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    b, a = signal.butter(order, normal_cutoff, btype='high')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return signal.filtfilt(b, a, data, axis=-1)


def cross_correlate(signal1: np.ndarray, signal2: np.ndarray, 
                   mode: str = 'full') -> np.ndarray:
    """
    Compute cross-correlation between two signals.
    
    Args:
        signal1: First input signal
        signal2: Second input signal
        mode: Correlation mode ('full', 'valid', 'same')
        
    Returns:
        Cross-correlation result
    """
    return np.correlate(signal1, signal2, mode=mode)


def autocorrelate(signal_data: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation of a signal.
    
    Args:
        signal_data: Input signal
        max_lag: Maximum lag to compute (None for full)
        
    Returns:
        Autocorrelation result
    """
    correlation = np.correlate(signal_data, signal_data, mode='full')
    
    if max_lag is not None:
        center = len(correlation) // 2
        start = max(0, center - max_lag)
        end = min(len(correlation), center + max_lag + 1)
        correlation = correlation[start:end]
        
    return correlation


def hilbert_transform(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute Hilbert transform of data.
    
    Args:
        data: Input data array
        axis: Axis along which to compute transform
        
    Returns:
        Hilbert transform result
    """
    return signal.hilbert(data, axis=axis)


def envelope_detection(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Detect envelope of signal using Hilbert transform.
    
    Args:
        data: Input signal data
        axis: Axis along which to compute envelope
        
    Returns:
        Signal envelope
    """
    analytic_signal = signal.hilbert(data, axis=axis)
    return np.abs(analytic_signal)


def phase_unwrap(phase: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Unwrap phase discontinuities.
    
    Args:
        phase: Input phase array (radians)
        axis: Axis along which to unwrap
        
    Returns:
        Unwrapped phase array
    """
    return np.unwrap(phase, axis=axis)


def detrend_linear(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Remove linear trend from data.
    
    Args:
        data: Input data array
        axis: Axis along which to detrend
        
    Returns:
        Detrended data array
    """
    return signal.detrend(data, axis=axis, type='linear')


def detrend_constant(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Remove constant (DC) component from data.
    
    Args:
        data: Input data array
        axis: Axis along which to detrend
        
    Returns:
        Detrended data array
    """
    return signal.detrend(data, axis=axis, type='constant')


def upsample(data: np.ndarray, factor: int, axis: int = -1) -> np.ndarray:
    """
    Upsample data by integer factor using zero-padding in frequency domain.
    
    Args:
        data: Input data array
        factor: Upsampling factor
        axis: Axis along which to upsample
        
    Returns:
        Upsampled data array
    """
    if factor <= 1:
        return data.copy()
        
    # FFT, zero-pad, IFFT
    fft_data = fft(data, axis=axis)
    
    # Zero-pad in frequency domain
    padded_fft = zero_pad(fft_data, fft_data.shape[axis] * factor, axis=axis)
    
    # IFFT and scale
    upsampled = ifft(padded_fft, axis=axis)
    return upsampled * factor


def downsample(data: np.ndarray, factor: int, axis: int = -1,
               anti_alias: bool = True) -> np.ndarray:
    """
    Downsample data by integer factor.
    
    Args:
        data: Input data array
        factor: Downsampling factor
        axis: Axis along which to downsample
        anti_alias: Whether to apply anti-aliasing filter
        
    Returns:
        Downsampled data array
    """
    if factor <= 1:
        return data.copy()
        
    # Apply anti-aliasing filter if requested
    if anti_alias:
        # Design lowpass filter with cutoff at Nyquist/factor
        cutoff = 0.5 / factor
        b, a = signal.butter(5, cutoff, btype='low')
        data = signal.filtfilt(b, a, data, axis=axis)
        
    # Downsample by taking every factor-th sample
    indices = slice(None, None, factor)
    slicing = [slice(None)] * data.ndim
    slicing[axis] = indices
    
    return data[tuple(slicing)]


def complex_to_magnitude_phase(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert complex data to magnitude and phase.
    
    Args:
        data: Complex input data
        
    Returns:
        Tuple of (magnitude, phase) arrays
    """
    magnitude = np.abs(data)
    phase = np.angle(data)
    return magnitude, phase


def magnitude_phase_to_complex(magnitude: np.ndarray, 
                              phase: np.ndarray) -> np.ndarray:
    """
    Convert magnitude and phase to complex data.
    
    Args:
        magnitude: Magnitude array
        phase: Phase array (radians)
        
    Returns:
        Complex data array
    """
    return magnitude * np.exp(1j * phase)


def rms(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Compute root mean square of data.
    
    Args:
        data: Input data array
        axis: Axis along which to compute RMS (None for all)
        
    Returns:
        RMS value(s)
    """
    return np.sqrt(np.mean(np.abs(data)**2, axis=axis))


def snr_estimate(signal_data: np.ndarray, noise_data: np.ndarray) -> float:
    """
    Estimate signal-to-noise ratio.
    
    Args:
        signal_data: Signal data array
        noise_data: Noise data array
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(np.abs(signal_data)**2)
    noise_power = np.mean(np.abs(noise_data)**2)
    
    if noise_power == 0:
        return float('inf')
        
    snr_linear = signal_power / noise_power
    return 10 * np.log10(snr_linear)


def frequency_shift(data: np.ndarray, shift_freq: float, 
                   sample_rate: float, axis: int = -1) -> np.ndarray:
    """
    Apply frequency shift to data.
    
    Args:
        data: Input data array
        shift_freq: Frequency shift in Hz
        sample_rate: Sampling rate in Hz
        axis: Axis along which to apply shift
        
    Returns:
        Frequency-shifted data
    """
    length = data.shape[axis]
    t = np.arange(length) / sample_rate
    
    # Create frequency shift vector
    shift_shape = [1] * data.ndim
    shift_shape[axis] = length
    shift_vector = np.exp(2j * np.pi * shift_freq * t).reshape(shift_shape)
    
    return data * shift_vector
