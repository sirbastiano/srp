"""
Doppler frequency estimation for Sentinel-1 Level-0 processing.

This module provides functions for estimating Doppler centroid frequency
and Doppler rate from SAR data.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict
import logging
from scipy import signal
from scipy.optimize import minimize_scalar

from .signal_processing import (
    fft_1d, ifft_1d, cross_correlate, autocorrelate,
    apply_window, envelope_detection
)
from .constants import CENTER_FREQ, WAVELENGTH, SPEED_OF_LIGHT

logger = logging.getLogger(__name__)


class DopplerEstimator:
    """
    Doppler frequency estimation for SAR processing.
    
    This class provides various methods for estimating Doppler centroid
    frequency and Doppler rate from SAR signal data.
    """
    
    def __init__(self, prf: float, processing_bandwidth: float = 1000.0):
        """
        Initialize Doppler estimator.
        
        Args:
            prf: Pulse repetition frequency in Hz
            processing_bandwidth: Processing bandwidth in Hz
        """
        self.prf = prf
        self.processing_bandwidth = processing_bandwidth
        self._doppler_ambiguity = prf / 2
        
    def estimate_doppler_centroid_correlation(self, data: np.ndarray,
                                            correlation_length: int = 512) -> float:
        """
        Estimate Doppler centroid using correlation method.
        
        Args:
            data: Input SAR data (1D azimuth signal or 2D range-azimuth)
            correlation_length: Length for correlation calculation
            
        Returns:
            Estimated Doppler centroid frequency in Hz
        """
        if data.ndim == 1:
            return self._estimate_dc_correlation_1d(data, correlation_length)
        elif data.ndim == 2:
            # Average over range bins
            dc_estimates = []
            for range_bin in range(data.shape[1]):
                if np.any(data[:, range_bin] != 0):  # Skip empty range bins
                    dc = self._estimate_dc_correlation_1d(data[:, range_bin], correlation_length)
                    dc_estimates.append(dc)
                    
            return np.median(dc_estimates) if dc_estimates else 0.0
        else:
            raise ValueError('Input data must be 1D or 2D')
            
    def _estimate_dc_correlation_1d(self, signal_data: np.ndarray,
                                  correlation_length: int) -> float:
        """
        Estimate Doppler centroid for 1D signal using correlation.
        
        Args:
            signal_data: 1D azimuth signal
            correlation_length: Length for correlation
            
        Returns:
            Doppler centroid estimate in Hz
        """
        if len(signal_data) < correlation_length * 2:
            correlation_length = len(signal_data) // 4
            
        # Extract two segments for correlation
        mid_point = len(signal_data) // 2
        start_idx = mid_point - correlation_length
        end_idx = mid_point + correlation_length
        
        segment1 = signal_data[start_idx:mid_point]
        segment2 = signal_data[mid_point:end_idx]
        
        # Calculate cross-correlation
        correlation = cross_correlate(segment1, np.conj(segment2), mode='full')
        
        # Find peak and estimate phase
        peak_idx = np.argmax(np.abs(correlation))
        phase = np.angle(correlation[peak_idx])
        
        # Convert phase to Doppler frequency
        # Phase difference corresponds to Doppler shift
        time_separation = correlation_length / self.prf
        doppler_freq = phase / (2 * np.pi * time_separation)
        
        # Resolve ambiguity
        doppler_freq = self._resolve_doppler_ambiguity(doppler_freq)
        
        return doppler_freq
        
    def estimate_doppler_centroid_fft(self, data: np.ndarray,
                                    window_type: str = 'hamming') -> float:
        """
        Estimate Doppler centroid using FFT-based spectral analysis.
        
        Args:
            data: Input SAR data
            window_type: Window function for spectral analysis
            
        Returns:
            Estimated Doppler centroid frequency in Hz
        """
        if data.ndim == 1:
            return self._estimate_dc_fft_1d(data, window_type)
        elif data.ndim == 2:
            # Average over range bins
            dc_estimates = []
            for range_bin in range(data.shape[1]):
                if np.any(data[:, range_bin] != 0):
                    dc = self._estimate_dc_fft_1d(data[:, range_bin], window_type)
                    dc_estimates.append(dc)
                    
            return np.median(dc_estimates) if dc_estimates else 0.0
        else:
            raise ValueError('Input data must be 1D or 2D')
            
    def _estimate_dc_fft_1d(self, signal_data: np.ndarray,
                          window_type: str) -> float:
        """
        Estimate Doppler centroid for 1D signal using FFT.
        
        Args:
            signal_data: 1D azimuth signal
            window_type: Window function type
            
        Returns:
            Doppler centroid estimate in Hz
        """
        # Apply window function
        windowed_signal = apply_window(signal_data, window_type)
        
        # Compute power spectrum
        spectrum = fft_1d(windowed_signal)
        power_spectrum = np.abs(spectrum)**2
        
        # Create frequency vector
        freqs = np.fft.fftfreq(len(signal_data), 1/self.prf)
        freqs = np.fft.fftshift(freqs)
        power_spectrum = np.fft.fftshift(power_spectrum)
        
        # Find spectral centroid
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0
            
        weighted_freq = np.sum(freqs * power_spectrum)
        doppler_centroid = weighted_freq / total_power
        
        # Resolve ambiguity
        doppler_centroid = self._resolve_doppler_ambiguity(doppler_centroid)
        
        return doppler_centroid
        
    def estimate_doppler_centroid_energy_balance(self, data: np.ndarray) -> float:
        """
        Estimate Doppler centroid using energy balance method.
        
        Args:
            data: Input SAR data
            
        Returns:
            Estimated Doppler centroid frequency in Hz
        """
        if data.ndim == 1:
            signal_data = data
        elif data.ndim == 2:
            # Use middle range bin or average
            signal_data = data[:, data.shape[1]//2]
        else:
            raise ValueError('Input data must be 1D or 2D')
            
        # Split spectrum into upper and lower halves
        spectrum = fft_1d(signal_data)
        power_spectrum = np.abs(spectrum)**2
        
        N = len(power_spectrum)
        mid_point = N // 2
        
        # Energy in upper and lower halves
        lower_energy = np.sum(power_spectrum[:mid_point])
        upper_energy = np.sum(power_spectrum[mid_point:])
        
        total_energy = lower_energy + upper_energy
        if total_energy == 0:
            return 0.0
            
        # Energy balance indicates Doppler shift
        energy_ratio = (upper_energy - lower_energy) / total_energy
        
        # Convert to frequency (empirical relationship)
        doppler_freq = energy_ratio * self.prf / 4
        
        return self._resolve_doppler_ambiguity(doppler_freq)
        
    def estimate_doppler_rate(self, data: np.ndarray, 
                            dc_estimates: np.ndarray) -> np.ndarray:
        """
        Estimate Doppler rate from SAR data.
        
        Args:
            data: Input SAR data (azimuth x range)
            dc_estimates: Doppler centroid estimates for each range bin
            
        Returns:
            Doppler rate estimates for each range bin
        """
        if data.ndim != 2:
            raise ValueError('Input data must be 2D for Doppler rate estimation')
            
        num_azimuth, num_range = data.shape
        doppler_rates = np.zeros(num_range)
        
        # Time vector for azimuth direction
        azimuth_times = np.arange(num_azimuth) / self.prf
        
        for range_bin in range(num_range):
            if np.any(data[:, range_bin] != 0):
                doppler_rates[range_bin] = self._estimate_doppler_rate_range_bin(
                    data[:, range_bin], azimuth_times, dc_estimates[range_bin])
                    
        return doppler_rates
        
    def _estimate_doppler_rate_range_bin(self, signal_data: np.ndarray,
                                       times: np.ndarray, dc_estimate: float) -> float:
        """
        Estimate Doppler rate for a single range bin.
        
        Args:
            signal_data: 1D signal for range bin
            times: Time vector
            dc_estimate: Doppler centroid estimate
            
        Returns:
            Doppler rate in Hz/s
        """
        # Use short-time spectral analysis
        window_length = min(256, len(signal_data) // 4)
        overlap = window_length // 2
        
        # Calculate instantaneous Doppler frequency
        doppler_history = []
        time_history = []
        
        for start_idx in range(0, len(signal_data) - window_length, 
                             window_length - overlap):
            end_idx = start_idx + window_length
            window_signal = signal_data[start_idx:end_idx]
            window_time = times[start_idx + window_length//2]
            
            # Estimate instantaneous Doppler
            inst_doppler = self._estimate_dc_fft_1d(window_signal, 'hamming')
            
            doppler_history.append(inst_doppler)
            time_history.append(window_time)
            
        if len(doppler_history) < 2:
            return 0.0
            
        # Fit linear trend to get Doppler rate
        doppler_history = np.array(doppler_history)
        time_history = np.array(time_history)
        
        # Linear regression
        coeffs = np.polyfit(time_history, doppler_history, 1)
        doppler_rate = coeffs[0]  # Slope is Doppler rate
        
        return doppler_rate
        
    def estimate_multilook_doppler_centroid(self, data: np.ndarray,
                                          looks: int = 4) -> Tuple[float, float]:
        """
        Estimate Doppler centroid using multilook processing.
        
        Args:
            data: Input SAR data
            looks: Number of looks for averaging
            
        Returns:
            Tuple of (doppler_centroid, confidence)
        """
        if data.ndim == 1:
            signal_data = data
        elif data.ndim == 2:
            signal_data = data[:, data.shape[1]//2]  # Use middle range bin
        else:
            raise ValueError('Input data must be 1D or 2D')
            
        # Split data into looks
        look_length = len(signal_data) // looks
        if look_length < 64:  # Minimum length for reliable estimation
            return self.estimate_doppler_centroid_fft(signal_data), 0.0
            
        dc_estimates = []
        
        for look in range(looks):
            start_idx = look * look_length
            end_idx = start_idx + look_length
            look_data = signal_data[start_idx:end_idx]
            
            dc_estimate = self.estimate_doppler_centroid_fft(look_data)
            dc_estimates.append(dc_estimate)
            
        dc_estimates = np.array(dc_estimates)
        
        # Calculate mean and confidence (inverse of variance)
        mean_dc = np.mean(dc_estimates)
        variance = np.var(dc_estimates)
        confidence = 1.0 / (variance + 1e-10)  # Avoid division by zero
        
        return mean_dc, confidence
        
    def _resolve_doppler_ambiguity(self, doppler_freq: float) -> float:
        """
        Resolve Doppler ambiguity using PRF.
        
        Args:
            doppler_freq: Raw Doppler frequency estimate
            
        Returns:
            Ambiguity-resolved Doppler frequency
        """
        # Doppler frequencies are ambiguous modulo PRF
        while doppler_freq > self._doppler_ambiguity:
            doppler_freq -= self.prf
        while doppler_freq < -self._doppler_ambiguity:
            doppler_freq += self.prf
            
        return doppler_freq
        
    def estimate_doppler_bandwidth(self, data: np.ndarray) -> float:
        """
        Estimate Doppler bandwidth of the signal.
        
        Args:
            data: Input SAR data
            
        Returns:
            Estimated Doppler bandwidth in Hz
        """
        if data.ndim == 1:
            signal_data = data
        elif data.ndim == 2:
            signal_data = data[:, data.shape[1]//2]
        else:
            raise ValueError('Input data must be 1D or 2D')
            
        # Compute power spectrum
        windowed_signal = apply_window(signal_data, 'hamming')
        spectrum = fft_1d(windowed_signal)
        power_spectrum = np.abs(spectrum)**2
        power_spectrum = np.fft.fftshift(power_spectrum)
        
        # Find -3dB bandwidth
        max_power = np.max(power_spectrum)
        threshold = max_power / 2  # -3dB
        
        # Find indices where power exceeds threshold
        above_threshold = power_spectrum >= threshold
        indices = np.where(above_threshold)[0]
        
        if len(indices) == 0:
            return 0.0
            
        # Calculate bandwidth
        bandwidth_samples = indices[-1] - indices[0] + 1
        bandwidth_hz = bandwidth_samples * self.prf / len(power_spectrum)
        
        return bandwidth_hz
        
    def quality_assessment(self, data: np.ndarray, 
                          dc_estimate: float) -> Dict[str, float]:
        """
        Assess quality of Doppler centroid estimate.
        
        Args:
            data: Input SAR data
            dc_estimate: Doppler centroid estimate to assess
            
        Returns:
            Dictionary of quality metrics
        """
        if data.ndim == 1:
            signal_data = data
        elif data.ndim == 2:
            signal_data = data[:, data.shape[1]//2]
        else:
            raise ValueError('Input data must be 1D or 2D')
            
        # Calculate various quality metrics
        metrics = {}
        
        # Signal-to-noise ratio estimate
        signal_power = np.mean(np.abs(signal_data)**2)
        noise_power = np.var(np.abs(signal_data)**2) / len(signal_data)
        metrics['snr_db'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Spectral peak sharpness
        spectrum = fft_1d(apply_window(signal_data, 'hamming'))
        power_spectrum = np.abs(spectrum)**2
        power_spectrum = np.fft.fftshift(power_spectrum)
        
        max_power = np.max(power_spectrum)
        mean_power = np.mean(power_spectrum)
        metrics['peak_sharpness'] = max_power / (mean_power + 1e-10)
        
        # Doppler bandwidth
        metrics['doppler_bandwidth'] = self.estimate_doppler_bandwidth(signal_data)
        
        # Confidence based on multiple estimates
        dc_fft = self.estimate_doppler_centroid_fft(signal_data)
        dc_corr = self.estimate_doppler_centroid_correlation(signal_data)
        
        diff = abs(dc_fft - dc_corr)
        metrics['estimate_consistency'] = np.exp(-diff / 100)  # Confidence metric
        
        return metrics
