"""
Main S1 Decoder class for Sentinel-1 Level-0 processing.

This module provides the main S1Decoder class that orchestrates the entire
processing pipeline from Level-0 packets to SAR images.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging

from .packet import L0Packet
from .state_vectors import StateVectors
from .doppler import DopplerEstimator
from .image_formation import (
    pulse_compression, get_reference_function, generate_chirp,
    azimuth_frequency_ufr, azimuth_time_ufr, get_azimuth_blocks,
    range_migration_correction, multilook_processing
)
from .signal_processing import (
    fft_1d, ifft_1d, apply_window, bandpass_filter,
    frequency_shift, detrend_constant
)
from .constants import (
    SWATH_NAMES, INSTRUMENT_MODES, F_REF, CENTER_FREQ, WAVELENGTH,
    PROCESSING_PARAMS
)

logger = logging.getLogger(__name__)


class S1Decoder:
    """
    Main decoder class for Sentinel-1 Level-0 processing.
    
    This class provides the complete processing pipeline from Level-0 packets
    to focused SAR images, including range compression, azimuth compression,
    and various processing options.
    """
    
    def __init__(self, filename: Optional[Union[str, Path]] = None):
        """
        Initialize S1 Decoder.
        
        Args:
            filename: Path to Level-0 data file (optional)
        """
        self._filename: Optional[Path] = Path(filename) if filename else None
        self._flat_packets: List[L0Packet] = []
        self._times: List[float] = []
        self._swath_counts: Dict[str, int] = {}
        self._echo_packets: Dict[str, List[List[L0Packet]]] = {}
        self._cal_packets: Dict[str, List[L0Packet]] = {}
        self._state_vectors: Optional[StateVectors] = None
        
        if self._filename:
            self._set_packets()
            self._set_state_vectors()
            
    def _set_packets(self) -> None:
        """Load and organize packets from file."""
        if not self._filename or not self._filename.exists():
            raise FileNotFoundError(f'File not found: {self._filename}')
            
        logger.info(f'Loading packets from {self._filename}')
        
        # Load all packets
        self._flat_packets = L0Packet.get_packets(self._filename)
        logger.info(f'Loaded {len(self._flat_packets)} packets')
        
        # Extract times
        self._times = [packet.get_time() for packet in self._flat_packets]
        
        # Organize packets by swath and signal type
        self._organize_packets()
        
    def _organize_packets(self) -> None:
        """Organize packets by swath and signal type."""
        self._swath_counts.clear()
        self._echo_packets.clear()
        self._cal_packets.clear()
        
        echo_packets_list = {}
        
        for packet in self._flat_packets:
            swath_name = packet.get_swath_name()
            signal_type = packet.get_signal_type()
            
            # Skip packets with unknown swath names
            if swath_name == 'UNKNOWN':
                continue
                
            # Count packets per swath
            if swath_name not in self._swath_counts:
                self._swath_counts[swath_name] = 0
            self._swath_counts[swath_name] += 1
            
            # Organize by signal type
            if signal_type == 'ECHO':
                if swath_name not in echo_packets_list:
                    echo_packets_list[swath_name] = []
                echo_packets_list[swath_name].append(packet)
            elif packet.is_calibration_packet():
                if swath_name not in self._cal_packets:
                    self._cal_packets[swath_name] = []
                self._cal_packets[swath_name].append(packet)
                
        # Convert to 2D structure for burst processing
        for swath_name, packets in echo_packets_list.items():
            if len(packets) > 0:
                blocks, block_size = get_azimuth_blocks(packets)
                self._echo_packets[swath_name] = blocks
            
        logger.info(f'Organized packets into {len(self._swath_counts)} swaths')
        for swath, count in self._swath_counts.items():
            logger.info(f'  {swath}: {count} packets')
            
    def _set_state_vectors(self) -> None:
        """Extract state vectors from packets."""
        self._state_vectors = StateVectors(self._flat_packets)
        logger.info('Extracted state vectors from packets')
        
    def get_state_vectors(self) -> StateVectors:
        """
        Get state vectors.
        
        Returns:
            StateVectors object
        """
        if self._state_vectors is None:
            raise ValueError('State vectors not initialized')
        return self._state_vectors
        
    def get_swath_names(self) -> List[str]:
        """
        Get list of available swath names.
        
        Returns:
            List of swath names
        """
        return list(self._swath_counts.keys())
        
    def get_burst_count(self, swath: str) -> int:
        """
        Get number of bursts for a swath.
        
        Args:
            swath: Swath name
            
        Returns:
            Number of bursts
        """
        self._validate_request(swath)
        
        if swath in self._echo_packets:
            return len(self._echo_packets[swath])
        return 0
        
    def get_burst(self, swath: str, burst: int) -> np.ndarray:
        """
        Get raw burst data for specified swath and burst.
        
        Args:
            swath: Swath name
            burst: Burst number
            
        Returns:
            Complex burst data array
        """
        self._validate_request(swath, burst)
        
        if swath not in self._echo_packets:
            raise ValueError(f'No echo packets for swath {swath}')
            
        if burst >= len(self._echo_packets[swath]):
            raise ValueError(f'Burst {burst} not available for swath {swath}')
            
        # Extract burst packets
        burst_packets = self._echo_packets[swath][burst]
        
        # Decode packet data
        burst_data = []
        for packet in burst_packets:
            decoded_data = packet.decode_packet_data()
            if decoded_data.size > 0:  # Only add non-empty decoded data
                burst_data.append(decoded_data)
            
        # Combine into 2D array (azimuth x range)
        if burst_data:
            # Handle case where decoded arrays have different lengths
            try:
                return np.array(burst_data)
            except ValueError:
                # If arrays have different shapes, find max length and pad
                max_length = max(len(arr) for arr in burst_data)
                padded_data = []
                for arr in burst_data:
                    if len(arr) < max_length:
                        # Pad with zeros
                        padded = np.zeros(max_length, dtype=arr.dtype)
                        padded[:len(arr)] = arr
                        padded_data.append(padded)
                    else:
                        padded_data.append(arr)
                return np.array(padded_data)
        else:
            return np.array([], dtype=complex)
            
    def get_swath(self, swath: str) -> np.ndarray:
        """
        Get complete swath data (all bursts concatenated).
        
        Args:
            swath: Swath name
            
        Returns:
            Complex swath data array with padded bursts to handle different sample counts
        """
        self._validate_request(swath)
        
        burst_count = self.get_burst_count(swath)
        if burst_count == 0:
            return np.array([])
            
        # First pass: collect all bursts and find maximum sample count
        swath_data = []
        max_samples = 0
        
        for burst_idx in range(burst_count):
            burst_data = self.get_burst(swath, burst_idx)
            if burst_data.size > 0:
                swath_data.append(burst_data)
                max_samples = max(max_samples, burst_data.shape[1])
        
        if not swath_data:
            return np.array([])
            
        # Second pass: pad all bursts to have the same number of samples
        padded_data = []
        for burst_array in swath_data:
            current_samples = burst_array.shape[1]
            if current_samples < max_samples:
                # Pad with zeros on the right
                padding = max_samples - current_samples
                padded_burst = np.pad(
                    burst_array, 
                    ((0, 0), (0, padding)), 
                    mode='constant', 
                    constant_values=0
                )
                padded_data.append(padded_burst)
            else:
                padded_data.append(burst_array)
        
        return np.concatenate(padded_data, axis=0)
            
    def get_range_compressed_burst(self, swath: str, burst: int,
                                 range_doppler: bool = False) -> np.ndarray:
        """
        Get range-compressed burst data.
        
        Args:
            swath: Swath name
            burst: Burst number
            range_doppler: Whether to apply range-Doppler correction
            
        Returns:
            Range-compressed burst data
        """
        # Get raw burst data
        burst_data = self.get_burst(swath, burst)
        
        if burst_data.size == 0:
            return np.array([])
            
        # Apply range compression
        return self._range_compress(burst_data, range_doppler)
        
    def get_range_compressed_swath(self, swath: str,
                                 range_doppler: bool = False) -> np.ndarray:
        """
        Get range-compressed swath data.
        
        Args:
            swath: Swath name
            range_doppler: Whether to apply range-Doppler correction
            
        Returns:
            Range-compressed swath data
        """
        self._validate_request(swath)
        
        # Determine processing mode
        instrument_mode = INSTRUMENT_MODES.get(swath, 'SM')
        
        if instrument_mode == 'SM':
            return self._get_range_compressed_swath_sm(swath, range_doppler)
        elif instrument_mode == 'IW':
            return self._get_range_compressed_swath_iw(swath, range_doppler)
        else:
            # Default processing
            swath_data = self.get_swath(swath)
            return self._range_compress(swath_data, range_doppler)
            
    def get_azimuth_compressed_burst(self, swath: str, burst: int) -> np.ndarray:
        """
        Get azimuth-compressed burst data.
        
        Args:
            swath: Swath name
            burst: Burst number
            
        Returns:
            Azimuth-compressed burst data
        """
        # Get range-compressed data first
        range_compressed = self.get_range_compressed_burst(swath, burst)
        
        if range_compressed.size == 0:
            return np.array([])
            
        # Apply azimuth compression
        return self._azimuth_compress(range_compressed, swath, burst)
        
    def get_azimuth_compressed_swath(self, swath: str) -> np.ndarray:
        """
        Get azimuth-compressed swath data.
        
        Args:
            swath: Swath name
            
        Returns:
            Azimuth-compressed swath data
        """
        self._validate_request(swath)
        
        # Determine processing mode
        instrument_mode = INSTRUMENT_MODES.get(swath, 'SM')
        
        if instrument_mode == 'SM':
            return self._get_azimuth_compressed_swath_sm(swath)
        elif instrument_mode == 'IW':
            return self._get_azimuth_compressed_swath_iw(swath)
        else:
            # Default processing
            range_compressed = self.get_range_compressed_swath(swath)
            return self._azimuth_compress(range_compressed, swath)
            
    def _range_compress(self, data: np.ndarray, range_doppler: bool = False) -> np.ndarray:
        """
        Apply range compression to data.
        
        Args:
            data: Input raw data
            range_doppler: Whether to apply range-Doppler correction
            
        Returns:
            Range-compressed data
        """
        if data.ndim == 1:
            return self._range_compress_1d(data, range_doppler)
        elif data.ndim == 2:
            # Apply to each azimuth line
            compressed_data = np.zeros_like(data)
            for az_idx in range(data.shape[0]):
                compressed_data[az_idx, :] = self._range_compress_1d(
                    data[az_idx, :], range_doppler)
            return compressed_data
        else:
            raise ValueError('Input data must be 1D or 2D')
            
    def _range_compress_1d(self, signal: np.ndarray, range_doppler: bool) -> np.ndarray:
        """
        Apply range compression to 1D signal.
        
        Args:
            signal: Input 1D signal
            range_doppler: Whether to apply range-Doppler correction
            
        Returns:
            Range-compressed signal
        """
        # Generate reference chirp
        # In practice, chirp parameters would come from packet data
        duration = 27.12e-6  # Typical S1 chirp duration
        bandwidth = 100e6    # Typical S1 bandwidth
        sample_rate = 117.6e6  # Typical S1 sample rate
        
        chirp = generate_chirp(duration, bandwidth, sample_rate)
        
        # Create matched filter
        reference = get_reference_function(chirp)
        
        # Apply pulse compression
        compressed_signal = pulse_compression(signal, reference)
        
        if range_doppler:
            # Apply range-Doppler correction (frequency domain)
            compressed_signal = self._apply_range_doppler_correction(compressed_signal)
            
        return compressed_signal
        
    def _azimuth_compress(self, data: np.ndarray, swath: str, 
                        burst: Optional[int] = None) -> np.ndarray:
        """
        Apply azimuth compression to range-compressed data.
        
        Args:
            data: Range-compressed data
            swath: Swath name
            burst: Burst number (optional)
            
        Returns:
            Azimuth-compressed data
        """
        if data.ndim != 2:
            raise ValueError('Input data must be 2D for azimuth compression')
            
        # Get processing parameters
        prf = self._estimate_prf(swath)
        doppler_estimator = DopplerEstimator(prf)
        
        # Estimate Doppler centroid
        dc_estimates = np.zeros(data.shape[1])
        for range_bin in range(data.shape[1]):
            if np.any(data[:, range_bin] != 0):
                dc_estimates[range_bin] = doppler_estimator.estimate_doppler_centroid_fft(
                    data[:, range_bin])
                    
        # Apply azimuth compression
        # Choose method based on instrument mode
        instrument_mode = INSTRUMENT_MODES.get(swath, 'SM')
        
        if instrument_mode == 'IW':
            # TOPS mode processing
            return self._azimuth_compress_tops(data, dc_estimates, prf, swath)
        else:
            # Standard stripmap processing
            return self._azimuth_compress_stripmap(data, dc_estimates, prf)
            
    def _azimuth_compress_stripmap(self, data: np.ndarray, dc_estimates: np.ndarray,
                                 prf: float) -> np.ndarray:
        """
        Azimuth compression for stripmap mode.
        
        Args:
            data: Range-compressed data
            dc_estimates: Doppler centroid estimates
            prf: Pulse repetition frequency
            
        Returns:
            Azimuth-compressed data
        """
        # Get initial packet for parameters
        if not self._flat_packets:
            raise ValueError('No packets available')
            
        initial_packet = self._flat_packets[0]
        
        # Processing parameters
        burst_duration = len(data) / prf
        dc_rate = 0.0  # Simplified - should be estimated
        processing_bandwidth = PROCESSING_PARAMS['processing_bandwidth']
        
        # Apply frequency domain azimuth compression
        return azimuth_frequency_ufr(
            data, dc_estimates, initial_packet, dc_rate,
            burst_duration, prf, processing_bandwidth)
            
    def _azimuth_compress_tops(self, data: np.ndarray, dc_estimates: np.ndarray,
                             prf: float, swath: str) -> np.ndarray:
        """
        Azimuth compression for TOPS mode.
        
        Args:
            data: Range-compressed data
            dc_estimates: Doppler centroid estimates
            prf: Pulse repetition frequency
            swath: Swath name
            
        Returns:
            Azimuth-compressed data
        """
        # TOPS processing is more complex and requires beam pattern correction
        # This is a simplified implementation
        
        # Get azimuth FM rate estimates
        az_fm_rate = self._estimate_azimuth_fm_rates(data.shape[1])
        
        # Get initial packet
        initial_packet = self._flat_packets[0]
        
        # Processing parameters
        burst_duration = len(data) / prf
        dc_rate = 0.0  # Should be estimated from data
        processing_bandwidth = PROCESSING_PARAMS['processing_bandwidth']
        swath_number = self._get_swath_number(swath)
        
        # Apply time domain azimuth compression
        return azimuth_time_ufr(
            data, dc_estimates, az_fm_rate, initial_packet, dc_rate,
            burst_duration, prf, processing_bandwidth, swath_number)
            
    def _get_range_compressed_swath_sm(self, swath: str, 
                                     range_doppler: bool) -> np.ndarray:
        """Get range-compressed data for stripmap mode."""
        swath_data = self.get_swath(swath)
        return self._range_compress(swath_data, range_doppler)
        
    def _get_range_compressed_swath_iw(self, swath: str,
                                     range_doppler: bool) -> np.ndarray:
        """Get range-compressed data for interferometric wide swath mode."""
        # IW mode processing requires burst-by-burst processing
        burst_count = self.get_burst_count(swath)
        
        if burst_count == 0:
            return np.array([])
            
        # Process each burst separately
        compressed_bursts = []
        for burst_idx in range(burst_count):
            burst_data = self.get_range_compressed_burst(swath, burst_idx, range_doppler)
            if burst_data.size > 0:
                compressed_bursts.append(burst_data)
                
        if compressed_bursts:
            return np.concatenate(compressed_bursts, axis=0)
        else:
            return np.array([])
            
    def _get_azimuth_compressed_swath_sm(self, swath: str) -> np.ndarray:
        """Get azimuth-compressed data for stripmap mode."""
        range_compressed = self.get_range_compressed_swath(swath)
        return self._azimuth_compress(range_compressed, swath)
        
    def _get_azimuth_compressed_swath_iw(self, swath: str) -> np.ndarray:
        """Get azimuth-compressed data for interferometric wide swath mode."""
        # Process each burst separately then combine
        burst_count = self.get_burst_count(swath)
        
        if burst_count == 0:
            return np.array([])
            
        compressed_bursts = []
        for burst_idx in range(burst_count):
            burst_data = self.get_azimuth_compressed_burst(swath, burst_idx)
            if burst_data.size > 0:
                compressed_bursts.append(burst_data)
                
        if compressed_bursts:
            return np.concatenate(compressed_bursts, axis=0)
        else:
            return np.array([])
            
    def _validate_request(self, swath: str, burst: Optional[int] = None) -> None:
        """
        Validate processing request parameters.
        
        Args:
            swath: Swath name
            burst: Burst number (optional)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not self._flat_packets:
            raise ValueError('No packets loaded')
            
        if swath not in self._swath_counts:
            available_swaths = list(self._swath_counts.keys())
            raise ValueError(f'Swath {swath} not available. Available: {available_swaths}')
            
        if burst is not None:
            burst_count = self.get_burst_count(swath)
            if burst < 0 or burst >= burst_count:
                raise ValueError(f'Burst {burst} not valid for swath {swath} (0-{burst_count-1})')
                
    def _estimate_prf(self, swath: str) -> float:
        """
        Estimate pulse repetition frequency for swath.
        
        Args:
            swath: Swath name
            
        Returns:
            PRF in Hz
        """
        if swath in self._echo_packets and self._echo_packets[swath]:
            # Get PRF from first packet
            first_burst = self._echo_packets[swath][0]
            if first_burst:
                pri = first_burst[0].get_pri()  # PRI in microseconds
                if pri > 0:
                    return 1e6 / pri  # Convert to Hz
                    
        # Default PRF for different modes
        instrument_mode = INSTRUMENT_MODES.get(swath, 'SM')
        if instrument_mode == 'SM':
            return 1200.0  # Hz
        elif instrument_mode == 'IW':
            return 486.0   # Hz
        else:
            return 1000.0  # Hz
            
    def _apply_range_doppler_correction(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply range-Doppler correction to compressed signal.
        
        Args:
            signal: Range-compressed signal
            
        Returns:
            Range-Doppler corrected signal
        """
        # Range-Doppler correction compensates for range cell migration
        # This is a simplified implementation
        return signal  # Placeholder
        
    def _estimate_azimuth_fm_rates(self, num_range_bins: int) -> np.ndarray:
        """
        Estimate azimuth FM rates for range bins.
        
        Args:
            num_range_bins: Number of range bins
            
        Returns:
            FM rate estimates for each range bin
        """
        # Simplified FM rate estimation
        # In practice, this would use geometry and orbit information
        fm_rates = np.zeros((num_range_bins, 1))
        
        # Typical FM rate values for Sentinel-1
        base_fm_rate = -2300.0  # Hz/s
        fm_rates[:, 0] = base_fm_rate
        
        return fm_rates
        
    def _get_swath_number(self, swath: str) -> int:
        """
        Get numeric swath number from swath name.
        
        Args:
            swath: Swath name
            
        Returns:
            Swath number
        """
        for num, name in SWATH_NAMES.items():
            if name == swath:
                return num
        return 0
