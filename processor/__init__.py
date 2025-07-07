"""
SarPyx Processor Module

A Python refactoring of the sentinel1_decode C++ library for processing
Sentinel-1 Level-0 SAR data.

This module provides classes and functions for:
- Decoding Sentinel-1 Level-0 packets
- Signal processing (range and azimuth compression)
- Image formation
- State vector processing
- Doppler estimation

Author: Refactored from C++ implementation by Andrew Player
"""

from .constants import (
    PRIMARY_HEADER, SECONDARY_HEADER, CENTER_FREQ, WAVELENGTH, SPEED_OF_LIGHT,
    F_REF, PI, SWATH_NAMES, BAQ_MODES, SIGNAL_TYPE, POLARISATION
)
from .packet import L0Packet, HCode, Quad
from .decoder import S1Decoder
from .signal_processing import (
    fft_1d, ifft_1d, apply_window, bandpass_filter, lowpass_filter,
    cross_correlate, hilbert_transform, envelope_detection
)
from .image_formation import (
    pulse_compression, get_reference_function, generate_chirp,
    azimuth_frequency_ufr, azimuth_time_ufr, multilook_processing
)
from .state_vectors import StateVectors, StateVector
from .doppler import DopplerEstimator

__version__ = "1.0.0"
__author__ = "SarPyx Team"

__all__ = [
    # Core classes
    'L0Packet',
    'S1Decoder', 
    'StateVectors',
    'StateVector',
    'DopplerEstimator',
    'HCode',
    'Quad',
    
    # Image formation functions
    'pulse_compression',
    'get_reference_function',
    'generate_chirp',
    'azimuth_frequency_ufr',
    'azimuth_time_ufr',
    'multilook_processing',
    
    # Signal processing functions
    'fft_1d',
    'ifft_1d',
    'apply_window',
    'bandpass_filter',
    'lowpass_filter',
    'cross_correlate',
    'hilbert_transform',
    'envelope_detection',
    
    # Constants
    'PRIMARY_HEADER',
    'SECONDARY_HEADER',
    'CENTER_FREQ',
    'WAVELENGTH',
    'SPEED_OF_LIGHT',
    'F_REF',
    'PI',
    'SWATH_NAMES',
    'BAQ_MODES',
    'SIGNAL_TYPE',
    'POLARISATION',
]
