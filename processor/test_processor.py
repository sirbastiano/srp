#!/usr/bin/env python3
"""
Test script to verify the processor module installation and functionality.

This script runs basic tests to ensure all components are working correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add the processor module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from processor import (
        L0Packet, S1Decoder, StateVectors, StateVector, DopplerEstimator,
        pulse_compression, get_reference_function, generate_chirp,
        fft_1d, ifft_1d, apply_window, bandpass_filter,
        PRIMARY_HEADER, SECONDARY_HEADER, CENTER_FREQ, WAVELENGTH
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_constants():
    """Test that constants are properly defined."""
    print("\n=== Testing Constants ===")
    
    assert len(PRIMARY_HEADER) == 8, "PRIMARY_HEADER should have 8 elements"
    assert len(SECONDARY_HEADER) > 40, "SECONDARY_HEADER should have many elements"
    assert CENTER_FREQ > 5e9, "CENTER_FREQ should be ~5.4 GHz"
    assert WAVELENGTH > 0.05, "WAVELENGTH should be ~5.5 cm"
    
    print("✓ Constants test passed")


def test_packet_creation():
    """Test packet creation and basic functionality."""
    print("\n=== Testing Packet Creation ===")
    
    # Create a packet
    packet = L0Packet(packet_index=0)
    assert packet.packet_index == 0, "Packet index should be 0"
    
    # Test property access
    primary_header = packet.primary_header
    secondary_header = packet.secondary_header
    assert isinstance(primary_header, dict), "Primary header should be dict"
    assert isinstance(secondary_header, dict), "Secondary header should be dict"
    
    print("✓ Packet creation test passed")


def test_state_vectors():
    """Test state vector functionality."""
    print("\n=== Testing State Vectors ===")
    
    # Create state vectors
    state_vectors = StateVectors()
    
    # Add a test state vector
    position = np.array([7000000.0, 0.0, 0.0])  # 7000 km from Earth center
    velocity = np.array([0.0, 7500.0, 0.0])    # 7.5 km/s orbital velocity
    state_vector = StateVector(time=0.0, position=position, velocity=velocity)
    
    state_vectors.add_state_vector(state_vector)
    
    # Test interpolation
    interpolated = state_vectors.interpolate_state_vector(0.0)
    np.testing.assert_allclose(interpolated.position, position, rtol=1e-10)
    np.testing.assert_allclose(interpolated.velocity, velocity, rtol=1e-10)
    
    print("✓ State vectors test passed")


def test_doppler_estimator():
    """Test Doppler estimation functionality."""
    print("\n=== Testing Doppler Estimator ===")
    
    # Create estimator
    prf = 1000.0  # Hz
    estimator = DopplerEstimator(prf)
    
    # Create test signal with known Doppler shift
    num_samples = 1024
    doppler_freq = 100.0  # Hz
    time_vec = np.arange(num_samples) / prf
    
    # Generate signal with Doppler shift
    test_signal = np.exp(2j * np.pi * doppler_freq * time_vec)
    test_signal += 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    
    # Estimate Doppler centroid
    estimated_dc = estimator.estimate_doppler_centroid_fft(test_signal)
    
    # Check if estimate is reasonable (within 10 Hz of true value)
    assert abs(estimated_dc - doppler_freq) < 10.0, f"Doppler estimate error too large: {abs(estimated_dc - doppler_freq)}"
    
    print("✓ Doppler estimator test passed")


def test_signal_processing():
    """Test signal processing functions."""
    print("\n=== Testing Signal Processing ===")
    
    # Test FFT functions
    test_signal = np.random.randn(256) + 1j * np.random.randn(256)
    
    # Forward and inverse FFT should be identity
    fft_result = fft_1d(test_signal)
    ifft_result = ifft_1d(fft_result)
    np.testing.assert_allclose(test_signal, ifft_result, rtol=1e-10)
    
    # Test windowing
    windowed = apply_window(test_signal, 'hamming')
    assert windowed.shape == test_signal.shape, "Windowed signal should have same shape"
    
    print("✓ Signal processing test passed")


def test_image_formation():
    """Test image formation functions."""
    print("\n=== Testing Image Formation ===")
    
    # Test chirp generation
    duration = 10e-6  # 10 microseconds
    bandwidth = 50e6  # 50 MHz
    sample_rate = 100e6  # 100 MHz
    
    chirp = generate_chirp(duration, bandwidth, sample_rate)
    expected_length = int(duration * sample_rate)
    assert len(chirp) == expected_length, f"Chirp length mismatch: {len(chirp)} vs {expected_length}"
    
    # Test reference function generation
    reference = get_reference_function(chirp)
    assert len(reference) == len(chirp), "Reference should have same length as chirp"
    
    # Test pulse compression
    # Create a simple signal with the chirp embedded
    signal_length = 2000
    signal = np.zeros(signal_length, dtype=complex)
    start_idx = 100
    end_idx = start_idx + len(chirp)
    
    # Make sure we don't exceed signal bounds
    if end_idx <= signal_length:
        signal[start_idx:end_idx] = chirp
    else:
        # Truncate chirp if needed
        available_length = signal_length - start_idx
        signal[start_idx:] = chirp[:available_length]
        
    signal += 0.01 * (np.random.randn(signal_length) + 1j * np.random.randn(signal_length))  # Add noise
    
    compressed = pulse_compression(signal, reference)
    assert len(compressed) > 0, "Compressed signal should not be empty"
    
    print("✓ Image formation test passed")


def test_decoder_creation():
    """Test decoder creation (without file)."""
    print("\n=== Testing Decoder Creation ===")
    
    # Create decoder without file
    decoder = S1Decoder()
    
    # Test basic functionality
    swath_names = decoder.get_swath_names()
    assert isinstance(swath_names, list), "Swath names should be a list"
    
    print("✓ Decoder creation test passed")


def run_all_tests():
    """Run all tests."""
    print("Running processor module tests...")
    
    try:
        test_constants()
        test_packet_creation()
        test_state_vectors()
        test_doppler_estimator()
        test_signal_processing()
        test_image_formation()
        test_decoder_creation()
        
        print("\n" + "="*50)
        print("✓ All tests passed successfully!")
        print("The processor module is working correctly.")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
