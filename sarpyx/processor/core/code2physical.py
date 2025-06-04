"""
Sentinel-1 Parameter Transformations: From Bytecodes to Physical Values

This module provides comprehensive transformation functions to convert raw bytecode
values from Sentinel-1 data packets into meaningful physical parameters.

All transformations follow the exact specifications from the Sentinel-1 documentation
and match the implementation in the original decoder.
"""

from typing import Dict, Union
import struct


# Reference constants
F_REF = 37.53472224e6  # Reference frequency in Hz (37.53472224 MHz)


# ============================================================================
# PRIMARY HEADER TRANSFORMATIONS (6 bytes total)
# ============================================================================

def extract_packet_version_number(header_bytes: bytes) -> int:
    """
    Extract packet version number from primary header.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Packet version number (3 bits)
    """
    assert len(header_bytes) >= 2, 'Need at least 2 bytes for packet version'
    tmp16 = int.from_bytes(header_bytes[0:2], 'big')
    return tmp16 >> 13


def extract_packet_type(header_bytes: bytes) -> int:
    """
    Extract packet type from primary header.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Packet type (1 bit)
    """
    assert len(header_bytes) >= 2, 'Need at least 2 bytes for packet type'
    tmp16 = int.from_bytes(header_bytes[0:2], 'big')
    return (tmp16 >> 12) & 0x01


def extract_secondary_header_flag(header_bytes: bytes) -> int:
    """
    Extract secondary header flag from primary header.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Secondary header flag (1 bit)
    """
    assert len(header_bytes) >= 2, 'Need at least 2 bytes for secondary header flag'
    tmp16 = int.from_bytes(header_bytes[0:2], 'big')
    return (tmp16 >> 11) & 0x01


def extract_process_id(header_bytes: bytes) -> int:
    """
    Extract process ID (PID) from primary header.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Process ID (7 bits)
    """
    assert len(header_bytes) >= 2, 'Need at least 2 bytes for process ID'
    tmp16 = int.from_bytes(header_bytes[0:2], 'big')
    return (tmp16 >> 4) & 0x7F


def extract_packet_category(header_bytes: bytes) -> int:
    """
    Extract packet category (PCAT) from primary header.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Packet category (4 bits)
    """
    assert len(header_bytes) >= 2, 'Need at least 2 bytes for packet category'
    tmp16 = int.from_bytes(header_bytes[0:2], 'big')
    return tmp16 & 0xF


def extract_sequence_flags(header_bytes: bytes) -> int:
    """
    Extract sequence flags from primary header.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Sequence flags (2 bits)
    """
    assert len(header_bytes) >= 4, 'Need at least 4 bytes for sequence flags'
    tmp16 = int.from_bytes(header_bytes[2:4], 'big')
    return tmp16 >> 14


def extract_packet_sequence_count(header_bytes: bytes) -> int:
    """
    Extract packet sequence count from primary header.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Packet sequence count (14 bits)
    """
    assert len(header_bytes) >= 4, 'Need at least 4 bytes for packet sequence count'
    tmp16 = int.from_bytes(header_bytes[2:4], 'big')
    return tmp16 & 0x3FFF


def extract_packet_data_length(header_bytes: bytes) -> int:
    """
    Extract packet data length from primary header.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Packet data length in bytes (length is stored as N-1)
    """
    assert len(header_bytes) >= 6, 'Need 6 bytes for packet data length'
    tmp16 = int.from_bytes(header_bytes[4:6], 'big')
    return tmp16 + 1


# ============================================================================
# SECONDARY HEADER TRANSFORMATIONS (62 bytes total)
# ============================================================================

# Datation Service (Bytes 0-5)

def extract_coarse_time(header_bytes: bytes) -> int:
    """
    Extract coarse time from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Coarse time (32 bits)
    """
    assert len(header_bytes) >= 4, 'Need at least 4 bytes for coarse time'
    return int.from_bytes(header_bytes[0:4], 'big')


def extract_fine_time(header_bytes: bytes) -> float:
    """
    Extract fine time from secondary header and convert to fractional seconds.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Fine time in fractional seconds
    """
    assert len(header_bytes) >= 6, 'Need at least 6 bytes for fine time'
    raw_fine_time = int.from_bytes(header_bytes[4:6], 'big')
    return (raw_fine_time + 0.5) * (2**-16)


# Fixed Ancillary Data (Bytes 6-19)

def extract_sync_marker(header_bytes: bytes) -> int:
    """
    Extract sync marker from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Sync marker (32 bits, expected: 0x352EF853)
    """
    assert len(header_bytes) >= 10, 'Need at least 10 bytes for sync marker'
    return int.from_bytes(header_bytes[6:10], 'big')


def extract_data_take_id(header_bytes: bytes) -> int:
    """
    Extract data take ID from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Data take ID (32 bits)
    """
    assert len(header_bytes) >= 14, 'Need at least 14 bytes for data take ID'
    return int.from_bytes(header_bytes[10:14], 'big')


def extract_ecc_number(header_bytes: bytes) -> int:
    """
    Extract ECC number from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        ECC number (8 bits)
    """
    assert len(header_bytes) >= 15, 'Need at least 15 bytes for ECC number'
    return header_bytes[14]


def extract_test_mode(header_bytes: bytes) -> int:
    """
    Extract test mode from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Test mode (3 bits)
    """
    assert len(header_bytes) >= 16, 'Need at least 16 bytes for test mode'
    return (header_bytes[15] >> 4) & 0x07


def extract_rx_channel_id(header_bytes: bytes) -> int:
    """
    Extract Rx channel ID from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Rx channel ID (4 bits)
    """
    assert len(header_bytes) >= 16, 'Need at least 16 bytes for Rx channel ID'
    return header_bytes[15] & 0x0F


def extract_instrument_config_id(header_bytes: bytes) -> int:
    """
    Extract instrument configuration ID from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Instrument configuration ID (32 bits)
    """
    assert len(header_bytes) >= 20, 'Need at least 20 bytes for instrument config ID'
    return int.from_bytes(header_bytes[16:20], 'big')


# Sub-commutated Ancillary Data (Bytes 20-22)

def extract_subcom_data_word_index(header_bytes: bytes) -> int:
    """
    Extract sub-commutated data word index from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Sub-commutated data word index (8 bits)
    """
    assert len(header_bytes) >= 21, 'Need at least 21 bytes for subcom data word index'
    return header_bytes[20]


def extract_subcom_data_word(header_bytes: bytes) -> int:
    """
    Extract sub-commutated data word from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Sub-commutated data word (16 bits)
    """
    assert len(header_bytes) >= 23, 'Need at least 23 bytes for subcom data word'
    return int.from_bytes(header_bytes[21:23], 'big')


# Counters Service (Bytes 23-30)

def extract_space_packet_count(header_bytes: bytes) -> int:
    """
    Extract space packet count from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Space packet count (32 bits)
    """
    assert len(header_bytes) >= 27, 'Need at least 27 bytes for space packet count'
    return int.from_bytes(header_bytes[23:27], 'big')


def extract_pri_count(header_bytes: bytes) -> int:
    """
    Extract PRI count from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        PRI count (32 bits)
    """
    assert len(header_bytes) >= 31, 'Need at least 31 bytes for PRI count'
    return int.from_bytes(header_bytes[27:31], 'big')


# Radar Configuration (Bytes 31-57)

def extract_error_flag(header_bytes: bytes) -> int:
    """
    Extract error flag from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Error flag (1 bit)
    """
    assert len(header_bytes) >= 32, 'Need at least 32 bytes for error flag'
    return header_bytes[31] >> 7


def extract_baq_mode(header_bytes: bytes) -> int:
    """
    Extract BAQ mode from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        BAQ mode (5 bits)
    """
    assert len(header_bytes) >= 32, 'Need at least 32 bytes for BAQ mode'
    return header_bytes[31] & 0x1F


def extract_baq_block_length(header_bytes: bytes) -> int:
    """
    Extract BAQ block length from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        BAQ block length (8 bits)
    """
    assert len(header_bytes) >= 33, 'Need at least 33 bytes for BAQ block length'
    return header_bytes[32]


def extract_range_decimation(header_bytes: bytes) -> int:
    """
    Extract range decimation from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Range decimation code (8 bits)
    """
    assert len(header_bytes) >= 35, 'Need at least 35 bytes for range decimation'
    return header_bytes[34]


def range_dec_to_sample_rate(rgdec_code: int) -> float:
    """
    Convert range decimation code to sample rate in Hz.
    
    Args:
        rgdec_code: Range decimation code (0-11)
        
    Returns:
        Sample rate in Hz
        
    Raises:
        ValueError: If rgdec_code is not in lookup table
    """
    lookup_table = {
        0: 3 * F_REF,
        1: (8/3) * F_REF,
        3: (20/9) * F_REF,
        4: (16/9) * F_REF,
        5: (3/2) * F_REF,
        6: (4/3) * F_REF,
        7: (2/3) * F_REF,
        8: (12/7) * F_REF,
        9: (5/4) * F_REF,
        10: (6/13) * F_REF,
        11: (16/11) * F_REF
    }
    
    if rgdec_code not in lookup_table:
        raise ValueError(f'Invalid range decimation code: {rgdec_code}')
    
    return lookup_table[rgdec_code]


def extract_rx_gain(header_bytes: bytes) -> float:
    """
    Extract and convert Rx gain from secondary header to dB.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Rx gain in dB
    """
    assert len(header_bytes) >= 36, 'Need at least 36 bytes for Rx gain'
    raw_gain = header_bytes[35]
    return raw_gain * -0.5  # Scale by -0.5 dB per LSB


def extract_tx_pulse_ramp_rate(header_bytes: bytes) -> float:
    """
    Extract and convert Tx pulse ramp rate (TXPRR) from secondary header.
    
    This is one of the most complex transformations in the Sentinel-1 decoder.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Tx pulse ramp rate in Hz/s
    """
    assert len(header_bytes) >= 38, 'Need at least 38 bytes for TXPRR'
    tmp16 = int.from_bytes(header_bytes[36:38], 'big')
    
    # Extract sign bit (MSB) and magnitude (15 bits)
    txprr_sign = (-1) ** (1 - (tmp16 >> 15))
    magnitude = tmp16 & 0x7FFF
    
    # Apply scaling: F_REF² / 2²¹
    txprr = txprr_sign * magnitude * (F_REF**2) / (2**21)
    
    return txprr


def extract_tx_pulse_start_frequency(header_bytes: bytes) -> float:
    """
    Extract and convert Tx pulse start frequency (TXPSF) from secondary header.
    
    This transformation depends on the TXPRR value and is very complex.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Tx pulse start frequency in Hz
    """
    assert len(header_bytes) >= 40, 'Need at least 40 bytes for TXPSF'
    
    # First get TXPRR (needed for TXPSF calculation)
    txprr = extract_tx_pulse_ramp_rate(header_bytes)
    
    # Extract TXPSF raw value
    tmp16 = int.from_bytes(header_bytes[38:40], 'big')
    
    # Calculate additive component from TXPRR
    txpsf_additive = txprr / (4 * F_REF)
    
    # Extract sign bit and magnitude
    txpsf_sign = (-1) ** (1 - (tmp16 >> 15))
    magnitude = tmp16 & 0x7FFF
    
    # Apply scaling and combine components
    txpsf = txpsf_additive + txpsf_sign * magnitude * F_REF / (2**14)
    
    return txpsf


def extract_tx_pulse_length(header_bytes: bytes) -> float:
    """
    Extract and convert Tx pulse length from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Tx pulse length in seconds
    """
    assert len(header_bytes) >= 43, 'Need at least 43 bytes for Tx pulse length'
    tmp24 = int.from_bytes(header_bytes[40:43], 'big')
    return tmp24 / F_REF


def extract_rank(header_bytes: bytes) -> int:
    """
    Extract rank from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Rank (5 bits)
    """
    assert len(header_bytes) >= 44, 'Need at least 44 bytes for rank'
    return header_bytes[43] & 0x1F


def extract_pri(header_bytes: bytes) -> float:
    """
    Extract and convert PRI (Pulse Repetition Interval) from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        PRI in seconds
    """
    assert len(header_bytes) >= 47, 'Need at least 47 bytes for PRI'
    tmp24 = int.from_bytes(header_bytes[44:47], 'big')
    return tmp24 / F_REF


def extract_sampling_window_start_time(header_bytes: bytes) -> float:
    """
    Extract and convert SWST (Sampling Window Start Time) from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        SWST in seconds
    """
    assert len(header_bytes) >= 50, 'Need at least 50 bytes for SWST'
    tmp24 = int.from_bytes(header_bytes[47:50], 'big')
    return tmp24 / F_REF


def extract_sampling_window_length(header_bytes: bytes) -> float:
    """
    Extract and convert SWL (Sampling Window Length) from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        SWL in seconds
    """
    assert len(header_bytes) >= 53, 'Need at least 53 bytes for SWL'
    tmp24 = int.from_bytes(header_bytes[50:53], 'big')
    return tmp24 / F_REF


def extract_sas_ssb_flag(header_bytes: bytes) -> int:
    """
    Extract SAS SSB flag from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        SAS SSB flag (1 bit)
    """
    assert len(header_bytes) >= 54, 'Need at least 54 bytes for SAS SSB flag'
    return header_bytes[53] >> 7


def extract_polarisation(header_bytes: bytes) -> int:
    """
    Extract polarisation from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Polarisation (3 bits)
    """
    assert len(header_bytes) >= 54, 'Need at least 54 bytes for polarisation'
    return (header_bytes[53] >> 4) & 0x07


def extract_temperature_compensation(header_bytes: bytes) -> int:
    """
    Extract temperature compensation from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Temperature compensation (2 bits)
    """
    assert len(header_bytes) >= 54, 'Need at least 54 bytes for temperature compensation'
    return (header_bytes[53] >> 2) & 0x03


def extract_calibration_mode(header_bytes: bytes) -> int:
    """
    Extract calibration mode from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Calibration mode (2 bits)
    """
    assert len(header_bytes) >= 57, 'Need at least 57 bytes for calibration mode'
    return header_bytes[56] >> 6


def extract_tx_pulse_number(header_bytes: bytes) -> int:
    """
    Extract Tx pulse number from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Tx pulse number (5 bits)
    """
    assert len(header_bytes) >= 57, 'Need at least 57 bytes for Tx pulse number'
    return header_bytes[56] & 0x1F


def extract_signal_type(header_bytes: bytes) -> int:
    """
    Extract signal type from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Signal type (4 bits)
    """
    assert len(header_bytes) >= 58, 'Need at least 58 bytes for signal type'
    return header_bytes[57] >> 4


def extract_swap_flag(header_bytes: bytes) -> int:
    """
    Extract swap flag from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Swap flag (1 bit)
    """
    assert len(header_bytes) >= 58, 'Need at least 58 bytes for swap flag'
    return header_bytes[57] & 0x01


def extract_swath_number(header_bytes: bytes) -> int:
    """
    Extract swath number from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Swath number (8 bits)
    """
    assert len(header_bytes) >= 59, 'Need at least 59 bytes for swath number'
    return header_bytes[58]


def extract_number_of_quads(header_bytes: bytes) -> int:
    """
    Extract number of quads from secondary header.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Number of quads (16 bits)
    """
    assert len(header_bytes) >= 61, 'Need at least 61 bytes for number of quads'
    return int.from_bytes(header_bytes[59:61], 'big')


# ============================================================================
# USER DATA TRANSFORMATIONS
# ============================================================================

def ten_bit_unsigned_to_signed_int(ten_bit: int) -> int:
    """
    Convert 10-bit unsigned to signed integer using two's complement.
    
    This function is used for bypass mode (BAQ mode 0) sample conversion.
    
    Args:
        ten_bit: 10-bit unsigned integer value (0-1023)
        
    Returns:
        Signed integer value (-511 to +511)
    """
    assert 0 <= ten_bit <= 1023, f'10-bit value must be 0-1023, got {ten_bit}'
    
    # Extract sign bit (MSB) and apply two's complement
    sign = int((-1) ** ((ten_bit >> 9) & 0x1))
    magnitude = ten_bit & 0x1FF  # 9-bit magnitude
    
    return sign * magnitude


def extract_bypass_samples(user_data: bytes, num_samples: int) -> list[int]:
    """
    Extract and convert 10-bit bypass mode samples from user data.
    
    In bypass mode (BAQ mode 0), samples are 10-bit signed integers
    packed into bytes. Every 5 samples occupy 8 bytes.
    
    Args:
        user_data: Raw user data bytes
        num_samples: Number of samples to extract
        
    Returns:
        List of signed sample values
    """
    samples = []
    byte_idx = 0
    
    for sample_idx in range(num_samples):
        # Calculate bit position for this sample
        bit_offset = (sample_idx * 10) % 8
        
        if bit_offset == 0:
            # Sample starts at byte boundary
            ten_bit = ((user_data[byte_idx] << 2) | 
                      (user_data[byte_idx + 1] >> 6)) & 0x3FF
            byte_idx += 1
        elif bit_offset == 2:
            # Sample starts 2 bits into byte
            ten_bit = (((user_data[byte_idx] & 0x3F) << 4) | 
                      (user_data[byte_idx + 1] >> 4)) & 0x3FF
            byte_idx += 1
        elif bit_offset == 4:
            # Sample starts 4 bits into byte
            ten_bit = (((user_data[byte_idx] & 0x0F) << 6) | 
                      (user_data[byte_idx + 1] >> 2)) & 0x3FF
            byte_idx += 1
        elif bit_offset == 6:
            # Sample starts 6 bits into byte
            ten_bit = (((user_data[byte_idx] & 0x03) << 8) | 
                      user_data[byte_idx + 1]) & 0x3FF
            byte_idx += 2
        else:
            # This should never happen with 10-bit samples (bit_offset can only be 0,2,4,6)
            raise ValueError(f'Unexpected bit offset: {bit_offset}')
        
        # Convert to signed and add to results
        signed_sample = ten_bit_unsigned_to_signed_int(ten_bit)
        samples.append(signed_sample)
    
    return samples


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def decode_all_primary_header_parameters(header_bytes: bytes) -> Dict[str, Union[int, float]]:
    """
    Decode all parameters from primary header in one call.
    
    Args:
        header_bytes: Primary header bytes (6 bytes)
        
    Returns:
        Dictionary with all primary header parameters
    """
    assert len(header_bytes) >= 6, 'Primary header must be 6 bytes'
    
    return {
        'packet_version_number': extract_packet_version_number(header_bytes),
        'packet_type': extract_packet_type(header_bytes),
        'secondary_header_flag': extract_secondary_header_flag(header_bytes),
        'process_id': extract_process_id(header_bytes),
        'packet_category': extract_packet_category(header_bytes),
        'sequence_flags': extract_sequence_flags(header_bytes),
        'packet_sequence_count': extract_packet_sequence_count(header_bytes),
        'packet_data_length': extract_packet_data_length(header_bytes),
    }


def decode_all_secondary_header_parameters(header_bytes: bytes) -> Dict[str, Union[int, float]]:
    """
    Decode all parameters from secondary header in one call.
    
    Args:
        header_bytes: Secondary header bytes (62 bytes)
        
    Returns:
        Dictionary with all secondary header parameters
    """
    assert len(header_bytes) >= 62, 'Secondary header must be 62 bytes'
    
    return {
        # Datation Service
        'coarse_time': extract_coarse_time(header_bytes),
        'fine_time': extract_fine_time(header_bytes),
        
        # Fixed Ancillary Data
        'sync_marker': extract_sync_marker(header_bytes),
        'data_take_id': extract_data_take_id(header_bytes),
        'ecc_number': extract_ecc_number(header_bytes),
        'test_mode': extract_test_mode(header_bytes),
        'rx_channel_id': extract_rx_channel_id(header_bytes),
        'instrument_config_id': extract_instrument_config_id(header_bytes),
        
        # Sub-commutated Ancillary Data
        'subcom_data_word_index': extract_subcom_data_word_index(header_bytes),
        'subcom_data_word': extract_subcom_data_word(header_bytes),
        
        # Counters Service
        'space_packet_count': extract_space_packet_count(header_bytes),
        'pri_count': extract_pri_count(header_bytes),
        
        # Radar Configuration
        'error_flag': extract_error_flag(header_bytes),
        'baq_mode': extract_baq_mode(header_bytes),
        'baq_block_length': extract_baq_block_length(header_bytes),
        'range_decimation': extract_range_decimation(header_bytes),
        'rx_gain_db': extract_rx_gain(header_bytes),
        'tx_pulse_ramp_rate_hz_per_s': extract_tx_pulse_ramp_rate(header_bytes),
        'tx_pulse_start_frequency_hz': extract_tx_pulse_start_frequency(header_bytes),
        'tx_pulse_length_s': extract_tx_pulse_length(header_bytes),
        'rank': extract_rank(header_bytes),
        'pri_s': extract_pri(header_bytes),
        'sampling_window_start_time_s': extract_sampling_window_start_time(header_bytes),
        'sampling_window_length_s': extract_sampling_window_length(header_bytes),
        'sas_ssb_flag': extract_sas_ssb_flag(header_bytes),
        'polarisation': extract_polarisation(header_bytes),
        'temperature_compensation': extract_temperature_compensation(header_bytes),
        'calibration_mode': extract_calibration_mode(header_bytes),
        'tx_pulse_number': extract_tx_pulse_number(header_bytes),
        'signal_type': extract_signal_type(header_bytes),
        'swap_flag': extract_swap_flag(header_bytes),
        'swath_number': extract_swath_number(header_bytes),
        'number_of_quads': extract_number_of_quads(header_bytes),
    }


def decode_complete_packet_header(primary_header: bytes, secondary_header: bytes) -> Dict[str, Union[int, float]]:
    """
    Decode all parameters from both primary and secondary headers.
    
    Args:
        primary_header: Primary header bytes (6 bytes)
        secondary_header: Secondary header bytes (62 bytes)
        
    Returns:
        Dictionary with all header parameters
    """
    result = {}
    result.update(decode_all_primary_header_parameters(primary_header))
    result.update(decode_all_secondary_header_parameters(secondary_header))
    
    # Add derived parameters
    result['sample_rate_hz'] = range_dec_to_sample_rate(result['range_decimation'])
    
    return result


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_sync_marker(sync_marker: int) -> bool:
    """
    Validate that the sync marker has the expected value.
    
    Args:
        sync_marker: Extracted sync marker value
        
    Returns:
        True if sync marker is valid (0x352EF853)
    """
    return sync_marker == 0x352EF853


def validate_packet_version(version: int) -> bool:
    """
    Validate packet version number.
    
    Args:
        version: Packet version number
        
    Returns:
        True if version is valid (typically 0)
    """
    return version == 0


def validate_baq_mode(baq_mode: int) -> bool:
    """
    Validate BAQ mode value.
    
    Args:
        baq_mode: BAQ mode value
        
    Returns:
        True if BAQ mode is valid
    """
    valid_modes = {0, 3, 4, 5, 12, 13, 14}
    return baq_mode in valid_modes
