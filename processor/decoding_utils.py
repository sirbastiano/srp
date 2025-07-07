"""
Decoding utilities for Sentinel-1 Level-0 processing.

This module provides utility functions for bit manipulation, file I/O,
BAQ decoding, and Huffman decoding operations.
"""

import struct
import numpy as np
from typing import BinaryIO, List, Union, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def decode_bits(data: bytes, bit_offset: int, bit_length: int) -> int:
    """
    Extract bits from a byte array at specified offset and length.
    
    Args:
        data: Byte array containing the data
        bit_offset: Starting bit position (0-based)
        bit_length: Number of bits to extract
        
    Returns:
        Extracted value as integer
        
    Raises:
        ValueError: If parameters are invalid
    """
    if bit_length <= 0 or bit_length > 32:
        raise ValueError(f'Invalid bit length: {bit_length}')
        
    if bit_offset < 0:
        raise ValueError(f'Invalid bit offset: {bit_offset}')
        
    # Calculate byte positions
    start_byte = bit_offset // 8
    end_byte = (bit_offset + bit_length - 1) // 8
    
    if end_byte >= len(data):
        raise ValueError('Bit range exceeds data length')
        
    # Extract bytes and convert to integer
    value = 0
    for byte_idx in range(start_byte, end_byte + 1):
        value = (value << 8) | data[byte_idx]
        
    # Calculate shift amount to align the desired bits
    bits_in_last_byte = (bit_offset + bit_length - 1) % 8 + 1
    shift_amount = 8 - bits_in_last_byte
    
    # Shift and mask to get the desired bits
    value >>= shift_amount
    mask = (1 << bit_length) - 1
    value &= mask
    
    return value


def open_file(filename: Union[str, Path]) -> BinaryIO:
    """
    Open a binary file for reading.
    
    Args:
        filename: Path to the file
        
    Returns:
        Open binary file handle
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be opened
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f'File not found: {filename}')
        
    try:
        return open(filename, 'rb')
    except Exception as e:
        raise PermissionError(f'Cannot open file {filename}: {e}')


def linspace(start: float, stop: float, num: int) -> np.ndarray:
    """
    Create a linearly spaced array of values.
    
    Args:
        start: Starting value
        stop: Ending value
        num: Number of values
        
    Returns:
        Numpy array of linearly spaced values
    """
    return np.linspace(start, stop, num)


def huffman_decode(data: bytes, huffman_table: dict) -> List[int]:
    """
    Decode Huffman-encoded data using provided lookup table.
    
    Args:
        data: Huffman-encoded data bytes
        huffman_table: Huffman decoding lookup table
        
    Returns:
        List of decoded values
        
    Note:
        This is a simplified implementation. Real Huffman decoding
        requires more complex bit-by-bit processing.
    """
    decoded_values = []
    bit_buffer = 0
    bit_count = 0
    
    for byte in data:
        bit_buffer = (bit_buffer << 8) | byte
        bit_count += 8
        
        # Try to decode symbols from the bit buffer
        while bit_count > 0:
            # This is a simplified decoding - real implementation
            # would traverse the Huffman tree bit by bit
            found_symbol = False
            
            for code_length in range(1, min(bit_count + 1, 16)):  # Max 16-bit codes
                code = bit_buffer >> (bit_count - code_length)
                code &= (1 << code_length) - 1
                
                if code in huffman_table:
                    decoded_values.append(huffman_table[code])
                    bit_buffer &= (1 << (bit_count - code_length)) - 1
                    bit_count -= code_length
                    found_symbol = True
                    break
                    
            if not found_symbol:
                # No valid symbol found, skip one bit
                bit_count -= 1
                bit_buffer >>= 1
                
    return decoded_values


def baq_decode(data: bytes, baq_mode: str) -> np.ndarray:
    """
    Decode BAQ (Block Adaptive Quantization) compressed data.
    
    Args:
        data: BAQ-compressed data bytes
        baq_mode: BAQ compression mode string
        
    Returns:
        Decoded complex signal array
        
    Raises:
        NotImplementedError: For unsupported BAQ modes
        ValueError: If data format is invalid
    """
    if baq_mode == 'BYPASS':
        return _decode_bypass_baq(data)
    elif 'FBAQ' in baq_mode:
        return _decode_fbaq_baq(data, baq_mode)
    elif 'SMFBAQ' in baq_mode:
        return _decode_smfbaq_baq(data, baq_mode)
    else:
        raise NotImplementedError(f'BAQ mode not implemented: {baq_mode}')


def _decode_bypass_baq(data: bytes) -> np.ndarray:
    """
    Decode bypass mode BAQ data (no compression).
    
    Args:
        data: Raw uncompressed data
        
    Returns:
        Complex signal array
    """
    if not data or len(data) == 0:
        return np.array([], dtype=complex)
        
    # Handle cases where data length is not perfectly divisible by 4
    data_length = len(data)
    usable_length = (data_length // 4) * 4
    
    if usable_length == 0:
        logger.warning(f'Insufficient data for bypass decoding ({data_length} bytes)')
        return np.array([], dtype=complex)
        
    if usable_length != data_length:
        logger.warning(f'Truncating bypass data from {data_length} to {usable_length} bytes')
        data = data[:usable_length]
        
    # Unpack as 16-bit signed integers (I, Q pairs)
    num_samples = usable_length // 4
    try:
        values = struct.unpack(f'>{num_samples * 2}h', data)
    except struct.error as e:
        logger.error(f'Struct unpack error in bypass decoding: {e}')
        return np.array([], dtype=complex)
    
    # Convert to complex array
    i_samples = np.array(values[0::2], dtype=np.float32)
    q_samples = np.array(values[1::2], dtype=np.float32)
    
    return i_samples + 1j * q_samples


def _decode_fbaq_baq(data: bytes, baq_mode: str) -> np.ndarray:
    """
    Decode FBAQ (Flexible Block Adaptive Quantization) data.
    
    Args:
        data: FBAQ-compressed data
        baq_mode: Specific FBAQ mode
        
    Returns:
        Decoded complex signal array
        
    Note:
        This is a simplified implementation. Real FBAQ decoding requires
        complex bit manipulation and lookup tables.
    """
    if not data or len(data) == 0:
        return np.array([], dtype=complex)
        
    # Extract bit depth from mode string
    if '3 BIT' in baq_mode:
        bit_depth = 3
    elif '4 BIT' in baq_mode:
        bit_depth = 4
    elif '5 BIT' in baq_mode:
        bit_depth = 5
    else:
        logger.warning(f'Unknown FBAQ bit depth in mode: {baq_mode}, using default 4-bit')
        bit_depth = 4
    
    # Simplified FBAQ decoding
    # Real implementation would involve:
    # 1. Block header parsing
    # 2. Huffman decoding of quantized values
    # 3. Reconstruction using quantization tables
    # 4. Bit manipulation for sign and magnitude
    
    logger.warning(f'Using simplified FBAQ decoding for mode: {baq_mode}')
    
    # For now, return a placeholder complex array
    # In practice, this would be much more complex
    num_samples = max(1, len(data) // 2)  # Rough estimate, ensure at least 1
    return np.zeros(num_samples, dtype=np.complex64)


def _decode_smfbaq_baq(data: bytes, baq_mode: str) -> np.ndarray:
    """
    Decode SMFBAQ (Spectrally Matched FBAQ) data.
    
    Args:
        data: SMFBAQ-compressed data
        baq_mode: Specific SMFBAQ mode
        
    Returns:
        Decoded complex signal array
        
    Note:
        This is a simplified implementation. Real SMFBAQ decoding requires
        spectral matching and complex reconstruction algorithms.
    """
    # Extract bit depth from mode string
    if '3 BIT' in baq_mode:
        bit_depth = 3
    elif '4 BIT' in baq_mode:
        bit_depth = 4
    elif '5 BIT' in baq_mode:
        bit_depth = 5
    else:
        raise ValueError(f'Unknown SMFBAQ bit depth in mode: {baq_mode}')
    
    # Simplified SMFBAQ decoding
    logger.warning(f'Using simplified SMFBAQ decoding for mode: {baq_mode}')
    
    # For now, return a placeholder complex array
    num_samples = len(data) // 2  # Rough estimate
    return np.zeros(num_samples, dtype=np.complex64)


def sign_magnitude_decode(value: int, bit_width: int) -> int:
    """
    Decode a sign-magnitude encoded value.
    
    Args:
        value: Encoded value
        bit_width: Width in bits
        
    Returns:
        Decoded signed integer
    """
    if bit_width <= 0:
        return 0
        
    sign_bit = 1 << (bit_width - 1)
    magnitude_mask = sign_bit - 1
    
    magnitude = value & magnitude_mask
    is_negative = (value & sign_bit) != 0
    
    return -magnitude if is_negative else magnitude


def twos_complement_decode(value: int, bit_width: int) -> int:
    """
    Decode a two's complement encoded value.
    
    Args:
        value: Encoded value
        bit_width: Width in bits
        
    Returns:
        Decoded signed integer
    """
    if bit_width <= 0:
        return 0
        
    # Check if the sign bit is set
    sign_bit = 1 << (bit_width - 1)
    if value & sign_bit:
        # Negative number - extend sign bits
        return value - (1 << bit_width)
    else:
        # Positive number
        return value


def pack_bits(values: List[int], bit_widths: List[int]) -> bytes:
    """
    Pack multiple values into a byte array with specified bit widths.
    
    Args:
        values: List of values to pack
        bit_widths: List of bit widths for each value
        
    Returns:
        Packed byte array
        
    Raises:
        ValueError: If lengths don't match or values are invalid
    """
    if len(values) != len(bit_widths):
        raise ValueError('Values and bit_widths must have same length')
        
    bit_buffer = 0
    bit_count = 0
    result = bytearray()
    
    for value, width in zip(values, bit_widths):
        if width <= 0 or width > 32:
            raise ValueError(f'Invalid bit width: {width}')
            
        if value < 0 or value >= (1 << width):
            raise ValueError(f'Value {value} exceeds {width} bits')
            
        # Add value to bit buffer
        bit_buffer = (bit_buffer << width) | value
        bit_count += width
        
        # Extract complete bytes
        while bit_count >= 8:
            byte_value = (bit_buffer >> (bit_count - 8)) & 0xFF
            result.append(byte_value)
            bit_count -= 8
            bit_buffer &= (1 << bit_count) - 1
            
    # Handle remaining bits
    if bit_count > 0:
        byte_value = (bit_buffer << (8 - bit_count)) & 0xFF
        result.append(byte_value)
        
    return bytes(result)


def unpack_bits(data: bytes, bit_widths: List[int]) -> List[int]:
    """
    Unpack values from a byte array with specified bit widths.
    
    Args:
        data: Byte array to unpack
        bit_widths: List of bit widths for each value
        
    Returns:
        List of unpacked integer values
        
    Raises:
        ValueError: If bit widths are invalid or data insufficient
    """
    values = []
    bit_offset = 0
    
    for width in bit_widths:
        if width <= 0 or width > 32:
            raise ValueError(f'Invalid bit width: {width}')
            
        try:
            value = decode_bits(data, bit_offset, width)
            values.append(value)
            bit_offset += width
        except ValueError as e:
            raise ValueError(f'Error unpacking at bit offset {bit_offset}: {e}')
            
    return values


def calculate_checksum(data: bytes) -> int:
    """
    Calculate a simple checksum for data integrity verification.
    
    Args:
        data: Data bytes to checksum
        
    Returns:
        Checksum value
    """
    return sum(data) & 0xFFFF


def validate_sync_marker(data: bytes, expected_marker: int = 0x352EF853) -> bool:
    """
    Validate the sync marker in packet data.
    
    Args:
        data: Data containing sync marker
        expected_marker: Expected sync marker value
        
    Returns:
        True if sync marker is valid, False otherwise
    """
    if len(data) < 4:
        return False
        
    # Extract 32-bit sync marker from beginning of data
    marker = struct.unpack('>I', data[:4])[0]
    return marker == expected_marker
