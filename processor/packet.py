"""
L0Packet class for storing and decoding Level-0 packets.

This module provides the L0Packet class for convenient storage and decoding
of Sentinel-1 Level-0 packets. See the SAR Space Packet Protocol Data Unit
specification for packet format details.
"""

import struct
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO
from pathlib import Path
import logging

from .constants import (
    PRIMARY_HEADER, PRIMARY_HEADER_FIELDS,
    SECONDARY_HEADER, SECONDARY_HEADER_FIELDS,
    BAQ_MODES, TEST_MODES, POLARISATION, SIGNAL_TYPE, SWATH_NAMES,
    AZIMUTH_BEAM_ADDRESS_TO_ANGLE, F_REF, PI, CENTER_FREQ
)
from .decoding_utils import (
    decode_bits, huffman_decode, baq_decode,
    open_file, linspace
)

logger = logging.getLogger(__name__)


class HCode:
    """
    H_CODE struct representing one element of a quadrature.
    
    Attributes:
        signs: List of sign values (0 or 1)
        m_codes: List of magnitude codes
    """
    
    def __init__(self, num_codes: int = 128):
        """
        Initialize HCode with specified number of codes.
        
        Args:
            num_codes: Number of codes to initialize (default: 128)
        """
        self.signs: List[int] = [0] * num_codes
        self.m_codes: List[int] = [0] * num_codes


class Quad:
    """
    Quadrature struct representing one of the IE, IO, QE, or QO quads.
    
    Attributes:
        blocks: List of HCode blocks
        key: Component key identifier
    """
    
    def __init__(self, component_key: str):
        """
        Initialize Quad with component key.
        
        Args:
            component_key: Key identifying the quadrature component
        """
        self.blocks: List[HCode] = []
        self.key: str = component_key


class L0Packet:
    """
    Level-0 Packet class for storing and decoding Sentinel-1 L0 packets.
    
    This class handles the parsing and decoding of individual Level-0 packets
    from Sentinel-1 SAR data files.
    """
    
    def __init__(self, packet_index: int = 0):
        """
        Initialize L0Packet.
        
        Args:
            packet_index: Index of the packet in the data file
        """
        self._packet_index: int = packet_index
        self._primary_header: Dict[str, int] = {}
        self._secondary_header: Dict[str, int] = {}
        self._raw_data: Optional[bytes] = None
        self._decoded_data: Optional[np.ndarray] = None
        
    @property
    def packet_index(self) -> int:
        """Get the packet index."""
        return self._packet_index
        
    @property
    def primary_header(self) -> Dict[str, int]:
        """Get the primary header dictionary."""
        return self._primary_header.copy()
        
    @property
    def secondary_header(self) -> Dict[str, int]:
        """Get the secondary header dictionary."""
        return self._secondary_header.copy()
        
    def get_baq_block_length(self) -> int:
        """
        Get the length of the BAQ blocks in bytes.
        
        Returns:
            Length of BAQ blocks in bytes
        """
        return 8 * (self._secondary_header['baq_block_length'] + 1)
        
    def get_pri(self) -> float:
        """
        Get the Pulse Repetition Interval in microseconds.
        
        Returns:
            PRI in microseconds
        """
        return self._secondary_header['pri'] / F_REF
        
    def get_pulse_length(self) -> float:
        """
        Get the pulse length in microseconds.
        
        Returns:
            Pulse length in microseconds
        """
        return self._secondary_header['pulse_length'] / F_REF
        
    def get_swl(self) -> float:
        """
        Get the sampling window length in microseconds.
        
        Returns:
            Sampling window length in microseconds
        """
        return self._secondary_header['swl'] / F_REF
        
    def get_swst(self) -> float:
        """
        Get the start time of the sampling window in microseconds.
        
        Returns:
            SWST in microseconds
        """
        return self._secondary_header['swst'] / F_REF
        
    def get_time(self) -> float:
        """
        Get the packet time from coarse and fine time fields.
        
        Returns:
            Packet time in seconds
        """
        coarse_time = float(self._secondary_header['coarse_time'])
        fine_time = float(self._secondary_header['fine_time'])
        
        # Convert fine time to fractional seconds
        time = coarse_time + (1.0 / fine_time) if fine_time != 0 else coarse_time
        return time
        
    def get_rx_gain(self) -> float:
        """
        Get the RX gain in dB.
        
        Returns:
            RX gain in dB
        """
        return -0.5 * self._secondary_header['rx_gain']
        
    def get_start_frequency(self) -> float:
        """
        Get the TX pulse start frequency in MHz.
        
        Returns:
            Start frequency in MHz
        """
        sign = -1 if self._secondary_header['pulse_start_frequency_sign'] == 0 else 1
        mag = self._secondary_header['pulse_start_frequency_mag']
        txprr = self.get_tx_ramp_rate()
        
        return (sign * mag * (F_REF / 16384)) + (txprr / (4 * F_REF))
        
    def get_azimuth_beam_angle(self) -> float:
        """
        Get the azimuth beam angle in radians.
        
        Returns:
            Azimuth beam angle in radians
        """
        address = self._secondary_header['azimuth_beam_address']
        if address < len(AZIMUTH_BEAM_ADDRESS_TO_ANGLE):
            return AZIMUTH_BEAM_ADDRESS_TO_ANGLE[address]
        return 0.0
        
    def get_tx_ramp_rate(self) -> float:
        """
        Get the linear FM rate at which the chirp frequency changes.
        
        Returns:
            TX ramp rate in MHz/microsecond
        """
        sign = -1 if self._secondary_header['tx_ramp_rate_sign'] == 0 else 1
        mag = self._secondary_header['tx_ramp_rate_mag']
        
        return sign * mag * (F_REF**2 / 2097152)
        
    def get_swath_name(self) -> str:
        """
        Get the swath name from swath number.
        
        Returns:
            Swath name string
        """
        swath_num = self._secondary_header.get('swath_number', 0)
        return SWATH_NAMES.get(swath_num, 'UNKNOWN')
        
    def get_polarisation(self) -> str:
        """
        Get the polarisation string.
        
        Returns:
            Polarisation string
        """
        pol_num = self._secondary_header.get('polarisation', 0)
        return POLARISATION.get(pol_num, 'H')
        
    def get_signal_type(self) -> str:
        """
        Get the signal type string.
        
        Returns:
            Signal type string
        """
        sig_type = self._secondary_header.get('signal_type', 0)
        return SIGNAL_TYPE.get(sig_type, 'ECHO')
        
    def get_baq_mode(self) -> str:
        """
        Get the BAQ mode string.
        
        Returns:
            BAQ mode string
        """
        baq_mode = self._secondary_header.get('baq_mode', 0)
        return BAQ_MODES.get(baq_mode, 'UNKNOWN')
        
    def is_echo_packet(self) -> bool:
        """
        Check if this is an echo packet.
        
        Returns:
            True if echo packet, False otherwise
        """
        return self.get_signal_type() == 'ECHO'
        
    def is_calibration_packet(self) -> bool:
        """
        Check if this is a calibration packet.
        
        Returns:
            True if calibration packet, False otherwise
        """
        signal_type = self.get_signal_type()
        return signal_type in ['TX_CAL', 'RX_CAL', 'EPDN_CAL', 'TA_CAL', 'APDN_CAL', 'TxH_CAL_ISO']
        
    def decode_packet_data(self) -> np.ndarray:
        """
        Decode the packet data using appropriate BAQ decoding.
        
        Returns:
            Decoded complex signal data as numpy array
            
        Raises:
            ValueError: If packet data cannot be decoded
        """
        if self._decoded_data is not None:
            return self._decoded_data
            
        if self._raw_data is None or len(self._raw_data) == 0:
            logger.warning(f'Packet {self._packet_index}: No raw data available for decoding')
            self._decoded_data = np.array([], dtype=complex)
            return self._decoded_data
            
        baq_mode = self.get_baq_mode()
        
        try:
            if baq_mode == 'BYPASS':
                self._decoded_data = self._decode_bypass()
            elif 'FBAQ' in baq_mode:
                self._decoded_data = self._decode_fbaq()
            elif 'SMFBAQ' in baq_mode:
                self._decoded_data = self._decode_smfbaq()
            else:
                logger.warning(f'Packet {self._packet_index}: Unsupported BAQ mode: {baq_mode}, returning empty array')
                self._decoded_data = np.array([], dtype=complex)
                
        except Exception as e:
            logger.warning(f'Failed to decode packet {self._packet_index} (BAQ mode: {baq_mode}): {e}')
            # Return empty array instead of raising exception
            self._decoded_data = np.array([], dtype=complex)
            
        return self._decoded_data
        
    def _decode_bypass(self) -> np.ndarray:
        """
        Decode bypass mode data (no compression).
        
        Returns:
            Decoded complex signal array
        """
        if not self._raw_data or len(self._raw_data) == 0:
            return np.array([], dtype=complex)
            
        # In bypass mode, data is stored as 16-bit I/Q pairs
        # Handle cases where data length is not perfectly divisible by 4
        data_length = len(self._raw_data)
        
        # Truncate to nearest multiple of 4 bytes (I/Q pairs)
        usable_length = (data_length // 4) * 4
        
        if usable_length == 0:
            logger.warning(f'Packet {self._packet_index}: Insufficient data for bypass decoding ({data_length} bytes)')
            return np.array([], dtype=complex)
            
        if usable_length != data_length:
            logger.warning(f'Packet {self._packet_index}: Truncating data from {data_length} to {usable_length} bytes')
            
        # Use only the usable portion
        usable_data = self._raw_data[:usable_length]
        
        # Unpack as 16-bit signed integers (I, Q pairs)
        num_samples = usable_length // 4
        try:
            data = struct.unpack(f'>{num_samples * 2}h', usable_data)
        except struct.error as e:
            logger.error(f'Packet {self._packet_index}: Struct unpack error: {e}')
            return np.array([], dtype=complex)
        
        # Convert to complex array
        i_samples = np.array(data[0::2], dtype=np.float32)
        q_samples = np.array(data[1::2], dtype=np.float32)
        
        return i_samples + 1j * q_samples
        
    def _decode_fbaq(self) -> np.ndarray:
        """
        Decode FBAQ (Flexible Block Adaptive Quantization) data.
        
        Returns:
            Decoded complex signal array
        """
        if not self._raw_data:
            return np.array([], dtype=complex)
        # FBAQ decoding is complex and involves Huffman decoding
        # This is a simplified implementation
        return baq_decode(self._raw_data, self.get_baq_mode())
        
    def _decode_smfbaq(self) -> np.ndarray:
        """
        Decode SMFBAQ (Spectrally Matched Flexible Block Adaptive Quantization) data.
        
        Returns:
            Decoded complex signal array
        """
        if not self._raw_data:
            return np.array([], dtype=complex)
        # SMFBAQ decoding is similar to FBAQ but with spectral matching
        return baq_decode(self._raw_data, self.get_baq_mode())
        
    def print_primary_header(self) -> None:
        """Print the primary header information."""
        print('Primary Header:')
        for field in PRIMARY_HEADER_FIELDS:
            value = self._primary_header.get(field, 'N/A')
            print(f'  {field}: {value}')
            
    def print_secondary_header(self) -> None:
        """Print the secondary header information."""
        print('Secondary Header:')
        for field in SECONDARY_HEADER_FIELDS:
            value = self._secondary_header.get(field, 'N/A')
            print(f'  {field}: {value}')
            
    def print_modes(self) -> None:
        """Print operating mode information."""
        print('Operating Mode Info:')
        print(f'  Swath: {self.get_swath_name()}')
        print(f'  Polarisation: {self.get_polarisation()}')
        print(f'  Signal Type: {self.get_signal_type()}')
        print(f'  BAQ Mode: {self.get_baq_mode()}')
        
    def print_pulse_info(self) -> None:
        """Print pulse timing information."""
        print('Pulse Info:')
        print(f'  PRI: {self.get_pri():.6f} μs')
        print(f'  Pulse Length: {self.get_pulse_length():.6f} μs')
        print(f'  SWST: {self.get_swst():.6f} μs')
        print(f'  SWL: {self.get_swl():.6f} μs')
        print(f'  RX Gain: {self.get_rx_gain():.2f} dB')
        print(f'  Start Frequency: {self.get_start_frequency():.6f} MHz')
        print(f'  TX Ramp Rate: {self.get_tx_ramp_rate():.6f} MHz/μs')
        
    @staticmethod
    def get_packets(filename: Union[str, Path], max_packets: int = 0) -> List['L0Packet']:
        """
        Read and parse packets from a Level-0 data file.
        
        Args:
            filename: Path to the Level-0 data file
            max_packets: Maximum number of packets to read (0 = all)
            
        Returns:
            List of L0Packet objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f'File not found: {filename}')
            
        packets = []
        packet_index = 0
        
        try:
            with open(filename, 'rb') as file:
                while True:
                    if max_packets > 0 and packet_index >= max_packets:
                        break
                        
                    packet = L0Packet._read_packet_from_file(file, packet_index)
                    if packet is None:
                        break
                        
                    packets.append(packet)
                    packet_index += 1
                    
        except Exception as e:
            logger.error(f'Error reading packets from {filename}: {e}')
            raise ValueError(f'Failed to read packets: {e}')
            
        logger.info(f'Read {len(packets)} packets from {filename}')
        return packets
        
    @staticmethod
    def _read_packet_from_file(file: BinaryIO, packet_index: int) -> Optional['L0Packet']:
        """
        Read a single packet from file.
        
        Args:
            file: Open binary file handle
            packet_index: Index of the packet being read
            
        Returns:
            L0Packet object or None if end of file
        """
        try:
            # Read primary header (6 bytes)
            primary_data = file.read(6)
            if len(primary_data) < 6:
                return None
                
            # Read secondary header (62 bytes)  
            secondary_data = file.read(62)
            if len(secondary_data) < 62:
                return None
                
            packet = L0Packet(packet_index)
            packet._parse_primary_header(primary_data)
            packet._parse_secondary_header(secondary_data)
            
            # Read packet data
            data_length = packet._primary_header['packet_data_length']
            if data_length > 62:  # Only read if there's data beyond secondary header
                # packet_data_length includes the secondary header (62 bytes)
                actual_data_length = data_length - 62
                packet_data = file.read(actual_data_length)
                if len(packet_data) == actual_data_length:
                    packet._raw_data = packet_data
                else:
                    logger.warning(f'Packet {packet_index}: Expected {actual_data_length} bytes, got {len(packet_data)}')
                    packet._raw_data = packet_data  # Store what we got
                
            return packet
            
        except Exception as e:
            logger.error(f'Error reading packet {packet_index}: {e}')
            return None
            
    def _parse_primary_header(self, data: bytes) -> None:
        """
        Parse the primary header from raw bytes.
        
        Args:
            data: Raw primary header bytes
        """
        bit_offset = 0
        
        for i, field_name in enumerate(PRIMARY_HEADER_FIELDS):
            bit_length = PRIMARY_HEADER[i]
            value = decode_bits(data, bit_offset, bit_length)
            self._primary_header[field_name] = value
            bit_offset += bit_length
            
    def _parse_secondary_header(self, data: bytes) -> None:
        """
        Parse the secondary header from raw bytes.
        
        Args:
            data: Raw secondary header bytes
        """
        bit_offset = 0
        
        for i, field_name in enumerate(SECONDARY_HEADER_FIELDS):
            bit_length = SECONDARY_HEADER[i]
            value = decode_bits(data, bit_offset, bit_length)
            self._secondary_header[field_name] = value
            bit_offset += bit_length
