"""
Constants and lookup tables for Sentinel-1 Level-0 processing.

This module contains all the constants, lookup tables, and data structures
necessary for decoding Level-0 Products. The data is based on:
- "SAR Space Packet Protocol Data Unit" specification
- "Sentinel-1-Level-0-Data-Decoding-Package" documentation

References:
- https://sentinels.copernicus.eu/documents/247904/2142675/Sentinel-1-SAR-Space-Packet-Protocol-Data-Unit.pdf
- https://sentinel.esa.int/documents/247904/0/Sentinel-1-Level-0-Data-Decoding-Package.pdf
"""

import numpy as np
from typing import List, Dict, Any

# Mathematical constants
PI = np.pi
SPEED_OF_LIGHT = 299792458.0  # m/s

# Sentinel-1 specific constants
CENTER_FREQ = 5.405000454334350e+9  # Hz
WAVELENGTH = SPEED_OF_LIGHT / CENTER_FREQ  # m
F_REF = 37.53472224  # MHz
DELTA_T_SUPPRESSED = (320 / (8 * F_REF)) * 1e-6  # seconds

# Header sizes
PRIMARY_HEADER_SIZE = 6  # bytes
SECONDARY_HEADER_SIZE = 62  # bytes
WORD_SIZE = 16  # bits

# Table 2.4-1 from Page 13 - Primary Header bit allocations
PRIMARY_HEADER: List[int] = [
    3,   # Packet Version Number
    1,   # Packet Type
    1,   # Secondary Header Flag
    7,   # Process ID
    4,   # Process Category
    2,   # Sequence Flags
    14,  # Packet Sequence Count 
    16   # Packet Data Length
]

# Primary Header field names
PRIMARY_HEADER_FIELDS: List[str] = [
    'packet_version_number',
    'packet_type',
    'secondary_header_flag',
    'process_id',
    'process_category',
    'sequence_flags',
    'packet_sequence_count',
    'packet_data_length'
]

# Tables 3.2-1 -> 3.2-19 from Pages 15 -> 54 - Secondary Header bit allocations
SECONDARY_HEADER: List[int] = [
    32,  # Coarse Time
    16,  # Fine Time
    32,  # Sync Marker
    32,  # Data Take ID
    8,   # ECC Number
    1,   # N/A
    3,   # Test Mode
    4,   # RX Channel ID
    32,  # Instrument Configuration ID
    8,   # Sub-Commutative Ancillary Data Word Index
    16,  # Sub-Commutative Ancillary Data Word
    32,  # Counter Service
    32,  # PRI Count
    1,   # Error Flag
    2,   # N/A
    5,   # BAQ Mode
    8,   # BAQ Block Length
    8,   # N/A
    8,   # Range Decimation
    8,   # RX Gain
    1,   # TX Ramp Rate Sign
    15,  # TX Ramp Rate Magnitude
    1,   # Pulse Start Frequency Sign
    15,  # Pulse Start Frequency Magnitude
    24,  # Pulse Length
    3,   # N/A
    5,   # Rank
    24,  # PRI
    24,  # SWST
    24,  # SWL
    1,   # SAS SSB MESSAGE > SSB Flag
    3,   # SAS SSB MESSAGE > Polarisation
    2,   # SAS SSB MESSAGE > Temperature Compensation
    2,   # SAS SSB MESSAGE > N/A
    4,   # SAS SSB MESSAGE > Elevation Beam Address
    2,   # SAS SSB MESSAGE > N/A
    10,  # SAS SSB MESSAGE > Azimuth Beam Address
    16,  # SAS SSB MESSAGE > SAS Test
    4,   # N/A
    1,   # Signal Type
    3,   # N/A
    5,   # Swap flag
    8,   # Swath Number
    1    # N/A
]

# Secondary header field names
SECONDARY_HEADER_FIELDS: List[str] = [
    'coarse_time',
    'fine_time',
    'sync_marker',
    'data_take_id',
    'ecc_number',
    'na_1',
    'test_mode',
    'rx_channel_id',
    'instrument_configuration_id',
    'sub_commutative_ancillary_data_word_index',
    'sub_commutative_ancillary_data_word',
    'counter_service',
    'pri_count',
    'error_flag',
    'na_2',
    'baq_mode',
    'baq_block_length',
    'na_3',
    'range_decimation',
    'rx_gain',
    'tx_ramp_rate_sign',
    'tx_ramp_rate_mag',
    'pulse_start_frequency_sign',
    'pulse_start_frequency_mag',
    'pulse_length',
    'na_4',
    'rank',
    'pri',
    'swst',
    'swl',
    'ssb_flag',
    'polarisation',
    'temperature_compensation',
    'na_5',
    'elevation_beam_address',
    'na_6',
    'azimuth_beam_address',
    'sas_test',
    'na_7',
    'signal_type',
    'na_8',
    'swap_flag',
    'swath_number',
    'na_9'
]

# BAQ modes lookup table
BAQ_MODES: Dict[int, str] = {
    0: 'BYPASS',
    3: 'FBAQ 3 BIT',
    4: 'FBAQ 4 BIT',
    5: 'FBAQ 5 BIT',
    12: 'SMFBAQ 3 BIT',
    13: 'SMFBAQ 4 BIT',
    14: 'SMFBAQ 5 BIT'
}

# Test modes lookup table
TEST_MODES: Dict[int, str] = {
    0: 'DEFAULT',
    1: 'CONTINGENCY',
    2: 'SPARE',
    3: 'SPARE',
    4: 'SPARE',
    5: 'SPARE',
    6: 'SPARE',
    7: 'SPARE'
}

# Polarisation lookup table
POLARISATION: Dict[int, str] = {
    0: 'H',
    1: 'V',
    2: 'RH',
    3: 'RV'
}

# Signal type lookup table
SIGNAL_TYPE: Dict[int, str] = {
    0: 'ECHO',
    1: 'NOISE',
    8: 'TX_CAL',
    9: 'RX_CAL',
    10: 'EPDN_CAL',
    11: 'TA_CAL',
    12: 'APDN_CAL',
    15: 'TxH_CAL_ISO'
}

# Swath lookup table
SWATH_NAMES: Dict[int, str] = {
    0: 'S1',
    1: 'S2', 
    2: 'S3',
    3: 'S4',
    4: 'S5',
    5: 'S6',
    6: 'IW1',
    7: 'IW2',
    8: 'IW3',
    9: 'EW1',
    10: 'EW2',
    11: 'EW3',
    12: 'EW4',
    13: 'EW5',
    14: 'WV1',
    15: 'WV2',
    16: 'GP',
    17: 'SPARE',
    18: 'SPARE',
    19: 'SPARE',
    20: 'SPARE',
    21: 'SPARE',
    22: 'SPARE',
    23: 'SPARE',
    24: 'SPARE',
    25: 'SPARE',
    26: 'SPARE',
    27: 'SPARE',
    28: 'SPARE',
    29: 'SPARE',
    30: 'SPARE',
    31: 'EN'
}

# Instrument mode lookup
INSTRUMENT_MODES: Dict[str, str] = {
    'S1': 'SM',
    'S2': 'SM',
    'S3': 'SM',
    'S4': 'SM',
    'S5': 'SM',
    'S6': 'SM',
    'IW1': 'IW',
    'IW2': 'IW',
    'IW3': 'IW',
    'EW1': 'EW',
    'EW2': 'EW',
    'EW3': 'EW',
    'EW4': 'EW',
    'EW5': 'EW',
    'WV1': 'WV',
    'WV2': 'WV'
}

# Range of azimuth beam angles (radians) for azimuth beam address lookup
AZIMUTH_BEAM_ADDRESS_TO_ANGLE = np.linspace(-0.018, 0.018, 1024)

# Huffman decode lookup tables (simplified representation)
# In practice, these would be more complex data structures
HUFFMAN_DECODE_TABLES: Dict[str, Any] = {
    'brc': {},  # BAQ Rate Control
    'thidx': {},  # Threshold Index
    'sm': {},  # Sign/Magnitude
    'neg': {},  # Negative values
    'pos': {}   # Positive values
}

# Replica chirp parameters
REPLICA_CHIRP_PARAMS: Dict[str, float] = {
    'bandwidth': 100e6,  # 100 MHz
    'duration': 27.12e-6,  # 27.12 microseconds
    'sample_rate': 117.6e6  # 117.6 MHz
}

# Window functions for processing
WINDOW_FUNCTIONS: Dict[str, str] = {
    'hamming': 'hamming',
    'hanning': 'hanning',
    'blackman': 'blackman',
    'kaiser': 'kaiser'
}

# Processing parameters
PROCESSING_PARAMS: Dict[str, Any] = {
    'range_looks': 1,
    'azimuth_looks': 1,
    'doppler_bandwidth': 1000.0,  # Hz
    'processing_bandwidth': 117.6e6,  # Hz
    'orbit_state_vectors_interval': 10.0,  # seconds
    'max_doppler_search_range': 2000.0  # Hz
}
