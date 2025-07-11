# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:39:18 2022.

@author: richa
"""
# Constant used to scale several data fields
F_REF = 37.53472224 * 1e6

# Useful for processing radar data
SPEED_OF_LIGHT_MPS = 299792458.0
TX_FREQ_HZ = 5.405e9
TX_WAVELENGTH_M = SPEED_OF_LIGHT_MPS / TX_FREQ_HZ
WGS84_SEMI_MAJOR_AXIS_M = 6378137
WGS84_SEMI_MINOR_AXIS_M = 6356752.3142

# Packet metadata dataframe indices
PACKET_NUM_FIELD_NAME = 'Packet Number'
BURST_NUM_FIELD_NAME = 'Azimuth Block Number'

# Packet metadata dataframe field names
PACKET_VER_NUM_FIELD_NAME = 'Packet Version Number'
PACKET_TYPE_FIELD_NAME = 'Packet Type'
SECONDARY_HEADER_FIELD_NAME = 'Secondary Header Flag'
PID_FIELD_NAME = 'PID'
PCAT_FIELD_NAME = 'PCAT'
SEQUENCE_FLAGS_FIELD_NAME = 'Sequence Flags'
PACKET_SEQUENCE_COUNT_FIELD_NAME = 'Packet Sequence Count'
PACKET_DATA_LEN_FIELD_NAME = 'Packet Data Length'
COARSE_TIME_FIELD_NAME = 'Coarse Time'
FINE_TIME_FIELD_NAME = 'Fine Time'
SYNC_FIELD_NAME = 'Sync'
DATA_TAKE_ID_FIELD_NAME = 'Data Take ID'
ECC_NUM_FIELD_NAME = 'ECC Number'
TEST_MODE_FIELD_NAME = 'Test Mode'
RX_CHAN_ID_FIELD_NAME = 'Rx Channel ID'
INSTRUMENT_CONFIG_ID_FIELD_NAME = 'Instrument Configuration ID'
SUBCOM_ANC_DATA_WORD_INDEX_FIELD_NAME = 'Sub-commutated Ancilliary Data Word Index'
SUBCOM_ANC_DATA_WORD_FIELD_NAME = 'Sub-commutated Ancilliary Data Word'
SPACE_PACKET_COUNT_FIELD_NAME = 'Space Packet Count'
PRI_COUNT_FIELD_NAME = 'PRI Count'
ERROR_FLAG_FIELD_NAME = 'Error Flag'
BAQ_MODE_FIELD_NAME = 'BAQ Mode'
BAQ_BLOCK_LEN_FIELD_NAME = 'BAQ Block Length'
RANGE_DEC_FIELD_NAME = 'Range Decimation'
RX_GAIN_FIELD_NAME = 'Rx Gain'
TX_RAMP_RATE_FIELD_NAME = 'Tx Ramp Rate'
TX_PULSE_START_FREQ_FIELD_NAME = 'Tx Pulse Start Frequency'
TX_PULSE_LEN_FIELD_NAME = 'Tx Pulse Length'
RANK_FIELD_NAME = 'Rank'
PRI_FIELD_NAME = 'PRI'
SWST_FIELD_NAME = 'SWST'
SWL_FIELD_NAME = 'SWL'
SAS_SSB_FLAG_FIELD_NAME = 'SAS SSB Flag'
POLARIZATION_FIELD_NAME = 'Polarisation' # Can I irritate Americans and Brits AT THE SAME TIME?!
TEMP_COMP_FIELD_NAME = 'Temperature Compensation'
CAL_MODE_FIELD_NAME = 'Calibration Mode'
TX_PULSE_NUM_FIELD_NAME = 'Tx Pulse Number'
SIGNAL_TYPE_FIELD_NAME = 'Signal Type'
SWAP_FLAG_FIELD_NAME = 'Swap Flag'
SWATH_NUM_FIELD_NAME = 'Swath Number'
NUM_QUADS_FIELD_NAME = 'Number of Quads'

# Subcommed data output dataframe field names
X_POS_FIELD_NAME = "X-axis position ECEF"
Y_POS_FIELD_NAME = "Y-axis position ECEF"
Z_POS_FIELD_NAME = "Z-axis position ECEF"
X_VEL_FIELD_NAME = "X-axis velocity ECEF"
Y_VEL_FIELD_NAME = "Y-axis velocity ECEF"
Z_VEL_FIELD_NAME = "Z-axis velocity ECEF"
POD_SOLN_DATA_TIMESTAMP_FIELD_NAME = "POD Solution Data Timestamp"
Q0_FIELD_NAME = "Q0 Attitude Quaternion"
Q1_FIELD_NAME = "Q1 Attitude Quaternion"
Q2_FIELD_NAME = "Q2 Attitude Quaternion"
Q3_FIELD_NAME = "Q3 Attitude Quaternion"
X_ANG_RATE_FIELD_NAME = "Omega-X Angular Rate"
Y_ANG_RATE_FIELD_NAME = "Omega-Y Angular Rate"
Z_ANG_RATE_FIELD_NAME = "Omega-Z Angular Rate"
ATTITUDE_DATA_TIMESTAMP_FIELD_NAME = "Attitude Data Timestamp"