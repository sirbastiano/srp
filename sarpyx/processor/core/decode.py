import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from s1isp.decoder import (
    EUdfDecodingMode, 
    SubCommutatedDataDecoder,
    decode_stream, 
    decoded_stream_to_dict, 
    decoded_subcomm_to_dict
)
from .encoding import parameter_transformations as pt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_echo_bursts(records: List[Any]) -> Tuple[List[List[Any]], List[int]]:
    """Extract echo bursts from radar records and return burst data with indexes.

    This function filters radar records to extract only echo signals, then groups
    them by number of quads to create separate bursts. It's specifically designed
    for stripmap mode which typically has two bursts.

    Args:
        records (List[Any]): List of records from ISP decoder containing radar data.
                           Each record should have radar_configuration_support.ses.signal_type
                           and radar_sample_count.number_of_quads attributes.

    Returns:
        Tuple[List[List[Any]], List[int]]: A tuple containing:
            - List of echo burst sublists grouped by number of quads
            - List of indexes indicating burst start/end positions in original records

    Raises:
        AssertionError: If records list is empty or no echo records are found.
        ValueError: If unable to extract proper burst structure.
    """
    assert records, 'Records list cannot be empty'
    
    signal_types = {'noise': 1, 'tx_cal': 8, 'echo': 0}
    echo_signal_type = signal_types['echo']
    
    # Filter echo records
    filtered_records = [
        record for record in records 
        if record[1].radar_configuration_support.ses.signal_type == echo_signal_type
    ]
    
    assert filtered_records, 'No echo records found in the input data'
    
    # Find first echo record index in original records
    echo_start_idx = None
    for idx, record in enumerate(records):
        if record[1].radar_configuration_support.ses.signal_type == echo_signal_type:
            echo_start_idx = idx
            break
    
    assert echo_start_idx is not None, 'Echo start index not found'
    
    # Extract number of quads for burst grouping
    def get_number_of_quads(record: Any) -> int:
        """Extract number of quads from a record."""
        return record[1].radar_sample_count.number_of_quads
    
    # Get unique quad counts for burst separation
    first_nq = get_number_of_quads(filtered_records[0])
    last_nq = get_number_of_quads(filtered_records[-1])
    
    # Create unique list of quad counts
    unique_quad_counts = list(dict.fromkeys([first_nq, last_nq]))  # Preserves order, removes duplicates
    
    # Group bursts by number of quads
    bursts = []
    for quad_count in unique_quad_counts:
        burst = [record for record in filtered_records if get_number_of_quads(record) == quad_count]
        if burst:  # Only add non-empty bursts
            bursts.append(burst)
    
    assert bursts, 'No valid bursts found after filtering'
    
    # Calculate burst boundary indexes
    indexes = [echo_start_idx]
    current_idx = echo_start_idx
    
    for burst in bursts:
        current_idx += len(burst)
        indexes.append(current_idx)
    
    return bursts, indexes


def save_pickle_file(file_path: Union[str, Path], data: Any) -> None:
    """Save data to a pickle file with robust error handling.

    Args:
        file_path (Union[str, Path]): Path where to save the pickle file.
        data (Any): Data object to be pickled and saved.

    Raises:
        OSError: If file cannot be written due to permissions or disk space.
        ValueError: If file_path is empty or invalid.
    """
    if not file_path:
        raise ValueError('File path cannot be empty')
    
    file_path = Path(file_path)
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f'Successfully saved pickle file: {file_path}')
    except Exception as e:
        logger.error(f'Failed to save pickle file {file_path}: {e}')
        raise


def extract_headers(file_path: Union[str, Path], mode: str = 's1isp', apply_transformations: bool = False) -> pd.DataFrame:
    """Extract metadata headers from radar file using specified decoding mode.

    Args:
        file_path (Union[str, Path]): Path to the radar data file.
        mode (str): Extraction mode. Currently only 's1isp' is fully supported.
        apply_transformations (bool): Whether to apply parameter transformations to convert raw values to physical units.

    Returns:
        pd.DataFrame: DataFrame containing the extracted metadata headers with optional transformations.

    Raises:
        ValueError: If mode is not supported or file_path is invalid.
        FileNotFoundError: If file_path does not exist.
        RuntimeError: If decoding fails or transformation fails.
    """
    supported_modes = ['richa', 's1isp']
    if mode not in supported_modes:
        raise ValueError(f"Mode must be one of {supported_modes}, got '{mode}'")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')

    if mode == 'richa':
        raise NotImplementedError('Richa mode requires meta_extractor implementation')
    
    try:
        records, _, _ = decode_stream(
            str(file_path),
            udf_decoding_mode=EUdfDecodingMode.NONE,
        )
        headers_data = decoded_stream_to_dict(records, enum_value=True)
        metadata_df = pd.DataFrame(headers_data)
        
        # Apply parameter transformations if requested
        if apply_transformations:
            try:
                metadata_df = _apply_parameter_transformations(metadata_df)
                logger.info(f'Applied parameter transformations to {len(metadata_df)} records')
            except Exception as e:
                logger.warning(f'Failed to apply parameter transformations: {e}')
                # Continue with raw values if transformations fail
        
        logger.info(f'Successfully extracted {len(metadata_df)} header records')
        return metadata_df
        
    except Exception as e:
        logger.error(f'Failed to extract headers from {file_path}: {e}')
        raise RuntimeError(f'Header extraction failed: {e}') from e


def decode_radar_file(input_file: Union[str, Path], apply_transformations: bool = False) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Decode Sentinel-1 Level 0 radar file and extract burst data with ephemeris.

    This function performs complete decoding of a radar file, extracting both
    the radar echo data and associated metadata including ephemeris information.

    Args:
        input_file (Union[str, Path]): Path to the input Level 0 radar file.
        apply_transformations (bool): Whether to apply parameter transformations to convert raw values to physical units.

    Returns:
        Tuple[List[Dict[str, Any]], List[int]]: A tuple containing:
            - List of dictionaries with 'echo', 'metadata', and 'ephemeris' keys
            - List of indexes indicating burst positions in the original data

    Raises:
        FileNotFoundError: If input file does not exist.
        RuntimeError: If decoding process fails.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_file}')
    
    logger.info(f'Starting decode process for: {input_file}')
    
    try:
        # Decode the stream with UDF data
        records, _, subcom_data_records = decode_stream(
            str(input_file),
            udf_decoding_mode=EUdfDecodingMode.DECODE,
        )
        
        logger.info(f'Decoded {len(records)} records from file')
        
        # Process subcommutated data for ephemeris
        if subcom_data_records:
            subcom_decoder = SubCommutatedDataDecoder()
            subcom_decoded = subcom_decoder.decode(subcom_data_records)
            subcom_dict = decoded_subcomm_to_dict(subcom_decoded=subcom_decoded)
            ephemeris_df = pd.DataFrame(subcom_dict)
            logger.info(f'Extracted ephemeris data with {len(ephemeris_df)} records')
        else:
            logger.warning('No subcommutated data found, creating empty ephemeris DataFrame')
            ephemeris_df = pd.DataFrame()
        
        # Extract echo bursts
        echo_bursts, burst_indexes = extract_echo_bursts(records)
        logger.info(f'Extracted {len(echo_bursts)} echo bursts')
        
        # Process each burst
        processed_bursts = []
        for i, burst in enumerate(echo_bursts):
            try:
                # Extract metadata for this burst
                headers_data = decoded_stream_to_dict(burst, enum_value=True)
                burst_metadata = pd.DataFrame(headers_data)
                
                # Apply transformations if requested
                if apply_transformations:
                    burst_metadata = _apply_parameter_transformations(burst_metadata)
                
                # Extract radar data (UDF - User Data Field)
                radar_data = np.array([record.udf for record in burst])
                
                burst_dict = {
                    'echo': radar_data,
                    'metadata': burst_metadata,
                    'ephemeris': ephemeris_df
                }
                processed_bursts.append(burst_dict)
                
                logger.info(f'Processed burst {i}: {radar_data.shape} radar samples, '
                          f'{len(burst_metadata)} metadata records')
                
            except Exception as e:
                logger.error(f'Failed to process burst {i}: {e}')
                raise RuntimeError(f'Burst processing failed for burst {i}') from e
        
        return processed_bursts, burst_indexes
        
    except Exception as e:
        logger.error(f'Decoding failed for {input_file}: {e}')
        raise RuntimeError(f'File decoding failed: {e}') from e


class S1L0Decoder:
    """Minimal API for decoding Sentinel-1 Level 0 data files."""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize the decoder with logging configuration.
        
        Args:
            log_level: Logging level (default: INFO)
        """
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
    
    def decode_file(
        self, 
        input_file: Path | str, 
        output_dir: Optional[Path | str] = None,
        headers_only: bool = False,
        apply_transformations: bool = True
    ) -> Dict[str, Any]:
        """Decode a Sentinel-1 Level 0 .dat file.
        
        Args:
            input_file: Path to the input .dat file
            output_dir: Directory to save processed data (optional)
            headers_only: If True, extract only headers for quick preview
            apply_transformations: If True, apply parameter transformations to convert raw values to physical units
            
        Returns:
            Dictionary containing decoded data with keys:
            - 'burst_data': List of burst dictionaries (if headers_only=False)
            - 'headers': DataFrame with header information (if headers_only=True)
            - 'file_info': Basic file information
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If decoding fails
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f'Input file not found: {input_path}')
        
        self.logger.info(f'Processing file: {input_path}')
        self.logger.info(f'File size: {input_path.stat().st_size / (1024**2):.1f} MB')
        
        result = {
            'file_info': {
                'path': str(input_path),
                'size_mb': input_path.stat().st_size / (1024**2),
                'filename': input_path.name
            }
        }
        
        try:
            if headers_only:
                # Extract headers only for quick preview
                self.logger.info('Extracting headers only...')
                headers_df = extract_headers(input_path, mode='s1isp', apply_transformations=apply_transformations)
                
                result['headers'] = headers_df
                result['num_records'] = len(headers_df)
                
                self.logger.info(f'Extracted {len(headers_df)} header records')
                
            else:
                # Full decoding
                self.logger.info('Starting full decode process...')
                burst_data, burst_indexes = decode_radar_file(input_path, apply_transformations=apply_transformations)
                
                result['burst_data'] = burst_data
                result['burst_indexes'] = burst_indexes
                result['num_bursts'] = len(burst_data)
                
                # Add burst summaries
                burst_summaries = []
                for i, burst in enumerate(burst_data):
                    summary = {
                        'burst_id': i,
                        'echo_shape': burst['echo'].shape,
                        'metadata_records': len(burst['metadata']),
                        'ephemeris_records': len(burst['ephemeris'])
                    }
                    burst_summaries.append(summary)
                
                result['burst_summaries'] = burst_summaries
                
                self.logger.info(f'Successfully decoded {len(burst_data)} bursts')
            
            # Save data if output directory is specified
            if output_dir is not None:
                save_path = self._save_data(result, input_path, Path(output_dir))
                result['saved_to'] = str(save_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f'Decoding failed: {e}')
            raise ValueError(f'Failed to decode file: {e}') from e
    
    def _save_data(
        self, 
        decoded_data: Dict[str, Any], 
        input_path: Path, 
        output_dir: Path
    ) -> Path:
        """Save decoded data to pickle files.
        
        Args:
            decoded_data: Dictionary containing decoded data
            input_path: Original input file path
            output_dir: Directory to save files
            
        Returns:
            Path to the output directory
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        file_stem = input_path.stem
        
        self.logger.info(f'Saving processed data to: {output_dir}')
        
        if 'headers' in decoded_data:
            # Save headers only
            headers_path = output_dir / f'{file_stem}_headers.pkl'
            decoded_data['headers'].to_pickle(headers_path)
            self.logger.info(f'Saved headers: {headers_path}')
            
        elif 'burst_data' in decoded_data:
            # Save full burst data
            burst_data = decoded_data['burst_data']
            
            # Save ephemeris (once for all bursts)
            if burst_data and not burst_data[0]['ephemeris'].empty:
                ephemeris_path = output_dir / f'{file_stem}_ephemeris.pkl'
                burst_data[0]['ephemeris'].to_pickle(ephemeris_path)
                self.logger.info(f'Saved ephemeris: {ephemeris_path}')
            
            # Save each burst
            for i, burst in enumerate(burst_data):
                # Save metadata
                metadata_path = output_dir / f'{file_stem}_burst_{i}_metadata.pkl'
                burst['metadata'].to_pickle(metadata_path)
                
                # Save radar echo data
                echo_path = output_dir / f'{file_stem}_burst_{i}_echo.pkl'
                save_pickle_file(echo_path, burst['echo'])
                
                self.logger.info(f'Saved burst {i}: metadata and echo data')
        
        # Save summary info
        info_path = output_dir / f'{file_stem}_info.pkl'
        summary_info = {k: v for k, v in decoded_data.items() 
                       if k not in ['burst_data', 'headers']}
        save_pickle_file(info_path, summary_info)
        
        total_files = len(list(output_dir.glob(f'{file_stem}*.pkl')))
        self.logger.info(f'Created {total_files} output files')
        
        return output_dir


# Parameter Transformations Integration
# =====================================
# This module integrates parameter transformations from parameter_transformations.py
# to convert raw bytecode values from Sentinel-1 data packets into meaningful physical
# parameters. When apply_transformations=True is used:
#
# Key Physical Transformations Applied:
# • fine_time: Raw 16-bit → Seconds using (raw + 0.5) * 2^-16
# • rx_gain: Raw codes → dB using raw * -0.5
# • pri: Raw counts → Seconds using raw / F_REF
# • tx_pulse_length: Raw counts → Seconds using raw / F_REF
# • tx_ramp_rate: Raw 16-bit → Hz/s using sign/magnitude extraction + F_REF² scaling
# • tx_pulse_start_freq: Raw 16-bit → Hz using sign/magnitude + F_REF scaling
# • range_decimation: Raw codes → Sample rate (Hz) using lookup table
# • swst/swl: Raw counts → Seconds using raw / F_REF
#
# Additional Features:
# • Validation columns (sync_marker_valid, baq_mode_valid, etc.)
# • Descriptive columns (signal_type_name, polarization_name, etc.)
# • Derived columns (samples_per_line, data_take_hex, etc.)
#
# Usage:
# decoder = S1L0Decoder()
# result = decoder.decode_file(input_file, output_dir, apply_transformations=True)
# headers = extract_headers(file_path, apply_transformations=True)
# bursts = decode_radar_file(file_path, apply_transformations=True)
# =====================================


def _apply_parameter_transformations(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Apply parameter transformations using parameter_transformations.py API functions.
    
    This function integrates the parameter_transformations.py API directly to convert 
    raw bytecode values from Sentinel-1 data packets into meaningful physical parameters
    using the exact mathematical transformations defined in the API.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame containing raw metadata values from decoded packets.
        
    Returns:
        pd.DataFrame: DataFrame with transformed physical values and additional descriptive columns.
        
    Raises:
        Exception: If transformation of critical parameters fails.
    """
    transformed_df = metadata_df.copy()
    
    def safe_to_float(value: Any) -> Optional[float]:
        """Safely convert pandas scalar or other types to float, excluding complex numbers."""
        if value is None:
            return None
        if hasattr(value, '__array__') and pd.isna(value):
            return None
        if isinstance(value, complex):
            logger.warning(f'Cannot convert complex number to float: {value}')
            return None
        
        try:
            if isinstance(value, (np.number, np.ndarray)):
                return float(value.item())
            return float(value)
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f'Failed to convert value to float: {value}, error: {e}')
            return None
    
    # Apply fine time transformation: (raw + 0.5) * 2^-16 → seconds
    try:
        if 'fine_time' in transformed_df.columns:
            transformed_df['fine_time'] = (transformed_df['fine_time'] + 0.5) * (2**-16)
    except Exception as e:
        logger.warning(f'Error applying fine time transformation: {e}')
    
    # Apply Rx gain transformation: raw * -0.5 → dB
    try:
        if 'rx_gain' in transformed_df.columns:
            transformed_df['rx_gain'] = transformed_df['rx_gain'] * -0.5
    except Exception as e:
        logger.warning(f'Error applying Rx gain transformation: {e}')
    
    # Apply Tx pulse ramp rate transformation: 16-bit sign/magnitude → Hz/s
    try:
        if 'tx_ramp_rate' in transformed_df.columns:
            def transform_txprr(raw_value: Any) -> Optional[float]:
                """Transform Tx pulse ramp rate using extract_tx_pulse_ramp_rate logic."""
                converted = safe_to_float(raw_value)
                if converted is None:
                    return None
                    
                tmp16 = int(converted)
                txprr_sign = (-1) ** (1 - (tmp16 >> 15))
                magnitude = tmp16 & 0x7FFF
                txprr = txprr_sign * magnitude * (pt.F_REF**2) / (2**21)
                return txprr
            
            transformed_df['tx_ramp_rate'] = transformed_df['tx_ramp_rate'].apply(transform_txprr)
    except Exception as e:
        logger.warning(f'Error applying Tx ramp rate transformation: {e}')
    
    # Apply Tx pulse start frequency transformation: 16-bit + TXPRR dependency → Hz  
    try:
        if 'tx_pulse_start_freq' in transformed_df.columns and 'tx_ramp_rate' in metadata_df.columns:
            def transform_txpsf(row_idx: int) -> Optional[float]:
                """Transform Tx pulse start frequency using extract_tx_pulse_start_frequency logic."""
                raw_txpsf = transformed_df.loc[row_idx, 'tx_pulse_start_freq']
                raw_txprr = metadata_df.loc[row_idx, 'tx_ramp_rate'] 
                
                converted_txpsf = safe_to_float(raw_txpsf)
                converted_txprr = safe_to_float(raw_txprr)
                
                if converted_txpsf is None or converted_txprr is None:
                    return None
                
                # Calculate TXPRR for additive component
                tmp16_txprr = int(converted_txprr)
                txprr_sign = (-1) ** (1 - (tmp16_txprr >> 15))
                txprr_magnitude = tmp16_txprr & 0x7FFF
                txprr = txprr_sign * txprr_magnitude * (pt.F_REF**2) / (2**21)
                txpsf_additive = txprr / (4 * pt.F_REF)
                
                # Extract TXPSF sign bit and magnitude
                tmp16_txpsf = int(converted_txpsf)
                txpsf_sign = (-1) ** (1 - (tmp16_txpsf >> 15))
                txpsf_magnitude = tmp16_txpsf & 0x7FFF
                
                # Apply scaling and combine components
                txpsf = txpsf_additive + txpsf_sign * txpsf_magnitude * pt.F_REF / (2**14)
                return txpsf
            
            transformed_df['tx_pulse_start_freq'] = [
                transform_txpsf(i) for i in transformed_df.index
            ]
    except Exception as e:
        logger.warning(f'Error applying Tx start frequency transformation: {e}')
    
    # Apply Tx pulse length transformation: raw / F_REF → seconds
    try:
        if 'tx_pulse_length' in transformed_df.columns:
            transformed_df['tx_pulse_length'] = transformed_df['tx_pulse_length'] / pt.F_REF
    except Exception as e:
        logger.warning(f'Error applying Tx pulse length transformation: {e}')
    
    # Apply PRI transformation: raw / F_REF → seconds
    try:
        if 'pri' in transformed_df.columns:
            transformed_df['pri'] = transformed_df['pri'] / pt.F_REF
    except Exception as e:
        logger.warning(f'Error applying PRI transformation: {e}')
    
    # Apply sampling window transformations: raw / F_REF → seconds
    try:
        if 'swst' in transformed_df.columns:
            transformed_df['swst'] = transformed_df['swst'] / pt.F_REF
            
        if 'swl' in transformed_df.columns:
            transformed_df['swl'] = transformed_df['swl'] / pt.F_REF
    except Exception as e:
        logger.warning(f'Error applying sampling window transformations: {e}')
    
    # Apply range decimation to sample rate conversion using API lookup table
    try:
        if 'range_decimation' in transformed_df.columns:
            transformed_df['range_decimation'] = transformed_df['range_decimation'].apply(
                lambda x: pt.range_dec_to_sample_rate(int(x)) if pd.notna(x) else None
            )
    except Exception as e:
        logger.warning(f'Error applying range decimation transformation: {e}')
    
    # Apply additional descriptive transformations and validations from API
    try:
        # Signal type mapping
        if 'signal_type' in transformed_df.columns:
            signal_types = {0: 'echo', 1: 'noise', 8: 'tx_cal'}
            transformed_df['signal_type_name'] = transformed_df['signal_type'].map(signal_types)
            
        # Data take ID to hex representation
        if 'data_take_id' in transformed_df.columns:
            transformed_df['data_take_hex'] = transformed_df['data_take_id'].apply(
                lambda x: f'0x{int(x):08X}' if pd.notna(x) else None
            )
            
        # Number of quads to samples per line conversion
        if 'number_of_quads' in transformed_df.columns:
            transformed_df['samples_per_line'] = transformed_df['number_of_quads'] * 2
            
        # Polarization mapping
        if 'polarization' in transformed_df.columns:
            pol_mapping = {0: 'H', 1: 'V', 2: 'H+V', 3: 'H-V'}
            transformed_df['polarization_name'] = transformed_df['polarization'].map(pol_mapping)
            
        # Temperature compensation mapping
        if 'temperature_compensation' in transformed_df.columns:
            temp_comp_mapping = {0: 'disabled', 1: 'enabled', 2: 'reserved1', 3: 'reserved2'}
            transformed_df['temp_comp_name'] = transformed_df['temperature_compensation'].map(temp_comp_mapping)
        
        # Apply API validation functions
        if 'sync_marker' in transformed_df.columns:
            transformed_df['sync_marker_valid'] = transformed_df['sync_marker'].apply(
                lambda x: pt.validate_sync_marker(int(x)) if pd.notna(x) else False
            )
            
        if 'baq_mode' in transformed_df.columns:
            transformed_df['baq_mode_valid'] = transformed_df['baq_mode'].apply(
                lambda x: pt.validate_baq_mode(int(x)) if pd.notna(x) else False
            )
            
        if 'packet_version_number' in transformed_df.columns:
            transformed_df['packet_version_valid'] = transformed_df['packet_version_number'].apply(
                lambda x: pt.validate_packet_version(int(x)) if pd.notna(x) else False
            )
            
    except Exception as e:
        logger.warning(f'Error applying additional transformations: {e}')
    
    return transformed_df


# ------------- Main Function -------------
def main() -> int:
    """Main entry point for the radar file decoder.
    
    Handles command line arguments and orchestrates the decoding process.
    
    Returns:
        int: Exit code - 0 for success, 1 for failure.
    """
    parser = argparse.ArgumentParser(
                description='Decode Sentinel-1 Level 0 radar data files into processed bursts',
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
                        Examples:
                        python decode.py -i data.dat -o output_folder
                        python decode.py --inputfile /path/to/radar.dat --output /path/to/output
                        """,
            )
    
    parser.add_argument(
        '-i', '--inputfile', 
        type=str, 
        required=True,
        help='Path to input Level 0 radar data file (.dat)'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=True,
        help='Path to output directory for processed files'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug('Verbose logging enabled')
    
    # Validate inputs
    input_file = Path(args.inputfile)
    output_dir = Path(args.output)
    
    if not input_file.exists():
        logger.error(f'Input file does not exist: {input_file}')
        return 1
    
    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Output directory ready: {output_dir}')
    except Exception as e:
        logger.error(f'Failed to create output directory {output_dir}: {e}')
        return 1
    
    # Process the file
    try:
        file_stem = input_file.stem
        logger.info(f'Processing file: {file_stem}')
        
        # Decode the radar file
        burst_data, _ = decode_radar_file(input_file, apply_transformations=False)
        
        # Save processed data
        ephemeris_saved = False
        for burst_idx, burst in enumerate(burst_data):
            # Save ephemeris once (it's the same for all bursts)
            if not ephemeris_saved and not burst['ephemeris'].empty:
                ephemeris_path = output_dir / f'{file_stem}_ephemeris.pkl'
                burst['ephemeris'].to_pickle(ephemeris_path)
                logger.info(f'Saved ephemeris data: {ephemeris_path}')
                ephemeris_saved = True
            
            # Save burst-specific metadata
            metadata_path = output_dir / f'{file_stem}_pkt_{burst_idx}_metadata.pkl'
            burst['metadata'].to_pickle(metadata_path)
            
            # Save radar data
            radar_data_path = output_dir / f'{file_stem}_pkt_{burst_idx}.pkl'
            save_pickle_file(radar_data_path, burst['echo'])
            
            logger.info(f'Saved burst {burst_idx} data: metadata and radar arrays')
        
        logger.info(f'Successfully processed {len(burst_data)} bursts from {input_file}')
        return 0
        
    except Exception as e:
        logger.error(f'Processing failed: {e}')
        return 1


if __name__ == '__main__':
    exit(main())