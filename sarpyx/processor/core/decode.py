import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from s1isp.decoder import (
    EUdfDecodingMode, 
    SubCommutatedDataDecoder,
    decode_stream, 
    decoded_stream_to_dict, 
    decoded_subcomm_to_dict
)

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


def extract_headers(file_path: Union[str, Path], mode: str = 's1isp') -> pd.DataFrame:
    """Extract metadata headers from radar file using specified decoding mode.

    Args:
        file_path (Union[str, Path]): Path to the radar data file.
        mode (str): Extraction mode. Currently only 's1isp' is fully supported.

    Returns:
        pd.DataFrame: DataFrame containing the extracted metadata headers.

    Raises:
        ValueError: If mode is not supported or file_path is invalid.
        FileNotFoundError: If file_path does not exist.
        RuntimeError: If decoding fails.
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
        
        logger.info(f'Successfully extracted {len(metadata_df)} header records')
        return metadata_df
        
    except Exception as e:
        logger.error(f'Failed to extract headers from {file_path}: {e}')
        raise RuntimeError(f'Header extraction failed: {e}') from e


def decode_radar_file(input_file: Union[str, Path]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Decode Sentinel-1 Level 0 radar file and extract burst data with ephemeris.

    This function performs complete decoding of a radar file, extracting both
    the radar echo data and associated metadata including ephemeris information.

    Args:
        input_file (Union[str, Path]): Path to the input Level 0 radar file.

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


def main() -> None:
    """Main entry point for the radar file decoder.
    
    Handles command line arguments and orchestrates the decoding process.
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
        burst_data, _ = decode_radar_file(input_file)
        
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