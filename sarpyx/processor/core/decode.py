import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import s1isp
from s1isp.decoder import EUdfDecodingMode, decoded_subcomm_to_dict



def extract_echo_bursts(records: List[Any]) -> Tuple[List[List[Any]], List[int]]:
    """Extract echo bursts from radar records and return burst data with indexes.

    Args:
        records: The list of records from ISP decoder containing radar data.

    Returns:
        A tuple containing:
            - List of echo burst sublists grouped by number of quads
            - List of indexes indicating burst positions in original records

    Raises:
        AssertionError: If no echo records are found in the input.
        IndexError: If filtered records list is empty.
    """
    assert records, 'Records list cannot be empty'
    
    signal_types = {'noise': 1, 'tx_cal': 8, 'echo': 0}
    filtered = [x for x in records if x[1].radar_configuration_support.ses.signal_type == signal_types['echo']]
    
    assert filtered, 'No echo records found in the input data'
    
    # Find the first echo record index
    echo_start_idx = None
    for idx, x in enumerate(records):
        if x[1].radar_configuration_support.ses.signal_type == signal_types['echo']:
            echo_start_idx = idx
            break
    
    assert echo_start_idx is not None, 'No echo start index found'
    
    # Get number of quads 
    get_nq = lambda x: x[1].radar_sample_count.number_of_quads
    
    # Computing first and last number of quads:
    # In stripmap you only have two bursts 
    first_nq = get_nq(filtered[0])
    last_nq = get_nq(filtered[-1])
    nqs_list = [first_nq, last_nq]
    
    # Filtering the bursts
    bursts = [[x for x in filtered if get_nq(x) == nq] for nq in nqs_list]
    
    # Calculate proper indexes (fix syntax error with ++)
    burst_0_end = echo_start_idx + len(bursts[0])
    burst_1_end = burst_0_end + len(bursts[1])
    indexes = [echo_start_idx, burst_0_end, burst_1_end]
    
    return bursts, indexes


def pickle_save_file(path: Union[str, Path], datafile: Any) -> None:
    """Save data to a pickle file.

    Args:
        path: File path where to save the pickle file.
        datafile: Data object to be pickled and saved.

    Raises:
        IOError: If file cannot be written.
    """
    assert path, 'Path cannot be empty'
    
    with open(path, 'wb') as f:
        pickle.dump(datafile, f)


def header_extractor(filepath: Union[str, Path], mode: str = 's1isp') -> pd.DataFrame:
    """Extract metadata headers from radar file using specified mode.

    Args:
        filepath: Path to the radar data file.
        mode: Extraction mode, either 'richa' or 's1isp'.

    Returns:
        DataFrame containing the extracted metadata headers.

    Raises:
        ValueError: If mode is not 'richa' or 's1isp'.
        FileNotFoundError: If filepath does not exist.
    """
    assert mode in ['richa', 's1isp'], f"Mode must be 'richa' or 's1isp', got '{mode}'"
    assert Path(filepath).exists(), f'File not found: {filepath}'

    if mode == 'richa':
        # Commented out as the import is not available
        # meta = meta_extractor(filepath)
        raise NotImplementedError('Richa mode requires meta_extractor to be implemented')
    elif mode == 's1isp':
        records, offsets, subcom_data_records = s1isp.decoder.decode_stream(
            filepath,
            udf_decoding_mode=EUdfDecodingMode.NONE,
        )
        headers_data = s1isp.decoder.decoded_stream_to_dict(records, enum_value=True)
        meta = pd.DataFrame(headers_data)
    
    return meta


def decoder(inputfile: Union[str, Path]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Decode radar Level 0 file and extract bursts data.

    Args:
        inputfile: Path to the input Level 0 radar file.

    Returns:
        A tuple containing:
            - List of dictionaries with echo, metadata, and ephemeris data
            - List of indexes indicating burst positions

    Raises:
        FileNotFoundError: If input file does not exist.
    """
    assert Path(inputfile).exists(), f'Input file not found: {inputfile}'
    
    records, offsets, subcom_data_records = s1isp.decoder.decode_stream(
        inputfile,
        udf_decoding_mode=EUdfDecodingMode.DECODE,
    )
    
    # Process subcommutated data
    subcom_data = subcom_data_records
    subcom_data_decoded = s1isp.decoder.SubCommutatedDataDecoder().decode(subcom_data)
    subcom_data_decoded_dict = decoded_subcomm_to_dict(subcom_decoded=subcom_data_decoded)
    subcom_data_decoded_df = pd.DataFrame(subcom_data_decoded_dict)
    ephemeris = subcom_data_decoded_df
    
    # Extract echo bursts
    echo_bursts, indexes = extract_echo_bursts(records) 
    
    # Lambda function to extract echo data from records
    get_echo_arr = lambda x: x.udf
    
    # Process bursts
    bursts_lists = []
    for burst in echo_bursts:
        headers_data = s1isp.decoder.decoded_stream_to_dict(burst, enum_value=True)
        metadata = pd.DataFrame(headers_data)
        radar_data = np.array([get_echo_arr(x) for x in burst])
        bursts_lists.append({
            'echo': radar_data, 
            'metadata': metadata, 
            'ephemeris': ephemeris
        })
    
    return bursts_lists, indexes


def main() -> None:
    """Main function to handle command line arguments and process radar data."""
    parser = argparse.ArgumentParser(description='Decode Sentinel-1 Level 0 radar data files')
    parser.add_argument('-i', '--inputfile', type=str, help='Path to input .dat file', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to output folder', required=True)
    args = parser.parse_args()

    inputfile = args.inputfile
    output_folder = args.output
    
    assert Path(inputfile).exists(), f'Input file does not exist: {inputfile}'
    
    l0_name = Path(inputfile).stem
    os.makedirs(output_folder, exist_ok=True)
    
    print('Decoding Level 0 file...')
    l0file, indexes = decoder(inputfile)
    
    # Note: meta_extractor is commented out, so this will need to be implemented
    # total_metadata = meta_extractor(inputfile)
    
    for idx, burst in enumerate(l0file):
        ephemeris = burst['ephemeris']
        # Use burst metadata since total_metadata is not available
        metadata = burst['metadata']
        radar_data = burst['echo']
        
        # Save files
        ephemeris_path = os.path.join(output_folder, f'{l0_name}_ephemeris.pkl')
        metadata_path = os.path.join(output_folder, f'{l0_name}_pkt_{idx}_metadata.pkl')
        radar_data_path = os.path.join(output_folder, f'{l0_name}_pkt_{idx}.pkl')
        
        ephemeris.to_pickle(ephemeris_path)
        metadata.to_pickle(metadata_path)
        pickle_save_file(path=radar_data_path, datafile=radar_data)
        
        print(f'Saved burst {idx} data to {output_folder}')


if __name__ == '__main__':
    main()