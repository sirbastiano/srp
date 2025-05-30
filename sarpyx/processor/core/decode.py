from pathlib import Path, os
import argparse
import pickle
import pandas as pd
import numpy as np
import os

import s1isp
from s1isp.decoder import decoded_subcomm_to_dict
from s1isp.decoder import EUdfDecodingMode

from .backup_metahandler import meta_extractor

def extract_echo_bursts(records):
    """
    Splits a list into sublists based on given indexes.

    Parameters:
        records: The list of records from isp.
    Returns:
    list of list: A list containing the sublists created by splitting the original list at the given indexes.
    """
    
    
    signal_types = {'noise': 1, 'tx_cal': 8, 'echo': 0}
    filtered = [x for x in records if x[1].radar_configuration_support.ses.signal_type == signal_types['echo']]
    
    #### TMP, TODO: remove it
    # Initialize the index
    echo_start_idx = None

    # Iterate through records with index
    for idx, x in enumerate(records):
        if x[1].radar_configuration_support.ses.signal_type == signal_types['echo']:
            echo_start_idx = idx
            break  # Exit loop once the first match is found
    
    # Get number of quads 
    get_nq = lambda x: x[1].radar_sample_count.number_of_quads
    # Computing first and last number of quads:
    # In stripmap you only have two bursts 
    first_nq = get_nq(filtered[0])
    last_nq = get_nq(filtered[-1])
    # Creating list of number of quads:
    nqs_list = [first_nq, last_nq]
    # Filtering the bursts
    bursts = [[x for x in filtered if get_nq(x) in [nq]] for idx, nq in enumerate(nqs_list)]
    # TODO: remove indexes
    return bursts, [echo_start_idx, echo_start_idx+int(len(bursts[0])), echo_start_idx+int(len(bursts[0]))++int(len(bursts[1]))]

def picklesavefile(path, datafile):
    with open(path, 'wb') as f:
        pickle.dump(datafile, f)

def header_extractor(filepath, mode: str = 'richa'):
    """
    Function to extract the metadata either with richa or s1isp

        mode (str): 'richa' or 's1isp'
    """

    if mode == 'richa':
        meta = meta_extractor(filepath)
    elif mode == 's1isp':
        records, offsets, subcom_data_records = s1isp.decoder.decode_stream(
        filepath,
        # maxcount=6000,  # comment out this line to decode all the ISPs in the file
        udf_decoding_mode=EUdfDecodingMode.NONE, # to have the corresponding signal data
        )
        headers_data = s1isp.decoder.decoded_stream_to_dict(records, enum_value=True)
        meta = pd.DataFrame(headers_data)
    
    return meta    

def decoder(inputfile):
    records, offsets, subcom_data_records = s1isp.decoder.decode_stream(
    inputfile,
    # maxcount=6000,  # comment out this line to decode all the ISPs in the file
    udf_decoding_mode=EUdfDecodingMode.DECODE, # to have the corresponding signal data
    )
    
    ### Subcomm
    subcom_data = subcom_data_records
    subcom_data_decoded = s1isp.decoder.SubCommutatedDataDecoder().decode(subcom_data)
    subcom_data_decoded_dict = decoded_subcomm_to_dict(subcom_decoded=subcom_data_decoded)
    subcom_data_decoded_df = pd.DataFrame(subcom_data_decoded_dict)
    ephemeris = subcom_data_decoded_df
    
    # Echoes signal extraction
    # TODO: remove indexes
    echo_bursts, indexes = extract_echo_bursts(records) 
    
    # Lamda func to extract echo data from records
    get_echo_arr = lambda x: x.udf
    # Cycling to bursts
    bursts_lists = []
    
    for burst in echo_bursts:
        headers_data = s1isp.decoder.decoded_stream_to_dict(burst, enum_value=True)
        metadata = pd.DataFrame(headers_data)
        radar_data = np.array([get_echo_arr(x) for x in burst])
        bursts_lists.append({'echo':radar_data, 'metadata':metadata, 'ephemeris':ephemeris})
    return bursts_lists, indexes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--inputfile', type=str, help='Path to input .dat file', default=None)
    parser.add_argument('-o', '--output', type=str, help='Path to folder output files', default=None)
    args = parser.parse_args()

    inputfile = args.inputfile
    L0_name = Path(inputfile).stem
    
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    
    if inputfile is not None:
        print('Decoding Level 0 file...')
        l0file, indexes = decoder(inputfile)
        total_metadata = meta_extractor(inputfile)
        
        for idx, burst in enumerate(l0file):
            ephemeris = burst['ephemeris']
            # metadata = burst['metadata']
            metadata = total_metadata.iloc[indexes[idx]:indexes[idx+1]]
            radar_data = burst['echo']
            # Save
            ephemeris.to_pickle(os.path.join(output_folder, Path(inputfile).stem + '_ephemeris.pkl'))
            metadata.to_pickle(os.path.join(output_folder, Path(inputfile).stem + f'_pkt_{idx}_metadata.pkl'))
            picklesavefile(path=f'{output_folder}/{L0_name}_pkt_{idx}.pkl', datafile=radar_data)