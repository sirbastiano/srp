import logging
from pathlib import Path
import pandas as pd
import gc
import joblib
from tqdm import tqdm
import sys
# ------ Custom Imports ------
from sarpyx.processor.core.decode import S1L0Decoder
from sarpyx.utils.io import find_dat_file

# ------ Configure logging ------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('decoder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------ Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ------ Functions ------
def decode_s1_l0(input_file, output_dir):
    decoder = S1L0Decoder()
    decoder.decode_file(input_file, output_dir, save_to_zarr=True, headers_only=False)
    logger.info(f'✅ Decoding completed for {input_file.name} to {output_dir}')
    del decoder
    # Garbage collection
    gc.collect()


def retrieve_input_files(safe_folders, verbose=False):
    """Retrieve input files from SAFE folders."""
    pols = ['vh', 'vv', 'hh', 'hv']
    input_files = []
    folders_map = {x: [] for x in safe_folders}
    for folder in safe_folders:
        for pol in pols:
            try:
                input_file = find_dat_file(folder, pol)
                input_files.append(input_file)
                folders_map[folder].append(input_file)
                if verbose:
                    logger.info(f'📁 Found {input_file.name} in {folder.name}')
            except FileNotFoundError:
                continue
        else:
            if verbose:
                logger.info(f'📁 No .dat file found in {folder.name}, checking next folder...')
            pass
    return input_files, folders_map
            

if __name__ == '__main__':
    # ------ Setup paths ------
    cwd = Path.cwd().parent
    data_dir = cwd / 'extracted_data'
    output_dir = cwd / 'decoded_data'

    # Find first SAFE folder and .dat file
    safe_folders = [f for f in data_dir.rglob('*.SAFE') if f.is_dir()]
    input_files, folders_map = retrieve_input_files(safe_folders)
    if not input_files:
        logger.error('❌ No input files found. Please check the data directory.')
        sys.exit(1)
    
    # dump input_files and folders_map to joblib files
    joblib.dump(input_files, output_dir / 'info_input_files.joblib')
    joblib.dump(folders_map, output_dir / 'info_folders_map.joblib')

    for input_file in tqdm(input_files, desc="Processing files"):
        if input_file.is_file():
            logger.info(f'🔍 Processing {input_file.name}...')
            decode_s1_l0(input_file, output_dir)
        else:
            logger.warning(f'❌ {input_file.name} is not a valid file, skipping...')
    # Print summary of processed files
    logger.info(f'🔍 Processed {len(input_files)} files from {len(folders_map)} SAFE folders.')
    sys.exit(0)