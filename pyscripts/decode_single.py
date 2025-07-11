import logging
from pathlib import Path
import pandas as pd
import gc
from tqdm import tqdm
import sys
import argparse
from typing import List, Dict, Tuple
from sarpyx.processor.core.decode import S1L0Decoder
from sarpyx.utils.io import find_dat_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('decoder.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def decode_s1_l0(input_file: Path, output_dir: Path) -> None:
    """
    Decode S1 L0 file to zarr format.

    Args:
        input_file (Path): Path to the input .dat file.
        output_dir (Path): Directory to save decoded output.

    Returns:
        None
    """
    decoder = S1L0Decoder()
    decoder.decode_file(input_file, output_dir, save_to_zarr=True, headers_only=False)
    logger.info(f'✅ Decoding completed for {input_file.name} to {output_dir}')
    del decoder
    gc.collect()

def retrieve_input_files(safe_folders: List[Path], verbose: bool = False) -> Tuple[List[Path], Dict[Path, List[Path]]]:
    """
    Retrieve input files from SAFE folders.

    Args:
        safe_folders (List[Path]): List of SAFE folder paths.
        verbose (bool): Whether to log verbose output.

    Returns:
        Tuple[List[Path], Dict[Path, List[Path]]]: (input_files, folders_map)
    """
    pols = ['vh', 'vv', 'hh', 'hv']
    input_files: List[Path] = []
    folders_map: Dict[Path, List[Path]] = {folder: [] for folder in safe_folders}
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
        if verbose and not folders_map[folder]:
            logger.info(f'📁 No .dat file found in {folder.name}, checking next folder...')
    return input_files, folders_map

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description='Decode S1 L0 products to zarr format')
    parser.add_argument('--input', type=str, required=True, help='Path to .dat file or SAFE folder')
    return parser.parse_args()

def main() -> None:
    """
    Decodes Sentinel-1 Level-0 (.dat or .SAFE) files and saves the output to a specified directory.

    This function parses command-line arguments to determine the input path, validates the input,
    retrieves relevant files, and processes each file using the decode_s1_l0 function. The results
    are stored in the 'decoded_data' directory. It supports both single .dat files and .SAFE folders.

    Args:
        --input (str): Path to the input .dat file or .SAFE folder.

    Raises:
        AssertionError: If the input path does not exist or no valid input files are found.
        SystemExit: If the input is not a valid .dat file or .SAFE folder.
    Main function to decode S1 L0 files.

    Returns:
        None
    """
    args = parse_arguments()
    output_dir = Path('/Data_large/marine/PythonProjects/SAR/sarpyx/decoded_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    assert input_path.exists(), f'Input path does not exist: {input_path}'

    if input_path.is_file() and input_path.suffix == '.dat':
        input_files = [input_path]
        folders_map = {input_path.parent: [input_path]}
    elif input_path.is_dir() and input_path.name.endswith('.SAFE'):
        input_files, folders_map = retrieve_input_files([input_path], verbose=True)
    else:
        logger.error(f'❌ Invalid input: {input_path}. Must be a .dat file or .SAFE folder.')
        sys.exit(1)

    assert input_files, f'No valid input files found in: {input_path}'

    for input_file in input_files:
        if input_file.is_file():
            logger.info(f'🔍 Processing {input_file.name}...')
            decode_s1_l0(input_file, output_dir)
        else:
            logger.warning(f'❌ {input_file.name} is not a valid file, skipping...')

    logger.info(f'🔍 Processed {len(input_files)} files from {len(folders_map)} SAFE folders.')

if __name__ == '__main__':
    main()
