# ===== Imports =====
import os, sys
import logging
from pathlib import Path
import zipfile

# ===== Setup Logging =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unzip_files.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== Helpers =====
output_dir = '/Data_large/marine/PythonProjects/SAR/sarpyx/extracted_data'

def extract_zip(zip_path, extract_to):
    """Extracts a zip file to the specified directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully extracted {zip_path} to {extract_to}")
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        

if __name__ == '__main__':
    # ===== Setup =====
    locations = os.listdir('../../Data')
    path = Path('../../Data')
    logger.info(f'Found {len(locations)} locations in {path}')

    # ===== Main Code =====
    for idx, location in enumerate(locations):
        if location.startswith('.'):
            continue
        if not os.path.isdir(path / location):
            continue
        files = os.listdir(path / location)
        
        for file in files:
            logger.info(f'Processing {idx+1}/{len(locations)}: {location} - {file}')
            if file.endswith('.zip'):
                # Extract each file to output_dir/location/filename/
                zip_path = path / location / file
                extract_location = Path(output_dir) / location
                extract_location.mkdir(parents=True, exist_ok=True)
                extract_zip(str(zip_path), str(extract_location))
            
            else:
                continue
        logger.info(f'Finished processing {location}')
    logger.info('All files processed.')
    sys.exit(0)        