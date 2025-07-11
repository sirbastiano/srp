# write a script to unzip the downloaded file
# the zip files are in Data/RAW 
# each filename starts with "S1"
import os
import zipfile
from pathlib import Path
import logging
import argparse 


# set up logging
# Create a logger with the name "myapp"
logger = logging.getLogger("Unzipper")

# Set the logging level to INFO
logger.setLevel(logging.INFO)

# Create a console handler and set its logging level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Define a formatter for the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

def main(folder_path: str):
    folder_path = args.i
    print('Folder selected:', folder_path)
    zip_dir = Path(folder_path)
    # get a list of all the zip files
    zipFiles = list(zip_dir.iterdir())
    zipFiles = [x for x in zipFiles if x.is_file()]
    zipFiles = [x for x in zipFiles if x.name.startswith("S1")]
    assert len(zipFiles) > 0, f"No zip files found in {zip_dir}"
    logger.info(f"Found {len(zipFiles)} zip files in {zip_dir}")
    
    for f in zipFiles:
        logger.info (f"Unzipping: {f}")
        with zipfile.ZipFile(f, 'r') as zip_ref:
            logger.info(f"Extracting {f} to {zip_dir}")
            zip_ref.extractall(zip_dir)
        
        # remove the zip file
        logger.info (f"Removing zipfile: {f}")
        os.remove(f)
        
if __name__ == "__main__":
    """ This is executed when run from the command line:
        python -m utils.downloader.unzipper
    """
    # Create a file handler only when running as main
    file_handler = logging.FileHandler("unzip.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set search parameters with argparse
    parser = argparse.ArgumentParser()
    # create a wkt with: https://wktmap.com/
    parser.add_argument("-i", help="Folder Path", default=None)
    
    args = parser.parse_args()
    
    main(args)