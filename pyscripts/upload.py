# pip install --upgrade huggingface_hub
from huggingface_hub import HfApi
import logging
import sys
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def upload_to_huggingface(folder_path: str) -> None:
    """Uploads a local folder to the Hugging Face Hub.

    Args:
        folder_path (str): Path to the local folder.

    Raises:
        Exception: If an error occurs during upload.
        AssertionError: If folder_path does not exist.
    """
    assert os.path.exists(folder_path), f'Path does not exist: {folder_path}'
    
    # Extract repo_id from folder name
    repo_id = 'sirbastiano94/Maya4'
    assert repo_id.strip(), 'Repository ID cannot be empty'
    
    api = HfApi()
    try:
        logging.info(f'Uploading folder "{folder_path}" to Hugging Face repo "{repo_id}" (dataset)...')
        api.upload_large_folder(folder_path=folder_path, repo_id=repo_id, repo_type='dataset')
        logging.info('Upload completed successfully.')
    except Exception as e:
        logging.error(f'Upload failed: {e}')
        raise

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python upload.py <folder_path>')
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    upload_to_huggingface(folder_path=folder_path)
