from huggingface_hub import snapshot_download, delete_file, delete_folder, list_repo_tree, file_download
import os
from pathlib import Path
import argparse


def download_folder_from_hf(
    repo_id: str, 
    folder_name: str, 
    local_dir: str = '/Users/roberto.delprete/Library/CloudStorage/OneDrive-ESA/Desktop/Repos/SARPYX/data',
    repo_type: str = 'dataset'
) -> str:
    """Download a specific folder from Hugging Face Hub repository.
    
    Args:
        repo_id (str): The repository ID (e.g., 'username/repo-name').
        folder_name (str): The name of the folder to download.
        local_dir (str): Local directory where to save the folder. Defaults to './downloads'.
        repo_type (str): Type of repository ('dataset', 'model', 'space'). Defaults to 'dataset'.
    
    Returns:
        str: Path to the downloaded folder.
        
    Raises:
        ValueError: If repo_id or folder_name is empty.
        Exception: If download fails.
    """
    assert repo_id.strip(), 'repo_id cannot be empty'
    assert folder_name.strip(), 'folder_name cannot be empty'
    
    try:
        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        print(f'Downloading folder "{folder_name}" from repository "{repo_id}"...')
        
        # Download the specific folder
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=f'{folder_name}/**',
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        folder_path = os.path.join(downloaded_path, folder_name)
        print(f'Successfully downloaded to: {folder_path}')
        
        return folder_path
        
    except Exception as e:
        print(f'Error downloading folder: {str(e)}')
        raise




def parse_arguments() -> argparse.Namespace:
    """
        Parse command-line arguments for downloading a folder from a Hugging Face Hub repository.

        This function defines and parses the following command-line arguments:
        - `--repo_id` (str, required): The repository ID in the format "username/repo-name".
        - `--folder_name` (str, required): The name of the folder to download from the repository.
        - `--local_dir` (str, optional): The local directory where the folder will be saved. Defaults to './downloads'.
        - `--repo_type` (str, optional): The type of repository. Can be one of 'dataset', 'model', or 'space'. Defaults to 'dataset'.

            argparse.Namespace: An object containing the parsed arguments as attributes.
        Parse command-line arguments.

        Returns:
            argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Download a folder from a Hugging Face Hub repository.')
    parser.add_argument('--repo_id', type=str, required=True, default='sirbastiano94/Maya4' ,help='The repository ID (e.g., "username/repo-name").')
    parser.add_argument('--folder_name', type=str, required=True, help='The name of the folder to download.')
    parser.add_argument('--local_dir', type=str, default='./downloads', help='Local directory where to save the folder.')
    parser.add_argument('--repo_type', type=str, default='dataset', choices=['dataset', 'model', 'space'], help='Type of repository.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Download the specific folder using parsed arguments
    downloaded_folder_path = download_folder_from_hf(
        repo_id=args.repo_id,
        folder_name=args.folder_name,
        local_dir=args.local_dir,
        repo_type=args.repo_type
    )

    print(f'Folder downloaded to: {downloaded_folder_path}')
