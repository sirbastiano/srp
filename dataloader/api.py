# dataloader/api.py 
"""
This module provides functions to filter files by modalities and download files from the Hugging Face Hub.
It includes functions to filter files based on specified modalities and to download specific files from a dataset repository on the Hugging Face Hub.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_tree
import huggingface_hub
from multiprocessing import Pool
from typing import Optional


def filter_files_by_modalities(files: list, filters: list) -> list:
    """
    Filter files by multiple modalities.

    Args:
        files (list): List of RepoFile objects.
        filters (list): List of modality strings to filter by.

    Returns:
        list: Filtered list of RepoFile objects.

    Raises:
        AssertionError: If filters is not a list.
    """
    assert isinstance(filters, list), 'filters must be a list'
    return [x for x in files if any(f in x.path for f in filters)]

def download_file_from_hf(repo_id: str, filename: str, local_dir: str) -> Path:
    """
    Download a specific file from a Hugging Face Hub dataset.

    Args:
        repo_id (str): Repository ID (e.g., 'sirbastiano94/Maya4').
        filename (str): Path to the file in the repository.
        local_dir (str): Local directory to save the file.

    Returns:
        Path: Path to the downloaded file.

    Raises:
        FileNotFoundError: If the file is not found in the repository or after download.
    """
    print(f'Downloading "{filename}" from "{repo_id}" to "{local_dir}"...')
    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=filename,
            force_download=True,
            local_dir_use_symlinks=False,
            local_dir=local_dir,
        )
        downloaded_path = Path(downloaded_file)
        assert downloaded_path.exists(), f'File "{downloaded_path}" not found after download.'
        print(f'Successfully downloaded "{filename}" to "{downloaded_path}".')
        return downloaded_path
    except Exception as e:
        print(f'Download failed: {e}')
        raise

def list_files_in_repo(repo_id: str, path_in_repo: str, filters: list) -> list:
    """
    List all files in a Hugging Face Hub repository.

    Args:
        repo_id (str): Repository ID (e.g., 'sirbastiano94/Maya4').
        path_in_repo (str): Path in the repository to list files from.
        filters (list): List of modality strings to filter by. Allowed: ['raw', 'rc', 'rcmc', 'az']

    Returns:
        list: List of RepoFile objects.

    Raises:
        AssertionError: If filters is not a list or contains invalid modalities.
    """
    
    assert isinstance(filters, list), 'filters must be a list'
    allowed_modalities = ['raw', 'rc', 'rcmc', 'az']
    assert all(f in allowed_modalities for f in filters), f'Each filter must be one of {allowed_modalities}'
    
    tree = list_repo_tree(
        repo_id=repo_id,
        repo_type='dataset',
        path_in_repo=path_in_repo,
        recursive=True,
        ) # type: ignore
    files = [x for x in tree if isinstance(x, huggingface_hub.hf_api.RepoFile)]
    filtered_files = filter_files_by_modalities(files, filters)
    return filtered_files



def parser():    
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    
    import argparse
    parser = argparse.ArgumentParser(description='Download files from a Hugging Face Hub dataset.')
    parser.add_argument('--repo_id', type=str, default=__REPO_ID, help='Repository ID (default: sirbastiano94/Maya4)')
    parser.add_argument('--path_in_repo', type=str, default='s1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05d3c0.zarr', 
                        help='Path in the repository to list files from (default: s1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05dc0.zarr)')      
    parser.add_argument('--filters', type=str, nargs='+', default=['rc', 'az'],
                        help='List of modalities to filter by (default: ["rc", "az"])')
    parser.add_argument('--local_dir', type=str, default=None, help='Local directory to download files to (default: None, current directory)')
    return parser.parse_args()




def download_wrapper(file: huggingface_hub.hf_api.RepoFile, repo_id: str, output_dir: Optional[str]) -> None:
    """
    Wrapper function to download a file using the Hugging Face Hub.

    Args:
        file (RepoFile): File object to download.
        repo_id (str): Repository ID.
        output_dir (Optional[str]): Local directory to save the file.

    Returns:
        None
    """
    filename = file.path
    try:
        download_file_from_hf(repo_id, filename, output_dir)
    except FileNotFoundError as e:
        print(f'Error: {e}')

def down(repo_id: str = 'sirbastiano94/Maya4', 
         path_in_repo: str = 's1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05d3c0.zarr', 
         filters: list = ['rc', 'az'], 
         output_dir: Optional[str] = None) -> None:
    """
    Main function to filter files by modalities and download them from the Hugging Face Hub.

    Args:
        repo_id (str): Repository ID.
        path_in_repo (str): Path in the repository.
        filters (list): List of modalities to filter by.
        output_dir (Optional[str]): Local directory to download files to.

    Returns:
        None

    Raises:
        AssertionError: If filters is not a list.
    """
    # List files in the repository
    files = list_files_in_repo(repo_id, path_in_repo, filters)
    print(f'Found {len(files)} files matching the filter criteria.')

    # Download filtered files, use multiprocessing if more than one file
    if len(files) > 1:
        with Pool() as pool:
            pool.starmap(download_wrapper, [(file, repo_id, output_dir) for file in files])
    else:
        for file in files:
            download_wrapper(file, repo_id, output_dir)



if __name__ == '__main__':
    down()