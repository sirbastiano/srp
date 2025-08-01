# dataloader/api.py 
"""
This module provides functions to filter files by modalities and download files from the Hugging Face Hub.
It includes functions to filter files based on specified modalities and to download specific files from a dataset repository on the Hugging Face Hub.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_tree
from huggingface_hub.hf_api import RepoFile
import huggingface_hub
from multiprocessing import Pool
from typing import List, Optional, Union
import numpy as np

from utils import get_chunk_name_from_coords


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

def download_file_from_hf(repo_id: str, filename: str, local_dir: Union[str, os.PathLike]) -> Path:
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
            local_dir=str(local_dir)
        )
        downloaded_path = Path(downloaded_file)
        assert downloaded_path.exists(), f'File "{downloaded_path}" not found after download.'
        print(f'Successfully downloaded "{filename}" to "{downloaded_path}".')
        return downloaded_path
    except Exception as e:
        print(f'Download failed: {e}')
        raise
def list_base_files_in_repo(repo_id: str, path_in_repo: str= "", relative_path: bool=False) -> list:
    """
    Efficiently list file names in the base directory of a Hugging Face dataset repository.

    Args:
        repo_id (str): Repository ID (e.g., 'sirbastiano94/Maya4').
        path_in_repo (str): Path in the repository to list files from. Defaults to empty string for base directory.
        relative_path (bool): If True, return only file names (not full paths).

    Returns:
        list: List of file and folder names (str) in the base directory.
    """
    tree = list_repo_tree(
        repo_id=repo_id,
        repo_type='dataset',
        path_in_repo=path_in_repo,
        recursive=False,
    )
    names = [x.path for x in tree if hasattr(x, 'path')]
    if relative_path:
        names = [os.path.basename(name) for name in names]
    return names

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

def download_metadata(
    repo_id: str = 'sirbastiano94/Maya4', 
    zarr_archive: str = 's1a-s1-raw-s-hh-20240130t151239-20240130t151254-052337-06541b.zarr', 
    base_dir: str = "",
    local_dir: Optional[str|os.PathLike] = None, 
    download_all: bool = True
) -> Path:
    """
    Download metadata files from a Zarr archive on HuggingFace, only if not found locally.

    Args:
        repo_id (str): HuggingFace repo id.
        zarr_archive (str): Path to the Zarr archive in the repo.
        base_dir (str): Base directory within the Zarr archive to look for metadata files.
        local_dir (str): Local directory for temporary storage.
        download_all (bool): If True, download all metadata files. If False, only download the main metadata file.

    Returns:
        Path or None: Path to the main metadata file if found/downloaded, else None.
    """
    
    if base_dir == "":
        base_remote_metadata_path = f"{zarr_archive}"
        base_local_metadata_path = Path(local_dir) / zarr_archive
    else:
        base_remote_metadata_path = f"{zarr_archive}/{base_dir}"
        base_local_metadata_path = Path(local_dir) / zarr_archive / base_dir
    
    meta_candidates = ['.zgroup', '.zarray', 'zarr.json']
    misc_metadata = ['.zattrs']
    files = list_base_files_in_repo(repo_id, path_in_repo=base_remote_metadata_path, relative_path=True)
    print(f"Files in {base_remote_metadata_path}: {files}")
    # Download additional metadata if not found locally
    if download_all:
        for meta in misc_metadata:
            meta_path = base_local_metadata_path / meta
            if meta in files and not meta_path.exists():
                download_file_from_hf(repo_id, f"{base_remote_metadata_path}/{meta}", Path(local_dir))
    meta_found = False
    for meta in meta_candidates:
        if meta in files:
            meta_path = base_local_metadata_path / meta
            if not meta_path.exists():
                download_file_from_hf(repo_id, f"{base_remote_metadata_path}/{meta}", Path(local_dir))
            meta_found = True
    if meta_found:
        return meta_path
    raise FileNotFoundError(f"No metadata file (.zarray or zarr.json) found in {base_remote_metadata_path}.")

def download_metadata_from_product(
    zfile_name: str = 's1a-s1-raw-s-hh-20240130t151239-20240130t151254-052337-06541b.zarr',
    local_dir: Union[str, os.PathLike] = 'data',
    repo_id: str = 'sirbastiano94/Maya4',
    levels: List[str] = ['raw', 'rc', 'rcmc', 'az']
) -> os.PathLike:
    """
    Download metadata files from a Zarr archive on HuggingFace, only if not found locally.

    Args:
        zfile_name (str): Name of the Zarr file.
        local_dir (Union[str, os.PathLike]): Local directory for temporary storage.
        repo_id (str): HuggingFace repo id.
        zarr_archive (str): Path to the Zarr archive in the repo.

    Returns:
        Path: Path to the downloaded metadata files.
    """
    meta_file_path = download_metadata(
        repo_id=repo_id,
        zarr_archive=zfile_name,
        local_dir=local_dir,
        base_dir=''
    )
    for level in levels:
        meta_file_path = download_metadata(
            repo_id=repo_id,
            zarr_archive=zfile_name,
            local_dir=local_dir, 
            base_dir=level
        )
    return meta_file_path


def fetch_chunk_from_hf_zarr(
    level: str, 
    y: int, 
    x: int, 
    local_dir:Union[str, os.PathLike],
    repo_id: str = 'sirbastiano94/Maya4', 
    zarr_archive: str = 's1a-s1-raw-s-hh-20240130t151239-20240130t151254-052337-06541b.zarr'
    ) -> Path:
    """
    Download only the chunk containing (y, x) from a Zarr archive on HuggingFace.

    Args:
    repo_id (str): HuggingFace repo id.
    zarr_archive (str): Path to the Zarr archive in the repo.
    level (str): Zarr group/array name (e.g., 'rcmc', 'az').
    y (int): Y coordinate.
    x (int): X coordinate.
    local_dir (str): Local directory for temporary storage.

    Returns:
    str: path to the downloaded chunk file.
    """
    # List all files in the zarr archive directory, excluding zarr metadata files
    download_metadata(repo_id = repo_id,zarr_archive = zarr_archive, local_dir = local_dir)
    zarray_meta_file = download_metadata(repo_id = repo_id,zarr_archive = zarr_archive, base_dir = level, local_dir = local_dir)
    import json
    with open(zarray_meta_file) as f:
        zarr_meta = json.load(f)
    shape = zarr_meta['shape']

    if zarr_meta.get('zarr_format', 2) == 3:
        chunks = zarr_meta['chunk_grid']['configuration']['chunk_shape']
    else:
        chunks = zarr_meta['chunks']

    # Compute chunk indices for (y, x)
    chunk_fname = get_chunk_name_from_coords(
        y=y, 
        x=x,  
        zarr_file_name=zarr_archive, 
        level=level, 
        chunks=chunks, 
        version=zarr_meta.get('zarr_format', 2)
    )

    # Download chunk file
    chunk_file = download_file_from_hf(
        repo_id,
        chunk_fname,
        local_dir
    )

    return chunk_file


if __name__ == '__main__':
    down()