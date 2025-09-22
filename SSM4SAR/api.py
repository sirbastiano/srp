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
    # Normalize path for current OS and ensure it exists
    normalized_local_dir = Path(local_dir).resolve()
    normalized_local_dir.mkdir(parents=True, exist_ok=True)
    
    # Force POSIX-style paths for HuggingFace Hub on any OS
    import os
    import tempfile
    
    # Set environment variables to force proper path handling
    original_env = {}
    force_env_vars = {
        'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
        'HF_HUB_CACHE': str(normalized_local_dir / '.cache' / 'huggingface'),
        'HF_HOME': str(normalized_local_dir / '.cache' / 'huggingface'),
        'TMPDIR': str(normalized_local_dir / '.tmp'),
        'TMP': str(normalized_local_dir / '.tmp'),
        'TEMP': str(normalized_local_dir / '.tmp')
    }
    
    # Store original values and set new ones
    for key, value in force_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    # Create all necessary directories
    cache_dir = normalized_local_dir / '.cache' / 'huggingface' / 'download'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    tmp_dir = normalized_local_dir / '.tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create the specific zarr archive directory that HF will use
    filename_parts = filename.split('/')
    if len(filename_parts) > 1 and filename_parts[0].endswith('.zarr'):
        zarr_cache_dir = cache_dir / filename_parts[0]
        zarr_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Downloading "{filename}" from "{repo_id}" to "{normalized_local_dir}"...')
    print(f'Cache directory: {cache_dir}')
    print(f'Temp directory: {tmp_dir}')
    
    # Force platform-specific path handling
    import platform
    if platform.system() == 'Linux':
        # Additional Linux-specific environment variables to prevent Windows paths
        os.environ['PATHSEP'] = ':'
        os.environ['PLATFORM'] = 'linux'
        # Unset any Windows-related variables that might exist
        windows_vars = ['WINDIR', 'SYSTEMROOT', 'COMSPEC', 'PATHEXT']
        for var in windows_vars:
            if var in os.environ:
                del os.environ[var]
    
    try:
        # First try: Download directly to local_dir without explicit cache_dir
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=filename,
            force_download=False,
            local_dir_use_symlinks=False,
            local_dir=str(normalized_local_dir), 
            resume_download=True
            # Note: Removed cache_dir to let HF use default behavior
        )
        
        downloaded_path = Path(downloaded_file)
        assert downloaded_path.exists(), f'File "{downloaded_path}" not found after download.'
        print(f'Successfully downloaded "{filename}" to "{downloaded_path}".')
        return downloaded_path
        
    except Exception as e:
        print(f'Download failed for "{filename}": {e}')
        print(f'Local directory: {normalized_local_dir}')
        print(f'Directory exists: {normalized_local_dir.exists()}')
        print(f'Cache directory: {cache_dir}')
        print(f'Cache directory exists: {cache_dir.exists()}')
        
        # Try to create any missing directories in the error path
        error_str = str(e)
        if "No such file or directory" in error_str and ("incomplete" in error_str or ".tmp" in error_str):
            # Extract the problematic path from the error message
            if "'" in error_str:
                problem_path_str = error_str.split("'")[1]
                # Clean up Windows-style prefixes aggressively
                problem_path_str = problem_path_str.replace('\\\\?\\', '')
                problem_path_str = problem_path_str.replace('\\', '/')
                if problem_path_str.startswith('//'):
                    problem_path_str = problem_path_str[1:]
                
                print(f'Cleaned problematic path: {problem_path_str}')
                if problem_path_str.startswith('/'):
                    problem_path = Path(problem_path_str)
                    if not problem_path.parent.exists():
                        print(f'Creating missing directory: {problem_path.parent}')
                        problem_path.parent.mkdir(parents=True, exist_ok=True)
                        
                    # Retry the download once with explicit cache directory
                    print('Retrying download after creating missing directory...')
                    try:
                        # Try with explicit cache_dir
                        downloaded_file = hf_hub_download(
                            repo_id=repo_id,
                            repo_type='dataset',
                            filename=filename,
                            force_download=False,
                            local_dir_use_symlinks=False,
                            local_dir=str(normalized_local_dir), 
                            resume_download=True,
                            cache_dir=str(cache_dir)
                        )
                        
                        downloaded_path = Path(downloaded_file)
                        assert downloaded_path.exists(), f'File "{downloaded_path}" not found after retry.'
                        print(f'Successfully downloaded "{filename}" on retry to "{downloaded_path}".')
                        return downloaded_path
                    except Exception as retry_e:
                        print(f'Retry with explicit cache also failed: {retry_e}')
                        # Third attempt: force download without resume
                        try:
                            print('Final attempt: force download without resume...')
                            downloaded_file = hf_hub_download(
                                repo_id=repo_id,
                                repo_type='dataset',
                                filename=filename,
                                force_download=True,
                                local_dir_use_symlinks=False,
                                local_dir=str(normalized_local_dir), 
                                resume_download=False
                            )
                            
                            downloaded_path = Path(downloaded_file)
                            assert downloaded_path.exists(), f'File "{downloaded_path}" not found after final attempt.'
                            print(f'Successfully downloaded "{filename}" with force download to "{downloaded_path}".')
                            return downloaded_path
                        except Exception as final_e:
                            print(f'All download attempts failed. Final error: {final_e}')
                            raise final_e
        
        raise e
    
    finally:
        # Restore original environment variables
        for key, original_value in original_env.items():
            if original_value is not None:
                os.environ[key] = original_value
            elif key in os.environ:
                del os.environ[key]


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
def list_repos_by_author(author: str) -> list:
    """
    List all dataset repositories by a specific author on Hugging Face Hub.

    Args:
        author (str): Author's username.

    Returns:
        list: List of repository names (str).
    """
    from huggingface_hub import HfApi
    api = HfApi()
    repos = api.list_datasets(author=author)
    repo_names = [repo.id for repo in repos]
    return repo_names
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
    parser.add_argument('--repo_id', type=str, default='sirbastiano94/Maya4', help='Repository ID (default: sirbastiano94/Maya4)')
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
        # Handle None output_dir case
        local_dir = output_dir if output_dir is not None else Path.cwd() / 'downloads'
        download_file_from_hf(repo_id, filename, local_dir)
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
    
    # Normalize local directory path
    if local_dir is None:
        local_dir = Path.cwd() / 'downloads'
    normalized_local_dir = Path(local_dir).resolve()
    normalized_local_dir.mkdir(parents=True, exist_ok=True)
    
    if base_dir == "":
        base_remote_metadata_path = f"{zarr_archive}"
        base_local_metadata_path = normalized_local_dir / zarr_archive
    else:
        base_remote_metadata_path = f"{zarr_archive}/{base_dir}"
        base_local_metadata_path = normalized_local_dir / zarr_archive / base_dir
    
    # Ensure local metadata path exists
    base_local_metadata_path.mkdir(parents=True, exist_ok=True)
    
    meta_candidates = ['.zgroup', '.zarray', 'zarr.json']
    misc_metadata = ['.zattrs']
    files = list_base_files_in_repo(repo_id, path_in_repo=base_remote_metadata_path, relative_path=True)
    print(f"Files in {base_remote_metadata_path}: {files}")
    # Download additional metadata if not found locally
    if download_all:
        for meta in misc_metadata:
            meta_path = base_local_metadata_path / meta
            if meta in files and not meta_path.exists():
                download_file_from_hf(repo_id, f"{base_remote_metadata_path}/{meta}", normalized_local_dir)
    meta_found = False
    for meta in meta_candidates:
        if meta in files:
            meta_path = base_local_metadata_path / meta
            if not meta_path.exists():
                download_file_from_hf(repo_id, f"{base_remote_metadata_path}/{meta}", normalized_local_dir)
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
    local_dir: Union[str, os.PathLike],
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
        Path: path to the downloaded chunk file.
        
    Raises:
        FileNotFoundError: If metadata or chunk files cannot be downloaded.
        ValueError: If metadata format is invalid.
    """
    import json
    
    try:
        # Download base metadata first
        download_metadata(repo_id=repo_id, zarr_archive=zarr_archive, local_dir=local_dir)
        
        # Download level-specific metadata
        zarray_meta_file = download_metadata(
            repo_id=repo_id, 
            zarr_archive=zarr_archive, 
            base_dir=level, 
            local_dir=local_dir
        )
        
        # Parse metadata to get chunking information
        with open(zarray_meta_file, 'r') as f:
            zarr_meta = json.load(f)
            
        if 'shape' not in zarr_meta:
            raise ValueError(f"Invalid zarr metadata: missing 'shape' field in {zarray_meta_file}")
            
        shape = zarr_meta['shape']

        # Handle different zarr format versions
        zarr_format = zarr_meta.get('zarr_format', 2)
        if zarr_format == 3:
            if 'chunk_grid' not in zarr_meta or 'configuration' not in zarr_meta['chunk_grid']:
                raise ValueError(f"Invalid zarr v3 metadata: missing chunk_grid configuration in {zarray_meta_file}")
            chunks = zarr_meta['chunk_grid']['configuration']['chunk_shape']
        else:
            if 'chunks' not in zarr_meta:
                raise ValueError(f"Invalid zarr v2 metadata: missing 'chunks' field in {zarray_meta_file}")
            chunks = zarr_meta['chunks']

        # Validate coordinates are within bounds
        if y < 0 or y >= shape[0] or x < 0 or x >= shape[1]:
            raise ValueError(f"Coordinates ({y}, {x}) are out of bounds for shape {shape}")

        # Compute chunk indices for (y, x)
        chunk_fname = get_chunk_name_from_coords(
            y=y, 
            x=x,  
            zarr_file_name=zarr_archive, 
            level=level, 
            chunks=chunks, 
            version=zarr_format
        )

        # Download chunk file with retry mechanism
        try:
            chunk_file = download_file_from_hf(
                repo_id,
                chunk_fname,
                local_dir
            )
        except Exception as download_error:
            raise FileNotFoundError(
                f"Failed to download chunk {chunk_fname} for coordinates ({y}, {x}) "
                f"from {repo_id}/{zarr_archive}/{level}: {download_error}"
            ) from download_error

        return chunk_file
        
    except Exception as e:
        print(f"Error in fetch_chunk_from_hf_zarr: {e}")
        raise


if __name__ == '__main__':
    down()