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
import threading
import time
import hashlib
from contextlib import contextmanager
import sys

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.n = 0
        
        def update(self, n):
            self.n += n
        
        def set_description(self, desc):
            self.desc = desc
            print(f"\r{desc}", end='', flush=True)
        
        def close(self):
            print()  # New line after completion

from utils import get_chunk_name_from_coords

# Global locks for thread-safe downloading
_download_locks = {}
_locks_lock = threading.Lock()


@contextmanager
def file_download_lock(file_path: str):
    """
    Thread-safe context manager for file downloads.
    Ensures only one thread downloads a specific file at a time.
    
    Args:
        file_path (str): Unique identifier for the file being downloaded
    """
    # Create a hash-based lock key to handle long paths
    lock_key = hashlib.md5(file_path.encode()).hexdigest()
    
    # Get or create lock for this file
    with _locks_lock:
        if lock_key not in _download_locks:
            _download_locks[lock_key] = threading.Lock()
        lock = _download_locks[lock_key]
    
    # Acquire the file-specific lock
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        # Clean up lock if no longer needed (optional optimization)
        with _locks_lock:
            if lock_key in _download_locks and not lock.locked():
                # Small delay to avoid race conditions
                time.sleep(0.001)
                if not lock.locked():
                    _download_locks.pop(lock_key, None)


def create_colored_progress_bar(desc: str, total: Optional[int] = None, color: str = 'blue'):
    """
    Create a colored progress bar that will be cleared after completion.
    
    Args:
        desc (str): Description for the progress bar
        total (Optional[int]): Total number of items (None for indeterminate)
        color (str): Color name for the progress bar
    
    Returns:
        tqdm: Configured progress bar object
    """
    # ANSI color codes
    colors = {
        'blue': '\033[34m',
        'green': '\033[32m', 
        'yellow': '\033[33m',
        'red': '\033[31m',
        'cyan': '\033[36m',
        'magenta': '\033[35m',
        'reset': '\033[0m'
    }
    
    color_code = colors.get(color.lower(), colors['blue'])
    reset_code = colors['reset']
    
    # Use a default total of 1 to avoid tqdm boolean evaluation issues
    # We'll update the total later when we know the actual file size
    display_total = total if total is not None else 1
    
    try:
        return tqdm(
            total=display_total,
            desc=f"{color_code}{desc}{reset_code}",
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False,  # This makes the bar disappear after completion
            bar_format=f"{color_code}{{desc}}: {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}]{{postfix}}{reset_code}",
            dynamic_ncols=True,
            position=0
        )
    except Exception:
        # Return a simple fallback progress bar that won't cause issues
        class SimpleFallbackProgressBar:
            def __init__(self):
                self.total = display_total
                self.n = 0
                self.desc = desc
            
            def update(self, n):
                self.n += n
            
            def set_description(self, desc):
                self.desc = desc
                print(f"\\r{desc}", end='', flush=True)
            
            def close(self):
                print()  # New line after completion
                
        return SimpleFallbackProgressBar()


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

def download_file_from_hf(repo_id: str, filename: str, local_dir: Union[str, os.PathLike], show_progress: bool = True) -> Path:
    """
    Download a specific file from a Hugging Face Hub dataset with thread-safe locking.

    Args:
        repo_id (str): Repository ID (e.g., 'sirbastiano94/Maya4').
        filename (str): Path to the file in the repository.
        local_dir (str): Local directory to save the file.
        show_progress (bool): Whether to show colored progress bar.

    Returns:
        Path: Path to the downloaded file.

    Raises:
        FileNotFoundError: If the file is not found in the repository or after download.
    """
    # Disable progress bars when running in DataLoader worker processes
    # to prevent tqdm issues in multiprocessing contexts
    try:
        import torch
        if torch.utils.data.get_worker_info() is not None:
            show_progress = False
    except (ImportError, AttributeError):
        pass
    
    lock_id = f"{repo_id}:{filename}"
    local_path = Path(local_dir) / filename
    
    # Check if file already exists and is complete
    if local_path.exists():
        if local_path.stat().st_size > 0:
            if show_progress:
                print(f'\\033[34m\\u2713 File "{filename}" already exists locally\\033[0m')
            return local_path
    # Use file-specific lock to prevent concurrent downloads of the same file
    with file_download_lock(lock_id):
        # Double-check if file was downloaded by another thread while waiting for lock
        if local_path.exists() and local_path.stat().st_size > 0:
            if show_progress:
                print(f'\\033[34m\\u2713 File "{filename}" was downloaded by another thread\\033[0m')
            return local_path
        
        # Create progress bar
        pbar = None
        if show_progress:
            try:
                pbar = create_colored_progress_bar(
                    desc=f'Downloading {os.path.basename(filename)}',
                    color='blue'
                )
            except Exception:
                # Fallback to simple print if progress bar creation fails
                print(f'\\033[34mDownloading {os.path.basename(filename)}...\\033[0m')
        
        try:
            if show_progress and pbar is not None:
                try:
                    pbar.set_description(f'\\033[34mDownloading {os.path.basename(filename)}\\033[0m')
                    pbar.update(0)
                except Exception:
                    # If progress bar fails, just continue without it
                    pass
            
            # Configure HuggingFace download with custom progress handling
            if show_progress:
                # Disable HF's built-in progress bar and use our custom one
                import os as os_module
                original_tqdm_disable = os_module.environ.get('TQDM_DISABLE', '')
                os_module.environ['TQDM_DISABLE'] = '1'
                try:
                    downloaded_file = hf_hub_download(
                        repo_id=repo_id,
                        repo_type='dataset',
                        filename=filename,
                        force_download=False,
                        local_dir_use_symlinks=True,
                        local_dir=str(local_dir), 
                        resume_download=True,
                        local_files_only=False,
                    )
                finally:
                    # Restore original TQDM_DISABLE setting
                    if original_tqdm_disable:
                        os_module.environ['TQDM_DISABLE'] = original_tqdm_disable
                    else:
                        os_module.environ.pop('TQDM_DISABLE', None)
            else:
                # Completely silent download - disable HF progress bars
                import os as os_module
                original_tqdm_disable = os_module.environ.get('TQDM_DISABLE', '')
                os_module.environ['TQDM_DISABLE'] = '1'
                try:
                    print(f"Trying to download file {filename} from repo {repo_id}")
                    downloaded_file = hf_hub_download(
                        repo_id=repo_id,
                        repo_type='dataset',
                        filename=filename,
                        force_download=False,
                        local_dir_use_symlinks=True,
                        local_dir=str(local_dir), 
                        resume_download=False,
                        local_files_only=False,
                    )
                finally:
                    # Restore original TQDM_DISABLE setting
                    if original_tqdm_disable:
                        os_module.environ['TQDM_DISABLE'] = original_tqdm_disable
                    else:
                        os_module.environ.pop('TQDM_DISABLE', None)
            
            downloaded_path = Path(downloaded_file)
            
            if not downloaded_path.exists():
                raise FileNotFoundError(f'File "{downloaded_path}" not found after download.')
            
            if show_progress and pbar is not None:
                try:
                    file_size = downloaded_path.stat().st_size if downloaded_path.exists() else 0
                    pbar.total = file_size
                    pbar.update(file_size)
                    pbar.set_description(f'\\033[34m\\u2713 Downloaded {os.path.basename(filename)}\\033[0m')
                    time.sleep(0.1)  # Brief pause to show completion
                    pbar.close()
                except Exception:
                    # If progress bar fails, show simple completion message
                    print(f'\\033[34m\\u2713 Downloaded {os.path.basename(filename)}\\033[0m')
            elif show_progress:
                print(f'\\033[34m\\u2713 Downloaded {os.path.basename(filename)}\\033[0m')
            
            return downloaded_path
            
        except Exception as e:
            if show_progress and pbar is not None:
                try:
                    pbar.set_description(f'\\033[31m\\u2717 Failed to download {os.path.basename(filename)}\\033[0m')
                    time.sleep(0.1)
                    pbar.close()
                except Exception:
                    pass
            if show_progress:
                print(f'\\033[31m\\u2717 Failed to download {os.path.basename(filename)}: {e}\\033[0m')
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
    
    import argparse
    parser = argparse.ArgumentParser(description='Download files from a Hugging Face Hub dataset.')
    parser.add_argument('--repo_id', type=str, default='sirbastiano94/Maya4', help='Repository ID (default: sirbastiano94/Maya4)')
    parser.add_argument('--path_in_repo', type=str, default='s1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05d3c0.zarr', 
                        help='Path in the repository to list files from (default: s1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05dc0.zarr)')      
    parser.add_argument('--filters', type=str, nargs='+', default=['rc', 'az'],
                        help='List of modalities to filter by (default: ["rc", "az"])')
    parser.add_argument('--local_dir', type=str, default=None, help='Local directory to download files to (default: None, current directory)')
    return parser.parse_args()




def download_wrapper(file: huggingface_hub.hf_api.RepoFile, repo_id: str, output_dir: Optional[str], show_progress: bool = True) -> None:
    """
    Wrapper function to download a file using the Hugging Face Hub with thread safety.

    Args:
        file (RepoFile): File object to download.
        repo_id (str): Repository ID.
        output_dir (Optional[str]): Local directory to save the file.
        show_progress (bool): Whether to show colored progress bars.

    Returns:
        None
    """
    filename = file.path
    local_dir = output_dir or "."
    try:
        download_file_from_hf(repo_id, filename, local_dir, show_progress=show_progress)
    except FileNotFoundError as e:
        if show_progress:
            print(f'\033[31mError downloading {filename}: {e}\033[0m')
        else:
            print(f'Error: {e}')

def down(repo_id: str = 'sirbastiano94/Maya4', 
         path_in_repo: str = 's1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05d3c0.zarr', 
         filters: list = ['rc', 'az'], 
         output_dir: Optional[str] = None,
         show_progress: bool = True,
         max_workers: Optional[int] = None) -> None:
    """
    Main function to filter files by modalities and download them from the Hugging Face Hub.
    Uses thread-safe downloading to prevent concurrent downloads of the same files.

    Args:
        repo_id (str): Repository ID.
        path_in_repo (str): Path in the repository.
        filters (list): List of modalities to filter by.
        output_dir (Optional[str]): Local directory to download files to.
        show_progress (bool): Whether to show colored progress bars.
        max_workers (Optional[int]): Maximum number of worker processes for parallel downloads.

    Returns:
        None

    Raises:
        AssertionError: If filters is not a list.
    """
    # List files in the repository
    files = list_files_in_repo(repo_id, path_in_repo, filters)
    
    if show_progress:
        print(f'\033[36mFound {len(files)} files matching the filter criteria\033[0m')
    else:
        print(f'Found {len(files)} files matching the filter criteria.')

    # Download filtered files
    if len(files) > 1 and max_workers != 1:
        # Use multiprocessing for multiple files, but each process will use thread-safe downloading
        with Pool(processes=max_workers) as pool:
            pool.starmap(download_wrapper, [(file, repo_id, output_dir, show_progress) for file in files])
    else:
        # Sequential download for single files or when max_workers=1
        for file in files:
            download_wrapper(file, repo_id, output_dir, show_progress)

def download_metadata(
    repo_id: str = 'sirbastiano94/Maya4', 
    zarr_archive: str = 's1a-s1-raw-s-hh-20240130t151239-20240130t151254-052337-06541b.zarr', 
    base_dir: str = "",
    local_dir: Optional[Union[str, os.PathLike]] = None, 
    download_all: bool = True,
    show_progress: bool = True
) -> Path:
    """
    Download metadata files from a Zarr archive on HuggingFace, only if not found locally.
    Uses thread-safe downloading to prevent concurrent downloads of the same files.

    Args:
        repo_id (str): HuggingFace repo id.
        zarr_archive (str): Path to the Zarr archive in the repo.
        base_dir (str): Base directory within the Zarr archive to look for metadata files.
        local_dir (str): Local directory for temporary storage.
        download_all (bool): If True, download all metadata files. If False, only download the main metadata file.
        show_progress (bool): Whether to show colored progress bars.

    Returns:
        Path: Path to the main metadata file if found/downloaded.
        
    Raises:
        FileNotFoundError: If no metadata file is found.
    """
    # Handle None local_dir
    if local_dir is None:
        local_dir = "."
    
    local_dir_path = Path(local_dir)
    
    if base_dir == "":
        base_remote_metadata_path = f"{zarr_archive}"
        base_local_metadata_path = local_dir_path / zarr_archive
    else:
        base_remote_metadata_path = f"{zarr_archive}/{base_dir}"
        base_local_metadata_path = local_dir_path / zarr_archive / base_dir
    
    meta_candidates = ['.zgroup', '.zarray', 'zarr.json']
    misc_metadata = ['.zattrs']
    
    # Use thread-safe listing (this is generally safe as it's read-only)
    files = list_base_files_in_repo(repo_id, path_in_repo=base_remote_metadata_path, relative_path=True)
    
    if show_progress:
        print(f"\033[36mChecking metadata in {base_remote_metadata_path}\033[0m")
    
    # Download additional metadata if not found locally
    if download_all:
        for meta in misc_metadata:
            meta_path = base_local_metadata_path / meta
            if meta in files and not meta_path.exists():
                download_file_from_hf(repo_id, f"{base_remote_metadata_path}/{meta}", local_dir_path, show_progress=show_progress)
    
    meta_found = False
    meta_path = None
    
    for meta in meta_candidates:
        if meta in files:
            meta_path = base_local_metadata_path / meta
            if not meta_path.exists():
                download_file_from_hf(repo_id, f"{base_remote_metadata_path}/{meta}", local_dir_path, show_progress=show_progress)
            meta_found = True
            break  # Return the first found metadata file
    
    if meta_found and meta_path is not None:
        return meta_path
    
    raise FileNotFoundError(f"No metadata file (.zarray or zarr.json) found in {base_remote_metadata_path}.")

def download_metadata_from_product(
    zfile_name: str = 's1a-s1-raw-s-hh-20240130t151239-20240130t151254-052337-06541b.zarr',
    local_dir: Union[str, os.PathLike] = 'data',
    repo_id: str = 'sirbastiano94/Maya4',
    levels: List[str] = ['raw', 'rc', 'rcmc', 'az'],
    show_progress: bool = True
) -> os.PathLike:
    """
    Download metadata files from a Zarr archive on HuggingFace, only if not found locally.
    Uses thread-safe downloading to prevent concurrent downloads of the same files.

    Args:
        zfile_name (str): Name of the Zarr file.
        local_dir (Union[str, os.PathLike]): Local directory for temporary storage.
        repo_id (str): HuggingFace repo id.
        levels (List[str]): List of processing levels to download metadata for.
        show_progress (bool): Whether to show colored progress bars.

    Returns:
        Path: Path to the downloaded metadata files.
    """
    # Download main metadata
    meta_file_path = download_metadata(
        repo_id=repo_id,
        zarr_archive=zfile_name,
        local_dir=local_dir,
        base_dir='',
        show_progress=show_progress
    )
    
    # Download metadata for each level
    for level in levels:
        meta_file_path = download_metadata(
            repo_id=repo_id,
            zarr_archive=zfile_name,
            local_dir=local_dir, 
            base_dir=level,
            show_progress=show_progress
        )
    
    return meta_file_path


def fetch_chunk_from_hf_zarr(
    level: str, 
    y: int, 
    x: int, 
    local_dir: Union[str, os.PathLike],
    repo_id: str = 'sirbastiano94/Maya4', 
    zarr_archive: str = 's1a-s1-raw-s-hh-20240130t151239-20240130t151254-052337-06541b.zarr',
    show_progress: bool = False  # Default to False for dataloader usage
) -> Path:
    """
    Download only the chunk containing (y, x) from a Zarr archive on HuggingFace.
    Uses thread-safe downloading to prevent concurrent downloads of the same chunks.

    Args:
        repo_id (str): HuggingFace repo id.
        zarr_archive (str): Path to the Zarr archive in the repo.
        level (str): Zarr group/array name (e.g., 'rcmc', 'az').
        y (int): Y coordinate.
        x (int): X coordinate.
        local_dir (str): Local directory for temporary storage.
        show_progress (bool): Whether to show colored progress bars.

    Returns:
        str: path to the downloaded chunk file.
    """
    # Download metadata with thread safety
    download_metadata(repo_id=repo_id, zarr_archive=zarr_archive, local_dir=local_dir, show_progress=show_progress)
    zarray_meta_file = download_metadata(
        repo_id=repo_id, 
        zarr_archive=zarr_archive, 
        base_dir=level, 
        local_dir=local_dir,
        show_progress=show_progress
    )
    
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

    # Download chunk file with thread safety
    chunk_file = download_file_from_hf(
        repo_id,
        chunk_fname,
        local_dir,
        show_progress=show_progress
    )

    return chunk_file


if __name__ == '__main__':
    down()