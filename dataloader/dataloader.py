import math
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
import numpy as np

from zarr.storage import LocalStore
import zarr

from typing import List, Tuple, Dict, Optional, Union, Callable
import json
import pandas as pd
import dask.array as da
import time 
import os
import functools
import math
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from utils import get_chunk_name_from_coords, get_part_from_filename, get_sample_visualization, get_zarr_version, parse_product_filename, get_balanced_sample_files, SampleFilter
from api import list_base_files_in_repo, list_repos_by_author
from utils import minmax_normalize, minmax_inverse, extract_stripmap_mode_from_filename, RC_MAX, RC_MIN, GT_MAX, GT_MIN
from api import fetch_chunk_from_hf_zarr, download_metadata_from_product
import matplotlib.pyplot as plt
from normalization import BaseTransformModule, SARTransform
from matplotlib.figure import Figure

class LazyCoordinateRange:
    """
    Lazy replacement for np.arange that doesn't materialize the array.
    """
    def __init__(self, start: int, stop: int, step: int = 1):
        self.start = start
        self.stop = stop
        self.step = step
        self._length = max(0, (stop - start + step - 1) // step)
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index >= len(self) or index < 0:
            raise IndexError("Index out of range")
        return self.start + index * self.step
    
    def __iter__(self):
        current = self.start
        while current < self.stop:
            yield current
            current += self.step

class LazyCoordinateGenerator:
    """
    Lazy generator for coordinate tuples that generates coordinates on-demand.
    Supports different ordering patterns (row, col, chunk) and block patterns.
    """
    
    def __init__(self, y_range: LazyCoordinateRange, x_range: LazyCoordinateRange, 
                 patch_order: str = "row", block_pattern: Optional[Tuple[int, int]] = None,
                 zfile: Optional[os.PathLike] = None, dataset: Optional['SARZarrDataset'] = None):
        self.y_range = y_range
        self.x_range = x_range
        self.patch_order = patch_order
        self.block_pattern = block_pattern
        self.zfile = zfile
        self.dataset = dataset
        
        # Pre-calculate total length without materializing arrays
        self._length = len(y_range) * len(x_range)
    
    def __iter__(self):
        """Generate coordinates lazily based on the specified order and block pattern."""
        if self.patch_order == "row":
            yield from self._generate_row_order()
        elif self.patch_order == "col":
            yield from self._generate_col_order()
        elif self.patch_order == "chunk":
            yield from self._generate_chunk_order()
        else:
            raise ValueError(f"Unknown patch_order: {self.patch_order}")
    
    def _generate_row_order(self):
        """Generate coordinates in row-major order with optional block pattern."""
        if self.block_pattern is not None:
            yield from self._generate_block_pattern_row()
        else:
            for y in self.y_range:
                for x in self.x_range:
                    yield (y, x)
    
    def _generate_col_order(self):
        """Generate coordinates in column-major order with optional block pattern."""
        if self.block_pattern is not None:
            yield from self._generate_block_pattern_col()
        else:
            for x in self.x_range:
                for y in self.y_range:
                    yield (y, x)
    
    def _generate_chunk_order(self):
        """Generate coordinates in chunk-aware order for optimal cache performance."""
        if self.dataset is None or self.zfile is None:
            yield from self._generate_row_order()
            return
        
        try:
            # Get chunk information
            sample = self.dataset.get_store_at_level(self.zfile, self.dataset.level_from)
            ch, cw = sample.chunks
            
            # Group coordinates by chunks without materializing all coordinates
            chunk_coords = {}
            for y in self.y_range:
                for x in self.x_range:
                    cy, cx = y // ch, x // cw
                    chunk_key = (cy, cx)
                    if chunk_key not in chunk_coords:
                        chunk_coords[chunk_key] = []
                    chunk_coords[chunk_key].append((y, x))
            
            # Sort chunk keys and yield coordinates chunk by chunk
            sorted_chunks = sorted(chunk_coords.keys())
            for chunk_key in sorted_chunks:
                coords = chunk_coords[chunk_key]
                # Sort coordinates within chunk (x first, then y for cache locality)
                coords.sort(key=lambda coord: (coord[1], coord[0]))
                for coord in coords:
                    yield coord
                    
        except Exception:
            yield from self._generate_row_order()
    
    def _generate_block_pattern_row(self):
        """Generate coordinates with block pattern in row-major order."""
        if self.block_pattern is None:
            raise ValueError("block_pattern must be set for block pattern generation.")
        block_size, _ = self.block_pattern
        
        for y in self.y_range:
            # Process x coordinates in blocks
            x_start = 0
            while x_start < len(self.x_range):
                x_end = min(x_start + block_size, len(self.x_range))
                for x_idx in range(x_start, x_end):
                    x = self.x_range[x_idx]  # Get actual x coordinate
                    yield (y, x)
                x_start = x_end
    
    def _generate_block_pattern_col(self):
        """Generate coordinates with block pattern in column-major order."""
        if self.block_pattern is None:
            raise ValueError("block_pattern must be set for block pattern generation.")
        block_size, _ = self.block_pattern
        
        for x in self.x_range:
            # Process y coordinates in blocks
            y_start = 0
            while y_start < len(self.y_range):
                y_end = min(y_start + block_size, len(self.y_range))
                for y_idx in range(y_start, y_end):
                    y = self.y_range[y_idx]  # Get actual y coordinate
                    yield (y, x)
                y_start = y_end
    
    def __len__(self):
        """Return total number of coordinates."""
        return self._length
    
    def __getitem__(self, index: int) -> Tuple[int, int]:
        """
        Support indexing for compatibility with existing code.
        Note: This is less efficient than iteration for large datasets.
        """
        if index < 0:
            index += len(self)
        if index >= len(self) or index < 0:
            raise IndexError("Index out of range")
        
        # Calculate coordinates without materializing arrays
        if self.patch_order == "row":
            y_idx = index // len(self.x_range)
            x_idx = index % len(self.x_range)
            return (self.y_range[y_idx], self.x_range[x_idx])
        elif self.patch_order == "col":
            x_idx = index // len(self.y_range)
            y_idx = index % len(self.y_range)
            return (self.y_range[y_idx], self.x_range[x_idx])
        else:
            # For chunk order, fall back to list conversion (less efficient)
            coords = list(self)
            return coords[index]


class SARZarrDataset(Dataset):
    """
    PyTorch Dataset for loading SAR (Synthetic Aperture Radar) data patches from Zarr format archives.

    This class supports efficient patch sampling from multiple Zarr files, with both rectangular and parabolic patch extraction. It handles both local and remote (Hugging Face) Zarr stores, with on-demand patch downloading and LRU chunk caching for performance.

    Features:
        - Loads SAR data patches from Zarr stores (local or remote), supporting real and complex-valued data.
        - Multiple patch sampling modes: "rectangular", "parabolic".
        - Efficient patch coordinate indexing and caching for fast repeated access.
        - Optional patch transformation and visualization utilities.
        - Handles both input and target SAR processing levels (e.g., "rcmc" and "az").
        - Supports saving/loading patch indices to avoid recomputation.
        - Implements chunk-level LRU caching for efficient repeated access.
        - Flexible filtering by part, year, month, polarization, and stripmap mode via SampleFilter.
        - Supports positional encoding and concatenation of patches.

    Args:
        data_dir (str): Directory containing Zarr files.
        filters (SampleFilter, optional): Filter for selecting data by part, year, etc.
        author (str, optional): Author or dataset identifier. Defaults to 'Maya4'.
        online (bool, optional): If True, enables remote Hugging Face access. Defaults to False.
        return_whole_image (bool, optional): If True, returns the whole image as a single patch. Defaults to False.
        transform (callable, optional): Optional transform to apply to both input and target patches.
        patch_size (Tuple[int, int], optional): Size of the patch (height, width). Defaults to (256, 256). If the patch width or height is set to a negative value, it will be computed based on the image dimensions minus the buffer.
        complex_valued (bool, optional): If True, returns complex-valued tensors. If False, returns real and imaginary parts stacked. Defaults to False.
        level_from (str, optional): Key for the input SAR processing level. Defaults to "rcmc".
        level_to (str, optional): Key for the target SAR processing level. Defaults to "az".
        patch_mode (str, optional): Patch extraction mode: "rectangular", or "parabolic". Defaults to "rectangular".
        parabola_a (float, optional): Curvature parameter for parabolic patch mode. Defaults to 0.001.
        save_samples (bool, optional): If True, saves computed patch indices to disk. Defaults to True.
        buffer (Tuple[int, int], optional): Buffer (margin) to avoid sampling near image edges. Defaults to (100, 100).
        stride (Tuple[int, int], optional): Stride for patch extraction. Defaults to (50, 50).
        max_base_sample_size (Tuple[int, int], optional): Maximum base sample size. Defaults to (-1, -1).
        backend (str, optional): Backend for loading Zarr data, either "zarr" or "dask". Defaults to "zarr".
        verbose (bool, optional): If True, prints verbose output. Defaults to True.
        cache_size (int, optional): Maximum number of chunks to cache in memory.
        positional_encoding (bool, optional): If True, adds positional encoding to input patches. Defaults to True.
        dataset_length (int, optional): Optional override for dataset length.
        max_products (int, optional): Maximum number of Zarr products to use. Defaults to 10.
        samples_per_prod (int, optional): Number of patches to sample per product. Defaults to 1000.
        concatenate_patches (bool, optional): If True, concatenates patches along the specified axis.
        concat_axis (int, optional): Axis along which to concatenate patches.
        max_stripmap_modes (int, optional): Maximum number of stripmap modes.
        use_positional_as_token (bool, optional): If True, uses positional encoding as a token.

    Attributes:
        data_dir (str): Directory containing Zarr files.
        patch_size (Tuple[int, int]): Patch size (height, width).
        level_from (str): Input SAR processing level.
        level_to (str): Target SAR processing level.
        patch_mode (str): Patch extraction mode.
        buffer (Tuple[int, int]): Buffer for patch extraction.
        stride (Tuple[int, int]): Stride for patch extraction.
        cache_size (int): LRU cache size for chunk loading.
        positional_encoding (bool): Whether to add positional encoding.
        dataset_length (int): Optional override for dataset length.
        ... (see code for additional attributes)

    Example:
        >>> dataset = SARZarrDataset("/path/to/zarrs", patch_size=(128, 128), cache_size=1000)
        >>> x, y = dataset[("path/to/zarr", 100, 100)]
        >>> dataset.visualize_item(("path/to/zarr", 100, 100))
    """
    def __init__(
        self,
        data_dir: str,
        filters: Optional[SampleFilter] = None,
        author: str = 'Maya4',
        online: bool = False,
        return_whole_image: bool = False,
        transform: Optional[SARTransform] = None,
        patch_size: Tuple[int, int] = (256, 256),
        complex_valued: bool = False,
        level_from: str = "rcmc",
        level_to: str = "az",
        patch_mode: str = "rectangular",      # rectangular, or parabolic
        parabola_a: Optional[float] = 0.001,         # curvature parameter for parabolic mode
        save_samples: bool = True, 
        buffer: Tuple[int, int] = (100, 100), 
        stride: Tuple[int, int] = (50, 50), 
        block_pattern: Optional[Tuple[int, int]] = None,
        max_base_sample_size: Tuple[int, int] = (-1, -1),
        backend: str = "zarr",  # "zarr" or "dask"
        verbose: bool= True, 
        cache_size: int = 1000, 
        positional_encoding: bool = True, 
        dataset_length: Optional[int] = None, 
        max_products: int = 10, 
        samples_per_prod: int = 1000, 
        concatenate_patches: bool = False,  # wether to concatenate 1D patches into 2D patches (useful for transformers from rcmc to az)
        concat_axis: int = 1,               # Axis on which patches have to be concatenated: 0=vertical, 1=horizontal
        max_stripmap_modes: int = 6,
        use_positional_as_token: bool = False,
        use_balanced_sampling: bool = True,  # whether to use balanced sampling from dataset splits, 
        split: str = "train"               # dataset split to use for balanced sampling
    ):
        self.data_dir = Path(data_dir)
        self.filters = filters if filters is not None else SampleFilter()
        self.author = author
        self.return_whole_image = return_whole_image
        self.transform = transform
        self._patch_size = patch_size
        self.level_from = level_from
        self.level_to = level_to
        
        self.patch_mode = patch_mode
        self.parabola_a = parabola_a
        self.complex_valued = complex_valued
        self.buffer = buffer
        self.stride = stride
        self.block_pattern = block_pattern
        self.backend = backend
        self.verbose = verbose
        self.save_samples = save_samples
        self.online = online
        self.positional_encoding = positional_encoding
        self._dataset_length = dataset_length
        self._max_products = max_products
        self._samples_per_prod = samples_per_prod
        self.max_stripmap_modes = max_stripmap_modes
        self._max_base_sample_size = max_base_sample_size
        self.concatenate_patches = concatenate_patches
        self.concat_axis = concat_axis
        self.use_positional_as_token = use_positional_as_token
        self.use_balanced_sampling = use_balanced_sampling
        self.split = split
        
        self._patch: Dict[str, np.ndarray] = {
            self.level_from: np.array([0]),
            self.level_to: np.array([0])
        }
        # Validate concatenation parameters
        # if self.concatenate_patches:
            # ph, pw = self._patch_size
            # if self.concat_axis == 0 and pw != 1:
            #     raise ValueError("For vertical concatenation (axis=0), patch width must be 1")
            # elif self.concat_axis == 1 and ph != 1:
            #     raise ValueError("For horizontal concatenation (axis=1), patch height must be 1")
        self._load_chunk = functools.lru_cache(maxsize=cache_size)(
            self._load_chunk_uncached
        )

        # self._samples_by_file: Dict[os.PathLike, List[Tuple[int,int]]] = {}
        self._y_coords: Dict[os.PathLike, np.ndarray] = {}
        self._x_coords: Dict[os.PathLike, np.ndarray] = {}
        # self._pos_encoding_out: Dict[str, np.ndarray] = {}
        # self.init_samples()
        self._initialize_stores()
        if self.verbose:
            print(f"Initialized dataloader with config: buffer={buffer}, stride={stride}, patch_size={patch_size}, complex_values={complex_valued}")
    def get_patch_size(self, zfile: Optional[Union[str, os.PathLike]]) -> Tuple[int, int]:
        """
        Retrieve the patch size for a given Zarr file based on its processing level.
        
        Args:
            zfile (os.PathLike): Path to the Zarr file.

        Returns:
            Tuple[int, int]: Patch size (height, width) for the specified processing level.
        """
        
        ph, pw = self._patch_size

        if ph > 0 and pw > 0 or zfile is None:
            return ph, pw
        arr = self.get_store_at_level(Path(zfile), self.level_from)
        if arr is not None:
            if self.backend == "zarr":
                y, x = arr.shape 
            elif self.backend == "dask":
                y, x = arr.shape[1:]
            if ph <= 0:
                ph = y - 2*self.buffer[0]
            if pw <= 0:
                pw = x - 2*self.buffer[1]
            return ph, pw
    def get_metadata(self, zfile: Union[str, os.PathLike], rows: Optional[Union[Tuple[int, int], slice]] = None):
        """
        Extract metadata from zarr attributes.
        
        Returns:
            pandas DataFrame containing metadata if available, None otherwise
        """
        arr = self.get_store(Path(zfile))
        print(f"Available zarr attributes: {list(arr.attrs.keys())}")
        if 'metadata' in arr.attrs:
            import pandas as pd
            #print(arr.attrs['metadata'].keys())
            metadata = pd.DataFrame(arr.attrs['metadata']['data'])
            
            if rows is not None and isinstance(rows, slice):
                start, stop, step_val = rows.indices(len(metadata))
                row_indices = list(range(start, stop, step_val or 1))
                metadata = metadata.iloc[row_indices].reset_index(drop=True)
        else:
            metadata = None
            
        if 'ephemeris' in arr.attrs:
            import pandas as pd
            ephemeris = pd.DataFrame(arr.attrs['ephemeris']['data'])
            
        else:
            ephemeris = None
        return metadata, ephemeris

    def get_samples_by_file(self, zfile: Union[str, os.PathLike]) -> List[Tuple[int,int]]:
        """
        Get the list of patch coordinates for a given Zarr file.

        Args:
            zfile (os.PathLike): Path to the Zarr file.
        Returns:
            List[Tuple[int, int]]: List of (y, x) coordinates for patches in the specified Zarr file.
        """
        try: 
            return self._files.loc[self._files['full_name'] == Path(zfile)]['samples'].values[0]
        except Exception:
            return None
    def _set_samples_for_file(self, zfile: Union[str, os.PathLike], samples: List[Tuple[int,int]]):
        """
        Set the list of patch coordinates for a given Zarr file.

        Args:
            zfile (os.PathLike): Path to the Zarr file.
            samples (List[Tuple[int, int]]): List of (y, x) coordinates for patches in the specified Zarr file.
        """
        idx = self._files.index[self._files['full_name'] == Path(zfile)]
        if len(idx) > 0:
            self._files.at[idx[0], 'samples'] = samples
    def _build_file_list(self):
        """
        Retrieve the list of Zarr files to use, either from the local directory or from a remote Hugging Face repository.
        Filters files using the provided glob pattern and limits to max_products.
        """
        if self.online:
            repos = list_repos_by_author(self.author)
            if self.verbose:
                print(f"Found {len(repos)} repositories by author '{self.author}': {repos}")
            self.remote_files = {}
            for repo in repos:
                repo_files = list_base_files_in_repo(
                    repo_id=repo,
                )
                # Convert glob pattern to regex for matching
                if self.verbose:
                    print(f"Found {len(repo_files)} files in the remote repository: '{repo}'")
                repo_name = repo.split('/')[-1]
                self.remote_files[repo_name] = repo_files
            records = [r for r in (parse_product_filename(os.path.join(self.data_dir, part, f)) for part, files in self.remote_files.items() for f in files) if r is not None]
            print(f"Total files found in remote repository: {len(records)}")
            df = pd.DataFrame(records)
        else:
            print(f"Files in local directory {self.data_dir}: {[f.name for f in sorted(self.data_dir.glob('*'))]}")
            records = [r for r in (parse_product_filename(f) for f in sorted(self.data_dir.glob("*"))) if r is not None]
            df = pd.DataFrame(records)
        # Drop records without acquisition_date and ensure datetime type
        self._files = self.filters._filter_products(df)
        self._files.sort_values(by=['full_name'], inplace=True)
        # Apply balanced sampling if enabled
        if self.use_balanced_sampling:
                balanced_files = get_balanced_sample_files(
                    max_samples=self._max_products, 
                    data_dir=self.data_dir,
                    sample_filter=self.filters,
                    config_path=str(self.data_dir),
                    verbose=True,  # self.verbose
                    split_type=self.split, 
                    repos=self.filters.parts if self.filters.parts else ['PT1', 'PT2', 'PT3', 'PT4']
                )
                if balanced_files:
                    # Filter the files to only include balanced selection
                    balanced_paths = [Path(f) for f in balanced_files]
                    self._files = self._files[self._files['full_name'].isin(balanced_paths)]
                    if self.verbose:
                        print(f"Applied balanced sampling: selected {len(self._files)} files from {len(balanced_files)} balanced candidates")
                else:
                    if self.verbose:
                        print("Warning: Balanced sampling returned no files, falling back to standard selection")
                    self._files = self._files.iloc[:self._max_products]
        else:
            self._files = self._files.iloc[:self._max_products]
        if self.verbose:
            print(f"Selected files: {len(self._files)} total")
            print(f"Files: {self._files['full_name'].tolist()}")

    def _append_file_to_stores(self, zfile: Union[str, os.PathLike]):
        """
        Appends a Zarr file to the stores dictionary, opening it in read-only mode.
        This method is used to initialize the dataset with a specific Zarr file.
        
        Args:
            zfile (os.PathLike): Path to the Zarr file to be added.
        """
        if not Path(zfile).exists():
            return
        # Only return if the entry does not exist at all
        if not os.path.exists(get_part_from_filename(zfile)):
            os.makedirs(os.path.join(self.data_dir, get_part_from_filename(zfile)), exist_ok=True)
        idx = self._files.index[self._files['full_name'] == Path(zfile)]
        if len(idx) > 0 and self._files.at[idx[0], 'store'] is not None:
            return  # Do not return, just continue
        if self.backend == "zarr":
            idx = self._files.index[self._files['full_name'] == Path(zfile)]
            if len(idx) > 0:
                #print(f"Opening Zarr store for file {zfile} at index {idx[0]}")
                self._files.at[idx[0], 'store'] = self.open_archive(zfile)
        elif self.backend == "dask":
            self._files.loc[self._files['full_name'] == Path(zfile), 'store'] = {}
            for level in (self.level_from, self.level_to):
                complete_path = os.path.join(zfile, level) 
                idx = self._files.index[self._files['full_name'] == Path(zfile)]
                if len(idx) > 0:
                    self._files.at[idx[0], 'store'] = {}
                    self._files.at[idx[0], 'store'][level] = self.open_archive(complete_path) 
        else:
            raise ValueError(f"Unknown backend {self.backend}")

    def _initialize_stores(self):
        """
        Initializes data stores based on the selected backend.

        For the "zarr" backend, opens each file as a Zarr store in read-only mode and stores it in `self._files` in the 'stores' attribute.
        For the "dask" backend, creates a dictionary for each file, loading data for each specified level using Dask arrays
        with the given patch size for rechunking.
        Raises:
            ValueError: If an unknown backend is specified.
        """
        t0 = time.time()
        self._build_file_list()
        if self.verbose:
            dt = time.time() - t0
            print(f"Files list calculation took {dt:.2f} seconds.")
            
        t0 = time.time()
        for zfile in self.get_files():
            self._append_file_to_stores(Path(zfile))
        if self.verbose:
            dt = time.time() - t0
            print(f"Zarr stores initialization took {dt:.2f} seconds.")

    def get_files(self) -> List[os.PathLike]:
        """
        Returns the list of Zarr files used in the dataset.

        Returns:
            List[os.PathLike]: List of Zarr file paths.
        """

        return self._files['full_name'].tolist()

    def open_archive(self, zfile: os.PathLike) -> (zarr.Group | zarr.Array):
        """
        Open a Zarr archive and return the root group or array, depending on backend.

        Args:
            zfile (os.PathLike): Path to the Zarr file.

        Returns:
            zarr.Group or zarr.Array: The root group or array of the Zarr archive.
        """
        if self.backend == "dask":
            return da.from_zarr(zfile)
        elif self.backend == "zarr":
            return zarr.open(zfile, mode='r')
        else: 
            raise ValueError(f"Unknown backend {self.backend}")
        
    def get_store_at_level(self, zfile: Union[os.PathLike, str], level: str) -> zarr.Array:
        level_dir = Path(zfile) / level
        if not Path(zfile).exists() or not level_dir.exists() or not level_dir.exists():
            if self.online:
                zfile_name = os.path.basename(zfile)
                part = get_part_from_filename(zfile)
                repo_id = self.author + '/' + part
                download_metadata_from_product(
                    zfile_name=str(zfile_name),
                    local_dir=os.path.join(self.data_dir, part),
                    levels=[level, self.level_from, self.level_to],
                    repo_id=repo_id
                )
            else:
                raise ValueError(f"Levels {self.level_from} and {self.level_to} not found in Zarr store {zfile}.")
        self._append_file_to_stores(zfile=Path(zfile))
        return self._files.loc[self._files['full_name'] == Path(zfile), 'store'].values[0][level]
    
    def get_store(self, zfile: Union[os.PathLike, str]) -> zarr.Group:
        if not Path(zfile).exists():
            if self.online:
                zfile_name = os.path.basename(zfile)
                part = get_part_from_filename(zfile)
                repo_id = self.author + '/' + part
                download_metadata_from_product(
                    zfile_name=str(zfile_name),
                    local_dir=os.path.join(self.data_dir, part),
                    levels=[self.level_from, self.level_to], 
                    repo_id=repo_id
                )
            else:
                raise ValueError(f"Levels {self.level_from} and {self.level_to} not found in Zarr store {zfile}.")
        self._append_file_to_stores(zfile=Path(zfile))
        return self._files.loc[self._files['full_name'] == str(zfile), 'store'].values[0]
    def get_max_base_sample_size(self, zfile: Union[str, os.PathLike]):
        ph, pw =   self._max_base_sample_size
        if ph == -1:
            ph = self.get_store_at_level(Path(zfile), self.level_from).shape[0]
        if pw == -1:
            pw = self.get_store_at_level(Path(zfile), self.level_from).shape[1]
        return ph, pw
    def get_whole_sample_shape(self, zfile: os.PathLike) -> Tuple[int, int]:
        """
        Get the shape of the whole sample (image) at the specified level from the Zarr file.

        Args:
            zfile (os.PathLike): Path to the Zarr file.

        Returns:
            Tuple[int, int]: Shape of the whole sample (height, width).
        """
        ph, pw = self.get_store_at_level(zfile, self.level_from).shape
        return (ph - self.buffer[0] * 2, pw - self.buffer[1] * 2)
    
    def calculate_patches_from_store(self, zfile: os.PathLike, patch_order: str = "row", window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None):
        """
        Compute and store valid patch coordinates for a given Zarr file, according to the patch mode and parameters.
        Downloads metadata if needed (in online mode) and ensures all patches fit within image bounds.

        Args:
            zfile (os.PathLike): Path to the Zarr file.
            patch_order (str): Ordering strategy for patch coordinates. Options are "row", "col", or "chunk".
        """
        level_from_dir = Path(zfile) / self.level_from
        level_t_dir = Path(zfile) / self.level_to
        if not Path(zfile).exists() or not level_from_dir.exists() or not level_t_dir.exists():
            if self.online:
                zfile_name = os.path.basename(zfile)
                part = get_part_from_filename(zfile)
                repo_id = self.author + '/' + part
                download_metadata_from_product(
                    zfile_name=str(zfile_name),
                    local_dir=os.path.join(self.data_dir, part),
                    levels=[self.level_from, self.level_to], 
                    repo_id=repo_id
                )
            else:
                raise ValueError(f"Levels {self.level_from} and {self.level_to} not found in Zarr store {zfile}.")
        
        self._append_file_to_stores(Path(zfile))

        h, w = self.get_whole_sample_shape(zfile) 
        
        if self.return_whole_image:
            coords = [(0, 0)]
            self._set_samples_for_file(zfile, coords)
            return
        
        stride_y, stride_x = self.stride
        ph, pw = self.get_patch_size(zfile)
        
        if self.patch_mode == "rectangular":
            y_min, y_max = self.buffer[0], h - self.buffer[0]
            x_min, x_max = self.buffer[1], w - self.buffer[1]

            if window is not None:
                y_min, y_max = max(window[0][0], y_min), min(window[1][0], y_max)
                x_min, x_max = max(window[0][1], x_min), min(window[1][1], x_max)
                stride_x, stride_y = min(stride_x, x_max - x_min), min(stride_y, y_max - y_min)
                
            if self.concatenate_patches:
                mph, mpw = self.get_max_base_sample_size(zfile)
                mph, mpw = min(mph, x_max-x_min), min(mpw, y_max-y_min)
                if self.concat_axis == 0:
                    # Use lazy coordinate ranges instead of np.arange
                    y_range = LazyCoordinateRange(y_min, y_max - mph + 1, mph)
                    x_range = LazyCoordinateRange(x_min, x_max - stride_x + 1, stride_x)
                elif self.concat_axis == 1:
                    y_range = LazyCoordinateRange(y_min, y_max - stride_y + 1, stride_y)
                    x_range = LazyCoordinateRange(x_min, x_max - mpw + 1, mpw)
                else:
                    raise ValueError(f"Invalid concat_axis: {self.concat_axis}. Must be 0 (vertical) or 1 (horizontal).")
            else:
                # Use lazy coordinate ranges instead of np.arange - THIS IS THE KEY FIX!
                y_range = LazyCoordinateRange(y_min, y_max - stride_y + 1, stride_y)
                x_range = LazyCoordinateRange(x_min, x_max - stride_x + 1, stride_x)

        else:
            raise ValueError(f"Unknown patch_mode {self.patch_mode}")
        
        # Store a lazy coordinate generator with lazy ranges
        lazy_coords = LazyCoordinateGenerator(
            y_range=y_range, 
            x_range=x_range,
            patch_order=patch_order,
            block_pattern=self.block_pattern,
            zfile=zfile,
            dataset=self
        )
        # print(f"Calculated {len(lazy_coords)} patches for file {zfile} with patch size {ph}x{pw}, buffer {self.buffer}, stride {self.stride}, in mode {self.patch_mode}")
        # print(f"Y range: {y_range.start} to {y_range.stop} with step {y_range.step}, X range: {x_range.start} to {x_range.stop} with step {x_range.step}")
        self._set_samples_for_file(zfile, lazy_coords)

    def reorder_samples(self, zfile: os.PathLike, patch_order: str = "row") -> 'LazyCoordinateGenerator':
        """
        Create a lazy coordinate generator instead of pre-computing all coordinates.
        """
        existing_samples = self.get_samples_by_file(zfile)
        if isinstance(existing_samples, LazyCoordinateGenerator):
            return LazyCoordinateGenerator(
                y_range=existing_samples.y_range,
                x_range=existing_samples.x_range,
                patch_order=patch_order,
                block_pattern=self.block_pattern,
                zfile=zfile,
                dataset=self
            )
        else:
            # Fallback for existing implementations
            return existing_samples
    
    def __len__(self):
        """
        Return the total number of patches in the dataset (samples_per_prod * max_products).
        """
        return self._samples_per_prod * self._max_products

    def _get_base_sample(self, zfile: os.PathLike, y: int, x: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the raw input and target patches from the Zarr store at the specified coordinates,
        without applying any transformation or positional encoding.

        This method extracts patches from the specified Zarr file and coordinates, using either
        rectangular or parabolic patch extraction based on the dataset configuration. The returned
        patches are in their original format (complex-valued), suitable for further processing or
        transformation.

        Args:
            zfile (os.PathLike): Path to the Zarr file.
            y (int): y-coordinate of the top-left corner of the patch.
            x (int): x-coordinate of the top-left corner of the patch.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the input patch (from level_from)
            and the target patch (from level_to), both as NumPy arrays.

        Notes:
            - If `patch_mode` is "parabolic", patches are sampled using the `_sample_parabolic_patch` method.
            - Otherwise, rectangular patches are extracted using `_get_sample`.
            - No normalization, transformation, or positional encoding is applied.
        """
        if self.patch_mode == "parabolic":
            patch_from = self._sample_parabolic_patch(zfile, self.level_from, x, y)
            patch_to = self._sample_parabolic_patch(zfile, self.level_to, x, y)
        else:
            ph, pw = self.get_patch_size(zfile)
            sh, sw = self.get_whole_sample_shape(zfile)
            bsh, bsw = self.get_max_base_sample_size(zfile)
            sh, sw = min(sh, bsh), min(sw, bsw)
            if self.concatenate_patches:
                if self.concat_axis == 0:
                    patch_from = self._get_sample(zfile, self.level_from, y, x, sh, pw)
                    patch_to = self._get_sample(zfile, self.level_to,y, x, sh, pw)
                elif self.concat_axis == 1:
                    patch_from = self._get_sample(zfile, self.level_from, y, x, ph, sw)
                    patch_to = self._get_sample(zfile, self.level_to, y, x, ph, sw)
            else:
                patch_from = self._get_sample(zfile, self.level_from, y, x, ph, pw)
                
                # if pw == 1:
                #     original_patch = self.get_store_at_level(zfile, self.level_from)[y:y+ph, x]
                # elif ph == 1:
                #     original_patch = self.get_store_at_level(zfile, self.level_from)[y, x:x+pw]
                # else:
                #     original_patch = self.get_store_at_level(zfile, self.level_from)[y:y+ph,x:x+pw]
                # #print(f"Original patch shape")
                # assert (patch_from.squeeze() ==original_patch).all(), f"Patch data mismatch! Patch: {patch_from.squeeze()}, original patch: {original_patch}"
                # print(f"Comparison is ok between the patches: {patch_from.squeeze()[:50]} and {original_patch[:50]}")
                
                patch_to = self._get_sample(zfile, self.level_to, y, x, ph, pw)
                # if pw == 1:
                #     original_patch = self.get_store_at_level(zfile, self.level_to)[y:y+ph, x]
                # elif ph == 1:
                #     original_patch = self.get_store_at_level(zfile, self.level_to)[y, x:x+pw]
                # else:
                #     original_patch = self.get_store_at_level(zfile, self.level_to)[y:y+ph,x:x+pw]
                # #print(f"Original patch shape")
                # assert (patch_to.squeeze() ==original_patch).all(), f"Patch data mismatch! Patch: {patch_to.squeeze()}, original patch: {original_patch}"
                # print(f"Comparison is ok between the patches: {patch_from.squeeze()[:50]} and {original_patch[:50]}")
                # assert (patch_from == patch_to).all(), f"Patch data mismatch! Patch from: {patch_from.squeeze()}, Patch to: {patch_to.squeeze()}"
        return patch_from, patch_to

    def __getitem__(self, idx: Tuple[str, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Load a patch from the dataset given a (zfile, y, x) tuple.

        Args:
            idx (Tuple[str, int, int]): A tuple containing the zfile path name, y-coordinate, and x-coordinate
                specifying the location of the patch to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target patches as torch tensors.
                The format and type of the tensors depend on the `complex_valued` attribute:
                    - If `complex_valued` is False, tensors have shape (2, patch_height, patch_width) with real and imaginary parts stacked.
                    - If `complex_valued` is True, tensors retain their complex dtype.

        Raises:
            IndexError: If the provided index is out of bounds.
            KeyError: If the specified levels are not found in the zarr store.

        Notes:
            - If `patch_mode` is "parabolic", patches are sampled using the `_sample_parabolic_patch` method.
            - If a `transform` is provided, it is applied to both input and target patches.
        """
        zfile, y, x = idx
        # Extract stride number from filename if present
        stripmap_mode = extract_stripmap_mode_from_filename(os.path.basename(zfile))
        zfile = Path(zfile)
        start_time = time.time()

        t0 = time.time()

        # Retrieve the horizontal size (width) from the store for self.level_from
        sample_height, sample_width = self.get_whole_sample_shape(zfile) 
        if self.verbose:
            dt = time.time() - t0
            print(f"Sample shape for {zfile} at level {self.level_from}: {sample_height}x{sample_width} took {dt:.4f} seconds")
            
        t0 = time.time()
        patch_from, patch_to = self._get_base_sample(zfile, y, x)
        if self.verbose:
            dt = time.time() - t0
            print(f"Base sample loading for {zfile} at ({y}, {x}) took {dt:.4f} seconds")
            
            
        if self.transform:
            t0 = time.time()
            patch_from = self.transform(patch_from, self.level_from)
            patch_to = self.transform(patch_to, self.level_to)
            if self.verbose:
                dt = time.time() - t0
                print(f"Patch transformation took {dt:.4f} seconds")
        #print(f"Patch shape before stacking: {patch_from.shape}, {patch_to.shape}")
        if not self.complex_valued:
            t0 = time.time()
            patch_from = np.stack((np.real(patch_from), np.imag(patch_from)), axis=-1).astype(np.float32)
            patch_to = np.stack((np.real(patch_to), np.imag(patch_to)), axis=-1).astype(np.float32)
            if self.verbose:
                dt = time.time() - t0
                print(f"Complex to real conversion took {dt:.4f} seconds")
        #print(f"Shape before positional encoding: {patch_from.shape}")
        if self.positional_encoding:
            t0 = time.time()
            global_column_index = x + stripmap_mode * sample_width
            
            patch_from = self.add_position_embedding(patch_from, (y, global_column_index), (sample_height, sample_width * self.max_stripmap_modes), level=self.level_from)
            patch_to = self.add_position_embedding(patch_to, (y, global_column_index), (sample_height, sample_width * self.max_stripmap_modes), level=self.level_to)
            if self.verbose:
                dt = time.time() - t0
                print(f"Patch positional encoding took {dt:.4f} seconds")
        #print(f"Shape after positional encoding: {patch_from.shape}, {patch_to.shape}")
        if self.concatenate_patches:
            t0 = time.time()
            # Get concatenated patches instead of single patch
            patch_from = self._get_concatenated_patch(patch_from, zfile)
            #print(f"Concatenated patch from: {patch_from}")
            patch_to = self._get_concatenated_patch(patch_to, zfile)
            #print(f"Concatenated patch to: {patch_to}")
            if self.verbose:
                dt = time.time() - t0
                print(f"Concatenated patch shapes: from={patch_from.shape}, to={patch_to.shape} took {dt:.4f} seconds")
        # Optionally concatenate positional embedding as next token
        if self.use_positional_as_token:
            # patch_from shape: (1000, seq_len, 2)
            backscatter = patch_from[..., 0]
            pos_embed = patch_from[..., 1]
            patch_from = np.concatenate([backscatter, pos_embed[:, :, :1]], axis=-1)  # (1000, seq_len+1)
        x_tensor = torch.from_numpy(patch_from)
        y_tensor = torch.from_numpy(patch_to)
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"Loading patch ({zfile}, {y}, {x}) took {elapsed:.4f} seconds. Stripmap mode: {stripmap_mode}")
        #print(f"Opening zfile: {zfile}, y: {y}, x: {x}, stripmap_mode: {stripmap_mode}, shape: {x_tensor.shape}, {y_tensor.shape}")
        return x_tensor, y_tensor
    
    def _download_sample_if_missing(self, zfile: os.PathLike, level: str, y: int, x: int) -> Path:
        """
        Downloads a missing sample patch from the Hugging Face Zarr archive if it is not already available locally.

        Args:
            zfile (os.PathLike): Path to the Zarr file in the dataset.
            level (str): The processing level to retrieve data from (e.g., "rcmc", "az").
            y (int): The y-coordinate of the patch.
            x (int): The x-coordinate of the patch.

        Returns:
            Path: Path to the downloaded chunk file.
        """
        zfile_name = os.path.basename(zfile)
        part = get_part_from_filename(zfile)
        repo_id = self.author + '/' + part
        if not zfile.exists():
            if self.online:
                meta_file = download_metadata_from_product(
                    zfile_name=str(zfile_name),
                    local_dir=os.path.join(self.data_dir, part),
                    levels=[self.level_from, self.level_to], 
                    repo_id=repo_id
                )
                with open(meta_file) as f:
                    zarr_meta = json.load(f)
                version = zarr_meta.get('zarr_format', 2)
                self.calculate_patches_from_store(zfile)
            else:
                raise FileNotFoundError(f"Zarr file {zfile} does not exist.")

        chunk_name = get_chunk_name_from_coords(y, x, zarr_file_name=zfile_name, level=level, chunks=self.get_store_at_level(zfile, level).chunks, version=get_zarr_version(zfile))
        chunk_path = self.data_dir / part / chunk_name
        
        if not chunk_path.exists():
            if self.verbose:
                print(f"Chunk {chunk_name} not found locally. Downloading from Hugging Face Zarr archive...")
            fetch_chunk_from_hf_zarr(level=level, y=y, x=x, zarr_archive=zfile_name, local_dir=os.path.join(self.data_dir, part), repo_id=repo_id)
        return chunk_path


    def _load_chunk_uncached(self, zfile: os.PathLike, level: str, cy: int, cx: int) -> np.ndarray:
        """
        Load a single chunk at chunk coordinates (cy, cx).
        This is the cacheable unit - individual chunks.
        
        Args:
            zfile (os.PathLike): Path to the Zarr file.
            level (str): Zarr group/level name.
            cy (int): Chunk y-index.
            cx (int): Chunk x-index.
            
        Returns:
            np.ndarray: Single chunk data.
        """
        arr = self.get_store_at_level(zfile, level)
        ch, cw = arr.chunks
        
        # Calculate actual array coordinates for this chunk
        chunk_y_start = cy * ch
        chunk_x_start = cx * cw
        chunk_y_end = min(chunk_y_start + ch, arr.shape[0])
        chunk_x_end = min(chunk_x_start + cw, arr.shape[1])
        
        # Ensure the chunk is downloaded if not already available
        if self.online:
            self._download_sample_if_missing(zfile, level, chunk_y_start, chunk_x_start)
        
        # Load the actual chunk data
        chunk_data = arr[chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
        
        return chunk_data.astype(np.complex128)
    
    def _get_sample_from_cached_chunks_vectorized(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int) -> np.ndarray:
        """
        Optimized version that dispatches to specialized methods based on patch geometry.
        """
        arr = self.get_store_at_level(zfile, level)
        ch, cw = arr.chunks
        
        # Dispatch to optimized methods based on patch type
        if ph == 1:
            # Horizontal strip - most optimized
            return self._get_horizontal_strip_optimized(zfile, level, y, x, pw, ch, cw)
        elif pw == 1:
            # Vertical strip
            return self._get_vertical_strip_optimized(zfile, level, y, x, ph, ch, cw)
        elif ph <= ch and pw <= cw:
            # Small rectangular patch that likely fits in single chunk
            return self._get_small_patch_optimized(zfile, level, y, x, ph, pw, ch, cw)
        else:
            # Large rectangular patch - needs multi-chunk handling
            return self._get_large_patch_optimized(zfile, level, y, x, ph, pw, ch, cw)
    def _get_large_patch_optimized(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int, ch: int, cw: int) -> np.ndarray:
        """
        Optimized for large rectangular patches spanning multiple chunks.
        Uses direct extraction instead of creating large temporary arrays.
        """
        # Calculate chunk spans
        cy_start, cx_start = y // ch, x // cw
        cy_end, cx_end = (y + ph - 1) // ch, (x + pw - 1) // cw
        
        # Pre-allocate final patch only
        if not hasattr(self, "_patch") or level not in self._patch.keys() or self._patch[level].shape != (ph, pw):
            self._patch[level] = np.zeros((ph, pw), dtype=np.complex128)
        # Direct extraction without temporary arrays
        for cy in range(cy_start, cy_end + 1):
            for cx in range(cx_start, cx_end + 1):
                chunk = self._load_chunk(zfile, level, cy, cx)
                
                # Calculate intersection bounds
                chunk_y_start, chunk_x_start = cy * ch, cx * cw
                chunk_y_end = chunk_y_start + chunk.shape[0]
                chunk_x_end = chunk_x_start + chunk.shape[1]
                
                # Global coordinates
                y_start = max(y, chunk_y_start)
                y_end = min(y + ph, chunk_y_end)
                x_start = max(x, chunk_x_start)
                x_end = min(x + pw, chunk_x_end)
                
                if y_start < y_end and x_start < x_end:
                    # Source indices in chunk
                    src_y1, src_y2 = y_start - chunk_y_start, y_end - chunk_y_start
                    src_x1, src_x2 = x_start - chunk_x_start, x_end - chunk_x_start
                    
                    # Destination indices in patch
                    dst_y1, dst_y2 = y_start - y, y_end - y
                    dst_x1, dst_x2 = x_start - x, x_end - x
                    
                    # Direct copy
                    self._patch[level][dst_y1:dst_y2, dst_x1:dst_x2] = chunk[src_y1:src_y2, src_x1:src_x2]

        return self._patch[level]
    def _get_small_patch_optimized(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int, ch: int, cw: int) -> np.ndarray:
        """
        Optimized for small patches that likely fit in a single chunk.
        """
        cy, cx = y // ch, x // cw
        chunk = self._load_chunk(zfile, level, cy, cx)
        
        dy, dx = y % ch, x % cw
        
        # Fast path: patch fits entirely in chunk
        if dy + ph <= chunk.shape[0] and dx + pw <= chunk.shape[1]:
            return chunk[dy:dy+ph, dx:dx+pw].copy()
        
        # Boundary case: use minimal allocation
        patch = np.zeros((ph, pw), dtype=np.complex128)
        max_h = min(ph, chunk.shape[0] - dy)
        max_w = min(pw, chunk.shape[1] - dx)
        
        if max_h > 0 and max_w > 0:
            patch[:max_h, :max_w] = chunk[dy:dy+max_h, dx:dx+max_w]
        
        return patch
    def _get_strip_optimized(self, zfile: os.PathLike, level: str, y: int, x: int, length: int, ch: int, cw: int, axis: int) -> np.ndarray:
        """
        Optimized extraction for strips (vertical or horizontal).
        axis=0: vertical strip (shape=(length, 1)), axis=1: horizontal strip (shape=(1, length))
        """
        if axis == 0:
            # Vertical strip: patch_size=(length, 1)
            cy_start, cy_end = y // ch, (y + length - 1) // ch
            cx = x // cw
            if not hasattr(self, "_patch") or level not in self._patch.keys() or self._patch[level].shape[0] != length:
                self._patch[level] = np.zeros(length, dtype=np.complex128)
                #print(f"Allocating patch for vertical strip: {self._patch.shape}")
            current_pos = 0
            for cy in range(cy_start, cy_end + 1):
                chunk = self._load_chunk(zfile, level, cy, cx)
                chunk_y_start = cy * ch
                global_start = max(y, chunk_y_start)
                global_end = min(y + length, chunk_y_start + chunk.shape[0])
                if global_start < global_end:
                    dx = x % cw
                    if dx < chunk.shape[1]:
                        local_start = global_start - chunk_y_start
                        local_end = global_end - chunk_y_start
                        slice_height = local_end - local_start
                        self._patch[level][current_pos:current_pos + slice_height] = chunk[local_start:local_end, dx]
                        current_pos += slice_height
            return self._patch[level].reshape(length, 1)
        elif axis == 1:
            # Horizontal strip: patch_size=(1, length)
            cx_start, cx_end = x // cw, (x + length - 1) // cw
            cy = y // ch
            if not hasattr(self, "_patch") or level not in self._patch.keys() or self._patch[level].shape[0] != length:
                self._patch[level] = np.zeros(length, dtype=np.complex128)
                #print(f"Allocating patch for horizontal strip: {self._patch.shape}")

            current_pos = 0
            for cx in range(cx_start, cx_end + 1):
                chunk = self._load_chunk(zfile, level, cy, cx)
                chunk_x_start = cx * cw
                global_start = max(x, chunk_x_start)
                global_end = min(x + length, chunk_x_start + chunk.shape[1])
                if global_start < global_end:
                    dy = y % ch
                    if dy < chunk.shape[0]:
                        local_start = global_start - chunk_x_start
                        local_end = global_end - chunk_x_start
                        slice_width = local_end - local_start
                        self._patch[level][current_pos:current_pos + slice_width] = chunk[dy, local_start:local_end]
                        current_pos += slice_width
            return self._patch[level].reshape(1, length)
        else:
            raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")

    # For backward compatibility, you can alias the old methods:
    def _get_vertical_strip_optimized(self, zfile, level, y, x, ph, ch, cw):
        return self._get_strip_optimized(zfile, level, y, x, ph, ch, cw, axis=0)

    def _get_horizontal_strip_optimized(self, zfile, level, y, x, pw, ch, cw):
        return self._get_strip_optimized(zfile, level, y, x, pw, ch, cw, axis=1)

    def _get_sample_from_cached_chunks(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int) -> np.ndarray:
        """
        Highly optimized version with fast single-chunk path.
        """
        arr = self.get_store_at_level(zfile, level)
        ch, cw = arr.chunks
        
        # Pre-calculate chunk coordinates
        cy_start, cx_start = y // ch, x // cw
        cy_end, cx_end = (y + ph - 1) // ch, (x + pw - 1) // cw
        
        # Fast path: single chunk (handles ~80-90% of cases)
        if cy_start == cy_end and cx_start == cx_end:
            chunk = self._load_chunk(zfile, level, cy_start, cx_start)
            dy, dx = y % ch, x % cw  
            
            # Bounds check only once
            if dy + ph <= chunk.shape[0] and dx + pw <= chunk.shape[1]:
                return chunk[dy:dy+ph, dx:dx+pw].copy()  # Direct slice
            else:
                # Handle boundary case
                # print("HEYYYYYYYYY")
                patch = np.zeros((ph, pw), dtype=np.complex128)
                max_h = min(ph, chunk.shape[0] - dy)
                max_w = min(pw, chunk.shape[1] - dx)
                patch[:max_h, :max_w] = chunk[dy:dy+max_h, dx:dx+max_w]
                return patch
        
        # Multi-chunk path (only when necessary)
        return self._get_sample_from_cached_chunks_vectorized(zfile, level, y, x, ph, pw)
    
    def _get_sample(self, zfile: os.PathLike, level: str, y, x, ph: int = 0, pw: int = 0) -> np.ndarray:
        """
        Retrieve a sample patch from the Zarr store at the specified level and coordinates.

        Args:
            zfile (os.PathLike): Path to the Zarr file.
            level (str): Processing level (e.g., 'rcmc', 'az').
            y (int): y-coordinate of the patch.
            x (int): x-coordinate of the patch.
            ph (int): Patch height.
            pw (int): Patch width.

        Returns:
            np.ndarray: The desired patch as a NumPy array.
        """
        t0 = time.time()
        if self.backend == "dask":
            sample = self.get_store_at_level(zfile, level)
            if ph == 0 and pw == 0: 
                arr = sample[y, x].compute() 
            elif ph == 0:   
                arr = sample[y, x:x+pw].compute()
            elif pw == 0:
                arr = sample[y:y+ph, x].compute()
            else: 
                arr = sample[y:y+ph, x:x+pw].compute()
            delta_t = time.time() - t0
            if self.verbose:
                print(f"Loading {level} data took {delta_t:.2f} seconds.")
            return arr.astype(np.complex128)
        elif self.backend == "zarr":
            patch = self._get_sample_from_cached_chunks(zfile, level, y, x, ph, pw)
            # print(f"Original patch: {patch}")
            # if pw == 1:
            #     original_patch = self.get_store_at_level(zfile, level)[y:y+ph, x]
            # elif ph == 1:
            #     original_patch = self.get_store_at_level(zfile, level)[y, x:x+pw]
            # else:
            #     original_patch = self.get_store_at_level(zfile, level)[y:y+ph,x:x+pw]
            # #print(f"Original patch shape")
            # assert (patch.squeeze() ==original_patch).all(), f"Patch data mismatch! Patch: {patch.squeeze()}, original patch: {original_patch}"
            return patch
        else:
            raise ValueError(f"Unknown backend {self.backend}")
    
    def _get_concatenated_patch(self, patch: np.ndarray, zfile: os.PathLike) -> np.ndarray:
        """
        Get concatenated patches by sampling multiple 1D patches and concatenating them.
        
        For concat_axis=0 (vertical concatenation):
        - patch_size = (n, 1000) - horizontal strips  
        - stride = (1000, 1000)
        - Concatenate 10 horizontal strips vertically -> result: (10*n, 1000)
        
        For concat_axis=1 (horizontal concatenation):
        - patch_size = (1000, n) - vertical strips
        - stride = (1000, 1000) 
        - Concatenate 10 vertical strips horizontally -> result: (1000, 10*nm)
        
        Args:
            zfile: Path to Zarr file
            y, x: Starting coordinates
            
        Returns:
            Tuple of concatenated patches (from_level, to_level)
        """
        ph, pw = self.get_patch_size(zfile)
        #print(patch.shape)
        if len(patch.shape) == 2:
            patch = patch[..., np.newaxis]
        fph, fpw, c = patch.shape
        stride_y, stride_x = self.stride
        
        if self.concat_axis == 0:
            axis = 1                
        elif self.concat_axis == 1:
            axis = 0
        else:
            raise ValueError(f"Invalid concat_axis: {self.concat_axis}")

        x_starts = np.arange(0, fpw - pw + 1, stride_x)
        y_starts = np.arange(0, fph - ph + 1, stride_y)
        #print(f"X_starts={x_starts}, Y_starts={y_starts}")
        blocks = [patch[y:y+ph, x:x+pw, :] for x in x_starts for y in y_starts]
        patch = np.concatenate(blocks, axis=axis) 
        return patch
    def get_patch_visualization(
        self, 
        patch: np.ndarray, 
        level: str,
        zfile: Optional[Union[str, os.PathLike]]= None,
        remove_positional_encoding: bool = True,
        restore_complex: bool = None,
        prepare_for_plotting: bool = True,
        vminmax: Optional[Union[Tuple[float, float], str]] = 'auto',
        plot_type: str = "magnitude", 
        denormalize: bool = True,
        ) -> np.ndarray:
        """
        Convert a model-generated patch back to its original visualization format.
        
        This method reverses the operations performed in __getitem__:
        1. Removes positional encoding if present
        2. Restores complex format if needed
        3. Reverses concatenation operations to restore original patch structure
        
        Args:
            patch (np.ndarray): Model-generated patch to visualize
            remove_positional_encoding (bool): Whether to remove positional encoding channels
            restore_complex (bool): Whether to restore complex format. If None, uses self.complex_valued
            vminmax: Value range for visualization
            prepare_for_plotting: Whether to prepare the patch for plotting (e.g., by adding color channels)
            figsize: Figure size for plotting
            
        Returns:
            np.ndarray: Visualized patch
        """
        if restore_complex is None:
            restore_complex = not self.complex_valued
        if isinstance(patch, torch.Tensor):
            patch = patch.detach().cpu().numpy()
        #print(f"Patch shape at the beginning: {patch.shape}")
        # Step 1: Handle input format and remove positional encoding
        if remove_positional_encoding and self.positional_encoding:
            if patch.ndim == 3:
                # Remove last 2 channels (positional encoding)
                if patch.shape[-1] > 2 and np.iscomplexobj(patch):
                    patch = patch[..., :-2]
                elif patch.shape[-1] == 2 and np.iscomplexobj(patch):
                    patch = patch[..., :-1]
            elif patch.ndim == 2 and np.iscomplexobj(patch):
                #The positional encoding is concatenated along one single dimension as complex number
                patch = patch[...: :-1]
            elif patch.ndim == 4:
                patch = patch[..., :-2]
        #print(f"Patch shape after positional encoding removal: {patch.shape}")
        # Step 2: Restore complex format if needed
        if not restore_complex and patch.shape[-1] == 2:
            # Convert from stacked real/imaginary to complex
            patch = patch[..., 0] + 1j * patch[..., 1]
        elif restore_complex and patch.ndim == 3 and patch.shape[-1] == 2:
            # Already in real/imaginary stacked format, convert to complex for processing
            patch = patch[..., 0] + 1j * patch[..., 1]
        #print(f"Patch shape after complex restoration: {patch.shape}")
        # Step 3: Reverse concatenation operations
        if self.concatenate_patches:
            #print(f"Patch shape before concatenation: {patch.shape}")
            patch = self._reverse_concatenation(patch, zfile=zfile)
        #print(f"Patch shape in the end: {patch.shape}")
        if self.transform is not None and denormalize:
            patch = self.transform.inverse(patch, level)
        if prepare_for_plotting:
            patch, vmin, vmax = get_sample_visualization(data=patch, plot_type=plot_type, vminmax=vminmax)
        return patch
    def _reverse_concatenation(self, patch: np.ndarray, zfile: Union[str, os.PathLike]) -> np.ndarray:
        """
        Reverse the concatenation operation performed in _get_concatenated_patch.
        
        For concat_axis=0 (vertical concatenation):
        - Input: (ph, num_patches) where ph is the original patch height
        - Output: (ph * num_patches, 1) - reconstructed as vertical strip
        
        For concat_axis=1 (horizontal concatenation):  
        - Input: (num_patches, pw) where pw is the original patch width
        - Output: (1, pw * num_patches) - reconstructed as horizontal strip
        
        Args:
            patch (np.ndarray): Concatenated patch from model
            
        Returns:
            np.ndarray: Reconstructed patch in original strip format
        """
        stride_y, stride_x = self.stride
        original_ph, original_pw = self.get_patch_size(zfile=zfile)
        mph, mpw = self.get_max_base_sample_size(zfile=zfile)
        original_ph, original_pw = min(mph, original_ph), min(mpw, original_pw)
        if len(patch.shape) == 3:
            patch = patch.squeeze(axis=2)
        elif len(patch.shape) != 2:
            raise ValueError(f"Expected patch to be 2D or 3D, got shape {patch.shape}")
        if self.concat_axis == 1:
            # Reverse vertical concatenation
            # Original: multiple (1, pw) patches stacked vertically -> (num_patches, pw)
            # Reverse: (num_patches, pw) -> (ph_total, pw) where ph_total = num_patches * stride_y
            
            num_patches, pw = patch.shape
            
            # Calculate total height based on stride
            pw_total = num_patches * stride_x
            
            # Create output array
            output = np.zeros((original_ph, pw_total), dtype=patch.dtype)
            
            # Place each patch row at its correct position
            for i in range(num_patches):
                start_y = i * stride_x
                end_y = start_y + pw  # Each original patch was height 1
                output[:, start_y:end_y] = patch[i:i+original_ph, :]
            
            return output
            
        elif self.concat_axis == 0:
            # Reverse horizontal concatenation  
            # Original: multiple (ph, 1) patches stacked horizontally -> (ph, num_patches)
            # Reverse: (ph, num_patches) -> (ph, pw_total) where pw_total = num_patches * stride_x
            
            ph, num_patches = patch.shape
            
            # Calculate total width based on stride
            ph_total = (num_patches-1) * stride_y + ph
            
            # Create output array
            output = np.zeros((num_patches * stride_y // original_pw, original_pw), dtype=patch.dtype)
            
            # Place each patch column at its correct position
            for i in range(0, num_patches, original_pw):

                start_x = i * stride_y // original_pw
                end_x = start_x + ph  # Each original patch was width 1
                #print(patch[:, i:i+1][:50])
                #print(f"Copying patch {i} of coordinates ({0}:{patch.shape[0]}, {i}:{i+original_pw}) of shape {patch[:, i:i+original_pw].shape} to output[{start_x}:{end_x}, :{original_pw}]")
                output[start_x:end_x, :original_pw] = patch[:, i:i+original_pw]

            #print(f"Reverse concatenated shape: {output.shape}")
            return output
            
        else:
            raise ValueError(f"Invalid concat_axis: {self.concat_axis}")
    def visualize_item(self, 
                       idx: Tuple[str, int, int],
                       show: bool = True,
                       vminmax: Optional[Union[Tuple[float, float], str]] = (0, 1000),
                       figsize: Tuple[int, int] = (15, 15)) -> None:
        """
        Visualizes a data sample at the specified index using matplotlib.
        This method retrieves the input and target data corresponding to the given index,
        generates their magnitude visualizations, and displays them side by side as images.
        The visualization can be shown interactively or closed after creation.
        Args:
            idx (Tuple[str, int, int]): The index tuple specifying the data sample to visualize.
            show (bool, optional): If True, displays the plot interactively. If False, closes the figure after creation. Defaults to True.
            vminmax (Optional[Union[Tuple[float, float], str]], optional): Minimum and maximum values for color scaling, or a string (e.g., 'raw') for automatic scaling. Defaults to (0, 1000).
            figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (15, 15).
        Returns:
            None
        """
        import matplotlib.pyplot as plt
        zfile, y, x = idx
        x, y = self._get_base_sample(Path(zfile), y, x) # x, y) 
        imgs = []
        img, vmin, vmax = get_sample_visualization(data=x, plot_type="magnitude", vminmax=vminmax)
        imgs.append({'name': self.level_from, 'img': img, 'vmin': vmin, 'vmax': vmax})
        img, vmin, vmax = get_sample_visualization(data=y, plot_type="magnitude", vminmax=vminmax)
        imgs.append({'name': self.level_to, 'img': img, 'vmin': vmin, 'vmax': vmax})
                
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for i in range(2):
            im = axes[i].imshow(
                imgs[i]['img'],
                aspect='auto',
                cmap='viridis',
                vmin=imgs[i]['vmin'],
                vmax=imgs[i]['vmax']
            )

            axes[i].set_title(f"{imgs[i]['name'].upper()} product")
            axes[i].set_xlabel('Range')
            axes[i].set_ylabel('Azimuth')
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            # axis equal aspect ratio
            axes[i].set_aspect('equal', adjustable='box')
         
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)


    def _sample_parabolic_patch(self, zfile: os.PathLike, level: str, x_center: int, y_start: int) -> np.ndarray:
        """
        Samples a parabolic patch using chunk-aware caching for maximum performance.
        
        For the input level (level_from), samples along the full parabolic curve.
        For the target level (level_to), samples only the central pixels (straight line) for focused comparison.

        Args:
            file_idx (int): Index of the zarr store.
            level (str): Level or key to access the appropriate data within the source.
            x_center (int): Center x-coordinate of the parabolic patch.
            y_start (int): Starting y-coordinate (row index) for the top of the patch.

        Returns:
            np.ndarray: A 2D NumPy array of shape `patch_size` containing the sampled patch, with dtype `np.complex128`.

        Notes:
            - Uses chunk-aware sampling to leverage the existing LRU caching mechanism.
            - Groups coordinates by chunks to minimize chunk loading operations.
            - For level_from: samples full parabolic curve with optimized chunk access.
            - For level_to: samples only central column and uses vectorized broadcasting for replication.
        """
        ph, pw = self.get_patch_size(zfile)
        patch = np.zeros((ph, pw), dtype=np.complex128)
        
        if level == self.level_to:
            # For target level, sample only the central column (straight line)
            # Use chunk-aware sampling for better cache utilization
            center_x = x_center
            y_coords = y_start + np.arange(ph)
            
            # Determine which chunks we need for the center column
            sample = self.get_store_at_level(zfile, level)
            if self.backend == "zarr":
                ch, cw = sample.chunks
                
                # Group y-coordinates by chunks to minimize chunk loads
                chunk_groups = {}
                for i, y in enumerate(y_coords):
                    cy = y // ch
                    if cy not in chunk_groups:
                        chunk_groups[cy] = []
                    chunk_groups[cy].append((i, y))
                
                # Sample from each chunk efficiently
                for cy, coords_in_chunk in chunk_groups.items():
                    cx = center_x // cw
                    chunk = self._load_chunk(zfile, level, cy, cx)
                    
                    for i, y in coords_in_chunk:
                        dy = y - cy * ch
                        dx = center_x - cx * cw
                        if dy < chunk.shape[0] and dx < chunk.shape[1]:
                            patch[i, pw // 2] = chunk[dy, dx]
            else:
                # For dask backend, use original approach
                for i in range(ph):
                    patch[i, pw // 2] = self._get_sample(zfile, level, y_coords[i], center_x, 1, 1).item()
            
            # Vectorized replication of center column using broadcasting
            center_column = patch[:, pw // 2]
            patch = np.broadcast_to(center_column[:, np.newaxis], (ph, pw)).copy()
                    
        else:
            # For input level (level_from), use chunk-aware parabolic sampling
            a = self.parabola_a
            
            # Vectorized coordinate generation
            i_indices = np.arange(ph)
            j_indices = np.arange(pw)
            I_grid, J_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
            
            Y_coords = y_start + I_grid
            offsets = J_grid - pw // 2
            X_coords = x_center + a * offsets ** 2
            X_coords = X_coords.astype(int)
            
            if self.backend == "zarr":
                # Group coordinates by chunks for efficient cache utilization
                sample = self.get_store_at_level(zfile, level)
                ch, cw = sample.chunks
                
                chunk_groups = {}
                for i in range(ph):
                    for j in range(pw):
                        y, x = Y_coords[i, j], X_coords[i, j]
                        cy, cx = y // ch, x // cw
                        chunk_key = (cy, cx)
                        if chunk_key not in chunk_groups:
                            chunk_groups[chunk_key] = []
                        chunk_groups[chunk_key].append((i, j, y, x))
                
                # Sample from each chunk efficiently
                for (cy, cx), coords_in_chunk in chunk_groups.items():
                    chunk = self._load_chunk(zfile, level, cy, cx)
                    
                    for i, j, y, x in coords_in_chunk:
                        dy, dx = y - cy * ch, x - cx * cw
                        if dy < chunk.shape[0] and dx < chunk.shape[1]:
                            patch[i, j] = chunk[dy, dx]
            else:
                # For dask backend, use original vectorized approach
                for i in range(ph):
                    for j in range(pw):
                        patch[i, j] = self._get_sample(zfile, level, Y_coords[i, j], X_coords[i, j], 1, 1).item()
        
        return patch
    
    def add_position_embedding(self, inp: np.ndarray, pos: Tuple[int, int], max_length: Tuple[int, int], level:str) -> np.ndarray:
        """
        Add both horizontal and vertical position embedding to the input numpy array.
        Uses cached position arrays for better performance.

        Args:
            inp (np.ndarray): Input array of shape (ph, pw, 2) for real-valued, or (ph, pw) for complex-valued.
            pos (Tuple[int, int]): Position tuple (y_offset, x_offset) for global positioning.
            max_length (Tuple[int, int]): Maximum length tuple (max_y, max_x) for normalization.

        Returns:
            np.ndarray: Output array with position embeddings appended as the last channels.
                        - For real-valued input (ph, pw, 2): returns (ph, pw, 4) - adds 2 position channels
                        - For complex input (ph, pw): returns (ph, pw, 4) - converts to real + adds 2 position channels
        """
        y_offset, x_offset = pos
        max_y, max_x = max_length
        ph, pw = inp.shape[:2]
        if inp.ndim == 2:
            inp = inp.reshape(ph, pw, 1)
        elif inp.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input patch for positional encoding. Got patch of shape: {inp.shape}")
        ph, pw, ch = inp.shape
        # # Handle input format conversion
        # if inp.ndim == 3 and inp.shape[2] == 2:
        #     # Real-valued input (ph, pw, 2)
        #     ph, pw, channels = inp.shape
        #     inp_flat = inp.reshape(-1, channels)
        # elif inp.ndim == 2:  
        #     # Complex-valued input (ph, pw) - convert to real format
        #     ph, pw = inp.shape
        #     # Convert complex to real format: stack real and imaginary parts
        #     #inp_real = np.stack([np.real(inp), np.imag(inp)], axis=-1)
        #     #inp_flat = inp_real.reshape(-1, 2)
        #     inp_flat = inp.reshape(-1, 1)
        #     channels = 1
        # else:
        #     raise ValueError("Input shape not recognized for positional embedding.")

        # Check if cached position arrays exist and have correct dimensions
        cache_key = (ph, pw)
        
        if (not hasattr(self, '_y_positions') or 
            not hasattr(self, '_x_positions') or 
            not hasattr(self, '_pos_cache_key') or
            self._pos_cache_key != cache_key):
            
            # Recompute position arrays
            self._y_positions = np.repeat(np.arange(ph), pw).reshape(ph, pw, 1)  # [0,0,0,...,pw times, 1,1,1,...,pw times, ...]
            self._x_positions = np.tile(np.arange(pw), ph).reshape(ph, pw, 1)    # [0,1,2,...,pw-1, 0,1,2,...,pw-1, ...]
            self._pos_cache_key = cache_key
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Recomputed position arrays for patch size {cache_key}")
        
        # if level not in self._pos_encoding_out.keys():
        #     self._pos_encoding_out[level] = np.zeros((ph, pw, ch + 2), dtype=inp.dtype) 
        
        # Add global offsets to cached arrays
        global_y_positions = y_offset + self._y_positions
        global_x_positions = x_offset + self._x_positions
        
        # Normalize positions to [0, 1] range
        y_position_embedding = (global_y_positions / max_y)
        x_position_embedding = (global_x_positions / max_x)
        
        if np.iscomplexobj(inp):
            # Create a complex positional embedding: real=x, imag=y
            pos_embedding = x_position_embedding[..., 0] + 1j * y_position_embedding[..., 0]
            pos_embedding = pos_embedding[..., np.newaxis]  # shape (ph, pw, 1)
            out = np.concatenate((inp, pos_embedding), axis=-1)
        else:
            out = np.concatenate((inp, y_position_embedding, x_position_embedding), axis=-1)
        return out
        # self._pos_encoding_out[level][..., :ch] = inp
        # self._pos_encoding_out[level][..., ch] = y_position_embedding[..., 0]
        # self._pos_encoding_out[level][..., ch + 1] = x_position_embedding[..., 0]
        # Reshape back to patch format with additional channels
        #out = out.reshape(ph, pw, channels + 2)
        
        # return self._pos_encoding_out[level]
class KPatchSampler(Sampler):
    """
    PyTorch Sampler that yields (file_idx, y, x) tuples for patch sampling.
    Draws k patches from each file in round-robin, with optional shuffling of files and patches.

    Args:
        dataset (SARZarrDataset): The dataset to sample patches from.
        samples_per_prod (int): Number of patches to sample per file. If 0, all patches are sampled.
        shuffle_files (bool): Whether to shuffle the order of files between epochs.
        shuffle_patches (bool): Whether to shuffle the patches within each file.
        seed (int): Random seed for reproducibility.
        verbose (bool): If True, prints timing and sampling info.
        max_products (int): Maximum number of products to sample from.
        patch_order (str): Order in which to sample patches from each file. Options are "row", "col", or "chunk".
        samples_per_prod (int): Number of patches to sample per product. If 0, all patches are sampled.
    """
    def __init__(
        self,
        dataset: SARZarrDataset,
        samples_per_prod: int = 0,
        shuffle_files: bool = True,
        seed: int = 42, 
        verbose: bool = True, 
        patch_order: str = "row",
    ):
        self.dataset = dataset
        self.samples_per_prod = samples_per_prod
        self.shuffle_files = shuffle_files
        self.patch_order = patch_order
        self.seed = seed
        self.verbose = verbose
        self.beginning = True
        self.coords: Dict[Path, List[Tuple[int, int]]] = {}
        self.zfiles = None
    def __iter__(self):
        """
        Iterate over the dataset, yielding (file_idx, y, x) tuples for patch sampling.
        Shuffles files and/or patches if enabled.
        """
        self.beginning = False
        rng = np.random.default_rng(self.seed)
        files = self.dataset.get_files() #files.copy()
        if self.verbose:
            print(files)
        if self.shuffle_files:
            rng.shuffle(files)
        for idx, f in enumerate(files):
            # Mark patches as loaded for this file
            if self.zfiles is not None and idx not in self.zfiles and f not in self.zfiles:
                if self.verbose:
                    print(f"Skipping file {f} as it's not in the filtered list.")
                continue
            if self.dataset.get_samples_by_file(f) is not None and len(self.dataset.get_samples_by_file(f)) == 0:
                # print(f"Calculating patches for file {f}")
                self.dataset.calculate_patches_from_store(f, patch_order=self.patch_order)
            self.coords[Path(f)] = self.get_coords_from_store(f)
            t0 = time.time()
            for y, x in self.coords[Path(f)]:
                if self.verbose:
                    print(f"Sampling from file {f}, patch at ({y}, {x})")
                yield (f, y, x)
            elapsed = time.time() - t0
            if self.verbose:
                print(f"Sampling {len(self.coords[Path(f)])} patches from file {f} took {elapsed:.2f} seconds.")
    def get_coords_from_store(self, zfile: Union[str, os.PathLike], window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None):
        if self.dataset.get_samples_by_file(zfile) is not None and len(self.dataset.get_samples_by_file(zfile)) == 0:
            # print(f"Calculating patches for file {zfile}")
            self.dataset.calculate_patches_from_store(Path(zfile), patch_order=self.patch_order, window=window)
        lazy_coords = self.dataset.get_samples_by_file(zfile)
        
        # If samples_per_prod is specified, limit the coordinates
        if self.samples_per_prod > 0:
            limited_coords = []
            for i, coord in enumerate(lazy_coords):
                if i >= self.samples_per_prod:
                    break
                limited_coords.append(coord)
            return limited_coords
        else:
            # Return the full lazy generator
            return lazy_coords
    def filter_by_zfiles(self, zfiles: Union[List[Union[str, os.PathLike]], str, int]) -> None:
        """
        Filter the dataset to only include samples from the specified list of zarr files.
        
        Args:
            zfiles (List[Union[str, os.PathLike]]): List of zarr file paths to include.
        """
        if isinstance(zfiles, (str, int, Path)):
            zfiles = [zfiles]
        self.zfiles = [Path(zf) if not isinstance(zf, (Path, int)) else zf for zf in zfiles]

    def __len__(self):
        """Return the total number of samples to be drawn by the sampler."""
        if self.beginning:
            return len(self.dataset)
        else:
            total = 0
            for zfile in self.dataset.get_files():
                lazy_coords = self.dataset.get_samples_by_file(zfile)
                if self.samples_per_prod > 0:
                    total += min(self.samples_per_prod, len(lazy_coords))
                else:
                    total += len(lazy_coords)
            return total

class SARDataloader(DataLoader):
    dataset: SARZarrDataset
    def __init__(self, dataset: SARZarrDataset, batch_size: int, sampler: KPatchSampler,  num_workers: int = 2,  pin_memory: bool= False, verbose: bool = False):
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=pin_memory)
        self.verbose = verbose
    def get_coords_from_zfile(self, zfile: Union[str, os.PathLike], window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        return self.sampler.get_coords_from_store(zfile, window=window)
    def filter_by_zfiles(self, zfiles: Union[List[Union[str, os.PathLike]], str, int]) -> None:
        """
        Filter the dataset to only include samples from the specified list of zarr files.
        
        Args:
            zfiles (List[Union[str, os.PathLike]]): List of zarr file paths to include.
        """
        self.sampler.filter_by_zfiles(zfiles)
        if self.verbose:
            print(f"Filtered dataset to {len(self.dataset)} samples from {len(zfiles)} files.")

    

def get_sar_dataloader(
    data_dir: str,
    filters: Optional[SampleFilter] = None,
    batch_size: int = 8,
    num_workers: int = 2,
    return_whole_image: bool = False,
    transform: Optional[Callable]=None,
    level_from: str = "rcmc",
    level_to: str = "az",
    patch_mode: str = "rectangular",
    patch_size: Tuple[int, int] = (512, 512),
    buffer: Tuple[int, int] = (100, 100),
    stride: Tuple[int, int] = (50, 50),
    block_pattern: Optional[Tuple[int, int]] = None,
    positional_encoding: bool = True,
    max_base_sample_size: Tuple[int, int] = (-1, -1),  # (-1, -1) means full image size
    backend: str = "zarr",  # "zarr" or "dask
    parabola_a: float = 0.001,
    shuffle_files: bool = True,
    patch_order: str = "row",  # "row", "col", or "chunk"
    complex_valued: bool = False,
    save_samples: bool = True, 
    verbose: bool = True, 
    cache_size: int = 10000, 
    max_products: int = 10, 
    samples_per_prod: int = 0,
    online: bool = True, 
    concatenate_patches: bool = False,
    concat_axis: int = 0,  # 0 for vertical, 1 for horizontal
    geographic_clustering: bool = False,  # Enable geographic clustering
    n_clusters: int = 10,  # Number of geographic clusters, 
    use_balanced_sampling: bool = True, 
    split: str = "train"
) -> SARDataloader:
    """
    Create and return a PyTorch DataLoader for SAR data using SARZarrDataset and KPatchSampler.

    Args:
        data_dir (str): Path to the directory containing SAR data.
        file_pattern (str, optional): Glob pattern for Zarr files. Defaults to "*.zarr".
        batch_size (int, optional): Number of samples per batch. Defaults to 8.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 2.
        return_whole_image (bool, optional): If True, returns the whole image. Defaults to False.
        transform (callable, optional): Optional transform to apply to samples.
        level_from (str, optional): Input SAR processing level. Defaults to "rcmc".
        level_to (str, optional): Target SAR processing level. Defaults to "az".
        patch_mode (str, optional): Patch extraction mode. Defaults to "rectangular".
        patch_size (Tuple[int, int], optional): Patch size. Defaults to (512, 512).
        buffer (Tuple[int, int], optional): Buffer size. Defaults to (100, 100).
        stride (Tuple[int, int], optional): Stride for patch extraction. Defaults to (50, 50).
        positional_encoding (bool, optional): If True, adds positional encoding to patches. Defaults to True.
        backend (str, optional): Backend for loading Zarr data. Defaults to "zarr".
        parabola_a (float, optional): Parabola parameter for patch extraction. Defaults to 0.001.
        shuffle_files (bool, optional): Shuffle file order. Defaults to True.
        patch_order (str, optional): Patch extraction order. Defaults to "row".
        complex_valued (bool, optional): If True, loads data as complex-valued. Defaults to False.
        save_samples (bool, optional): If True, saves sampled patches. Defaults to True.
        cache_size (int, optional): LRU cache size for chunks. Defaults to 10000.
        max_products (int, optional): Maximum number of products. Defaults to 10.
        samples_per_prod (int, optional): Number of patches per product. Defaults to 0 (all patches).
        online (bool, optional): If True, uses online data loading. Defaults to True.
        verbose (bool, optional): If True, prints additional info. Defaults to True.
        geographic_clustering (bool, optional): If True, clusters data by geographic location. Defaults to False.
        n_clusters (int, optional): Number of geographic clusters when clustering is enabled. Defaults to 10.
        split (str, optional): Dataset split to use (e.g., "train", "val", "test"). Defaults to "train".

    Returns:
        SARDataloader: PyTorch DataLoader for the SAR dataset.
    """
    dataset = SARZarrDataset(
        data_dir=data_dir,
        filters=filters,
        return_whole_image=return_whole_image,
        transform=transform,
        patch_size=patch_size,
        block_pattern=block_pattern,
        complex_valued=complex_valued,
        level_from=level_from,
        level_to=level_to,
        patch_mode=patch_mode,
        parabola_a=parabola_a, 
        save_samples= save_samples, 
        buffer = buffer, 
        stride=stride, 
        max_base_sample_size = max_base_sample_size,
        backend=backend, 
        verbose=verbose, 
        cache_size=cache_size, 
        online=online, 
        max_products=max_products, 
        samples_per_prod=samples_per_prod, 
        positional_encoding=positional_encoding, 
        concatenate_patches=concatenate_patches,
        concat_axis=concat_axis, 
        use_balanced_sampling=use_balanced_sampling,
        split=split
    )
    sampler = KPatchSampler(
        dataset,
        samples_per_prod=samples_per_prod,
        shuffle_files=shuffle_files,
        patch_order=patch_order,
        verbose=verbose
    )
    return SARDataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        verbose=verbose
    )

# Example usage
if __name__ == "__main__":
    
    # Create SARTransform using the factory method
    transforms = SARTransform.create_minmax_normalized_transform(
        normalize=True,
        rc_min=RC_MIN,
        rc_max=RC_MAX,
        gt_min=GT_MIN,
        gt_max=GT_MAX,
        complex_valued=True
    )
    
    loader = get_sar_dataloader(
       data_dir="/Data/sar_focusing",
       level_from="rc",
       level_to="az",
       batch_size=4,
       num_workers=4,
       patch_mode="rectangular", 
       complex_valued=False, 
       shuffle_files=False, 
       patch_order="col", 
       transform=transforms,
       max_base_sample_size=(-1, -1)
       #patch_mode="parabolic",
       #parabola_a=0.0005,
       #k=10
    )
    for i, (x_batch, y_batch) in enumerate(loader):
        print(f"Batch {i}: x {x_batch.shape}, y {y_batch.shape}")
        #break
