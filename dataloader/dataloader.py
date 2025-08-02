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
import dask.array as da
import time 
import os
import functools
from utils import get_chunk_name_from_coords, get_sample_visualization, get_zarr_version
from api import list_base_files_in_repo
from utils import normalize, extract_stride_number_from_filename, RC_MAX, RC_MIN, GT_MAX, GT_MIN
from api import fetch_chunk_from_hf_zarr
from api import download_metadata_from_product

class SARTransform(nn.Module):
    """
    PyTorch transform module for normalizing SAR data patches at different processing levels.

    This class allows you to specify a separate transformation function for each SAR processing level (e.g., 'raw', 'rc', 'rcmc', 'az').
    Each function should accept a NumPy array and return a transformed NumPy array.

    Args:
        transform_raw (callable, optional): Function to transform 'raw' level data.
        transform_rc (callable, optional): Function to transform 'rc' level data.
        transform_rcmc (callable, optional): Function to transform 'rcmc' level data.
        transform_az (callable, optional): Function to transform 'az' level data.
    """
    def __init__(
        self,
        transform_raw: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial]] = None,
        transform_rc: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial]] = None,
        transform_rcmc: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial]] = None,
        transform_az: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial]] = None,
    ):
        super(SARTransform, self).__init__()
        self.transforms: dict[str, Optional[Callable[[np.ndarray], np.ndarray]]] = {
            'raw': transform_raw if transform_raw else None,
            'rc': transform_rc if transform_rc else None,
            'rcmc': transform_rcmc if transform_rcmc else None,
            'az': transform_az if transform_az else None,
        }

    def forward(self, x: np.ndarray, level: str) -> np.ndarray:
        """
        Apply the appropriate transform to the input array for the specified SAR processing level.

        Args:
            x (np.ndarray): Input data array.
            level (str): SAR processing level ('raw', 'rc', 'rcmc', or 'az').

        Returns:
            np.ndarray: Transformed data array.
        """
        assert level in self.transforms, f"Transform for level '{level}' not defined."
        if self.transforms[level] is None:
            return x
        else:
            return self.transforms[level](x)

class SARZarrDataset(Dataset):
    """
    PyTorch Dataset for loading SAR (Synthetic Aperture Radar) data patches from Zarr format archives.

    Supports efficient patch sampling from multiple Zarr files, with rectangular, or parabolic patch extraction. Handles both local and remote (Hugging Face) Zarr stores, with on-demand patch downloading and LRU chunk caching.

    Features:
        - Loads SAR data patches from Zarr stores (local or remote), supporting real and complex-valued data.
        - Multiple patch sampling modes: "rectangular", "parabolic".
        - Efficient patch coordinate indexing and caching for fast repeated access.
        - Optional patch transformation and visualization utilities.
        - Handles both input and target SAR processing levels (e.g., "rcmc" and "az").
        - Supports saving/loading patch indices to avoid recomputation.
        - Implements chunk-level LRU caching for efficient repeated access.

    Args:
        data_dir (str): Directory containing Zarr files.
        file_pattern (str, optional): Glob pattern for Zarr files. Defaults to "*.zarr".
        repo_id (str, optional): Hugging Face repo ID for remote access. Defaults to 'sirbastiano94/Maya4'.
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
        backend (str, optional): Backend for loading Zarr data, either "zarr" or "dask". Defaults to "zarr".
        cache_size (int, optional): Maximum number of chunks to cache in memory.
        positional_encoding (bool, optional): If True, adds positional encoding to input patches. Defaults to True.
        dataset_length (int, optional): Optional override for dataset length.
        max_products (int, optional): Maximum number of Zarr products to use. Defaults to 10.
        samples_per_prod (int, optional): Number of patches to sample per product. Defaults to 1000.

    Example:
        >>> dataset = SARZarrDataset("/path/to/zarrs", patch_size=(128, 128), cache_size=1000)
        >>> x, y = dataset[("path/to/zarr", 100, 100)]
        >>> dataset.visualize_item(("path/to/zarr", 100, 100))
    """
    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "*.zarr",
        repo_id: str = 'sirbastiano94/Maya4',
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
        backend: str = "zarr",  # "zarr" or "dask"
        verbose: bool= True, 
        cache_size: int = 1000, 
        positional_encoding: bool = True, 
        dataset_length: Optional[int] = None, 
        max_products: int = 10, 
        samples_per_prod: int = 1000
    ):
        self.data_dir = Path(data_dir)
        self.file_pattern = file_pattern
        self.repo_id = repo_id
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
        self.backend = backend
        self.verbose = verbose
        self.save_samples = save_samples
        self.online = online
        self.positional_encoding = positional_encoding
        self._dataset_length = dataset_length
        self._max_products = max_products
        self._samples_per_prod = samples_per_prod
        
        self._load_chunk = functools.lru_cache(maxsize=cache_size)(
            self._load_chunk_uncached
        )

        self._stores: Dict[os.PathLike, zarr.core.Array] = {}
        self._samples_by_file: Dict[os.PathLike, List[Tuple[int,int]]] = {}
        self._y_coords: Dict[os.PathLike, np.ndarray] = {}
        self._x_coords: Dict[os.PathLike, np.ndarray] = {}
        #self.init_samples()
        self._initialize_stores()

    def _get_patch_size(self, zfile: os.PathLike) -> Tuple[int, int]:
        """
        Retrieve the patch size for a given Zarr file based on its processing level.
        
        Args:
            zfile (os.PathLike): Path to the Zarr file.

        Returns:
            Tuple[int, int]: Patch size (height, width) for the specified processing level.
        """
        ph, pw = self._patch_size
        if ph > 0 and pw > 0:
            return ph, pw
        else:
            if self.backend == "zarr":
                y, x = self._stores[zfile][self.level_from].shape
            elif self.backend == "dask":
                y, x = self._stores[zfile][self.level_from].shape[1:]
            if ph <= 0:
                ph = y - 2*self.buffer[0]
            if pw <= 0:
                pw = x - 2*self.buffer[1]
            return ph, pw

    def _get_file_list(self):
        """
        Retrieve the list of Zarr files to use, either from the local directory or from a remote Hugging Face repository.
        Filters files using the provided glob pattern and limits to max_products.
        """
        if self.online:
            import fnmatch
            self.remote_files = list_base_files_in_repo(
                repo_id=self.repo_id,
            )
            # Convert glob pattern to regex for matching
            regex_pattern = fnmatch.translate(self.file_pattern)
            filename_regex = re.compile(regex_pattern)
            if self.verbose:
                print(f"Found {len(self.remote_files)} files in the remote repository: '{self.remote_files}'")
            matched_files = [Path(self.data_dir).joinpath(f) for f in self.remote_files if filename_regex.match(f)]
            self.files = sorted(matched_files)[:min(self._max_products, len(matched_files))]
        else:
            self.files = sorted(self.data_dir.glob(self.file_pattern))
            self.files = self.files[:min(self._max_products, len(self.files))]
        if self.verbose:
            print(f"Selected only files:  {self.files}")

    def _append_file_to_stores(self, zfile: os.PathLike):
        """
        Appends a Zarr file to the stores dictionary, opening it in read-only mode.
        This method is used to initialize the dataset with a specific Zarr file.
        
        Args:
            zfile (os.PathLike): Path to the Zarr file to be added.
        """
        if not zfile.exists():
            return
        if self.backend == "zarr":
            store = LocalStore(str(zfile))
            self._stores[zfile] = zarr.open(store, mode='r')
        elif self.backend == "dask":
            self._stores[zfile] = {}
            for level in (self.level_from, self.level_to):
                complete_path = os.path.join(zfile, level) 
                self._stores[zfile][level] = da.from_zarr(complete_path) #.rechunk(self.patch_size)  
        else:
            raise ValueError(f"Unknown backend {self.backend}")

    def _initialize_stores(self):
        """
        Initializes data stores based on the selected backend.

        For the "zarr" backend, opens each file as a Zarr store in read-only mode and stores it in `self._stores`.
        For the "dask" backend, creates a dictionary for each file, loading data for each specified level using Dask arrays
        with the given patch size for rechunking.
        Raises:
            ValueError: If an unknown backend is specified.
        """
        t0 = time.time()
        self._get_file_list()
        if self.verbose:
            dt = time.time() - t0
            print(f"Files list calculation took {dt:.2f} seconds.")
            
        t0 = time.time()
        for zfile in self.files:
            self._append_file_to_stores(zfile)
        if self.verbose:
            df = time.time() - t0
            print(f"Zarr stores initialization took {df:.2f} seconds.")

    def calculate_patches_from_store(self, zfile:os.PathLike, patch_order: str = "row_major"):
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
                download_metadata_from_product(
                    zfile_name=str(zfile_name),
                    local_dir=self.data_dir, 
                    levels=[self.level_from, self.level_to], 
                    repo_id=self.repo_id
                )
            else:
                raise ValueError(f"Levels {self.level_from} and {self.level_to} not found in Zarr store {zfile}.")
        
        self._append_file_to_stores(zfile)
        # Print the directories (groups/arrays) inside the Zarr store

        h, w = self._stores[zfile][self.level_from].shape
        coords: List[Tuple[int,int]] = []
        if self.return_whole_image:
            coords = [(0, 0)]
        else:
            stride_y, stride_x = self.stride
            ph, pw = self._get_patch_size(zfile)
            if self.patch_mode == "rectangular":
                
                y_min, y_max = self.buffer[0], h - self.buffer[0]
                x_min, x_max = self.buffer[1], w - self.buffer[1]

                self._y_coords[zfile] = np.arange(y_min, y_max - stride_y + 1, stride_y)
                self._x_coords[zfile] = np.arange(x_min, x_max - stride_x + 1, stride_x)

            elif self.patch_mode == "parabolic":
                a = self.parabola_a
                
                # For parabolic patches, we need to ensure the entire parabolic curve fits within image bounds
                # The parabolic curve is x = x_center + a * (j - pw//2)^2 for j in [0, pw)
                # Maximum x offset occurs at the edges: max_offset = a * (pw//2)^2
                max_offset = int(np.ceil(a * (pw // 2) ** 2))
                
                # Apply buffer and ensure parabolic patch fits
                y_min, y_max = self.buffer[0], h - ph - self.buffer[0]
                x_min = self.buffer[1] + max_offset  # Left bound considering parabolic curve
                x_max = w - self.buffer[1] - max_offset  # Right bound considering parabolic curve        
                
                # Fully vectorized coordinate generation for maximum performance
                self._y_coords[zfile] = np.arange(y_min, y_max + 1, stride_y)
                self._x_coords[zfile] = np.arange(x_min, x_max + 1, stride_x)
            else:
                raise ValueError(f"Unknown patch_mode {self.patch_mode}")
        coords = self.reorder_samples(zfile, patch_order=patch_order)
        self._samples_by_file[zfile] = coords
            
    def reorder_samples(
        self, zfile: os.PathLike, patch_order: str = "row"
    ) -> List[Tuple[int, int]]:
        """
        Create ordered coordinates using np.meshgrid indexing parameter.
        
        Args:
            zfile: Path to the Zarr file
            patch_order: Ordering strategy for patch coordinates.

        Returns:
            List of (y, x) coordinate tuples in the specified order
        """
        
        if patch_order == "row":
            # Row-major: left-to-right, top-to-bottom (default 'xy' indexing)
            Y_grid, X_grid = np.meshgrid(
                self._y_coords[zfile], 
                self._x_coords[zfile], 
                indexing='ij' 
            )
            coords = list(zip(Y_grid.flatten().tolist(), X_grid.flatten().tolist()))
            
        elif patch_order == "col":
            # Column-major: top-to-bottom, left-to-right
            # Swap the order of coordinates in meshgrid and use different indexing
            X_grid, Y_grid = np.meshgrid(
                self._x_coords[zfile],
                self._y_coords[zfile], 
                indexing='ij'  
            )
            # For column-major, we want to iterate through Y first, then X
            coords = list(zip(Y_grid.flatten(), X_grid.flatten()))
            
        elif patch_order == "chunk":
            # Chunk-aware ordering: group by Zarr chunks
            coords = self._create_chunk_aware_coordinates(zfile)
            
        else:
            raise ValueError(f"Unknown patch_order: {patch_order}")
        
        return coords

    def _create_chunk_aware_coordinates(self, zfile: os.PathLike) -> List[Tuple[int, int]]:
        """
        Generate a list of (y, x) coordinate tuples ordered by their Zarr chunk locations for improved cache performance.

        This method creates a meshgrid of all possible (y, x) coordinates based on the internal coordinate arrays for the given Zarr file.
        It then determines the chunk index for each coordinate, sorts all coordinates so that those belonging to the same chunk are grouped together,
        and returns the sorted list. This chunk-aware ordering can significantly improve performance when accessing chunked array data.

        Args:
            zfile (os.PathLike): The path to the Zarr file for which to generate chunk-aware coordinates.

        Returns:
            List[Tuple[int, int]]: A list of (y, x) coordinate tuples, sorted such that coordinates within the same chunk are contiguous.

        Notes:
            - The chunk sizes are inferred from the Zarr store at the specified processing level (`self.level_from`).
            - The method assumes that `self._stores`, `self._y_coords`, and `self._x_coords` are properly initialized and populated.
            - The returned coordinate order is optimized for sequential chunk access, which can reduce I/O overhead and improve cache utilization.
        """

        # Get chunk size and patch size
        sample = self._stores[zfile][self.level_from]
        ch, cw = sample.chunks
        
        # Create coordinate grids
        Y_grid, X_grid = np.meshgrid(
            self._y_coords[zfile], 
            self._x_coords[zfile], 
            indexing='ij'
        )
        
        # Calculate chunk indices
        chunk_y_indices = Y_grid // ch
        chunk_x_indices = X_grid // cw
        
        # Create a combined sorting key for chunk-aware ordering
        # Primary sort: chunk_y, then chunk_x (process chunks row-major)
        # Secondary sort: x_coord within chunk, then y_coord within column
        max_chunk_x = chunk_x_indices.max() + 1
        max_x = X_grid.max() + 1
        
        # Multi-level sorting key:
        # 1. Chunk row (chunk_y)
        # 2. Chunk column (chunk_x) 
        # 3. X coordinate within chunk (for column-wise processing)
        # 4. Y coordinate within column (for vertical sampling)
        sort_key = (
            chunk_y_indices * max_chunk_x * max_x * Y_grid.max() +
            chunk_x_indices * max_x * Y_grid.max() +
            X_grid * Y_grid.max() +
            Y_grid
        )
        
        # Get sort indices
        sort_indices = np.argsort(sort_key.flatten())
        
        # Apply sorting to coordinates
        flat_y, flat_x = Y_grid.flatten(), X_grid.flatten()
        coords = [(int(flat_y[i]), int(flat_x[i])) for i in sort_indices]
        
        return coords
    
    def __len__(self):
        """
        Return the total number of patches in the dataset (samples_per_prod * max_products).
        """
        return self._samples_per_prod * self._max_products

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
        stride_number = extract_stride_number_from_filename(zfile.name)
        zfile = Path(zfile)
        t0 = time.time()

        # Retrieve the horizontal size (width) from the store for self.level_from
        width = self._stores[zfile][self.level_from].shape[1]
        # You can now use 'width' as needed in your logic

        if self.patch_mode == "parabolic":
            patch_from = self._sample_parabolic_patch(zfile, self.level_from, x, y)
            patch_to = self._sample_parabolic_patch(zfile, self.level_to, x, y)
        else:
            ph, pw = self._get_patch_size(zfile)
            patch_from = self._get_sample(zfile, self.level_from, y, x, ph, pw)
            patch_to = self._get_sample(zfile, self.level_to, y, x, ph, pw)

        if self.transform:
            patch_from = self.transform(patch_from, self.level_from)
            patch_to = self.transform(patch_to, self.level_to)

        if not self.complex_valued:
            patch_from = np.stack((patch_from.real, patch_from.imag), axis=-1).astype(np.float32)
            patch_to = np.stack((patch_to.real, patch_to.imag), axis=-1).astype(np.float32)

        if self.positional_encoding:
            index = y + stride_number * width
            patch_from = self.add_position_embedding(patch_from, index)
        x_tensor = torch.from_numpy(patch_from)
        y_tensor = torch.from_numpy(patch_to)
        if self.verbose:
            elapsed = time.time() - t0
            print(f"Loading patch ({zfile}, {y}, {x}) took {elapsed:.2f} seconds. Stride number: {stride_number}")

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
        if not zfile.exists():
            if self.online:
                meta_file = download_metadata_from_product(
                    zfile_name=str(zfile_name),
                    local_dir=self.data_dir, 
                    levels=[self.level_from, self.level_to], 
                    repo_id=self.repo_id
                )
                with open(meta_file) as f:
                    zarr_meta = json.load(f)
                version = zarr_meta.get('zarr_format', 2)
                self.calculate_patches_from_store(zfile)
            else:
                raise FileNotFoundError(f"Zarr file {zfile} does not exist.")
        
        chunk_name = get_chunk_name_from_coords(y, x, zarr_file_name=zfile_name, level=level, chunks=self._stores[zfile][level].chunks, version=get_zarr_version(zfile))
        chunk_path = self.data_dir / chunk_name
        
        if not chunk_path.exists():
            if self.verbose:
                print(f"Chunk {chunk_name} not found locally. Downloading from Hugging Face Zarr archive...")
            zfile_name = os.path.basename(zfile)
            fetch_chunk_from_hf_zarr(level=level, y=y, x=x, zarr_archive=zfile_name, local_dir=self.data_dir, repo_id=self.repo_id)
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
        arr = self._stores[zfile][level]
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
        
        return chunk_data.astype(np.complex64)
    
    def _get_sample_from_cached_chunks_vectorized(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int) -> np.ndarray:
        """
        Optimized version that dispatches to specialized methods based on patch geometry.
        """
        arr = self._stores[zfile][level]
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
        patch = np.zeros((ph, pw), dtype=np.complex64)
        print("Loading large patch from chunks...")
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
                    patch[dst_y1:dst_y2, dst_x1:dst_x2] = chunk[src_y1:src_y2, src_x1:src_x2]
        
        return patch
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
        patch = np.zeros((ph, pw), dtype=np.complex64)
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
            strip = np.zeros(length, dtype=np.complex64)
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
                        strip[current_pos:current_pos + slice_height] = chunk[local_start:local_end, dx]
                        current_pos += slice_height
            return strip.reshape(length, 1)
        elif axis == 1:
            # Horizontal strip: patch_size=(1, length)
            cx_start, cx_end = x // cw, (x + length - 1) // cw
            cy = y // ch
            strip = np.zeros(length, dtype=np.complex64)
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
                        strip[current_pos:current_pos + slice_width] = chunk[dy, local_start:local_end]
                        current_pos += slice_width
            return strip.reshape(1, length)
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
        arr = self._stores[zfile][level]
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
                patch = np.zeros((ph, pw), dtype=np.complex64)
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
            sample = self._stores[zfile][level]
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
            return arr.astype(np.complex64)
        elif self.backend == "zarr":
            patch = self._get_sample_from_cached_chunks(zfile, level, y, x, ph, pw)
            return patch
        else:
            raise ValueError(f"Unknown backend {self.backend}")
        
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
        x, y = self[idx]
        imgs = []
        img, vmin, vmax = get_sample_visualization(data=x.detach().cpu().numpy(), plot_type="magnitude", vminmax='raw')
        imgs.append({'name': self.level_from, 'img': img, 'vmin': vmin, 'vmax': vmax})
        img, vmin, vmax = get_sample_visualization(data=y.detach().cpu().numpy(), plot_type="magnitude")
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

            axes[i].set_title(f'{imgs[i]['name'].upper()} product')
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
            np.ndarray: A 2D NumPy array of shape `patch_size` containing the sampled patch, with dtype `np.complex64`.

        Notes:
            - Uses chunk-aware sampling to leverage the existing LRU caching mechanism.
            - Groups coordinates by chunks to minimize chunk loading operations.
            - For level_from: samples full parabolic curve with optimized chunk access.
            - For level_to: samples only central column and uses vectorized broadcasting for replication.
        """
        ph, pw = self._get_patch_size(zfile)
        patch = np.zeros((ph, pw), dtype=np.complex64)
        
        if level == self.level_to:
            # For target level, sample only the central column (straight line)
            # Use chunk-aware sampling for better cache utilization
            center_x = x_center
            y_coords = y_start + np.arange(ph)
            
            # Determine which chunks we need for the center column
            sample = self._stores[zfile][level]
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
                sample = self._stores[zfile][level]
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
    def add_position_embedding(self, inp: np.ndarray, item_index: int, max_length: int = 10000) -> np.ndarray:
        """
        Add a position embedding column to the input numpy array.

        Args:
            inp (np.ndarray): Input array of shape (ph, pw, 2) for real-valued, or (ph, pw) for complex-valued.
            item_index (int): Index of the item (used for position embedding).
            max_length (int): Maximum length for position embedding.

        Returns:
            np.ndarray: Output array with position embedding appended as the last channel.
        """
        # If input is real-valued (ph, pw, 2), flatten to (ph*pw, 2)
        if inp.ndim == 3 and inp.shape[2] == 2:
            ph, pw, channels = inp.shape
            inp_flat = inp.reshape(-1, channels)
        elif inp.ndim == 2:  # complex-valued (ph, pw)
            ph, pw = inp.shape
            inp_flat = inp.reshape(-1, 1)
        else:
            raise ValueError("Input shape not recognized for positional embedding.")

        # Create position embedding
        position_embedding = np.full((inp_flat.shape[0], 1), (item_index + 1) / (2 * max_length), dtype=inp_flat.dtype)
        out = np.concatenate((inp_flat, position_embedding), axis=1)

        # Reshape back to original patch shape with extra channel
        if inp.ndim == 3 and inp.shape[2] == 2:
            out = out.reshape(ph, pw, channels + 1)
        elif inp.ndim == 2:
            out = out.reshape(ph, pw, 2)
        return out

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
        self.files = dataset.files #list(dataset._samples_by_file.keys())
        self.verbose = verbose

    def __iter__(self):
        """
        Iterate over the dataset, yielding (file_idx, y, x) tuples for patch sampling.
        Shuffles files and/or patches if enabled.
        """
        rng = np.random.default_rng(self.seed)
        files = self.files.copy()
        print(files)
        if self.shuffle_files:
            rng.shuffle(files)
        for f in files:
            # Mark patches as loaded for this file
            self.dataset.calculate_patches_from_store(f, patch_order=self.patch_order)
            coords = self.dataset._samples_by_file[f].copy()
            t0 = time.time()

            n = len(coords) if self.samples_per_prod <= 0 else min(self.samples_per_prod, len(coords))
            for y, x in coords[:n]:
                if self.verbose:
                    print(f"Sampling from file {f}, patch at ({y}, {x})")
                yield (f, y, x)
            elapsed = time.time() - t0
            if self.verbose:
                print(f"Sampling {n} patches from file {f} took {elapsed:.2f} seconds.")

    def __len__(self):
        """
        Return the total number of samples to be drawn by the sampler.
        """
        if self.samples_per_prod > 0:
            return sum(min(self.samples_per_prod, len(v)) for v in self.dataset._samples_by_file.values())
        return self.dataset.total_patches

def get_sar_dataloader(
    data_dir: str,
    file_pattern: str = "*.zarr",
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
    online: bool = True
) -> DataLoader:
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

    Returns:
        DataLoader: PyTorch DataLoader for the SAR dataset.
    """
    dataset = SARZarrDataset(
        data_dir=data_dir,
        file_pattern=file_pattern,
        return_whole_image=return_whole_image,
        transform=transform,
        patch_size=patch_size,
        complex_valued=complex_valued,
        level_from=level_from,
        level_to=level_to,
        patch_mode=patch_mode,
        parabola_a=parabola_a, 
        save_samples= save_samples, 
        buffer = buffer, 
        stride=stride, 
        backend=backend, 
        verbose=verbose, 
        cache_size=cache_size, 
        online=online, 
        max_products=max_products, 
        samples_per_prod=samples_per_prod
    )
    sampler = KPatchSampler(
        dataset,
        samples_per_prod=samples_per_prod,
        shuffle_files=shuffle_files,
        patch_order=patch_order,
        verbose=verbose
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

# Example usage
if __name__ == "__main__":
    
    transforms = SARTransform(
        transform_raw = functools.partial(normalize, array_min=RC_MIN, array_max=RC_MAX),
        transform_rc = functools.partial(normalize, array_min=RC_MIN, array_max=RC_MAX),
        transform_rcmc =functools.partial(normalize, array_min=RC_MIN, array_max=RC_MAX),
        transform_az = functools.partial(normalize, array_min=GT_MIN, array_max=GT_MAX)
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
       transform=transforms
       #patch_mode="parabolic",
       #parabola_a=0.0005,
       #k=10
    )
    for i, (x_batch, y_batch) in enumerate(loader):
        print(f"Batch {i}: x {x_batch.shape}, y {y_batch.shape}")
        #break
