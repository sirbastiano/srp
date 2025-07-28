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
from utils import normalize, RC_MAX, RC_MIN, GT_MAX, GT_MIN
from api import fetch_chunk_from_hf_zarr
from api import download_metadata

class SARTransform(nn.Module):
    """
    A PyTorch transform that normalizes SAR data patches.

    Args:
        transform_raw (callable): Function to transform 'raw' level data.
        transform_rc (callable): Function to transform 'rc' level data.
        transform_rcmc (callable): Function to transform 'rcmc' level data.
        transform_az (callable): Function to transform 'az' level data.
        Each function should accept (x: np.ndarray, min: int, max: int).
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
        assert level in self.transforms, f"Transform for level '{level}' not defined."
        if self.transforms[level] is None:
            return x
        else:
            return self.transforms[level](x)

class SARZarrDataset(Dataset):
    """
    SARZarrDataset is a PyTorch Dataset for loading patches from SAR (Synthetic Aperture Radar) data stored in Zarr format.

    This dataset supports efficient patch sampling from multiple Zarr files, allowing for square, rectangular, or parabolic patch extraction. It groups patch coordinates by file for efficient indexing and supports direct access to patches via (file_idx, y, x) tuples.

    Key Features:
    - Loads SAR data patches from Zarr stores, supporting both real and complex-valued data.
    - Supports different patch sampling modes: "square", "rectangular", and "parabolic".
    - Efficiently indexes and caches patch coordinates for fast repeated access.
    - Allows optional transformation of patches and visualization utilities.
    - Handles both input and target SAR processing levels (e.g., "rcmc" and "az").
    - Supports saving and loading patch indices to avoid recomputation.
    - Implements a chunk-level LRU (Least Recently Used) caching mechanism for efficient repeated access to data chunks.

    Caching Mechanism:
    ------------------
    The class uses Python's functools.lru_cache to cache recently accessed data chunks. The `_load_chunk` method is wrapped with an LRU cache, whose size is controlled by the `cache_size` parameter. This means that when a patch is requested, the underlying chunk is loaded and stored in memory. If the same chunk is requested again (for a different patch or coordinate), it is retrieved from the cache, significantly speeding up repeated access and reducing disk I/O. When the cache is full, the least recently used chunks are evicted.

    Args:
        data_dir (str): Directory containing Zarr files.
        file_pattern (str, optional): Glob pattern for Zarr files. Defaults to "*hh*.zarr".
        return_whole_image (bool, optional): If True, returns the whole image as a single patch. Defaults to False.
        transform (callable, optional): Optional transform to apply to both input and target patches.
        patch_size (Tuple[int, int], optional): Size of the patch (height, width). Defaults to (256, 256).
        complex_valued (bool, optional): If True, returns complex-valued tensors. If False, returns real and imaginary parts stacked. Defaults to False.
        level_from (str, optional): Key for the input SAR processing level in the Zarr store. Defaults to "rcmc".
        level_to (str, optional): Key for the target SAR processing level in the Zarr store. Defaults to "az".
        patch_mode (str, optional): Patch extraction mode: "square", "rectangular", or "parabolic". Defaults to "square".
        parabola_a (Optional[float], optional): Curvature parameter for parabolic patch mode. Defaults to 0.001.
        save_samples (bool, optional): If True, saves computed patch indices to disk. Defaults to True.
        buffer (Optional[Tuple[int, int]], optional): Buffer (margin) to avoid sampling near image edges. Defaults to (100, 100).
        stride (Optional[Tuple[int, int]], optional): Stride for patch extraction. Defaults to (50, 50).
        backend (str, optional): Backend for loading Zarr data, either "zarr" or "dask". Defaults to "zarr".
        cache_size (int, optional): Maximum number of chunks to cache in memory for fast repeated access.

    Attributes:
        files (List[Path]): List of Zarr file paths.
        samples_by_file (Dict[int, List[Tuple[int, int]]]): Mapping from file index to list of patch coordinates.
        total_patches (int): Total number of patches in the dataset.

    Methods:
        __len__(): Returns the total number of patches.
        __getitem__(idx): Loads a patch given a (file_idx, y, x) tuple.
        visualize_item(idx, ...): Visualizes a data sample at the specified index.
        visualize_image_portion(file_idx, start_y, start_x, portion_height, portion_width, ...): Visualizes a whole portion of an image by merging multiple patches while respecting stride.
        _sample_parabolic_patch(data, y_start): Samples a parabolic patch from the data.

    Patch Coordinate Precomputation:
    --------------------------------
    For all patch modes, including parabolic, the valid patch coordinates are precomputed during initialization:
    - Square/Rectangular: Regular grid sampling with stride and buffer constraints
    - Parabolic: Center coordinates (x_center) that ensure the entire parabolic curve fits within image bounds
      The parabolic curve equation: x = x_center + parabola_a * (j - patch_width//2)^2
      Coordinates are validated to ensure all curve points remain within the image.

    Raises:
        AssertionError: If no valid patches are found in the dataset.
        ValueError: If an unknown patch_mode is specified.
        KeyError: If the specified levels are not found in the Zarr store.

    Example:
        >>> dataset = SARZarrDataset("/path/to/zarrs", patch_size=(128, 128), cache_size=1000)
        >>> x, y = dataset[(0, 100, 100)]
        >>> dataset.visualize_item((0, 100, 100))
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
        patch_mode: str = "square",      # square, rectangular, or parabolic
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
        self.patch_size = patch_size
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
        #self.init_samples()
        self._initialize_stores()
        
    def _get_file_list(self):
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
            self.files = sorted(self.data_dir.glob(self.file_pattern))[:min(self._max_products, len(self.files))]
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
                self._stores[zfile][level] = da.from_zarr(complete_path).rechunk(self.patch_size)  
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
        self._get_file_list()
        for zfile in self.files:
            self._append_file_to_stores(zfile)

    # def init_samples(self):
    #     """
    #     Initializes and indexes patch sample coordinates for each file in the dataset.
    #     This method generates or loads patch coordinates used for sampling image patches from the dataset files.
    #     It supports different patch modes ('square', 'rectangular', 'parabolic'), and can optionally return the whole image.
    #     The coordinates are grouped by file index and stored in `self._samples_by_file`.
    #     If a cached patch index file exists, it loads the coordinates from disk for efficiency.
    #     Otherwise, it computes the coordinates, optionally saves them to disk, and prints timing information if verbose mode is enabled.
    #     The method ensures that all generated patches fit within image bounds, taking into account patch size, stride, buffer, and (for parabolic patches) the parabolic curve parameters.
    #     Raises:
    #         ValueError: If an unknown patch mode is specified.
    #         AssertionError: If no valid patches are found after initialization.
    #     Side Effects:
    #         - Updates `self._samples_by_file` with patch coordinates grouped by file name.
    #         - Updates `self.total_patches` with the total number of patches.
    #         - Optionally saves patch indices to a JSON file.
    #         - Prints timing information if `self.verbose` is True.
    #         - Resets `self.store` and `self.file_idx` to None.
    #     """        
    #     self._initialize_stores()
    #     self.patch_index_file = self.data_dir / f"patch_index_{self.level_from}_{self.level_to}_{self.patch_mode}_{self.patch_size[0]}x{self.patch_size[1]}_a{self.parabola_a if self.patch_mode == 'parabolic' else 'none'}.json"
    #     if self.patch_index_file.exists():
    #         with open(self.patch_index_file, "r") as f:
    #             loaded = json.load(f)
    #         # Convert keys back to int and tuples
    #         self._samples_by_file = {Path(k): [tuple(coord) for coord in v] for k, v in loaded.items()}
    #     else:
    #         # Build sample indices grouped by file
    #         start_time = time.time()
    #         for zfile in self.files:
    #             self.calculate_patches_from_store(zfile)
    #         if self.save_samples:
    #             with open(self.patch_index_file, "w") as f:
    #                 json.dump({k: [list(coord) for coord in v] for k, v in self._samples_by_file.items()}, f)
    #         elapsed = time.time() - start_time  # End timing
    #         if self.verbose:
    #             print(f"Patch coordinate calculation took {elapsed:.2f} seconds.")

    #     self.total_patches = sum(len(v) for v in self._samples_by_file.values())
    #     assert self.total_patches > 0, "No valid patches found."
        
    #     self.store = None
    #     self.file_idx = None
        
    def calculate_patches_from_store(self, zfile:os.PathLike):
        store = zarr.open(zfile, mode="r")
        if not {self.level_from, self.level_to}.issubset(store.keys()):
            zfile_name = os.path.basename(zfile)
            if self.online:
                for base_file in ['', self.level_from, self.level_to]:
                    download_metadata(
                        zarr_archive=str(zfile_name),
                        local_dir=self.data_dir, 
                        base_dir=base_file
                    )
            else:
                raise ValueError(f"Levels {self.level_from} and {self.level_to} not found in Zarr store {zfile}.")
        h, w = store[self.level_from].shape
        coords: List[Tuple[int,int]] = []
        if self.return_whole_image:
            coords = [(0, 0)]
        else:
            if self.patch_mode in ("square", "rectangular"):
                ph, pw = self.patch_size
                y_min, y_max = self.buffer[0], h - self.buffer[0]
                x_min, x_max = self.buffer[1], w - self.buffer[1]
                
                y_coords = np.arange(y_min, y_max - self.stride[1] + 1, self.stride[1])
                x_coords = np.arange(x_min, x_max - self.stride[0] + 1, self.stride[0])
                
                # Create meshgrid and flatten to get all coordinate combinations
                Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
                # Use sample order: zip meshgrid as (y, x) in the order they appear
                coords = list(zip(Y_grid.flatten(), X_grid.flatten()))
            elif self.patch_mode == "parabolic":
                ph, pw = self.patch_size
                a = self.parabola_a
                
                # For parabolic patches, we need to ensure the entire parabolic curve fits within image bounds
                # The parabolic curve is x = x_center + a * (j - pw//2)^2 for j in [0, pw)
                # Maximum x offset occurs at the edges: max_offset = a * (pw//2)^2
                max_offset = int(np.ceil(a * (pw // 2) ** 2))
                
                # Apply buffer and ensure parabolic patch fits
                y_min, y_max = self.buffer[0], h - ph - self.buffer[0]
                x_min = self.buffer[1] + max_offset  # Left bound considering parabolic curve
                x_max = w - self.buffer[1] - max_offset  # Right bound considering parabolic curve
                
                # Generate coordinates with appropriate stride for parabolic patches
                stride_y, stride_x = self.stride
                
                # Fully vectorized coordinate generation for maximum performance
                y_coords = np.arange(y_min, y_max + 1, stride_y)
                x_centers = np.arange(x_min, x_max + 1, stride_x)
                
                # Create meshgrid for all combinations
                Y_grid, X_grid = np.meshgrid(y_coords, x_centers, indexing='ij')
                
                # Vectorized validation for all center positions at once
                j_indices = np.arange(pw)
                offsets = j_indices - pw // 2
                
                # For each x_center, calculate all x_curve positions
                # Shape: (len(y_coords), len(x_centers), pw)
                X_curves = X_grid[:, :, np.newaxis] + a * offsets[np.newaxis, np.newaxis, :]
                
                # Check bounds for all positions simultaneously
                valid_mask = np.all((X_curves >= 0) & (X_curves < w), axis=2)
                
                # Extract valid coordinates
                valid_y, valid_x = np.where(valid_mask)
                coords.extend([(Y_grid[i, j], X_grid[i, j]) for i, j in zip(valid_y, valid_x)])
            else:
                raise ValueError(f"Unknown patch_mode {self.patch_mode}")
        if coords:
            self._samples_by_file[zfile] = coords
    def __len__(self):
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
        zfile = Path(zfile)
        t0 = time.time() 
        if self.patch_mode == "parabolic":
            patch_from = self._sample_parabolic_patch(zfile, self.level_from, x, y)
            patch_to = self._sample_parabolic_patch(zfile, self.level_to, x, y)
        else:
            ph, pw = self.patch_size
            patch_from = self._get_sample(zfile, self.level_from, y, x, ph, pw) # data_x[y:y+ph, x:x+pw].astype(np.complex64)
            patch_to = self._get_sample(zfile, self.level_to, y, x, ph, pw) #data_y[y:y+ph, x:x+pw].astype(np.complex64)

        if self.transform:
            patch_from = self.transform(patch_from, self.level_from)
            patch_to = self.transform(patch_to, self.level_to)

        if not self.complex_valued:
            patch_from = np.stack((patch_from.real, patch_from.imag), axis=0).astype(np.float32)
            patch_to = np.stack((patch_to.real, patch_to.imag), axis=0).astype(np.float32)

        if self.positional_encoding:
            patch_from = self.add_position_embedding(patch_from, y)
            
        x_tensor = torch.from_numpy(patch_from)
        y_tensor = torch.from_numpy(patch_to)
        if self.verbose:
            elapsed = time.time() - t0
            print(f"Loading patch ({zfile}, {y}, {x}) took {elapsed:.2f} seconds.")

        return x_tensor, y_tensor

    def _load_chunk_uncached(self, zfile: os.PathLike, level: str, cy: int, cx: int) -> np.ndarray:
        """
        Reads and decompresses a full chunk from a Zarr store for a specified file, level, and chunk indices.

        Parameters:
            zfile (os.PathLike): Path to the Zarr file in the dataset.
            level (str): The resolution level or group within the Zarr store.
            cy (int): The vertical chunk index.
            cx (int): The horizontal chunk index.

        Returns:
            np.ndarray: The requested chunk as a NumPy array of type np.complex64.
        """
        arr = self._stores[zfile][level]
        ch, cw = arr.chunks
        y0, x0 = cy * ch, cx * cw
        self._download_sample_if_missing(zfile, level, y0, x0)
        chunk = arr[y0:y0+ch, x0:x0+cw]
        return chunk.astype(np.complex64)

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
        if not zfile.exists():
                if self.online:
                    for base_file in ['', self.level_from, self.level_to]:
                        download_metadata(
                            zarr_archive=str(zfile_name),
                            local_dir=self.data_dir, 
                            base_dir=base_file
                        )
                    self.calculate_patches_from_store(zfile)
                    self._append_file_to_stores(zfile)
                else:
                    raise FileNotFoundError(f"Zarr file {zfile} does not exist.")
        zfile_name = os.path.basename(zfile)
        chunk_name = get_chunk_name_from_coords(y, x, zarr_file_name=zfile_name, level=level, chunks=self._stores[zfile][level].chunks, version=get_zarr_version(zfile))
        chunk_path = self.data_dir / chunk_name
        
        if not chunk_path.exists():
            if self.verbose:
                print(f"Chunk {chunk_name} not found locally. Downloading from Hugging Face Zarr archive...")
            zfile_name = os.path.basename(zfile)
            fetch_chunk_from_hf_zarr(level=level, y=y, x=x, zarr_archive=zfile_name, local_dir=self.data_dir)
        return chunk_path

    def _get_sample(self, zfile: os.PathLike, level: str, y, x, ph: int = 0, pw: int = 0) -> np.ndarray:
        """
        Retrieves a sample patch from the Zarr store at the specified level and coordinates.
        
        Args:
            zfile (str): Path to the Zarr file in the dataset.
            level (str): The processing level to retrieve data from (e.g., "rcmc", "az").
            y (int): The y-coordinate of the patch.
            x (int): The x-coordinate of the patch.
            ph (int): Patch height.
            pw (int): Patch width.
        
        Returns:
            np.ndarray: The desired patch as numpy array.
        """
        sample = self._stores[zfile][level]
        t0 = time.time()
        if self.backend == "dask":
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
            ch, cw = sample.chunks
            cy, cx = y // ch, x // cw
            chunk = self._load_chunk(zfile, level, cy, cx)
            dy, dx = y - cy * ch, x - cx * cw
            patch = chunk[dy:dy+ph, dx:dx+pw]
            # pad if near array edge or chunk smaller
            h0, w0 = patch.shape
            
            if h0 < ph or w0 < pw:
                pad_h = ph - h0 if h0 < ph else 0
                pad_w = pw - w0 if w0 < pw else 0
                patch = np.pad(
                    patch,
                    ((0, pad_h), (0, pad_w)),
                    mode='constant',
                    constant_values=0
                )
            return patch
        else:
            raise ValueError(f"Unknown backend {self.backend}")
        
    def open_archive(self, zfile: os.PathLike) -> (zarr.Group | zarr.Array):
        """
        Opens a Zarr archive and returns the root group.
        
        Args:
            zfile (os.PathLike): Path to the Zarr file.
        
        Returns:
            zarr.Group: The root group of the Zarr archive.
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
        ph, pw = self.patch_size
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
        Adds a position embedding column to the input numpy array.

        Args:
            inp (np.ndarray): Input array of shape (N, D).
            item_index (int): Index of the item (used for position embedding).
            max_length (int): Maximum length for position embedding (default: 10000).

        Returns:
            np.ndarray: Output array with position embedding appended as the last column.
        """
        position_embedding = np.full((max_length, 1), (item_index + 1) / (2 * max_length))
        # If inp has shape (N, D), position_embedding should have shape (N, 1)
        if inp.shape[0] != max_length:
            position_embedding = position_embedding[:inp.shape[0]]
        out = np.concatenate((inp, position_embedding), axis=1)
        return out

    def visualize_image_portion(self, 
                               zfile: Union[str, os.PathLike],
                               start_y: int, 
                               start_x: int,
                               portion_height: int,
                               portion_width: int,
                               plot_type: str = 'magnitude',
                               show: bool = True,
                               vminmax: Optional[Union[Tuple[float, float], str]] = 'auto',
                               figsize: Tuple[int, int] = (15, 8),
                               save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualizes a whole portion of an image by merging multiple patches while handling stride properly.
        
        This method takes starting coordinates and a desired portion size, then samples and merges
        multiple patches to reconstruct the requested image portion. It handles stride properly by
        taking only the stride-sized portion from each patch (not the full patch) to avoid overlap.
        
        Args:
            zfile (Union[str, os.PathLike]): Path to the Zarr file to visualize from.
            start_y (int): Starting y-coordinate of the portion to visualize.
            start_x (int): Starting x-coordinate of the portion to visualize.
            portion_height (int): Height of the portion to visualize.
            portion_width (int): Width of the portion to visualize.
            plot_type (str, optional): Type of visualization ('magnitude', 'phase', 'real', 'imag'). Defaults to 'magnitude'.
            show (bool, optional): Whether to display the plot. Defaults to True.
            vminmax (Optional[Union[Tuple[float, float], str]], optional): Color scale limits or 'auto' for automatic scaling. Defaults to 'auto'.
            figsize (Tuple[int, int], optional): Figure size for matplotlib. Defaults to (15, 8).
            save_path (Optional[str], optional): Path to save the visualization. If None, image is not saved.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Merged arrays for input (level_from) and target (level_to) portions.
            
        Raises:
            ValueError: If the requested portion extends beyond available patches or image boundaries.
            
        Notes:
            - Only works with square/rectangular patch modes, not parabolic.
            - Uses stride-sized portions from each patch to create seamless merging without overlap.
            - For complex data, the visualization type determines what component is shown.
            - The merged result represents a continuous image portion reconstructed from patches.
        """
        import matplotlib.pyplot as plt
        
        if self.patch_mode == "parabolic":
            raise ValueError("Image portion visualization is not supported for parabolic patch mode.")
            
        if zfile not in self._samples_by_file:
            raise ValueError(f"File {zfile} not found in dataset.")
        
        # Get patch and stride information
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride
        
        # Calculate how many patches we need in each dimension
        patches_needed_h = (portion_height + stride_h - 1) // stride_h
        patches_needed_w = (portion_width + stride_w - 1) // stride_w
        
        # Initialize merged arrays
        merged_input = np.zeros((portion_height, portion_width), dtype=np.complex64)
        merged_target = np.zeros((portion_height, portion_width), dtype=np.complex64)
        
        # Track coverage for proper merging
        coverage_mask = np.zeros((portion_height, portion_width), dtype=bool)
        
        patches_found = 0
        patches_used = 0
        
        # Iterate through the grid of patches needed
        for i in range(patches_needed_h):
            for j in range(patches_needed_w):
                # Calculate patch coordinates in the original image
                patch_y = start_y + i * stride_h
                patch_x = start_x + j * stride_w
                
                # Check if this patch exists in our dataset
                patch_coord = (patch_y, patch_x)
                if patch_coord in self._samples_by_file[zfile]:
                    patches_found += 1
                    
                    # Get the patch data using __getitem__
                    try:
                        input_patch, target_patch = self.__getitem__((self.files.index(str(zfile)), patch_y, patch_x))
                        
                        # Convert back to numpy and handle complex conversion
                        if not self.complex_valued:
                            # Data is stored as [real, imag] channels, convert back to complex
                            if input_patch.dim() == 3:  # Check if it's a 3D tensor
                                input_patch_np = input_patch[0].numpy() + 1j * input_patch[1].numpy()
                                target_patch_np = target_patch[0].numpy() + 1j * target_patch[1].numpy()
                            else:
                                # Handle 2D case
                                input_patch_np = input_patch.numpy().astype(np.complex64)
                                target_patch_np = target_patch.numpy().astype(np.complex64)
                        else:
                            input_patch_np = input_patch.numpy()
                            target_patch_np = target_patch.numpy()
                        
                        # Calculate where this patch should go in the merged array
                        merge_start_y = i * stride_h
                        merge_start_x = j * stride_w
                        merge_end_y = min(merge_start_y + stride_h, portion_height)
                        merge_end_x = min(merge_start_x + stride_w, portion_width)
                        
                        # Calculate the actual patch region to copy (respecting stride, not full patch)
                        patch_copy_h = merge_end_y - merge_start_y
                        patch_copy_w = merge_end_x - merge_start_x
                        
                        # Copy the stride-sized portion from the patch to merged array
                        merged_input[merge_start_y:merge_end_y, merge_start_x:merge_end_x] = \
                            input_patch_np[:patch_copy_h, :patch_copy_w]
                        merged_target[merge_start_y:merge_end_y, merge_start_x:merge_end_x] = \
                            target_patch_np[:patch_copy_h, :patch_copy_w]
                        
                        coverage_mask[merge_start_y:merge_end_y, merge_start_x:merge_end_x] = True
                        patches_used += 1
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Could not load patch at ({patch_y}, {patch_x}): {e}")
                        continue
        
        if patches_used == 0:
            raise ValueError(f"No valid patches found for the requested portion starting at ({start_y}, {start_x})")
        
        if self.verbose:
            coverage_percent = np.sum(coverage_mask) / coverage_mask.size * 100
            print(f"Merged {patches_used}/{patches_found} available patches. Coverage: {coverage_percent:.1f}%")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Process data for visualization
        input_vis, input_vmin, input_vmax = get_sample_visualization(
            merged_input, plot_type=plot_type, vminmax=vminmax
        )
        target_vis, target_vmin, target_vmax = get_sample_visualization(
            merged_target, plot_type=plot_type, vminmax=vminmax
        )
        
        # Plot input (level_from)
        im1 = axes[0].imshow(input_vis, aspect='auto', cmap='viridis', 
                           vmin=input_vmin, vmax=input_vmax)
        axes[0].set_title(f'{self.level_from.upper()} - {plot_type.title()}')
        axes[0].set_xlabel('Range')
        axes[0].set_ylabel('Azimuth')
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=8)
        
        # Plot target (level_to)
        im2 = axes[1].imshow(target_vis, aspect='auto', cmap='viridis',
                           vmin=target_vmin, vmax=target_vmax)
        axes[1].set_title(f'{self.level_to.upper()} - {plot_type.title()}')
        axes[1].set_xlabel('Range')
        axes[1].set_ylabel('Azimuth')
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=8)
        
        # Add coverage information
        fig.suptitle(f'Image Portion: File {file_idx}, Start ({start_y}, {start_x}), '
                    f'Size ({portion_height}, {portion_width})\n'
                    f'Patches Used: {patches_used}, Coverage: {np.sum(coverage_mask) / coverage_mask.size * 100:.1f}%',
                    fontsize=12)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return merged_input, merged_target

class KPatchSampler(Sampler):
    """
    Sampler that yields (file_idx, y, x) tuples.
    Draws k patches from each file in round-robin, shuffling between epochs.

    Args:
        dataset (SARZarrDataset): The dataset to sample patches from.
        samples_per_prod (int): Number of patches to sample per file. If 0, all patches are sampled.
        shuffle_files (bool): Whether to shuffle the order of files between epochs.
        shuffle_patches (bool): Whether to shuffle the patches within each file.
        seed (int): Random seed for reproducibility of shuffling.
    """
    def __init__(
        self,
        dataset: SARZarrDataset,
        samples_per_prod: int = 0,
        shuffle_files: bool = True,
        shuffle_patches: bool = True,
        seed: int = 42, 
        verbose: bool = True, 
        max_products: int = 3, 
        window_size: Tuple[int, int] = (4000, 4000)
    ):
        self.dataset = dataset
        self.samples_per_prod = samples_per_prod
        self.shuffle_files = shuffle_files
        self.shuffle_patches = shuffle_patches
        self.seed = seed
        self.files = dataset.files #list(dataset._samples_by_file.keys())
        self.loaded_patches = {}
        self.verbose = verbose

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        files = self.files.copy()
        if self.shuffle_files:
            rng.shuffle(files)
        for f in files:
            # Mark patches as loaded for this file
            self.loaded_patches[f] = True
            if self.verbose:
                print(f"Downloading patches for file {f}")
            self.dataset.calculate_patches_from_store(f)
            coords = self.dataset._samples_by_file[f].copy()
            t0 = time.time()
            if self.shuffle_patches:
                rng.shuffle(coords)
            n = len(coords) if self.samples_per_prod <= 0 else min(self.samples_per_prod, len(coords))
            for y, x in coords[:n]:
                if self.verbose:
                    print(f"Sampling from file {f}, patch at ({y}, {x})")
                yield (f, y, x)
            elapsed = time.time() - t0
            if self.verbose:
                print(f"Sampling {n} patches from file {f} took {elapsed:.2f} seconds.")

    def __len__(self):
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
    patch_mode: str = "square",
    patch_size: Tuple[int, int] = (512, 512),
    buffer: Tuple[int, int] = (100, 100),
    stride: Tuple[int, int] = (50, 50),
    backend: str = "zarr",  # "zarr" or "dask
    parabola_a: float = 0.001,
    shuffle_files: bool = True,
    shuffle_patches: bool = True, 
    complex_valued: bool = False,
    save_samples: bool = True, 
    verbose: bool = True, 
    cache_size: int = 10000, 
    max_products: int = 10, 
    samples_per_prod: int = 0,
    online: bool = True
) -> DataLoader:
    """
    Creates and returns a DataLoader for SAR (Synthetic Aperture Radar) data using the SARZarrDataset and KPatchSampler.

    Args:
        data_dir (str): Path to the directory containing SAR data.
        batch_size (int, optional): Number of samples per batch. Defaults to 8.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 2.
        return_whole_image (bool, optional): If True, returns the whole image instead of patches. Defaults to False.
        transform (callable, optional): Optional transform to be applied on a sample.
        level_from (str, optional): The starting processing level of the SAR data. Defaults to "rcmc".
        level_to (str, optional): The target processing level of the SAR data. Defaults to "az".
        patch_mode (str, optional): Mode for patch extraction (e.g., "square"). Defaults to "square".
        patch_size (Tuple[int, int], optional): Size of the patches to extract. Defaults to (512, 512).
        buffer (Tuple[int, int], optional): Buffer size around patches. Defaults to (100, 100).
        stride (Tuple[int, int], optional): Stride for patch extraction. Defaults to (50, 50).
        backend (str, optional): Backend for loading Zarr data, either "zarr"
        parabola_a (float, optional): Parabola parameter for patch extraction. Defaults to 0.001.
        samples_per_prod (int, optional): Number of patches to sample per image. Defaults to 0 (all patches).
        shuffle_files (bool, optional): Whether to shuffle the order of files. Defaults to True.
        shuffle_patches (bool, optional): Whether to shuffle the order of patches. Defaults to True.
        complex_valued (bool, optional): If True, loads data as complex-valued. Defaults to False.
        save_samples (bool, optional): If True, saves sampled patches. Defaults to True.
        cache_size (int, optional): size of the cache for saving least recently accessed chunks
        max_products (int, optional): Maximum number of products consider for the dataloader. Defaults to 10.
        
        online (bool, optional): If True, uses online data loading. Defaults to True.
        verbose (bool, optional): If True, prints additional information during processing. Defaults to True

    Returns:
        DataLoader: A PyTorch DataLoader for the SAR dataset.
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
        shuffle_patches=shuffle_patches, 
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
       patch_mode="square", 
       complex_valued=False, 
       shuffle_files=False, 
       shuffle_patches=False, 
       transform=transforms
       #patch_mode="parabolic",
       #parabola_a=0.0005,
       #k=10
    )
    for i, (x_batch, y_batch) in enumerate(loader):
        print(f"Batch {i}: x {x_batch.shape}, y {y_batch.shape}")
        #break
