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

from utils import get_chunk_name_from_coords, get_part_from_filename, get_sample_visualization, get_zarr_version, parse_product_filename
from api import list_base_files_in_repo, list_repos_by_author
from utils import minmax_normalize, minmax_inverse, extract_stripmap_mode_from_filename, RC_MAX, RC_MIN, GT_MAX, GT_MIN
from api import fetch_chunk_from_hf_zarr, download_metadata_from_product
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from normalization import NormalizationScheme, NormalizationFactory
class EnhancedSARTransform(nn.Module):
    """
    Enhanced SAR transform with support for different normalization schemes.
    """
    
    def __init__(self, normalization_schemes: Dict[str, NormalizationScheme]):
        super().__init__()
        self.normalization_schemes = normalization_schemes
    
    def forward(self, x: np.ndarray, level: str) -> np.ndarray:
        """Apply normalization for the specified level."""
        assert level in self.normalization_schemes, f"No normalization scheme for level '{level}'"
        return self.normalization_schemes[level].normalize(x, level=level)
    
    def inverse(self, x: np.ndarray, level: str) -> np.ndarray:
        """Apply inverse normalization for the specified level."""
        assert level in self.normalization_schemes, f"No normalization scheme for level '{level}'"
        return self.normalization_schemes[level].denormalize(x, level=level)
    
    @classmethod
    def create_from_config(
        cls,
        input_scheme: str = 'minmax',
        output_scheme: str = 'minmax',
        complex_valued: bool = True,
        **kwargs
    ) -> 'EnhancedSARTransform':
        """Create transform from configuration.
        
        Args:
            input_scheme: Normalization scheme for input levels ('minmax', 'standard', 'robust', 'log', 'adaptive')
            output_scheme: Normalization scheme for output level ('minmax', 'standard', 'robust', 'log', 'adaptive')  
            complex_valued: Whether data is complex-valued
            **kwargs: Additional arguments for normalization schemes
            
        Returns:
            EnhancedSARTransform: Configured transform instance
        """
        schemes = NormalizationFactory.create_sar_schemes(
            input_scheme=input_scheme,
            output_scheme=output_scheme,
            complex_valued=complex_valued,
            **kwargs
        )
        
        return cls(schemes)

class BaseTransformModule(nn.Module):
    """Base class for SAR data transformations."""
    
    def __init__(self):
        super(BaseTransformModule, self).__init__()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply transformation to input data."""
        raise NotImplementedError("Subclasses must implement forward method")
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse transformation, if applicable."""
        raise NotImplementedError("Subclasses must implement inverse method")


class NormalizationModule(BaseTransformModule):
    """Normalization module for SAR data."""
    
    def __init__(self, data_min: float, data_max: float):
        super(NormalizationModule, self).__init__()
        self.data_min = data_min
        self.data_max = data_max
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize array to range [0, 1]."""
        # Apply normalization directly on numpy array
        return minmax_normalize(x, self.data_min, self.data_max)
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        # Apply inverse normalization
        return minmax_inverse(x, self.data_min, self.data_max)


class IdentityModule(BaseTransformModule):
    """Identity transformation that returns input unchanged."""
    
    def __init__(self):
        super(IdentityModule, self).__init__()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return x
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return x


class ComplexNormalizationModule(BaseTransformModule):
    """Complex-valued normalization module for SAR data."""
    
    def __init__(self, real_min: float, real_max: float, imag_min: float, imag_max: float):
        super(ComplexNormalizationModule, self).__init__()
        self.real_min = real_min
        self.real_max = real_max
        self.imag_min = imag_min
        self.imag_max = imag_max
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize complex array separately for real and imaginary parts."""
        if np.iscomplexobj(x):
            # Normalize real and imaginary parts separately
            # Use numpy functions to extract real and imaginary parts
            real_part = minmax_normalize(np.real(x), self.real_min, self.real_max)
            imag_part = minmax_normalize(np.imag(x), self.imag_min, self.imag_max)

            #real_part = np.clip(real_part, 0, 1)
            #imag_part = np.clip(imag_part, 0, 1)
            
            normalized = real_part + 1j * imag_part
        else:
            # Assume magnitude data
            normalized = minmax_normalize(x, self.real_min, self.real_max)

        return normalized
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if np.iscomplexobj(x):
            # Inverse normalize real and imaginary parts separately
            real_part = minmax_inverse(np.real(x), self.real_min, self.real_max)
            imag_part = minmax_inverse(np.imag(x), self.imag_min, self.imag_max)
            return real_part + 1j * imag_part
        else:
            # Assume magnitude data
            return minmax_inverse(x, self.real_min, self.real_max)

class AdaptiveNormalizationModule(BaseTransformModule):
    """Complex-valued normalization module for SAR data."""
    
    def __init__(self):
        super(AdaptiveNormalizationModule, self).__init__()

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize complex array separately for real and imaginary parts."""
        if np.iscomplexobj(x):
            # Normalize real and imaginary parts separately
            # Use numpy functions to extract real and imaginary parts
            re = np.real(x)
            im = np.imag(x)
            real_part = minmax_normalize(re, np.min(re), np.max(re))
            imag_part = minmax_normalize(im, np.min(im), np.max(im))

            #real_part = np.clip(real_part, 0, 1)
            #imag_part = np.clip(imag_part, 0, 1)
            
            normalized = real_part + 1j * imag_part
        else:
            # Assume magnitude data
            normalized = minmax_normalize(x, np.min(x), np.max(x))

        return normalized
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if np.iscomplexobj(x):
            # Inverse normalize real and imaginary parts separately
            real_part = minmax_inverse(np.real(x), np.min(x), np.max(x))
            imag_part = minmax_inverse(np.imag(x), np.min(x), np.max(x))
            return real_part + 1j * imag_part
        else:
            # Assume magnitude data
            return minmax_inverse(x, np.min(x), np.max(x))

class SARTransform(nn.Module):
    """
    PyTorch transform module for normalizing SAR data patches at different processing levels.

    This class uses separate PyTorch modules for each SAR processing level (e.g., 'raw', 'rc', 'rcmc', 'az').
    Each module can be any PyTorch nn.Module that implements the transformation.

    Args:
        transform_raw (BaseTransformModule, optional): Module to transform 'raw' level data.
        transform_rc (BaseTransformModule, optional): Module to transform 'rc' level data.
        transform_rcmc (BaseTransformModule, optional): Module to transform 'rcmc' level data.
        transform_az (BaseTransformModule, optional): Module to transform 'az' level data.
    """
    def __init__(
        self,
        transform_raw: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial, BaseTransformModule]] = None,
        transform_rc: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial, BaseTransformModule]] = None,
        transform_rcmc: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial, BaseTransformModule]] = None,
        transform_az: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial, BaseTransformModule]] = None,
    ):
        super(SARTransform, self).__init__()
        
        # Register transform modules
        self.transform_raw = transform_raw if transform_raw is not None else IdentityModule()
        self.transform_rc = transform_rc if transform_rc is not None else IdentityModule()
        self.transform_rcmc = transform_rcmc if transform_rcmc is not None else IdentityModule()
        self.transform_az = transform_az if transform_az is not None else IdentityModule()
        
        # Create a mapping for easy access
        self.transforms = {
            'raw': self.transform_raw,
            'rc': self.transform_rc,
            'rcmc': self.transform_rcmc,
            'az': self.transform_az,
        }

    def forward(self, x: np.ndarray, level: str) -> np.ndarray:
        """
        Apply the appropriate transform module to the input array for the specified SAR processing level.

        Args:
            x (np.ndarray): Input data array.
            level (str): SAR processing level ('raw', 'rc', 'rcmc', or 'az').

        Returns:
            np.ndarray: Transformed data array.
        """
        assert level in self.transforms, f"Transform for level '{level}' not defined."
        return self.transforms[level](x)
    def inverse(self, x: np.ndarray, level: str) -> np.ndarray:
        """
        Apply the appropriate inverse transform module to the input array for the specified SAR processing level.

        Args:
            x (np.ndarray): Input data array.
            level (str): SAR processing level ('raw', 'rc', 'rcmc', or 'az').

        Returns:
            np.ndarray: Inverse transformed data array.
        """
        assert level in self.transforms, f"Transform for level '{level}' not defined."
        if isinstance(self.transforms[level], BaseTransformModule):
            # If the transform is a custom module, use its inverse method
            return self.transforms[level].inverse(x)
        print("[WARNING] Inverse transform not defined, returning input unchanged.")
        return x

    @classmethod
    def create_minmax_normalized_transform(
        cls,
        normalize: bool = True,
        adaptive: bool = False,
        rc_min: float = RC_MIN,
        rc_max: float = RC_MAX,
        gt_min: float = GT_MIN,
        gt_max: float = GT_MAX,
        complex_valued: bool = True
    ):
        """
        Factory method to create a SARTransform with normalization modules.
        
        Args:
            normalize (bool): Whether to apply normalization.
            adaptive (bool): Whether to use adaptive normalization based on patch min/max.
            rc_min (float): Minimum value for RC data normalization.
            rc_max (float): Maximum value for RC data normalization.
            gt_min (float): Minimum value for ground truth data normalization.
            gt_max (float): Maximum value for ground truth data normalization.
            complex_valued (bool): Whether data is complex-valued.
        
        Returns:
            SARTransform: Configured transform instance.
        """
        if not normalize:
            return cls()
        if not adaptive:
            if complex_valued:
                raw_transform = ComplexNormalizationModule(rc_min, rc_max, rc_min, rc_max)
                rc_transform = ComplexNormalizationModule(gt_min, gt_max, gt_min, gt_max)
                rcmc_transform = ComplexNormalizationModule(gt_min, gt_max, gt_min, gt_max)
                az_transform = ComplexNormalizationModule(gt_min, gt_max, gt_min, gt_max)
            else:
                # Use simple normalization for magnitude data
                raw_transform = NormalizationModule(rc_min, rc_max)
                rc_transform = NormalizationModule(gt_min, gt_max)
                rcmc_transform = NormalizationModule(gt_min, gt_max)
                az_transform = NormalizationModule(gt_min, gt_max)
        else:
            raw_transform = AdaptiveNormalizationModule()
            rc_transform = AdaptiveNormalizationModule()
            rcmc_transform = AdaptiveNormalizationModule()
            az_transform = AdaptiveNormalizationModule()
        return cls(
            transform_raw=raw_transform,
            transform_rc=rc_transform,
            transform_rcmc=rcmc_transform,
            transform_az=az_transform
        )

class SampleFilter:
    def __init__(self, parts: List[str]=None, years: List[int] = None, months: List[int] = None, polarizations: List[str] = None, stripmap_modes: List[int] = None):
        """
        Initialize a filter for SAR dataset samples.

        Args:
            parts (List[str], optional): List of part names to include.
            years (List[int], optional): List of years to include.
            months (List[int], optional): List of months to include.
            polarizations (List[str], optional): List of polarizations to include.
            stripmap_modes (List[int], optional): List of stripmap modes to include.
        """
        """
        Initialize the SampleFilter with optional filtering criteria.

        Args:
            years (List[int], optional): List of years to filter.
            months (List[int], optional): List of months to filter.
            polarizations (List[str], optional): List of polarizations to filter.
            stripmap_modes (List[str], optional): List of stripmap modes to filter.
        """
        self.parts = parts if parts is not None else []
        self.years = years if years is not None else []
        self.months = months if months is not None else []
        self.polarizations = polarizations if polarizations is not None else []
        self.stripmap_modes = stripmap_modes if stripmap_modes is not None else []
    def get_filter_dict(self) -> Dict[str, List[Union[int, str]]]:
        """
        Return the filter as a dictionary for use in dataset selection.

        Returns:
            dict: Dictionary of filter criteria.
        """
        """
        Get the filter criteria as a dictionary.

        Returns:
            Dict[str, List[Union[int, str]]]: Dictionary with filter criteria.
        """
        filter_dict = {}
        if self.parts:
            filter_dict['part'] = self.parts
        if self.years:
            filter_dict['year'] = self.years
        if self.months:
            filter_dict['month'] = self.months
        if self.polarizations:
            filter_dict['polarization'] = self.polarizations
        if self.stripmap_modes:
            filter_dict['stripmap_mode'] = self.stripmap_modes
        return filter_dict
    def matches(self, record: dict) -> bool:
        """
        Check if a record matches the filter criteria.

        Args:
            record (dict): Metadata record to check.

        Returns:
            bool: True if record matches, False otherwise.
        """
        """
        Check if a given record matches the filter criteria.

        Args:
            record (dict): Dictionary containing product metadata.
        Returns:
            bool: True if the record matches all specified criteria, False otherwise.
        """
        if self.years and record.get('year') not in self.years:
            return False
        if self.months and record.get('month') not in self.months:
            return False
        if self.polarizations and record.get('polarization') not in self.polarizations:
            return False
        if self.stripmap_modes and str(record.get('stripmap_mode')) not in self.stripmap_modes:
            return False
        if self.parts and record.get('part') not in self.parts:
            return False
        return True
    def _filter_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filtering criteria to a DataFrame of product metadata.
        Args:
            df (pd.DataFrame): DataFrame containing product metadata.
        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        mask = pd.Series([True] * len(df))
        if len(self.years) > 0:
            mask &= df["acquisition_date"].dt.year.isin(self.years)
        if len(self.months) > 0:
            mask &= df["acquisition_date"].dt.month.isin(self.months)
        if len(self.stripmap_modes) > 0:
            mask &= df["stripmap_mode"].isin(self.stripmap_modes)
        if len(self.polarizations) > 0:
            mask &= df["polarization"].isin(self.polarizations)
        if len(self.parts) > 0:
            mask &= df["part"].isin(self.parts)
        return df[mask]

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
        use_positional_as_token: bool = False
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
        return self._files.loc[self._files['full_name'] == Path(zfile)]['samples'].values[0]
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
            self._files = self.filters._filter_products(pd.DataFrame(records))
        else:
            print(f"Files in local directory {self.data_dir}: {[f.name for f in sorted(self.data_dir.glob('*'))]}")
            records = [r for r in (parse_product_filename(f) for f in sorted(self.data_dir.glob("*"))) if r is not None]
            self._files = self.filters._filter_products(pd.DataFrame(records))
        self._files.sort_values(by=['full_name'], inplace=True)
        self._files = self._files.iloc[:self._max_products]
        if self.verbose:
            print(f"Selected only files:  {self._files}")

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
                print(f"Opening Zarr store for file {zfile} at index {idx[0]}")
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
        # Check if the zarr file exists and has valid structure
        zfile_path = Path(zfile)
        
        # For online mode, if the zarr directory doesn't exist or is empty, create a minimal structure
        if self.online and (not zfile_path.exists() or self._is_empty_zarr_directory(zfile_path)):
            print(f"Creating minimal zarr structure for online mode: {zfile_path}")
            zfile_path.mkdir(parents=True, exist_ok=True)
            
            # Create a minimal .zgroup file to make zarr.open work
            zgroup_path = zfile_path / '.zgroup'
            if not zgroup_path.exists():
                import json
                zgroup_content = {"zarr_format": 2}
                with open(zgroup_path, 'w') as f:
                    json.dump(zgroup_content, f)
                print(f"Created minimal .zgroup file at {zgroup_path}")
        
        # Check if zarr file has valid structure before opening
        if not self._has_valid_zarr_structure(zfile_path):
            if not self.online:
                raise FileNotFoundError(f"Zarr file {zfile} does not exist or is not a valid zarr archive.")
            else:
                # For online mode, we'll handle missing levels in get_store_at_level
                print(f"Warning: Zarr structure incomplete for {zfile}, but continuing in online mode")
        
        if self.backend == "dask":
            return da.from_zarr(zfile)
        elif self.backend == "zarr":
            return zarr.open(zfile, mode='r')
        else: 
            raise ValueError(f"Unknown backend {self.backend}")
    
    def _is_empty_zarr_directory(self, zfile_path: Path) -> bool:
        """Check if a zarr directory exists but is empty or has no valid zarr files."""
        if not zfile_path.exists():
            return True
        if not zfile_path.is_dir():
            return False
        # Check if directory is empty or has no zarr metadata files
        zarr_metadata_files = ['.zgroup', '.zarray', 'zarr.json']
        has_metadata = any((zfile_path / meta).exists() for meta in zarr_metadata_files)
        return not has_metadata
    
    def _has_valid_zarr_structure(self, zfile_path: Path) -> bool:
        """Check if a path contains a valid zarr structure."""
        if not zfile_path.exists():
            return False
        
        # Check for zarr metadata files
        zarr_metadata_files = ['.zgroup', '.zarray', 'zarr.json']
        has_metadata = any((zfile_path / meta).exists() for meta in zarr_metadata_files)
        
        return has_metadata  # For online mode, we don't require levels to exist here
        
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
    def calculate_patches_from_store(self, zfile:os.PathLike, patch_order: str = "row", window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None):
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
        # Print the directories (groups/arrays) inside the Zarr store

        h, w = self.get_whole_sample_shape(zfile) 
        coords: List[Tuple[int,int]] = []
        if self.return_whole_image:
            coords = [(0, 0)]
        else:
            stride_y, stride_x = self.stride
            ph, pw = self.get_patch_size(zfile)
            if self.patch_mode == "rectangular":
                
                y_min, y_max = self.buffer[0], h - self.buffer[0]
                x_min, x_max = self.buffer[1], w - self.buffer[1]

                if window is not None:
                    #print(f"Applying window: {window} to (({x_min}, {y_min}), ({x_max}, {y_max}))")
                    y_min, y_max = max(window[0][0], y_min), min(window[1][0], y_max)
                    x_min, x_max = max(window[0][1], x_min), min(window[1][1], x_max)
                    #print(f"Saving samples from coordinates ({x_min}, {y_min}) to ({x_max}, {y_max})")
                    stride_x, stride_y = min(stride_x, x_max - x_min), min(stride_y, y_max - y_min)
                if self.concatenate_patches:
                    mph, mpw = self.get_max_base_sample_size(zfile)
                    mph, mpw = min(mph, x_max-x_min), min(mpw, y_max-y_min)
                    if self.concat_axis == 0:
                        # Vertical concatenation: fix y-coordinate to 0, vary x-coordinates
                        self._y_coords[zfile] = np.arange(y_min, y_max - mph + 1, mph) 
                        self._x_coords[zfile] = np.arange(x_min, x_max - stride_x + 1, stride_x)
                    elif self.concat_axis == 1:
                        # Horizontal concatenation: fix y-coordinate to 0, vary x-coordinates
                        self._y_coords[zfile] = np.arange(y_min, y_max - stride_y + 1, stride_y)
                        self._x_coords[zfile] = np.arange(x_min, x_max - mpw + 1, mpw)  
                    else:
                        raise ValueError(f"Invalid concat_axis: {self.concat_axis}. Must be 0 (vertical) or 1 (horizontal).")
                else:
                    # Original logic when not concatenating patches
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
        self._set_samples_for_file(zfile, coords) #_samples_by_file[zfile] = coords
            
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
            - The method assumes that the attribute 'store' in `self._files`, `self._y_coords`, and `self._x_coords` are properly initialized and populated.
            - The returned coordinate order is optimized for sequential chunk access, which can reduce I/O overhead and improve cache utilization.
        """

        # Get chunk size and patch size
        sample = self.get_store_at_level(zfile=Path(zfile), level=self.level_from)
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
        # print(f"Opening zfile: {zfile}, y: {y}, x: {x}, stripmap_mode: {stripmap_mode}")
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
        plot_type: str = "magnitude"
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
                if patch.shape[-1] >= 2:
                    patch = patch[..., :-2]
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
        if not self.transform is None:
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
        for f in files:
            # Mark patches as loaded for this file
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
        self.dataset.calculate_patches_from_store(Path(zfile), patch_order=self.patch_order, window=window)
        coords = self.dataset.get_samples_by_file(zfile)  #self.dataset._samples_by_file[Path(zfile)].copy()
        n = len(coords) if self.samples_per_prod <= 0 else min(self.samples_per_prod, len(coords))
        return coords[:n]
    def __len__(self):
        """
        Return the total number of samples to be drawn by the sampler.
        """
        if self.beginning:
            return len(self.dataset)

        else:
            if self.samples_per_prod > 0:
                return sum(min(self.samples_per_prod, len(v)) for zfile in self.dataset.get_files() for v in [self.dataset.get_samples_by_file(zfile)])  # ._samples_by_file.values())
            else:
                return 0

class SARDataloader(DataLoader):
    dataset: SARZarrDataset
    def __init__(self, dataset: SARZarrDataset, batch_size: int, sampler: KPatchSampler,  num_workers: int = 2,  pin_memory: bool= True, verbose: bool = False):
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=pin_memory)
        self.verbose = verbose
    def get_coords_from_zfile(self, zfile: Union[str, os.PathLike], window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        return self.sampler.get_coords_from_store(zfile, window=window)

    

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
    concat_axis: int = 0  # 0 for vertical, 1 for horizontal
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

    Returns:
        SARDataloader: PyTorch DataLoader for the SAR dataset.
    """
    dataset = SARZarrDataset(
        data_dir=data_dir,
        filters=filters,
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
        max_base_sample_size = max_base_sample_size,
        backend=backend, 
        verbose=verbose, 
        cache_size=cache_size, 
        online=online, 
        max_products=max_products, 
        samples_per_prod=samples_per_prod, 
        positional_encoding=positional_encoding, 
        concatenate_patches=concatenate_patches,
        concat_axis=concat_axis
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
        pin_memory=True,
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
       data_dir="./Data/sar_focusing",
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
