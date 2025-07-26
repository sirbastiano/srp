import os
import numpy as np
import zarr
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List, Callable
import numcodecs # Ensure numcodecs is installed for compression
import pandas as pd
import shutil
import dask.array as da
from .io import gc_collect

 

# -----------  Functions -----------------

def save_array_to_zarr(array: 'np.ndarray',

                       file_path: str,

                       compressor_level: int = 9,

                       parent_product: str = None,

                       metadata_df: pd.DataFrame = None,

                       ephemeris_df: pd.DataFrame =None) -> None:

    """

    Save a numpy array to a Zarr file with maximum compression.

 

    Args:

        array (np.ndarray): The numpy array to save.

        file_path (str): The path to the output Zarr file.

        compressor_level (int): Compression level for Blosc (default is 9, maximum).

        metadata_df: Optional pandas DataFrame to save as zarr attributes.

        ephemeris_df: Optional pandas DataFrame with ephemeris data to save as zarr attributes.

 

    Returns:

        None

    """

    assert array is not None, 'Input array must not be None'

    assert isinstance(file_path, str) and file_path, 'file_path must be a non-empty string'

   

    # Use smaller chunks for better compression

    # Complex64 data often compresses better with smaller, square-ish chunks

    chunk_size = min(512, array.shape[0] // 4, array.shape[1] // 4)

    chunk_size = max(64, chunk_size)  # Ensure minimum chunk size

    chunks = (chunk_size, chunk_size)

   

    # Use maximum compression with zstd and byte shuffle

    codec = numcodecs.Blosc(

        cname='zstd', # Best compression ratio

        clevel=compressor_level, # Maximum compression level

        shuffle=numcodecs.Blosc.BITSHUFFLE # Better for floating point data

    )

   

    # Create Zarr array with specified shape, dtype, and compression

    zarr_array = zarr.open(

        file_path,

        mode='w',

        shape=array.shape,

        dtype=array.dtype,

        zarr_format=2,

        compressor=codec,

        chunks=chunks,

    )

    zarr_array[:] = array

   

    zarr_array.attrs['parent_product'] = parent_product if parent_product else 'unknown'

    zarr_array.attrs['creation_date'] = pd.Timestamp.now().isoformat()

    # Add metadata as attributes if provided

    if metadata_df is not None:

        # Handle NaN values by filling them with None or converting to string

        metadata_clean = metadata_df.fillna('null')

       

        # Convert DataFrame to dictionary for zarr attributes

        zarr_array.attrs['metadata'] = metadata_clean.to_dict('records')

        zarr_array.attrs['metadata_columns'] = list(metadata_df.columns)

        zarr_array.attrs['metadata_dtypes'] = metadata_df.dtypes.astype(str).to_dict()

        print(f'Added metadata with {len(metadata_df)} records as zarr attributes')

   

    # Add ephemeris data as attributes if provided

    if ephemeris_df is not None:

        ephemeris_clean = ephemeris_df.fillna('null')

        

        zarr_array.attrs['ephemeris'] = ephemeris_clean.to_dict('records')

        zarr_array.attrs['ephemeris_columns'] = list(ephemeris_df.columns)

        zarr_array.attrs['ephemeris_dtypes'] = ephemeris_df.dtypes.astype(str).to_dict()

        print(f'Added ephemeris with {len(ephemeris_df)} records as zarr attributes')

   

    print(f'Saved array to {file_path} with maximum compression (zstd-9, chunks={chunks})')

 

def _serialize_dict(data: Dict[str, Any]) -> Dict[str, Any]:

    """Serialize dictionary for Zarr attributes storage.

   

    Args:

        data: Dictionary to serialize

       

    Returns:

        Serialized dictionary compatible with Zarr attributes

    """

    serialized = {}

    for key, value in data.items():

        if isinstance(value, np.ndarray):

            # Convert numpy arrays to lists for JSON compatibility

            serialized[key] = value.tolist()

        elif isinstance(value, (np.integer, np.floating)):

            # Convert numpy scalars to Python types

            serialized[key] = value.item()

        elif isinstance(value, dict):

            # Recursively serialize nested dictionaries

            serialized[key] = _serialize_dict(value)

        else:

            serialized[key] = value

   

    return serialized

 

def dask_slice_saver(

    result: Dict[str, Union[np.ndarray, pd.DataFrame, Dict[str, Any]]],

    zarr_path: Union[str, Path],

    chunks: Optional[Union[str, Tuple[int, ...]]] = 'auto',

    clevel: int = 5

) -> None:

    """Save SAR processing results to Zarr format with optimized compression.

   

    This function saves the complete SAR processing pipeline results including all

    processing stages (raw, range compressed, RCMC, azimuth compressed) along with

    associated metadata and ephemeris data to a compressed Zarr store.

   

    The function handles automatic conversion of pandas DataFrames to JSON-serializable

    dictionaries and applies high-performance compression using Blosc with zstd codec

    and bit-shuffle for optimal storage efficiency.

   

    Args:

        result (Dict[str, Union[np.ndarray, pd.DataFrame, Dict[str, Any]]]):

            Processing results dictionary containing:

            - 'raw' (np.ndarray): Raw SAR data after initial processing

            - 'rc' (np.ndarray): Range compressed SAR data 

            - 'rcmc' (np.ndarray): Range Cell Migration Corrected data

            - 'az' (np.ndarray): Azimuth compressed (focused) SAR data

            - 'metadata' (Union[pd.DataFrame, Dict]): Acquisition metadata

            - 'ephemeris' (Union[pd.DataFrame, Dict]): Satellite ephemeris data

        zarr_path (Union[str, Path]):

            Output path for the Zarr store directory. Will be created if it doesn't exist.

        chunks (Optional[Union[str, Tuple[int, ...]]], optional):

            Chunking strategy for array storage. Options:

            - 'auto': Use automatic chunking with 4096 element maximum per dimension

            - Tuple[int, ...]: Custom chunk shape matching array dimensions

            - None: Use default chunking (4096,) * ndim

            Defaults to 'auto'.

        clevel (int, optional):

            Blosc compression level from 1 (fastest) to 9 (best compression).

            Higher values provide better compression at the cost of processing time.

            Defaults to 5 for balanced performance.

   

    Returns:

        None: Function performs file I/O operations only.

       

    Raises:

        AssertionError: If required array keys are missing from result dictionary.

        AssertionError: If required metadata keys are missing from result dictionary.

        AssertionError: If array values are not numpy.ndarray instances.

        ValueError: If metadata/ephemeris are not DataFrame or dict types.

        OSError: If zarr_path directory cannot be created or accessed.

       

    Examples:

        >>> # Save complete SAR processing results

        >>> processing_results = {

        ...     'raw': raw_data_array,

        ...     'rc': range_compressed_array,

        ...     'rcmc': rcmc_corrected_array,

        ...     'az': azimuth_focused_array,

        ...     'metadata': metadata_dataframe,

        ...     'ephemeris': ephemeris_dataframe

        ... }

        >>> dask_slice_saver(processing_results, '/path/to/output.zarr')

       

        >>> # Use custom chunking and high compression

        >>> dask_slice_saver(

        ...     processing_results,

        ...     '/path/to/output.zarr',

        ...     chunks=(1024, 1024),

        ...     clevel=9

        ... )

       

        >>> # Save with automatic chunking

        >>> dask_slice_saver(

        ...     processing_results,

        ...     Path('/data/sar_focused.zarr'),

        ...     chunks='auto',

        ...     clevel=7

        ... )

   

    Notes:

        - Uses Zarr v2 format for maximum compatibility

        - Applies zstd compression with bit-shuffle for optimal performance on SAR data

        - Automatically converts pandas DataFrames to 'records' format dictionaries

        - All arrays must have the same first dimension (azimuth/slow-time axis)

        - Metadata and ephemeris are stored as JSON-serializable attributes

        - Creates parent directories automatically if they don't exist

        - Overwrites existing arrays if zarr_path already exists

       

    Performance:

        - Chunking strategy significantly affects I/O performance

        - 'auto' chunking works well for most SAR datasets

        - Higher compression levels (7-9) recommended for archival storage

        - Lower compression levels (3-5) recommended for active processing

    """

    # Input validation

    required_arrays = ['raw', 'rc', 'rcmc', 'az']

    required_dicts = ['metadata', 'ephemeris']

   

    missing_arrays = set(required_arrays) - set(result.keys())

    missing_dicts = set(required_dicts) - set(result.keys())

   

    assert not missing_arrays, f'Missing required array keys: {missing_arrays}'

    assert not missing_dicts, f'Missing required metadata keys: {missing_dicts}'

   

    # Create Zarr group with v2 compatibility

    zarr_path = Path(zarr_path)

    zarr_path.parent.mkdir(parents=True, exist_ok=True)

   

    store = zarr.open_group(str(zarr_path), mode='w', zarr_format=2)

   

    # Configure compression codec

    compressor = numcodecs.Blosc(

        cname='zstd',

        clevel=clevel,

        shuffle=numcodecs.Blosc.BITSHUFFLE

    )