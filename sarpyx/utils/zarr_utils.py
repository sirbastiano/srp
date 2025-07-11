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
    
    # Save arrays with optimized compression
    for array_name in required_arrays:
        array_data = result[array_name]
        assert isinstance(array_data, np.ndarray), f'Expected {array_name} to be ndarray, got {type(array_data)}'
        
        # Determine optimal chunk size
        if chunks == 'auto':
            chunk_shape = tuple(min(dim_size, 4096) for dim_size in array_data.shape)
        elif isinstance(chunks, tuple):
            chunk_shape = chunks
        else:
            chunk_shape = (4096,) * array_data.ndim
        
        # Create array with Zarr v2 compatible compression
        zarr_array = store.create_array(
            name=array_name,
            shape=array_data.shape,
            dtype=array_data.dtype,
            chunks=chunk_shape,
            compressor=compressor,
            overwrite=True
        )
        zarr_array[:] = array_data
    
    # Handle metadata and ephemeris with flexible type conversion
    for dict_name in required_dicts:
        dict_data = result[dict_name]
        
        # Convert DataFrame to dictionary format
        if hasattr(dict_data, 'to_dict'):
            # Convert DataFrame to records format for better JSON compatibility
            dict_data = dict_data.to_dict('records')
        elif isinstance(dict_data, dict):
            # Already a dictionary, keep as-is
            pass
        else:
            raise ValueError(
                f'{dict_name} must be pandas DataFrame or dictionary, '
                f'got {type(dict_data)}'
            )
        
        # Serialize and store as attributes
        serialized_data = _serialize_dict({'data': dict_data})
        store.attrs[dict_name] = serialized_data


def concatenate_slices(
    slice_zarr_paths: List[Union[str, Path]],
    output_zarr_path: Union[str, Path],
    chunks: Optional[Union[str, Tuple[int, ...]]] = 'auto',
    clevel: int = 7,
) -> None:
    """Concatenate multiple slice Zarr files into a single Zarr store.
    
    Vertically concatenates arrays and merges dictionaries from multiple slice files.
    
    Args:
        slice_zarr_paths: List of paths to slice Zarr files to concatenate
        output_zarr_path: Path where to save the concatenated Zarr store
        chunks: Chunking strategy for output arrays
        clevel: Compression level (1-9)
        
    Raises:
        FileNotFoundError: If any slice Zarr file doesn't exist
        ValueError: If slice files have incompatible shapes or missing required data
    """
    # Validate input paths
    output_path = Path(output_zarr_path)
    assert len(slice_zarr_paths) > 0, 'At least one slice path must be provided'
    
    # Verify all slice files exist
    for slice_path in slice_zarr_paths:
        slice_path_obj = Path(slice_path)
        if not slice_path_obj.exists():
            raise FileNotFoundError(f'Slice Zarr file not found: {slice_path}')
    
    # Required arrays and dictionaries
    required_arrays = ['raw', 'rc', 'rcmc', 'az']
    required_dicts = ['metadata', 'ephemeris']
    
    # Read first slice to get structure and initialize concatenated data
    first_store = zarr.open(str(slice_zarr_paths[0]), mode='r')
    
    # Initialize concatenated arrays and dictionaries
    concatenated_arrays = {}
    concatenated_dicts = {}
    
    # Load all slices and concatenate
    all_arrays = {array_name: [] for array_name in required_arrays}
    all_dicts = {dict_name: [] for dict_name in required_dicts}
    
    for i, slice_path in enumerate(slice_zarr_paths):
        print(f'📖 Reading slice {i+1}/{len(slice_zarr_paths)}: {slice_path}')
        store = zarr.open(str(slice_path), mode='r')
        
        # Collect arrays
        for array_name in required_arrays:
            if array_name not in store:
                raise ValueError(f'Required array "{array_name}" not found in {slice_path}')
            array_data = store[array_name][:]
            all_arrays[array_name].append(array_data)
        
        # Collect dictionaries from attributes
        for dict_name in required_dicts:
            if dict_name not in store.attrs:
                raise ValueError(f'Required dictionary "{dict_name}" not found in {slice_path} attributes')
            dict_data = dict(store.attrs[dict_name])
            all_dicts[dict_name].append(dict_data)
    
    # Concatenate arrays vertically (along first axis)
    print('🔗 Concatenating arrays...')
    for array_name in required_arrays:
        concatenated_arrays[array_name] = np.concatenate(all_arrays[array_name], axis=0)
        print(f'   {array_name}: {concatenated_arrays[array_name].shape}')
    
    # Merge dictionaries (combine all entries)
    print('📋 Merging dictionaries...')
    for dict_name in required_dicts:
        concatenated_dicts[dict_name] = {}
        for i, dict_data in enumerate(all_dicts[dict_name]):
            for key, value in dict_data.items():
                # Create unique keys for each slice if there are conflicts
                if key in concatenated_dicts[dict_name]:
                    concatenated_dicts[dict_name][f'{key}_slice_{i}'] = value
                else:
                    concatenated_dicts[dict_name][key] = value
    
    # Create output Zarr store
    print(f'💾 Creating concatenated Zarr store at: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing store if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    store = zarr.open(str(output_path), mode='w')
    
    # Save concatenated arrays with compression
    for array_name in required_arrays:
        array_data = concatenated_arrays[array_name]
        
        # Determine appropriate chunk size
        if chunks == 'auto':
            chunk_shape = tuple(min(s, 4096) for s in array_data.shape)
        elif isinstance(chunks, tuple):
            chunk_shape = chunks
        else:
            chunk_shape = (4096,) * array_data.ndim
        
        print(f'   Saving {array_name} with shape {array_data.shape}, chunks {chunk_shape}')
        zarr_array = store.create_array(
            name=array_name,
            shape=array_data.shape,
            dtype=array_data.dtype,
            chunks=chunk_shape,
            compressors=[
                {
                    'name': 'blosc',
                    'configuration': {
                        'cname': 'zstd',
                        'clevel': clevel,
                        'shuffle': 'bitshuffle'
                    }
                }
            ],
            overwrite=True
        )
        zarr_array[:] = array_data
    
    # Save merged dictionaries as attributes
    for dict_name in required_dicts:
        store.attrs[dict_name] = concatenated_dicts[dict_name]
        print(f'   Saved {dict_name} with {len(concatenated_dicts[dict_name])} entries')
    
    print(f'✅ Successfully created concatenated Zarr store with {len(slice_zarr_paths)} slices')


def concatenate_slices_efficient(
    slice_zarr_paths: List[Union[str, Path]],
    output_zarr_path: Union[str, Path],
    chunks: Optional[Union[str, Tuple[int, ...]]] = 'auto',
    clevel: int = 5
) -> None:
    """Memory-efficiently concatenate multiple slice Zarr files into a single Zarr store.
    
    Processes slices one at a time to minimize memory usage by writing chunks directly
    to the output Zarr store without loading all data into memory simultaneously.
    
    Args:
        slice_zarr_paths: List of paths to slice Zarr files to concatenate
        output_zarr_path: Path where to save the concatenated Zarr store
        chunks: Chunking strategy for output arrays
        clevel: Compression level (1-9)
        
    Raises:
        FileNotFoundError: If any slice Zarr file doesn't exist
        ValueError: If slice files have incompatible shapes or missing required data
    """

    
    output_path = Path(output_zarr_path)
    assert len(slice_zarr_paths) > 0, 'At least one slice path must be provided'
    
    # Verify all slice files exist
    for slice_path in slice_zarr_paths:
        slice_path_obj = Path(slice_path)
        if not slice_path_obj.exists():
            raise FileNotFoundError(f'Slice Zarr file not found: {slice_path}')
    
    required_arrays = ['raw', 'rc', 'rcmc', 'az']
    required_dicts = ['metadata', 'ephemeris']
    
    print(f'🔍 Analyzing slice structures...')
    
    # First pass: analyze all slices to determine total shapes and validate compatibility
    slice_shapes = {}
    total_shapes = {}
    merged_dicts = {dict_name: {} for dict_name in required_dicts}
    
    for i, slice_path in enumerate(slice_zarr_paths):
        store = zarr.open(str(slice_path), mode='r')
        
        # Validate and collect array shapes
        current_shapes = {}
        for array_name in required_arrays:
            if array_name not in store:
                raise ValueError(f'Required array "{array_name}" not found in {slice_path}')
            
            shape = store[array_name].shape
            dtype = store[array_name].dtype
            current_shapes[array_name] = (shape, dtype)
            
            if i == 0:
                # Initialize total shapes with first slice
                total_shapes[array_name] = (shape, dtype)
            else:
                # Validate compatibility and update total shape
                prev_shape, prev_dtype = total_shapes[array_name]
                if shape[1:] != prev_shape[1:]:
                    raise ValueError(
                        f'Incompatible shapes for {array_name}: '
                        f'{prev_shape} vs {shape} in {slice_path}'
                    )
                if dtype != prev_dtype:
                    raise ValueError(
                        f'Incompatible dtypes for {array_name}: '
                        f'{prev_dtype} vs {dtype} in {slice_path}'
                    )
                # Update total shape (concatenate along axis 0)
                new_shape = (prev_shape[0] + shape[0],) + shape[1:]
                total_shapes[array_name] = (new_shape, dtype)
        
        slice_shapes[i] = current_shapes
        
        # Collect dictionaries
        for dict_name in required_dicts:
            if dict_name not in store.attrs:
                raise ValueError(f'Required dictionary "{dict_name}" not found in {slice_path} attributes')
            dict_data = dict(store.attrs[dict_name])
            for key, value in dict_data.items():
                if key in merged_dicts[dict_name]:
                    merged_dicts[dict_name][f'{key}_slice_{i}'] = value
                else:
                    merged_dicts[dict_name][key] = value
    
    print(f'📊 Total shapes determined:')
    for array_name in required_arrays:
        shape, dtype = total_shapes[array_name]
        print(f'   {array_name}: {shape} ({dtype})')
    
    # Remove existing output store if it exists
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create output Zarr store and initialize arrays
    print(f'🏗️ Creating output Zarr store at: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_store = zarr.open(str(output_path), mode='w')
    
    # Create output arrays with final shapes
    output_arrays = {}
    for array_name in required_arrays:
        final_shape, dtype = total_shapes[array_name]
        
        # Determine chunk size
        if chunks == 'auto':
            chunk_shape = tuple(min(s, 4096) for s in final_shape)
        elif isinstance(chunks, tuple):
            chunk_shape = chunks
        else:
            chunk_shape = (4096,) * len(final_shape)
        
        print(f'🔧 Creating {array_name} array: shape={final_shape}, chunks={chunk_shape}')
        output_arrays[array_name] = output_store.create_array(
            name=array_name,
            shape=final_shape,
            dtype=dtype,
            chunks=chunk_shape,
            compressors=[
                {
                    'name': 'blosc',
                    'configuration': {
                        'cname': 'zstd',
                        'clevel': clevel,
                        'shuffle': 'bitshuffle'
                    }
                }
            ],
            overwrite=True
        )
    
    # Second pass: copy data slice by slice
    print(f'📋 Copying data from {len(slice_zarr_paths)} slices...')
    current_offsets = {array_name: 0 for array_name in required_arrays}
    
    for i, slice_path in enumerate(slice_zarr_paths):
        print(f'   Processing slice {i+1}/{len(slice_zarr_paths)}: {Path(slice_path).name}')
        slice_store = zarr.open(str(slice_path), mode='r')
        
        for array_name in required_arrays:
            # Get slice data
            slice_array = slice_store[array_name]
            slice_shape = slice_array.shape
            
            # Calculate target slice in output array
            start_idx = current_offsets[array_name]
            end_idx = start_idx + slice_shape[0]
            
            # Copy data directly to output array
            output_arrays[array_name][start_idx:end_idx] = slice_array[:]
            
            # Update offset for next slice
            current_offsets[array_name] = end_idx
            
        print(f'     ✓ Copied slice {i+1} data')
    
    # Save merged dictionaries as attributes
    print(f'📝 Saving metadata and ephemeris...')
    for dict_name in required_dicts:
        output_store.attrs[dict_name] = merged_dicts[dict_name]
        print(f'   Saved {dict_name} with {len(merged_dicts[dict_name])} entries')
    
    print(f'✅ Successfully created memory-efficient concatenated Zarr store')
    print(f'   Output: {output_path}')
    print(f'   Total arrays: {len(required_arrays)}')
    for array_name in required_arrays:
        shape, _ = total_shapes[array_name]
        print(f'   {array_name}: {shape}')


# -----------  Classes -----------------
class ZarrManager:
    """
    A comprehensive class for managing Zarr files, providing convenient methods for data access, slicing, metadata extraction, analysis, visualization, and export.
    """
    
    def __init__(self, file_path: str) -> None:
        """
        Initialize with zarr file path.
        
        Args:
            file_path: Path to the zarr file or directory
        """
        self.file_path = file_path
        self.filename = Path(file_path).stem
        self._zarr_array = None
        self._metadata = None
        self._ephemeris = None
        self.echoes_shape = self.load().shape if self.load() is not None else None
        
    def load(self) -> zarr.Array:
        """
        Load the zarr array and cache it.
        
        Returns:
            zarr.Array: The loaded zarr array
        """
        if self._zarr_array is None:
            self._zarr_array = zarr.open(self.file_path, mode='r')
        return self._zarr_array
    
    def _create_output_dir(self, output_path: str) -> None:
        """
        Create output directory if it does not exist.
        
        Args:
            output_path: Path to the output directory
        """
        

        # Create a temporary directory for processing
        random_idx = np.random.randint(0, int(1e12))
        print(f'🔢 Random index for processing: {random_idx}')
        output_dir = Path(output_path)
        tmp_dir = output_dir / 'tmp' / str(random_idx)
        # create temporary directory for processing
        tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f'📂 Temporary directory created at: {tmp_dir}')
        
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f'Created output directory: {output_path}')
        else:
            print(f'Output directory already exists: {output_path}')
    
    @property
    def info(self) -> Dict[str, Any]:
        """
        Get basic info about the zarr array.
        
        Returns:
            Dict containing shape, dtype, chunks, nbytes, and size in MB
        """
        arr = self.load()
        return {
            'shape': arr.shape,
            'dtype': arr.dtype,
            'chunks': arr.chunks,
            'nbytes': arr.nbytes,
            'size_mb': arr.nbytes / (1024**2)
        }
    
    # ------------- Internal methods for block slicing -------------
    
    @gc_collect
    def drop(self, rows: Optional[Union[Tuple[int, int], slice, List[int]]] = None, 
             cols: Optional[Union[Tuple[int, int], slice, List[int]]] = None) -> Tuple[np.ndarray, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Drop specified rows and/or columns from the zarr array.
        
        This method provides flexible dropping capabilities for zarr arrays, supporting
        tuple-based indexing (start, end), slice objects, and lists of indices to drop.
        The method returns the array with specified rows/columns removed along with
        corresponding metadata and ephemeris information.
        
        Args:
            rows (Optional[Union[Tuple[int, int], slice, List[int]]]): 
                Row dropping specification. Can be:
                - Tuple of (start_idx, end_idx) for row range to drop
                - slice object for custom dropping
                - List of specific row indices to drop
                - None to keep all rows
                Defaults to None.
            cols (Optional[Union[Tuple[int, int], slice, List[int]]]): 
                Column dropping specification. Can be:
                - Tuple of (start_idx, end_idx) for column range to drop
                - slice object for custom dropping  
                - List of specific column indices to drop
                - None to keep all columns
                Defaults to None.
                
        Returns:
            Tuple[np.ndarray, Optional[pd.DataFrame], Optional[pd.DataFrame]]: A 3-element tuple containing:
                - np.ndarray: The array data with specified rows/columns dropped
                - Optional[pd.DataFrame]: Metadata with corresponding rows dropped (if rows were dropped)
                - Optional[pd.DataFrame]: Ephemeris information for the dataset
                
        Examples:
            >>> # Drop specific row and column ranges
            >>> data, meta, eph = zarr_obj.drop(rows=(100, 200), cols=(50, 150))
            >>> # Drop specific indices
            >>> data, meta, eph = zarr_obj.drop(rows=[10, 20, 30], cols=[5, 15, 25])
            >>> # Drop using slice objects
            >>> data, meta, eph = zarr_obj.drop(rows=slice(100, 200, 2))
            >>> # Drop only rows, keep all columns
            >>> data, meta, eph = zarr_obj.drop(rows=(0, 50))
        """
        arr = self.load()
        metadata = self.get_metadata()
        
        # Get array dimensions
        n_rows, n_cols = arr.shape
        
        # Convert drop specifications to boolean masks
        row_mask = np.ones(n_rows, dtype=bool)  # Keep all rows by default
        col_mask = np.ones(n_cols, dtype=bool)  # Keep all columns by default
        
        # Handle row dropping
        if rows is not None:
            if isinstance(rows, tuple):
                # Convert tuple to range of indices
                start_idx, end_idx = rows
                assert 0 <= start_idx < n_rows, f'Start row index {start_idx} out of bounds [0, {n_rows})'
                assert 0 <= end_idx <= n_rows, f'End row index {end_idx} out of bounds [0, {n_rows}]'
                assert start_idx < end_idx, f'Start index {start_idx} must be less than end index {end_idx}'
                row_mask[start_idx:end_idx] = False
            elif isinstance(rows, slice):
                # Convert slice to indices
                indices = list(range(*rows.indices(n_rows)))
                row_mask[indices] = False
            elif isinstance(rows, list):
                # Direct list of indices
                for idx in rows:
                    assert 0 <= idx < n_rows, f'Row index {idx} out of bounds [0, {n_rows})'
                row_mask[rows] = False
            else:
                raise ValueError(f'rows must be tuple, slice, list, or None, got {type(rows)}')
        
        # Handle column dropping
        if cols is not None:
            if isinstance(cols, tuple):
                # Convert tuple to range of indices
                start_idx, end_idx = cols
                assert 0 <= start_idx < n_cols, f'Start column index {start_idx} out of bounds [0, {n_cols})'
                assert 0 <= end_idx <= n_cols, f'End column index {end_idx} out of bounds [0, {n_cols}]'
                assert start_idx < end_idx, f'Start index {start_idx} must be less than end index {end_idx}'
                col_mask[start_idx:end_idx] = False
            elif isinstance(cols, slice):
                # Convert slice to indices
                indices = list(range(*cols.indices(n_cols)))
                col_mask[indices] = False
            elif isinstance(cols, list):
                # Direct list of indices
                for idx in cols:
                    assert 0 <= idx < n_cols, f'Column index {idx} out of bounds [0, {n_cols})'
                col_mask[cols] = False
            else:
                raise ValueError(f'cols must be tuple, slice, list, or None, got {type(cols)}')
        
        # Apply masks to get the data with dropped rows/columns
        filtered_arr = arr[row_mask, :][:, col_mask]
        
        # Handle metadata - drop corresponding rows if rows were dropped
        filtered_metadata = metadata
        if rows is not None and metadata is not None:
            filtered_metadata = metadata[row_mask].reset_index(drop=True)
        
        return (filtered_arr, filtered_metadata, self.get_ephemeris())
    
    
    @gc_collect
    def get_slice(self, rows: Optional[Union[Tuple[int, int], slice]] = None, 
                  cols: Optional[Union[Tuple[int, int], slice]] = None, 
                  step: Optional[int] = None) -> Tuple[np.ndarray, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Extract a slice from the zarr array with optional row and column indexing.
        
        This method provides flexible slicing capabilities for zarr arrays, supporting
        both tuple-based indexing (start, end) and slice objects. The method automatically
        handles the conversion between different indexing formats and returns the sliced
        data along with associated metadata and ephemeris information.
        
        Args:
            rows (Optional[Union[Tuple[int, int], slice]]): 
                Row indexing specification. Can be:
                - Tuple of (start_idx, end_idx) for row range
                - slice object for custom slicing
                - None to select all rows
                Defaults to None.
            cols (Optional[Union[Tuple[int, int], slice]]):
                Column indexing specification. Can be:
                - Tuple of (start_idx, end_idx) for column range  
                - slice object for custom slicing
                - None to select all columns
                Defaults to None.
            step (Optional[int]):
                Step size for slicing when using tuple indexing. Only applied
                when rows or cols are provided as tuples. Ignored for slice objects.
                Defaults to None.
                
        Returns:
            Tuple[np.ndarray, Optional[pd.DataFrame], Optional[pd.DataFrame]]: A 3-element tuple containing:
                - np.ndarray: The sliced array data
                - Optional[pd.DataFrame]: Metadata associated with the zarr array
                - Optional[pd.DataFrame]: Ephemeris information for the dataset
                
        Examples:
            >>> # Get full array
            >>> data, meta, eph = zarr_obj.get_slice()
            >>> # Get specific row and column ranges
            >>> data, meta, eph = zarr_obj.get_slice(rows=(100, 200), cols=(50, 150))
            >>> # Get every 2nd element in specified range
            >>> data, meta, eph = zarr_obj.get_slice(rows=(0, 100), step=2)
            >>> # Use slice objects for advanced indexing
            >>> data, meta, eph = zarr_obj.get_slice(rows=slice(None, None, 2))
        """
        arr = self.load()
        metadata = self.get_metadata()
        
        # Convert tuples to slices
        if isinstance(rows, tuple):
            rows = slice(rows[0], rows[1], step)
        if isinstance(cols, tuple):
            cols = slice(cols[0], cols[1], step)
            
        # Default to full array if no slicing specified
        if rows is None and cols is None:
            return (arr[:], metadata, self.get_ephemeris())
        elif rows is None:
            return (arr[:, cols], metadata, self.get_ephemeris())
        elif cols is None:
            return (arr[rows, :], metadata.iloc[rows], self.get_ephemeris())
        else:
            return (arr[rows, cols], metadata, self.get_ephemeris())



    def get_slice_block(self, N_blocks: int = 5, slice_idx: int = 0, outDict: bool = True, verbose: bool = False) -> None:
        """
        Internal method to handle block slicing for large arrays.
        
        This method is a placeholder for future implementation of block slicing
        logic, which can be useful for very large datasets that cannot fit into memory.
        """
        
        if slice_idx >= N_blocks:
            raise ValueError(f'Block number {slice_idx} exceeds total blocks {N_blocks}')
        # Calculate block size
        block_size = self.echoes_shape[0] // N_blocks # type: ignore
        start_row = slice_idx * block_size
        end_row = (slice_idx + 1) * block_size if slice_idx < N_blocks - 1 else self.echoes_shape[0] # type: ignore
        # Return the slice for the specified block
        if verbose:
            print(f'Extracting block {slice_idx + 1}/{N_blocks}: rows {start_row} to {end_row}')
        if start_row >= self.echoes_shape[0] or end_row > self.echoes_shape[0]:
            raise ValueError(f'Block {slice_idx} exceeds array bounds: {self.echoes_shape[0]} rows available')
        
        # Use get_slice to extract the block
        out = self.get_slice(rows=slice(start_row, end_row), cols=None)
        if outDict:
            return {'echo': out[0], 'metadata': out[1], 'ephemeris': out[2]}
        else:
            return out

    # ------------- End Internal methods for block slicing -------------



    def get_metadata(self) -> Optional['pd.DataFrame']:
        """
        Extract metadata from zarr attributes.
        
        Returns:
            pandas DataFrame containing metadata if available, None otherwise
        """
        if self._metadata is None:
            arr = self.load()
            if 'metadata' in arr.attrs:
                import pandas as pd
                self._metadata = pd.DataFrame(arr.attrs['metadata'])
        return self._metadata
    
    def get_ephemeris(self) -> Optional['pd.DataFrame']:
        """
        Extract ephemeris data from zarr attributes.
        
        Returns:
            pandas DataFrame containing ephemeris data if available, None otherwise
        """
        if self._ephemeris is None:
            arr = self.load()
            if 'ephemeris' in arr.attrs:
                import pandas as pd
                self._ephemeris = pd.DataFrame(arr.attrs['ephemeris'])
        return self._ephemeris
    
    @gc_collect
    def stats(self, sample_size: int = 1000) -> Dict[str, Optional[float]]:
        """
        Get statistical summary of the data.
        
        Args:
            sample_size: Number of samples to use for statistics (for large arrays)
            
        Returns:
            Dictionary containing statistical measures (mean, std, min, max, etc.)
        """
        arr = self.load()
        
        # Sample data for large arrays
        if arr.size > sample_size * 1000:
            step_row = max(1, arr.shape[0] // sample_size)
            step_col = max(1, arr.shape[1] // sample_size)
            sample = arr[::step_row, ::step_col]
        else:
            sample = arr[:]
            
        return {
            'mean': np.mean(sample),
            'std': np.std(sample),
            'min': np.min(sample),
            'max': np.max(sample),
            'real_mean': np.mean(np.real(sample)) if np.iscomplexobj(sample) else None,
            'imag_mean': np.mean(np.imag(sample)) if np.iscomplexobj(sample) else None,
            'magnitude_mean': np.mean(np.abs(sample)) if np.iscomplexobj(sample) else None,
            'phase_std': np.std(np.angle(sample)) if np.iscomplexobj(sample) else None
        } # type: ignore
    
    def visualize_slice(self, rows: Tuple[int, int] = (0, 100), 
                       cols: Tuple[int, int] = (0, 100), 
                       plot_type: str = 'magnitude') -> None:
        """
        Visualize a slice of the data.
        
        Args:
            rows: Tuple for row range (start, end)
            cols: Tuple for column range (start, end)
            plot_type: Type of plot - 'magnitude', 'phase', 'real', 'imag'
        """
        import matplotlib.pyplot as plt
        
        data_slice = self.get_slice(rows, cols)
        
        if plot_type == 'magnitude' and np.iscomplexobj(data_slice):
            plot_data = np.abs(data_slice)
            title = f'Magnitude - Rows {rows}, Cols {cols}'
        elif plot_type == 'phase' and np.iscomplexobj(data_slice):
            plot_data = np.angle(data_slice)
            title = f'Phase - Rows {rows}, Cols {cols}'
        elif plot_type == 'real':
            plot_data = np.real(data_slice)
            title = f'Real Part - Rows {rows}, Cols {cols}'
        elif plot_type == 'imag':
            plot_data = np.imag(data_slice)
            title = f'Imaginary Part - Rows {rows}, Cols {cols}'
        else:
            plot_data = data_slice
            title = f'Data - Rows {rows}, Cols {cols}'
        
        plt.figure(figsize=(10, 8))
        plt.imshow(plot_data, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.show()
    
    def get_compression_info(self) -> Dict[str, float]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary containing compression information (sizes, ratio, space saved)
        """
        arr = self.load()
        uncompressed_size = arr.nbytes
        
        # Calculate compressed size
        compressed_size = 0
        for root, dirs, files in os.walk(self.file_path):
            for file in files:
                compressed_size += os.path.getsize(os.path.join(root, file))
        
        return {
            'uncompressed_size_mb': uncompressed_size / (1024**2),
            'compressed_size_mb': compressed_size / (1024**2),
            'compression_ratio': uncompressed_size / compressed_size,
            'space_saved_percent': ((uncompressed_size - compressed_size) / uncompressed_size) * 100
        }
    
    def export_slice(self, output_path: str, 
                    rows: Optional[Union[Tuple[int, int], slice]] = None, 
                    cols: Optional[Union[Tuple[int, int], slice]] = None, 
                    format: str = 'npy') -> None:
        """
        Export a slice to various formats.
        
        Args:
            output_path: Path for output file
            rows: Row slice specification
            cols: Column slice specification
            format: Export format - 'npy', 'csv', 'hdf5'
        """
        data_slice = self.get_slice(rows, cols)
        
        if format == 'npy':
            np.save(output_path, data_slice)
        elif format == 'csv':
            if np.iscomplexobj(data_slice):
                # Save real and imaginary parts separately
                np.savetxt(output_path.replace('.csv', '_real.csv'), 
                          np.real(data_slice), delimiter=',')
                np.savetxt(output_path.replace('.csv', '_imag.csv'), 
                          np.imag(data_slice), delimiter=',')
            else:
                np.savetxt(output_path, data_slice, delimiter=',')
        elif format == 'hdf5':
            import h5py
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('data', data=data_slice)
        
        print(f'Exported slice to {output_path} in {format} format')
    
    def find_peaks(self, rows: Optional[Union[Tuple[int, int], slice]] = None, 
                   cols: Optional[Union[Tuple[int, int], slice]] = None, 
                   threshold_percentile: float = 95) -> Dict[str, Any]:
        """
        Find peaks in the data (useful for SAR analysis).
        
        Args:
            rows: Row slice to analyze
            cols: Column slice to analyze
            threshold_percentile: Percentile for peak detection (0-100)
            
        Returns:
            Dictionary containing peak information (indices, values, threshold, count)
        """
        data_slice = self.get_slice(rows, cols)
        magnitude = np.abs(data_slice) if np.iscomplexobj(data_slice) else data_slice
        
        threshold = np.percentile(magnitude, threshold_percentile)
        peak_locations = np.where(magnitude > threshold)
        
        return {
            'peak_indices': list(zip(peak_locations[0], peak_locations[1])),
            'peak_values': magnitude[peak_locations],
            'threshold': threshold,
            'num_peaks': len(peak_locations[0])
        }
    
    def memory_efficient_operation(self, operation: Callable[[np.ndarray], Any], 
                                  chunk_size: int = 1000) -> List[Any]:
        """
        Apply an operation to the data in chunks to save memory.
        
        Args:
            operation: Function to apply to each chunk
            chunk_size: Size of chunks to process (number of rows)
            
        Returns:
            List of results from applying operation to each chunk
        """
        arr = self.load()
        results = []
        
        for i in range(0, arr.shape[0], chunk_size):
            end_i = min(i + chunk_size, arr.shape[0])
            chunk = arr[i:end_i, :]
            result = operation(chunk)
            results.append(result)
        
        return results
    
    # -------- Chunking Methods -----------
    
    
    
    
    # -------- Chunking Methods End -----------
    
    
    @gc_collect
    def _export_raw(self):
        """
        Export raw SAR data, metadata, and ephemeris as a dictionary.

        Returns:
            dict: Dictionary with the following keys:
            - 'metadata' (Optional[pandas.DataFrame]): Metadata DataFrame, or None if unavailable.
            - 'ephemeris' (Optional[pandas.DataFrame]): Ephemeris DataFrame, or None if unavailable.
            - 'echo' (np.ndarray): Echo data as a NumPy array.
        """
        
        metadata = self.get_metadata()
        ephemeris = self.get_ephemeris()
        echo_zarr = self.load()
        echo_np = echo_zarr[:]
        del echo_zarr
        
        return {
            'metadata': metadata,
            'ephemeris': ephemeris,
            'echo': echo_np
        }
    
    def __repr__(self) -> str:
        """
        String representation of the ZarrManager.
        
        Returns:
            String with basic info about the zarr array
        """
        info = self.info
        return (f"ZarrManager(file_path='{self.file_path}', "
                f"shape={info['shape']}, dtype={info['dtype']}, "
                f"chunks={info['chunks']}, size_mb={info['size_mb']:.2f})")


class ProductHandler(ZarrManager):
    """
    A specialized ZarrManager subclass for handling concatenated Zarr data with multiple arrays.
    
    This class manages Zarr stores containing multiple arrays (az, raw, rc, rcmc) and their
    associated metadata and ephemeris information stored as JSON attributes.
    """
    
    def __init__(self, file_path: str) -> None:
        """
        Initialize with concatenated zarr file path.
        
        Args:
            file_path (str): Path to the concatenated zarr store directory
        """
        self.file_path = file_path
        self._zarr_store = None
        self._metadata = None
        self._ephemeris = None
        self._array_names = ['az', 'raw', 'rc', 'rcmc']
        self._arrays = {}
        
        # Get shapes of all arrays
        store = self._load_store()
        self.array_shapes = {name: store[name].shape for name in self._array_names if name in store}
        
    def _load_store(self) -> zarr.Group:
        """
        Load the zarr store (group) and cache it.
        
        Returns:
            zarr.Group: The loaded zarr store containing multiple arrays
        """
        if self._zarr_store is None:
            self._zarr_store = zarr.open(self.file_path, mode='r')
        return self._zarr_store
    
    def load(self) -> zarr.Group:
        """
        Override parent load method to return the zarr group instead of single array.
        
        Returns:
            zarr.Group: The loaded zarr store
        """
        return self._load_store()
    
    def get_array(self, array_name: str) -> zarr.Array:
        """
        Get a specific array from the concatenated zarr store.
        
        Args:
            array_name (str): Name of the array ('az', 'raw', 'rc', 'rcmc')
            
        Returns:
            zarr.Array: The requested zarr array
            
        Raises:
            ValueError: If array_name is not valid or not found in store
        """
        if array_name not in self._array_names:
            raise ValueError(f"Array name must be one of {self._array_names}, got '{array_name}'")
        
        store = self._load_store()
        if array_name not in store:
            raise ValueError(f"Array '{array_name}' not found in zarr store")
        
        if array_name not in self._arrays:
            self._arrays[array_name] = store[array_name]
        
        return self._arrays[array_name]
    
    @property
    def info(self) -> Dict[str, Any]:
        """
        Get comprehensive info about all arrays in the concatenated zarr store.
        
        Returns:
            Dict[str, Any]: Dictionary containing info for each array and total size
        """
        store = self._load_store()
        info_dict = {}
        total_size_mb = 0
        
        for array_name in self._array_names:
            if array_name in store:
                arr = store[array_name]
                array_info = {
                    'shape': arr.shape,
                    'dtype': arr.dtype,
                    'chunks': arr.chunks,
                    'nbytes': arr.nbytes,
                    'size_mb': arr.nbytes / (1024**2)
                }
                info_dict[array_name] = array_info
                total_size_mb += array_info['size_mb']
        
        info_dict['total_size_mb'] = total_size_mb
        info_dict['available_arrays'] = list(store.keys())
        
        return info_dict
    
    def get_metadata(self) -> Optional['pd.DataFrame']:
        """
        Extract metadata from zarr store attributes.
        
        Returns:
            Optional[pd.DataFrame]: Metadata DataFrame if available, None otherwise
        """
        if self._metadata is None:
            store = self._load_store()
            if 'metadata' in store.attrs:
                import pandas as pd
                self._metadata = pd.DataFrame(store.attrs['metadata'])
        return self._metadata
    
    def get_ephemeris(self) -> Optional['pd.DataFrame']:
        """
        Extract ephemeris data from zarr store attributes.
        
        Returns:
            Optional[pd.DataFrame]: Ephemeris DataFrame if available, None otherwise
        """
        if self._ephemeris is None:
            store = self._load_store()
            if 'ephemeris' in store.attrs:
                import pandas as pd
                self._ephemeris = pd.DataFrame(store.attrs['ephemeris'])
        return self._ephemeris
    
    @gc_collect
    def get_slice(self, 
                  array_names: Union[str, List[str]], 
                  rows: Optional[Union[Tuple[int, int], slice]] = None,
                  cols: Optional[Union[Tuple[int, int], slice]] = None,
                  step: Optional[int] = None,
                  include_metadata: bool = True) -> Dict[str, Any]:
        """
        Extract slices from specified arrays with associated metadata and ephemeris.
        
        Args:
            array_names (Union[str, List[str]]): Name(s) of arrays to slice ('az', 'raw', 'rc', 'rcmc')
            rows (Optional[Union[Tuple[int, int], slice]]): Row indexing specification
            cols (Optional[Union[Tuple[int, int], slice]]): Column indexing specification  
            step (Optional[int]): Step size for tuple-based indexing
            include_metadata (bool): Whether to include metadata and ephemeris in output
            
        Returns:
            Dict[str, Any]: Dictionary containing sliced arrays and optionally metadata/ephemeris
            
        Examples:
            >>> # Get slice from single array
            >>> result = manager.get_slice('raw', rows=(100, 200), cols=(50, 150))
            >>> # Get slices from multiple arrays
            >>> result = manager.get_slice(['raw', 'rc'], rows=(0, 500))
            >>> # Access results
            >>> raw_data = result['arrays']['raw']
            >>> metadata = result['metadata']
        """
        # Normalize array_names to list
        if isinstance(array_names, str):
            array_names = [array_names]
        
        # Validate array names
        for name in array_names:
            if name not in self._array_names:
                raise ValueError(f"Array name must be one of {self._array_names}, got '{name}'")
        
        # Convert tuples to slices
        if isinstance(rows, tuple):
            rows = slice(rows[0], rows[1], step)
        if isinstance(cols, tuple):
            cols = slice(cols[0], cols[1], step)
        
        # Extract data from each requested array
        result = {'arrays': {}}
        
        for array_name in array_names:
            arr = self.get_array(array_name)
            
            # Apply slicing
            if rows is None and cols is None:
                sliced_data = arr[:]
            elif rows is None:
                sliced_data = arr[:, cols]
            elif cols is None:
                sliced_data = arr[rows, :]
            else:
                sliced_data = arr[rows, cols]
            
            result['arrays'][array_name] = sliced_data
        
        # Include metadata and ephemeris if requested
        if include_metadata:
            metadata = self.get_metadata()
            ephemeris = self.get_ephemeris()
            
            # If rows are specified and metadata exists, slice metadata accordingly
            if rows is not None and metadata is not None:
                if isinstance(rows, slice):
                    # Convert slice to indices for pandas iloc
                    start, stop, step_val = rows.indices(len(metadata))
                    row_indices = list(range(start, stop, step_val or 1))
                    result['metadata'] = metadata.iloc[row_indices].reset_index(drop=True)
                else:
                    result['metadata'] = metadata
            else:
                result['metadata'] = metadata
            
            result['ephemeris'] = ephemeris
        
        # Add slice information for reference
        result['slice_info'] = {
            'rows': rows,
            'cols': cols,
            'array_names': array_names,
            'shapes': {name: result['arrays'][name].shape for name in array_names}
        }
        
        return result
    
    def get_array_slice(self, 
                       array_name: str,
                       rows: Optional[Union[Tuple[int, int], slice]] = None,
                       cols: Optional[Union[Tuple[int, int], slice]] = None,
                       step: Optional[int] = None) -> np.ndarray:
        """
        Get a slice from a single array (convenience method).
        
        Args:
            array_name (str): Name of the array to slice
            rows (Optional[Union[Tuple[int, int], slice]]): Row indexing specification
            cols (Optional[Union[Tuple[int, int], slice]]): Column indexing specification
            step (Optional[int]): Step size for tuple-based indexing
            
        Returns:
            np.ndarray: Sliced array data
        """
        result = self.get_slice(array_name, rows=rows, cols=cols, step=step, include_metadata=False)
        return result['arrays'][array_name]
    
    def get_metadata_slice(self, 
                          rows: Optional[Union[Tuple[int, int], slice]] = None) -> Optional['pd.DataFrame']:
        """
        Get a slice of metadata corresponding to specified rows.
        
        Args:
            rows (Optional[Union[Tuple[int, int], slice]]): Row indexing specification
            
        Returns:
            Optional[pd.DataFrame]: Sliced metadata DataFrame if available
        """
        metadata = self.get_metadata()
        if metadata is None:
            return None
        
        if rows is None:
            return metadata
        
        # Convert tuple to slice
        if isinstance(rows, tuple):
            rows = slice(rows[0], rows[1])
        
        # Apply slicing to metadata
        if isinstance(rows, slice):
            start, stop, step_val = rows.indices(len(metadata))
            row_indices = list(range(start, stop, step_val or 1))
            return metadata.iloc[row_indices].reset_index(drop=True)
        
        return metadata
    
    @gc_collect
    def compare_arrays(self, 
                      array_names: List[str],
                      rows: Optional[Union[Tuple[int, int], slice]] = None,
                      cols: Optional[Union[Tuple[int, int], slice]] = None,
                      comparison_type: str = 'magnitude') -> Dict[str, Any]:
        """
        Compare multiple arrays side by side with statistics.
        
        Args:
            array_names (List[str]): Names of arrays to compare
            rows (Optional[Union[Tuple[int, int], slice]]): Row slice for comparison
            cols (Optional[Union[Tuple[int, int], slice]]): Column slice for comparison
            comparison_type (str): Type of comparison ('magnitude', 'phase', 'real', 'imag')
            
        Returns:
            Dict[str, Any]: Comparison results with statistics and data
        """
        result = self.get_slice(array_names, rows=rows, cols=cols, include_metadata=False)
        comparison_data = {}
        stats = {}
        
        for array_name in array_names:
            data = result['arrays'][array_name]
            
            if comparison_type == 'magnitude' and np.iscomplexobj(data):
                processed_data = np.abs(data)
            elif comparison_type == 'phase' and np.iscomplexobj(data):
                processed_data = np.angle(data)
            elif comparison_type == 'real':
                processed_data = np.real(data)
            elif comparison_type == 'imag':
                processed_data = np.imag(data)
            else:
                processed_data = data
            
            comparison_data[array_name] = processed_data
            stats[array_name] = {
                'mean': np.mean(processed_data),
                'std': np.std(processed_data),
                'min': np.min(processed_data),
                'max': np.max(processed_data)
            }
        
        return {
            'data': comparison_data,
            'statistics': stats,
            'comparison_type': comparison_type,
            'slice_info': result['slice_info']
        }
    
    def export_arrays(self, 
                     output_dir: str,
                     array_names: Optional[List[str]] = None,
                     rows: Optional[Union[Tuple[int, int], slice]] = None,
                     cols: Optional[Union[Tuple[int, int], slice]] = None,
                     format: str = 'npy') -> None:
        """
        Export multiple arrays to files.
        
        Args:
            output_dir (str): Directory to save exported files
            array_names (Optional[List[str]]): Arrays to export (default: all)
            rows (Optional[Union[Tuple[int, int], slice]]): Row slice to export
            cols (Optional[Union[Tuple[int, int], slice]]): Column slice to export
            format (str): Export format ('npy', 'hdf5')
        """
        if array_names is None:
            array_names = self._array_names
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get sliced data
        result = self.get_slice(array_names, rows=rows, cols=cols, include_metadata=True)
        
        # Export arrays
        for array_name in array_names:
            if array_name in result['arrays']:
                data = result['arrays'][array_name]
                
                if format == 'npy':
                    output_path = Path(output_dir) / f'{array_name}.npy'
                    np.save(output_path, data)
                elif format == 'hdf5':
                    import h5py
                    output_path = Path(output_dir) / f'{array_name}.h5'
                    with h5py.File(output_path, 'w') as f:
                        f.create_dataset('data', data=data)
                
                print(f'Exported {array_name} to {output_path}')
        
        # Export metadata if available
        if result['metadata'] is not None:
            metadata_path = Path(output_dir) / 'metadata.csv'
            result['metadata'].to_csv(metadata_path, index=False)
            print(f'Exported metadata to {metadata_path}')
        
        # Export ephemeris if available  
        if result['ephemeris'] is not None:
            ephemeris_path = Path(output_dir) / 'ephemeris.csv'
            result['ephemeris'].to_csv(ephemeris_path, index=False)
            print(f'Exported ephemeris to {ephemeris_path}')
    
    def visualize_arrays(self, 
                        array_names: Union[str, List[str]],
                        rows: Tuple[int, int] = (0, 100),
                        cols: Tuple[int, int] = (0, 100),
                        plot_type: str = 'magnitude',
                        show: bool = True,
                        vminmax: Optional[Union[Tuple[float, float], str]] = (0, 1000),
                        figsize: Tuple[int, int] = (15, 15)) -> None:
        """
        Visualize multiple arrays side by side.
        
        Args:
            array_names (Union[str, List[str]]): Array name(s) to visualize
            rows (Tuple[int, int]): Row range for visualization
            cols (Tuple[int, int]): Column range for visualization
            plot_type (str): Type of plot ('magnitude', 'phase', 'real', 'imag')
            show (bool): Whether to show the plot or not
            vminmax (Optional[Union[Tuple[float, float], str]]): Min/max values for colorbar or 'auto'
            figsize (Tuple[int, int]): Figure size for matplotlib
        """
        import matplotlib.pyplot as plt
        
        if isinstance(array_names, str):
            array_names = [array_names]
        
        result = self.get_slice(array_names, rows=rows, cols=cols, include_metadata=False)
        
        fig, axes = plt.subplots(1, len(array_names), figsize=figsize)
        if len(array_names) == 1:
            axes = [axes]
        
        for i, array_name in enumerate(array_names):
            data = result['arrays'][array_name]
            
            if plot_type == 'magnitude' and np.iscomplexobj(data):
                plot_data = np.abs(data)
                title_suffix = 'Magnitude'
            elif plot_type == 'phase' and np.iscomplexobj(data):
                plot_data = np.angle(data)
                title_suffix = 'Phase'
            elif plot_type == 'real':
                plot_data = np.real(data)
                title_suffix = 'Real'
            elif plot_type == 'imag':
                plot_data = np.imag(data)
                title_suffix = 'Imaginary'
            else:
                plot_data = data
                title_suffix = 'Data'

            
            
            # Set vmin/vmax for 'raw' array, otherwise use phase or provided/default
            if array_name == 'raw':
                vmin, vmax = 0, 10
            elif plot_type == 'phase':
                vmin, vmax = -np.pi, np.pi
            elif vminmax == 'auto':
                mean_val = np.mean(plot_data)
                std_val = np.std(plot_data)
                vmin, vmax = mean_val - std_val, mean_val + std_val
            elif vminmax is not None:
                vmin, vmax = vminmax
            else:
                vmin, vmax = 0, 1000

            im = axes[i].imshow(
                plot_data,
                aspect='auto',
                cmap='viridis',
                vmin=vmin,
                vmax=vmax
            )

            axes[i].set_title(f'{array_name.upper()} - {title_suffix}')
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
    
    def __repr__(self) -> str:
        """
        String representation of the ConcatenatedZarrManager.
        
        Returns:
            str: String with basic info about the concatenated zarr store
        """
        info = self.info
        available_arrays = info.get('available_arrays', [])
        total_size = info.get('total_size_mb', 0)
        
        return (f"ConcatenatedZarrManager(file_path='{self.file_path}', "
                f"arrays={available_arrays}, total_size_mb={total_size:.2f})")
