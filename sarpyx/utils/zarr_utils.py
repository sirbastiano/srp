import numpy as np 
import zarr
import os
from typing import Optional, Tuple, Union, Dict, Any, List, Callable
import numcodecs # Ensure numcodecs is installed for compression
import pandas as pd
import gc


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

def gc_collect(func: Callable) -> Callable:
    """
    Decorator to perform garbage collection after function execution.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function with garbage collection.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper









# -----------  Classes -----------------
# TODO: Add method for handling metadata relative to single chunks.
class ZarrManager:
    """
    A comprehensive class for managing Zarr files, providing convenient methods for data access, slicing, metadata extraction, analysis, visualization, and export.
    Attributes:
        file_path (str): Path to the Zarr file or directory.
        _zarr_array (zarr.Array or None): Cached Zarr array object.
        _metadata (pandas.DataFrame or None): Cached metadata extracted from Zarr attributes.
        _ephemeris (pandas.DataFrame or None): Cached ephemeris data extracted from Zarr attributes.
    Methods:
        __init__(file_path: str) -> None
            Initialize the ZarrManager with the specified file path.
        load() -> zarr.Array
            Load and cache the Zarr array from the file path.
        info -> Dict[str, Any]
            Property returning basic information about the Zarr array (shape, dtype, chunks, nbytes, size in MB).
        get_slice(rows: Optional[Union[Tuple[int, int], slice]] = None, 
                  step: Optional[int] = None) -> np.ndarray
            Retrieve a slice of the Zarr array based on row and column specifications.
        get_metadata() -> Optional[pandas.DataFrame]
            Extract metadata from Zarr attributes as a pandas DataFrame, if available.
        get_ephemeris() -> Optional[pandas.DataFrame]
            Extract ephemeris data from Zarr attributes as a pandas DataFrame, if available.
        stats(sample_size: int = 1000) -> Dict[str, Optional[float]]
            Compute statistical summaries (mean, std, min, max, etc.) of the data, optionally using a sample for large arrays.
        visualize_slice(rows: Tuple[int, int] = (0, 100), 
                        plot_type: str = 'magnitude') -> None
            Visualize a slice of the data using matplotlib, supporting magnitude, phase, real, or imaginary plots.
        get_compression_info() -> Dict[str, float]
            Retrieve compression statistics, including uncompressed and compressed sizes, compression ratio, and space saved.
        export_slice(output_path: str, 
                     format: str = 'npy') -> None
            Export a slice of the data to various formats ('npy', 'csv', 'hdf5').
        find_peaks(rows: Optional[Union[Tuple[int, int], slice]] = None, 
                   threshold_percentile: float = 95) -> Dict[str, Any]
            Find peaks in the data above a specified percentile threshold.
        memory_efficient_operation(operation: Callable[[np.ndarray], Any], 
                                  chunk_size: int = 1000) -> List[Any]
            Apply a user-defined operation to the data in memory-efficient chunks.
        _export_raw()
            Export raw metadata, ephemeris, and echo data as a dictionary.
        __repr__() -> str
            Return a string representation of the ZarrManager instance, including file path and array info.
    """
    
    def __init__(self, file_path: str) -> None:
        """
        Initialize with zarr file path.
        
        Args:
            file_path: Path to the zarr file or directory
        """
        self.file_path = file_path
        self._zarr_array = None
        self._metadata = None
        self._ephemeris = None
        
    def load(self) -> zarr.Array:
        """
        Load the zarr array and cache it.
        
        Returns:
            zarr.Array: The loaded zarr array
        """
        if self._zarr_array is None:
            self._zarr_array = zarr.open(self.file_path, mode='r')
        return self._zarr_array
    
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
    
    def get_slice(self, rows: Optional[Union[Tuple[int, int], slice]] = None, 
                  cols: Optional[Union[Tuple[int, int], slice]] = None, 
                  step: Optional[int] = None) -> np.ndarray:
        """
        Get a slice of the zarr array.
        
        Args:
            rows: Tuple (start, end) or slice object for rows
            cols: Tuple (start, end) or slice object for columns  
            step: Step size for slicing
            
        Returns:
            numpy array with the requested slice
        """
        arr = self.load()
        
        # Convert tuples to slices
        if isinstance(rows, tuple):
            rows = slice(rows[0], rows[1], step)
        if isinstance(cols, tuple):
            cols = slice(cols[0], cols[1], step)
            
        # Default to full array if no slicing specified
        if rows is None and cols is None:
            return arr[:]
        elif rows is None:
            return arr[:, cols]
        elif cols is None:
            return arr[rows, :]
        else:
            return arr[rows, cols]
    
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
        }
    
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