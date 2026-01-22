"""NISAR Product Reader and Cutter Utilities.

This module provides tools for reading NISAR GSLC (Geocoded Single Look Complex)
products and subsetting them based on WKT polygon geometries.

Classes:
    NISARMetadata: Metadata container for NISAR products
    NISARReader: Reader for NISAR HDF5 products
    NISARCutter: Polygon-based product subsetting tool

Example:
    >>> from nisar_utils import NISARReader, NISARCutter
    >>> 
    >>> # Read product
    >>> reader = NISARReader('nisar_product.h5')
    >>> info = reader.info()
    >>> 
    >>> # Cut by WKT polygon
    >>> cutter = NISARCutter(reader)
    >>> wkt_polygon = 'POLYGON((x1 y1, x2 y2, x3 y3, x4 y4, x1 y1))'
    >>> subset = cutter.cut_by_wkt(wkt_polygon, 'HH')
    >>> cutter.save_subset(subset, 'output.tif')
"""

import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import rasterio
from rasterio.transform import Affine
from shapely import wkt
from shapely.geometry import Polygon
try:
    from shapely.vectorized import contains
except ImportError:
    from shapely import contains_xy as contains


@dataclass
class NISARMetadata:
    """Metadata container for NISAR GSLC products.
    
    Attributes:
        product_path: Path to the NISAR product file
        frequency: Frequency band (e.g., 'frequencyA')
        polarizations: List of available polarizations
        grid_name: Grid name
        shape: Data shape as (rows, cols)
        epsg: EPSG code for coordinate system
        x_spacing: Pixel spacing in X direction
        y_spacing: Pixel spacing in Y direction
        x_min: Minimum X coordinate
        y_min: Minimum Y coordinate
        x_max: Maximum X coordinate
        y_max: Maximum Y coordinate
    """
    
    product_path: str
    frequency: str
    polarizations: List[str]
    grid_name: str
    shape: Tuple[int, int]
    epsg: Optional[int]
    x_spacing: float
    y_spacing: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return bounds as (xmin, ymin, xmax, ymax)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)
    
    @property
    def transform(self) -> Affine:
        """Return affine transform for georeferencing."""
        return Affine.translation(self.x_min, self.y_max) * Affine.scale(
            self.x_spacing, -abs(self.y_spacing)
        )


class NISARReader:
    """Reader for NISAR GSLC (Geocoded Single Look Complex) products.
    
    This class provides methods to read NISAR HDF5 products and extract
    geocoded SAR data with proper geospatial metadata.
    
    Args:
        product_path: Path to the NISAR HDF5 product file
    
    Example:
        >>> reader = NISARReader('nisar_product.h5')
        >>> metadata = reader.get_metadata()
        >>> data = reader.read_data('HH')
    """
    
    def __init__(self, product_path: Union[str, Path]):
        """Initialize NISAR reader with product path.
        
        Args:
            product_path: Path to NISAR HDF5 file
            
        Raises:
            FileNotFoundError: If product file does not exist
        """
        self.product_path = Path(product_path)
        if not self.product_path.exists():
            raise FileNotFoundError(f'Product file not found: {self.product_path}')
        
        self._file = None
        self._metadata_cache = {}
    
    def __enter__(self):
        """Context manager entry."""
        self._file = h5py.File(self.product_path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def _get_science_path(self, frequency: str = 'frequencyA') -> str:
        """Construct path to science data.
        
        Args:
            frequency: Frequency band
            
        Returns:
            HDF5 path string
        """
        return f'science/LSAR/GSLC/grids/{frequency}'
    
    def get_available_polarizations(self, frequency: str = 'frequencyA') -> List[str]:
        """Get list of available polarizations in the product.
        
        Args:
            frequency: Frequency band (default: 'frequencyA')
            
        Returns:
            List of polarization strings (e.g., ['HH', 'HV', 'VH', 'VV'])
        """
        with h5py.File(self.product_path, 'r') as f:
            science_path = self._get_science_path(frequency)
            polarizations = []
            
            if science_path in f:
                for key in f[science_path].keys():
                    if key in ['HH', 'HV', 'VH', 'VV']:
                        polarizations.append(key)
            
            return sorted(polarizations)
    
    def get_metadata(self, frequency: str = 'frequencyA') -> NISARMetadata:
        """Extract metadata from NISAR product.
        
        Args:
            frequency: Frequency band (default: 'frequencyA')
            
        Returns:
            NISARMetadata object containing product metadata
            
        Raises:
            ValueError: If no polarizations found in product
        """
        if frequency in self._metadata_cache:
            return self._metadata_cache[frequency]
        
        with h5py.File(self.product_path, 'r') as f:
            science_path = self._get_science_path(frequency)
            
            # Get polarizations
            polarizations = self.get_available_polarizations(frequency)
            
            if not polarizations:
                raise ValueError(f'No polarizations found at {science_path}')
            
            # Read first polarization to get shape
            first_pol = polarizations[0]
            data_path = f'{science_path}/{first_pol}'
            dataset = f[data_path]
            shape = dataset.shape
            
            # Get coordinate information
            x_coords_path = f'{science_path}/xCoordinates'
            y_coords_path = f'{science_path}/yCoordinates'
            
            if x_coords_path in f and y_coords_path in f:
                x_coords = f[x_coords_path][:]
                y_coords = f[y_coords_path][:]
                
                x_min, x_max = float(x_coords.min()), float(x_coords.max())
                y_min, y_max = float(y_coords.min()), float(y_coords.max())
                
                # Calculate spacing
                x_spacing = (x_max - x_min) / (shape[1] - 1) if shape[1] > 1 else 0
                y_spacing = (y_max - y_min) / (shape[0] - 1) if shape[0] > 1 else 0
            else:
                # Fallback if coordinates not found
                x_min = y_min = 0.0
                x_max = float(shape[1])
                y_max = float(shape[0])
                x_spacing = y_spacing = 1.0
            
            # Try to get EPSG code
            epsg = None
            projection_path = f'{science_path}/projection'
            if projection_path in f:
                proj_group = f[projection_path]
                if 'epsg' in proj_group.attrs:
                    epsg = int(proj_group.attrs['epsg'])
            
            metadata = NISARMetadata(
                product_path=str(self.product_path),
                frequency=frequency,
                polarizations=polarizations,
                grid_name=frequency,
                shape=shape,
                epsg=epsg,
                x_spacing=x_spacing,
                y_spacing=y_spacing,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max
            )
            
            self._metadata_cache[frequency] = metadata
            return metadata
    
    def read_data(
        self,
        polarization: str,
        frequency: str = 'frequencyA',
        window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ) -> np.ndarray:
        """Read SAR data for a specific polarization.
        
        Args:
            polarization: Polarization to read (e.g., 'HH', 'HV', 'VH', 'VV')
            frequency: Frequency band (default: 'frequencyA')
            window: Optional window to read as ((row_start, row_stop), (col_start, col_stop))
            
        Returns:
            Complex numpy array containing SAR data
            
        Raises:
            ValueError: If polarization not found in product
        """
        with h5py.File(self.product_path, 'r') as f:
            science_path = self._get_science_path(frequency)
            data_path = f'{science_path}/{polarization}'
            
            if data_path not in f:
                raise ValueError(f'Polarization {polarization} not found at {data_path}')
            
            dataset = f[data_path]
            
            if window is not None:
                (row_start, row_stop), (col_start, col_stop) = window
                data = dataset[row_start:row_stop, col_start:col_stop]
            else:
                data = dataset[:]
            
            return data
    
    def get_coordinates(
        self,
        frequency: str = 'frequencyA'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get X and Y coordinate arrays.
        
        Args:
            frequency: Frequency band (default: 'frequencyA')
            
        Returns:
            Tuple of (x_coordinates, y_coordinates) as 1D arrays
        """
        with h5py.File(self.product_path, 'r') as f:
            science_path = self._get_science_path(frequency)
            
            x_coords = f[f'{science_path}/xCoordinates'][:]
            y_coords = f[f'{science_path}/yCoordinates'][:]
            
            return x_coords, y_coords
    
    def info(self) -> Dict:
        """Get comprehensive product information.
        
        Returns:
            Dictionary containing product information
        """
        metadata = self.get_metadata()
        
        return {
            'product_path': metadata.product_path,
            'frequency': metadata.frequency,
            'polarizations': metadata.polarizations,
            'grid': metadata.grid_name,
            'shape': metadata.shape,
            'epsg': metadata.epsg,
            'bounds': metadata.bounds,
            'x_spacing': metadata.x_spacing,
            'y_spacing': metadata.y_spacing,
        }


class NISARCutter:
    """Cut/subset NISAR products based on WKT polygons.
    
    This class provides functionality to subset NISAR GSLC products
    based on WKT polygon geometries, preserving georeferencing information.
    
    Args:
        reader: NISARReader instance
    
    Example:
        >>> reader = NISARReader('product.h5')
        >>> cutter = NISARCutter(reader)
        >>> wkt_polygon = 'POLYGON((lon1 lat1, lon2 lat2, ...))'
        >>> subset_data = cutter.cut_by_wkt(wkt_polygon, 'HH')
    """
    
    def __init__(self, reader: NISARReader):
        """Initialize cutter with a NISAR reader."""
        self.reader = reader
    
    def _wkt_to_polygon(self, wkt_string: str) -> Polygon:
        """Convert WKT string to Shapely Polygon.
        
        Args:
            wkt_string: Well-Known Text polygon string
            
        Returns:
            Shapely Polygon object
        """
        geom = wkt.loads(wkt_string)
        if not isinstance(geom, Polygon):
            raise ValueError(f'WKT geometry must be a Polygon, got {type(geom).__name__}')
        return geom
    
    def _get_pixel_window(
        self,
        polygon: Polygon,
        metadata: NISARMetadata
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate pixel window from polygon bounds.
        
        Args:
            polygon: Shapely Polygon defining the area of interest
            metadata: NISAR product metadata
            
        Returns:
            Window as ((row_start, row_stop), (col_start, col_stop))
        """
        # Get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds
        
        # Convert to pixel coordinates
        transform = metadata.transform
        inv_transform = ~transform
        
        # Transform polygon bounds to pixel space
        col_min, row_max = inv_transform * (minx, miny)
        col_max, row_min = inv_transform * (maxx, maxy)
        
        # Clip to image bounds and ensure integer coordinates
        row_start = max(0, int(np.floor(row_min)))
        row_stop = min(metadata.shape[0], int(np.ceil(row_max)))
        col_start = max(0, int(np.floor(col_min)))
        col_stop = min(metadata.shape[1], int(np.ceil(col_max)))
        
        # Ensure valid window
        if row_start >= row_stop or col_start >= col_stop:
            raise ValueError(f'Polygon does not intersect with product bounds: {metadata.bounds}')
        
        return ((row_start, row_stop), (col_start, col_stop))
    
    def _create_mask(
        self,
        polygon: Polygon,
        window: Tuple[Tuple[int, int], Tuple[int, int]],
        metadata: NISARMetadata
    ) -> np.ndarray:
        """Create binary mask for polygon within window.
        
        Args:
            polygon: Shapely Polygon defining the area of interest
            window: Pixel window as ((row_start, row_stop), (col_start, col_stop))
            metadata: NISAR product metadata
            
        Returns:
            Boolean mask array
        """
        (row_start, row_stop), (col_start, col_stop) = window
        rows = row_stop - row_start
        cols = col_stop - col_start
        
        # Create coordinate grids for the window
        transform = metadata.transform
        
        # Create mesh grid of pixel coordinates
        y_indices = np.arange(row_start, row_stop)
        x_indices = np.arange(col_start, col_stop)
        
        # Transform pixel coordinates to geographic coordinates
        xx, yy = np.meshgrid(x_indices, y_indices)
        
        # Apply affine transform
        x_coords = transform.c + xx * transform.a + yy * transform.b
        y_coords = transform.f + xx * transform.d + yy * transform.e
        
        # Create mask by checking point inclusion in polygon
        mask = np.zeros((rows, cols), dtype=bool)
        
        # Vectorized point-in-polygon test
        from shapely.vectorized import contains
        mask = contains(polygon, x_coords.ravel(), y_coords.ravel()).reshape((rows, cols))
        
        return mask
    
    def cut_by_wkt(
        self,
        wkt_polygon: str,
        polarization: str,
        frequency: str = 'frequencyA',
        apply_mask: bool = True
    ) -> Dict[str, Union[np.ndarray, NISARMetadata, Affine]]:
        """Cut NISAR product by WKT polygon.
        
        Args:
            wkt_polygon: Well-Known Text polygon string defining AOI
            polarization: Polarization to read (e.g., 'HH', 'HV', 'VH', 'VV')
            frequency: Frequency band (default: 'frequencyA')
            apply_mask: If True, apply polygon mask to data (pixels outside polygon set to NaN)
            
        Returns:
            Dictionary containing:
                - 'data': Subset SAR data array
                - 'mask': Boolean mask (True inside polygon)
                - 'metadata': Updated metadata for subset
                - 'transform': Affine transform for subset
                - 'window': Pixel window used for extraction
                - 'polygon': Original polygon geometry
        """
        # Parse WKT polygon
        polygon = self._wkt_to_polygon(wkt_polygon)
        
        # Get product metadata
        metadata = self.reader.get_metadata(frequency)
        
        # Calculate pixel window
        window = self._get_pixel_window(polygon, metadata)
        (row_start, row_stop), (col_start, col_stop) = window
        
        # Read data for window
        data = self.reader.read_data(polarization, frequency, window)
        
        # Create mask
        mask = self._create_mask(polygon, window, metadata)
        
        # Apply mask if requested
        if apply_mask:
            # Convert to float if integer type
            if not np.iscomplexobj(data):
                data = data.astype(float)
            # Create masked array or set values outside polygon to NaN
            masked_data = data.copy()
            if np.iscomplexobj(data):
                masked_data[~mask] = np.nan + 1j * np.nan
            else:
                masked_data[~mask] = np.nan
            data = masked_data
        
        # Update transform for subset
        transform = metadata.transform
        subset_transform = transform * Affine.translation(col_start, row_start)
        
        # Calculate new bounds
        subset_x_min = metadata.x_min + col_start * metadata.x_spacing
        subset_y_max = metadata.y_max - row_start * abs(metadata.y_spacing)
        subset_x_max = metadata.x_min + col_stop * metadata.x_spacing
        subset_y_min = metadata.y_max - row_stop * abs(metadata.y_spacing)
        
        # Create updated metadata for subset
        subset_metadata = NISARMetadata(
            product_path=metadata.product_path,
            frequency=frequency,
            polarizations=metadata.polarizations,
            grid_name=metadata.grid_name,
            shape=data.shape,
            epsg=metadata.epsg,
            x_spacing=metadata.x_spacing,
            y_spacing=metadata.y_spacing,
            x_min=subset_x_min,
            y_min=subset_y_min,
            x_max=subset_x_max,
            y_max=subset_y_max
        )
        
        return {
            'data': data,
            'mask': mask,
            'metadata': subset_metadata,
            'transform': subset_transform,
            'window': window,
            'polygon': polygon
        }
    
    def save_subset(
        self,
        subset_result: Dict,
        output_path: Union[str, Path],
        driver: str = 'GTiff',
        **kwargs
    ) -> None:
        """Save subset result to a georeferenced file.
        
        Args:
            subset_result: Result dictionary from cut_by_wkt()
            output_path: Output file path
            driver: Rasterio driver (default: 'GTiff') or 'HDF5'/'h5' for HDF5 format
            **kwargs: Additional arguments passed to rasterio.open() or h5py.File.create_dataset()
        """
        data = subset_result['data']
        metadata = subset_result['metadata']
        transform = subset_result['transform']
        output_path = Path(output_path)
        
        # Check if HDF5 format is requested
        if driver.upper() in ['HDF5', 'H5']:
            self._save_subset_hdf5(subset_result, output_path, **kwargs)
        else:
            self._save_subset_rasterio(subset_result, output_path, driver, **kwargs)
        
        print(f'Subset saved to: {output_path}')
    
    def _save_subset_hdf5(
        self,
        subset_result: Dict,
        output_path: Path,
        **kwargs
    ) -> None:
        """Save subset result to HDF5 file.
        
        Args:
            subset_result: Result dictionary from cut_by_wkt()
            output_path: Output file path
            **kwargs: Additional arguments passed to h5py.File.create_dataset()
        """
        data = subset_result['data']
        metadata = subset_result['metadata']
        transform = subset_result['transform']
        mask = subset_result['mask']
        
        with h5py.File(output_path, 'w') as f:
            # Create main data group
            data_group = f.create_group('data')
            
            # Save the SAR data
            ds = data_group.create_dataset('array', data=data, **kwargs)
            
            # Save mask
            mask_ds = data_group.create_dataset('mask', data=mask, compression='gzip')
            
            # Save metadata as attributes
            meta_group = f.create_group('metadata')
            meta_group.attrs['product_path'] = str(metadata.product_path)
            meta_group.attrs['frequency'] = metadata.frequency
            meta_group.attrs['grid_name'] = metadata.grid_name
            meta_group.attrs['shape'] = metadata.shape
            meta_group.attrs['epsg'] = metadata.epsg if metadata.epsg else -1
            meta_group.attrs['x_spacing'] = metadata.x_spacing
            meta_group.attrs['y_spacing'] = metadata.y_spacing
            meta_group.attrs['x_min'] = metadata.x_min
            meta_group.attrs['y_min'] = metadata.y_min
            meta_group.attrs['x_max'] = metadata.x_max
            meta_group.attrs['y_max'] = metadata.y_max
            
            # Save transform as array
            transform_arr = np.array([
                transform.a, transform.b, transform.c,
                transform.d, transform.e, transform.f
            ])
            meta_group.create_dataset('transform', data=transform_arr)
            
            # Save polarizations as string array
            if metadata.polarizations:
                dt = h5py.special_dtype(vlen=str)
                pol_ds = meta_group.create_dataset('polarizations', 
                                                   (len(metadata.polarizations),), 
                                                   dtype=dt)
                for i, pol in enumerate(metadata.polarizations):
                    pol_ds[i] = pol
    
    def _save_subset_rasterio(
        self,
        subset_result: Dict,
        output_path: Path,
        driver: str,
        **kwargs
    ) -> None:
        """Save subset result using rasterio (GeoTIFF, etc.).
        
        Args:
            subset_result: Result dictionary from cut_by_wkt()
            output_path: Output file path
            driver: Rasterio driver
            **kwargs: Additional arguments passed to rasterio.open()
        """
        data = subset_result['data']
        metadata = subset_result['metadata']
        transform = subset_result['transform']
        
        # Determine data type and bands
        if np.iscomplexobj(data):
            # Save as two-band (real, imaginary)
            count = 2
            dtype = rasterio.float32
            
            # Separate real and imaginary parts
            real_part = np.real(data).astype(np.float32)
            imag_part = np.imag(data).astype(np.float32)
            
            write_data = [real_part, imag_part]
        else:
            count = 1
            dtype = data.dtype
            write_data = [data]
        
        # Create rasterio profile
        profile = {
            'driver': driver,
            'height': data.shape[0],
            'width': data.shape[1],
            'count': count,
            'dtype': dtype,
            'crs': f'EPSG:{metadata.epsg}' if metadata.epsg else None,
            'transform': transform,
            'nodata': np.nan
        }
        profile.update(kwargs)
        
        # Write to file
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, band_data in enumerate(write_data, 1):
                dst.write(band_data, i)

