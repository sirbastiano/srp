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
import pyproj
from typing import Any, Dict, List, Tuple, Optional, Union, Sequence
from dataclasses import dataclass
from pathlib import Path
import rasterio
from rasterio.transform import Affine
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform
try:
    from shapely.vectorized import contains
except ImportError:
    from shapely import contains_xy as contains

from .meta import normalize_sar_timestamp


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
            # NISAR stores EPSG as the dataset value; attrs use 'epsg_code'
            epsg = None
            projection_path = f'{science_path}/projection'
            if projection_path in f:
                proj_ds = f[projection_path]
                try:
                    epsg = int(proj_ds[()])
                except Exception:
                    for attr_name in ('epsg_code', 'epsg'):
                        if attr_name in proj_ds.attrs:
                            epsg = int(proj_ds.attrs[attr_name])
                            break
            
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
    
    def _normalize_polarizations(
        self,
        polarization: Union[str, Sequence[str]]
    ) -> List[str]:
        """Normalize polarization input into a non-empty list of strings."""
        if isinstance(polarization, str):
            pols = [polarization]
        elif isinstance(polarization, Sequence):
            pols = [p for p in polarization]
        else:
            raise TypeError('polarization must be a string or a sequence of strings')

        if not pols:
            raise ValueError('polarization list cannot be empty')

        normalized: List[str] = []
        for pol in pols:
            if not isinstance(pol, str):
                raise TypeError('all polarization values must be strings')
            p = pol.strip().upper()
            if not p:
                raise ValueError('polarization values cannot be empty strings')
            normalized.append(p)

        return normalized

    def read_data(
        self,
        polarization: Union[str, Sequence[str]],
        frequency: str = 'frequencyA',
        window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ) -> np.ndarray:
        """Read SAR data for a specific polarization.
        
        Args:
            polarization: Polarization or list/sequence of polarizations to read
            frequency: Frequency band (default: 'frequencyA')
            window: Optional window to read as ((row_start, row_stop), (col_start, col_stop))
            
        Returns:
            Complex numpy array containing SAR data.
            Returns 2D array for a single polarization and 3D array (P, H, W) for multiple.
            
        Raises:
            ValueError: If polarization not found in product
        """
        pols = self._normalize_polarizations(polarization)

        with h5py.File(self.product_path, 'r') as f:
            science_path = self._get_science_path(frequency)
            arrays: List[np.ndarray] = []

            for pol in pols:
                data_path = f'{science_path}/{pol}'
                if data_path not in f:
                    raise ValueError(f'Polarization {pol} not found at {data_path}')

                dataset = f[data_path]
                if window is not None:
                    (row_start, row_stop), (col_start, col_stop) = window
                    arr = dataset[row_start:row_stop, col_start:col_stop]
                else:
                    arr = dataset[:]
                arrays.append(arr)

            if len(arrays) == 1:
                return arrays[0]

            return np.stack(arrays, axis=0)
    
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
        polarization: Union[str, Sequence[str]],
        frequency: str = 'frequencyA',
        apply_mask: bool = True
    ) -> Dict[str, Union[np.ndarray, NISARMetadata, Affine]]:
        """Cut NISAR product by WKT polygon.
        
        Args:
            wkt_polygon: Well-Known Text polygon string defining AOI
            polarization: Polarization string or list/sequence (e.g., 'HH' or ['HH', 'HV'])
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

        # Reproject polygon from WGS84 to product's native CRS if needed
        if metadata.epsg is not None and metadata.epsg != 4326:
            transformer = pyproj.Transformer.from_crs(4326, metadata.epsg, always_xy=True)
            polygon = shapely_transform(transformer.transform, polygon)

        # Calculate pixel window
        window = self._get_pixel_window(polygon, metadata)
        (row_start, row_stop), (col_start, col_stop) = window
        
        # Normalize selected polarizations and read data for window.
        selected_pols = self.reader._normalize_polarizations(polarization)
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
            mask_to_apply = mask if data.ndim == 2 else mask[np.newaxis, ...]
            if np.iscomplexobj(data):
                masked_data[~mask_to_apply] = np.nan + 1j * np.nan
            else:
                masked_data[~mask_to_apply] = np.nan
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
            polarizations=selected_pols,
            grid_name=metadata.grid_name,
            shape=(row_stop - row_start, col_stop - col_start),
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
    
    def cut_by_bbox(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        polarization: Union[str, Sequence[str]],
        frequency: str = 'frequencyA',
        apply_mask: bool = True,
    ) -> Dict[str, Union[np.ndarray, 'NISARMetadata', Affine]]:
        """Cut NISAR product by a bounding box already in the product's native CRS.

        Unlike cut_by_wkt, no CRS reprojection is performed. The bbox coordinates
        are used directly to compute the pixel window, which avoids the artificial
        expansion that occurs when a WGS84 rectangle is reprojected to UTM and its
        axis-aligned bounds are taken (causing adjacent tiles to overlap).

        Args:
            x_min: Left edge in the product's native CRS (metres).
            y_min: Bottom edge in the product's native CRS (metres).
            x_max: Right edge in the product's native CRS (metres).
            y_max: Top edge in the product's native CRS (metres).
            polarization: Polarization string or list/sequence (e.g. 'HH' or ['HH', 'HV']).
            frequency: Frequency band (default: 'frequencyA').
            apply_mask: If True, pixels outside the bbox are set to NaN.

        Returns:
            Same dict as cut_by_wkt: data, mask, metadata, transform, window, polygon.
        """
        from shapely.geometry import box as shapely_box

        metadata = self.reader.get_metadata(frequency)
        polygon = shapely_box(x_min, y_min, x_max, y_max)

        window = self._get_pixel_window(polygon, metadata)
        (row_start, row_stop), (col_start, col_stop) = window

        selected_pols = self.reader._normalize_polarizations(polarization)
        data = self.reader.read_data(polarization, frequency, window)
        mask = self._create_mask(polygon, window, metadata)

        if apply_mask:
            if not np.iscomplexobj(data):
                data = data.astype(float)
            masked_data = data.copy()
            mask_to_apply = mask if data.ndim == 2 else mask[np.newaxis, ...]
            if np.iscomplexobj(data):
                masked_data[~mask_to_apply] = np.nan + 1j * np.nan
            else:
                masked_data[~mask_to_apply] = np.nan
            data = masked_data

        subset_transform = metadata.transform * Affine.translation(col_start, row_start)

        subset_x_min = metadata.x_min + col_start * metadata.x_spacing
        subset_y_max = metadata.y_max - row_start * abs(metadata.y_spacing)
        subset_x_max = metadata.x_min + col_stop * metadata.x_spacing
        subset_y_min = metadata.y_max - row_stop * abs(metadata.y_spacing)

        subset_metadata = NISARMetadata(
            product_path=metadata.product_path,
            frequency=frequency,
            polarizations=selected_pols,
            grid_name=metadata.grid_name,
            shape=(row_stop - row_start, col_stop - col_start),
            epsg=metadata.epsg,
            x_spacing=metadata.x_spacing,
            y_spacing=metadata.y_spacing,
            x_min=subset_x_min,
            y_min=subset_y_min,
            x_max=subset_x_max,
            y_max=subset_y_max,
        )

        return {
            'data': data,
            'mask': mask,
            'metadata': subset_metadata,
            'transform': subset_transform,
            'window': window,
            'polygon': polygon,
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
        window = subset_result.get('window')
        
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

            # Write Abstracted_Metadata-compatible group so read_h5 / extract_core_metadata_sentinel work
            abst = f.create_group('metadata/Abstracted_Metadata')
            src_path = Path(metadata.product_path)
            if src_path.exists():
                with h5py.File(src_path, 'r') as src:
                    def _str(ds_val):
                        return ds_val.decode() if isinstance(ds_val, (bytes, bytearray)) else str(ds_val)

                    ident = src.get('science/LSAR/identification')
                    if ident is not None:
                        abst.attrs['MISSION']          = _str(ident['missionId'][()])
                        abst.attrs['PRODUCT_TYPE']     = _str(ident['productType'][()])
                        abst.attrs['ACQUISITION_MODE'] = _str(ident['productType'][()])
                        abst.attrs['PRODUCT']          = _str(ident['granuleId'][()])
                        abst.attrs['PASS']             = _str(ident['orbitPassDirection'][()])
                        abst.attrs['first_line_time']  = normalize_sar_timestamp(_str(ident['zeroDopplerStartTime'][()]))
                        abst.attrs['antenna_pointing'] = _str(ident['lookDirection'][()])

                    freq_base = f'science/LSAR/GSLC/grids/{metadata.frequency}'
                    if f'{freq_base}/centerFrequency' in src:
                        abst.attrs['radar_frequency'] = float(src[f'{freq_base}/centerFrequency'][()])
                    if f'{freq_base}/slantRangeSpacing' in src:
                        abst.attrs['range_spacing'] = float(src[f'{freq_base}/slantRangeSpacing'][()])

                    # Persist requested NISAR metadata in output for downstream use.
                    self._write_requested_nisar_metadata(src, f, metadata.frequency, window)

            abst.attrs['azimuth_spacing'] = abs(metadata.y_spacing)
            if metadata.polarizations:
                abst.attrs['mds1_tx_rx_polar'] = metadata.polarizations[0]
                if len(metadata.polarizations) > 1:
                    abst.attrs['mds2_tx_rx_polar'] = metadata.polarizations[1]

    def _copy_dataset_with_attrs(
        self,
        src: h5py.File,
        dst_parent: h5py.Group,
        src_path: str,
        dst_name: Optional[str] = None,
        data_slice: Optional[Any] = None,
    ) -> Optional[h5py.Dataset]:
        """Copy a dataset and its attributes from source product into output metadata."""
        if src_path not in src:
            return None

        src_ds = src[src_path]
        value = src_ds[()] if data_slice is None else src_ds[data_slice]
        out_name = dst_name or Path(src_path).name
        out_ds = dst_parent.create_dataset(out_name, data=value)
        for attr_name, attr_val in src_ds.attrs.items():
            out_ds.attrs[attr_name] = attr_val
        return out_ds

    def _copy_dataset_if_present(
        self,
        src: h5py.File,
        dst_parent: h5py.Group,
        src_path: str,
        dst_path: str,
    ) -> None:
        """Copy one dataset to a relative destination path if it exists in source."""
        if src_path not in src:
            return

        parts = [p for p in dst_path.split('/') if p]
        group = dst_parent
        for part in parts[:-1]:
            group = group.require_group(part)
        self._copy_dataset_with_attrs(src, group, src_path, dst_name=parts[-1])

    def _write_requested_nisar_metadata(
        self,
        src: h5py.File,
        dst: h5py.File,
        frequency: str,
        window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> None:
        """Save orbit vectors, angle metadata, and burst discovery metadata."""
        req = dst.require_group('metadata/nisar_requested_metadata')

        # Orbit vectors and related metadata.
        orbit_grp = req.require_group('orbit')
        orbit_paths = [
            'science/LSAR/GSLC/metadata/orbit/orbitType',
            'science/LSAR/GSLC/metadata/orbit/interpMethod',
            'science/LSAR/GSLC/metadata/orbit/time',
            'science/LSAR/GSLC/metadata/orbit/position',
            'science/LSAR/GSLC/metadata/orbit/velocity',
            'science/LSAR/GSLC/metadata/processingInformation/inputs/orbitFiles',
            'science/LSAR/identification/absoluteOrbitNumber',
            'science/LSAR/identification/orbitPassDirection',
        ]
        for src_path in orbit_paths:
            self._copy_dataset_with_attrs(src, orbit_grp, src_path)

        # Acquisition and incidence angle metadata.
        angles_grp = req.require_group('angles')
        self._copy_dataset_with_attrs(
            src,
            angles_grp,
            'science/LSAR/GSLC/metadata/sourceData/lookDirection',
            dst_name='sourceDataLookDirection',
        )
        self._copy_dataset_with_attrs(
            src,
            angles_grp,
            'science/LSAR/identification/lookDirection',
            dst_name='identificationLookDirection',
        )

        for freq in ('frequencyA', 'frequencyB'):
            for angle_name in ('nearRangeIncidenceAngle', 'farRangeIncidenceAngle'):
                src_path = f'science/LSAR/GSLC/metadata/sourceData/swaths/{freq}/{angle_name}'
                dst_name = f'{freq}_{angle_name}'
                self._copy_dataset_with_attrs(src, angles_grp, src_path, dst_name=dst_name)

        if window is not None:
            (row_start, row_stop), (col_start, col_stop) = window
            angle_slice = (slice(None), slice(row_start, row_stop), slice(col_start, col_stop))
        else:
            angle_slice = None

        incidence_ds = self._copy_dataset_with_attrs(
            src,
            angles_grp,
            'science/LSAR/GSLC/metadata/radarGrid/incidenceAngle',
            dst_name='localIncidenceAngle',
            data_slice=angle_slice,
        )
        elevation_ds = self._copy_dataset_with_attrs(
            src,
            angles_grp,
            'science/LSAR/GSLC/metadata/radarGrid/elevationAngle',
            dst_name='elevationAngle',
            data_slice=angle_slice,
        )

        # Add quick stats for convenience.
        if incidence_ds is not None:
            inc = incidence_ds[()]
            if inc.size > 0:
                angles_grp.attrs['localIncidenceAngle_min'] = float(np.nanmin(inc))
                angles_grp.attrs['localIncidenceAngle_max'] = float(np.nanmax(inc))
                angles_grp.attrs['localIncidenceAngle_mean'] = float(np.nanmean(inc))
            else:
                angles_grp.attrs['localIncidenceAngle_nodata_reason'] = 'empty window slice'
        if elevation_ds is not None:
            ele = elevation_ds[()]
            if ele.size > 0:
                angles_grp.attrs['elevationAngle_min'] = float(np.nanmin(ele))
                angles_grp.attrs['elevationAngle_max'] = float(np.nanmax(ele))
                angles_grp.attrs['elevationAngle_mean'] = float(np.nanmean(ele))
            else:
                angles_grp.attrs['elevationAngle_nodata_reason'] = 'empty window slice'

        angles_grp.attrs['frequency_for_subset'] = frequency

        # Burst metadata: copy if present, otherwise mark explicit absence.
        burst_grp = req.require_group('burst')
        burst_paths: List[str] = []

        def _find_burst_paths(name: str, obj: Union[h5py.Group, h5py.Dataset]) -> None:
            if 'burst' in name.lower() and isinstance(obj, h5py.Dataset):
                burst_paths.append(name)

        src.visititems(_find_burst_paths)
        burst_grp.attrs['burst_metadata_found'] = bool(burst_paths)

        if burst_paths:
            for path in burst_paths:
                relative_path = path if not path.startswith('/') else path[1:]
                self._copy_dataset_if_present(src, burst_grp, path, relative_path)
        else:
            burst_grp.attrs['note'] = (
                'No dataset/group containing "burst" found in source product. '
                'This is expected for many GSLC products.'
            )
    
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
        
        # Determine data type and bands. Multi-pol arrays use (P, H, W).
        if np.iscomplexobj(data):
            dtype = rasterio.float32
            if data.ndim == 2:
                count = 2
                write_data = [
                    np.real(data).astype(np.float32),
                    np.imag(data).astype(np.float32),
                ]
            elif data.ndim == 3:
                count = 2 * data.shape[0]
                write_data = []
                for i in range(data.shape[0]):
                    write_data.append(np.real(data[i]).astype(np.float32))
                    write_data.append(np.imag(data[i]).astype(np.float32))
            else:
                raise ValueError(f'Unsupported complex data shape for raster export: {data.shape}')
        else:
            dtype = data.dtype
            if data.ndim == 2:
                count = 1
                write_data = [data]
            elif data.ndim == 3:
                count = data.shape[0]
                write_data = [data[i] for i in range(data.shape[0])]
            else:
                raise ValueError(f'Unsupported data shape for raster export: {data.shape}')

        if data.ndim == 2:
            out_height, out_width = data.shape
        elif data.ndim == 3:
            out_height, out_width = data.shape[1], data.shape[2]
        else:
            raise ValueError(f'Unsupported data shape for raster export: {data.shape}')
        
        # Create rasterio profile
        profile = {
            'driver': driver,
            'height': out_height,
            'width': out_width,
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

