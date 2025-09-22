#!/usr/bin/env python3
"""
Dark Vessel Detection Database Creator - Python Implementation

This script processes Sentinel-1 SAR data to create a database for dark vessel detection,
focusing on GRD and SLC product processing as specified.

Author: Generated from MATLAB original
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds, xy
from rasterio.warp import transform_geom, calculate_default_transform, reproject
from rasterio import windows
import rasterio.transform
import rasterio.crs
try:
    import cv2
except ImportError:
    cv2 = None
from typing import Tuple, Dict, List, Optional, Union
import logging
from dataclasses import dataclass
import json
import struct
import argparse
from scipy.ndimage import zoom

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VesselData:
    """Container for vessel validation data."""
    latitude: float
    longitude: float
    x_pixel: float
    y_pixel: float
    length_pixel: float
    top_pixel: float
    left_pixel: float
    bottom_pixel: float
    right_pixel: float
    swath: int


@dataclass
class SARProducts:
    """Container for SAR product paths."""
    grd_product: str
    slc_product: str
    base_dir: str


class DarkVesselProcessor:
    """
    Processes Sentinel-1 SAR data for dark vessel detection database creation.
    
    This class handles the processing of GRD and SLC SAR products to generate
    georeferenced tiles and metadata for machine learning training.
    """
    
    def __init__(self, base_directory: str, tile_size: int = 512):
        """
        Initialize the processor.
        
        Args:
            base_directory: Base directory containing SAR products
            tile_size: Size of output tiles (default 512x512)
        """
        self.base_dir = Path(base_directory)
        self.tile_size = tile_size
        self.output_dirs = {
            'grd': self.base_dir / 'grdfile',
            'slc': self.base_dir / 'slcfile', 
            'xml': self.base_dir / 'xmlfile'
        }
        self._create_output_directories()
    
    def _create_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_vessel_data(self, validation_file: str) -> List[VesselData]:
        """
        Load vessel validation data from CSV file.
        
        Args:
            validation_file: Path to vessel validation CSV file
            
        Returns:
            List of VesselData objects
        """
        try:
            # Assuming CSV format with appropriate columns
            df = pd.read_csv(validation_file)
            vessels = []
            
            for _, row in df.iterrows():
                vessel = VesselData(
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    x_pixel=row['x_pixel'],
                    y_pixel=row['y_pixel'],
                    length_pixel=row['length_pixel'],
                    top_pixel=row['top_pixel'],
                    left_pixel=row['left_pixel'],
                    bottom_pixel=row['bottom_pixel'],
                    right_pixel=row['right_pixel'],
                    swath=row['swath']
                )
                vessels.append(vessel)
            
            logger.info(f"Loaded {len(vessels)} vessel records")
            return vessels
            
        except Exception as e:
            logger.error(f"Error loading vessel data: {e}")
            return []
    
    def parse_safe_manifest(self, safe_path: str) -> Dict:
        """
        Parse SAFE manifest.safe file to extract product information.
        
        Args:
            safe_path: Path to SAFE directory
            
        Returns:
            Dictionary containing manifest information
        """
        manifest_path = Path(safe_path) / 'manifest.safe'
        
        try:
            tree = ET.parse(manifest_path)
            root = tree.getroot()
            
            # Define namespaces
            namespaces = {
                'xfdu': 'urn:ccsds:schema:xfdu:1',
                's1': 'http://www.esa.int/safe/sentinel-1.0'
            }
            
            # Extract key information from manifest
            manifest_info = {
                'product_type': None,
                'polarizations': [],
                'swaths': [],
                'files': {'measurement': [], 'annotation': [], 'calibration': []}
            }
            
            # Look for all fileLocation elements
            for file_location in root.findall('.//*'):
                if file_location.tag.endswith('fileLocation'):
                    href = file_location.get('href', '').lstrip('./')
                    
                    if 'measurement' in href:
                        manifest_info['files']['measurement'].append(href)
                        # Extract polarization from filename
                        href_lower = href.lower()
                        if '-vv-' in href_lower:
                            manifest_info['polarizations'].append('VV')
                        elif '-vh-' in href_lower:
                            manifest_info['polarizations'].append('VH')
                        elif '-hv-' in href_lower:
                            manifest_info['polarizations'].append('HV')
                        elif '-hh-' in href_lower:
                            manifest_info['polarizations'].append('HH')
                    elif 'annotation' in href and 'calibration' not in href:
                        manifest_info['files']['annotation'].append(href)
                    elif 'calibration' in href:
                        manifest_info['files']['calibration'].append(href)
            
            # Remove duplicates
            for key in manifest_info['files']:
                manifest_info['files'][key] = list(set(manifest_info['files'][key]))
            manifest_info['polarizations'] = list(set(manifest_info['polarizations']))
            
            logger.info(f"Parsed manifest: {len(manifest_info['files']['measurement'])} measurement files, "
                       f"{len(manifest_info['files']['calibration'])} calibration files")
            
            return manifest_info
            
        except Exception as e:
            logger.error(f"Error parsing manifest {manifest_path}: {e}")
            return {}
    
    def process_grd_data(self, grd_safe_path: str, tile_id: int, 
                        corner_coords: List[Tuple[float, float]]) -> bool:
        """
        Process GRD (Ground Range Detected) data to create georeferenced tiles.
        
        Args:
            grd_safe_path: Path to GRD SAFE directory
            tile_id: Unique identifier for the tile
            corner_coords: Corner coordinates [(lon, lat), ...]
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            manifest_info = self.parse_safe_manifest(grd_safe_path)
            safe_path = Path(grd_safe_path)
            
            # Process VV and VH polarizations
            for measurement_file in manifest_info['files']['measurement']:
                file_path = safe_path / measurement_file
                
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Determine polarization from filename
                pol = 'vv' if 'vv' in str(file_path).lower() else 'vh'
                
                with rasterio.open(file_path) as src:
                    # Get bounding box from corner coordinates
                    lons, lats = zip(*corner_coords)
                    min_lon, max_lon = min(lons), max(lons)
                    min_lat, max_lat = min(lats), max(lats)
                    
                    # Transform to pixel coordinates
                    window = windows.from_bounds(
                        min_lon, min_lat, max_lon, max_lat, src.transform
                    )
                    
                    # Read data within window
                    data = src.read(1, window=window)
                    
                    # Apply histogram adjustment (equivalent to imadjust)
                    data_adj = self._adjust_histogram(data)
                    
                    # Save as TIFF
                    output_path = self.output_dirs['grd'] / f'DB_OPENSAR_VD_{tile_id}_GRD_{pol.upper()}.tiff'
                    
                    # Create new transform for the cropped area
                    new_transform = windows.transform(window, src.transform)
                    
                    with rasterio.open(
                        output_path, 'w',
                        driver='GTiff',
                        height=data_adj.shape[0],
                        width=data_adj.shape[1],
                        count=1,
                        dtype=data_adj.dtype,
                        crs=src.crs,
                        transform=new_transform
                    ) as dst:
                        dst.write(data_adj, 1)
                    
                    logger.info(f"Saved GRD {pol.upper()} tile: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing GRD data: {e}")
            return False
    
    def process_slc_data(self, slc_safe_path: str, tile_id: int,
                        corner_coords: List[Tuple[float, float]]) -> bool:
        """
        Process SLC (Single Look Complex) data to create amplitude tiles.
        
        Args:
            slc_safe_path: Path to SLC SAFE directory
            tile_id: Unique identifier for the tile
            corner_coords: Corner coordinates [(lon, lat), ...]
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            manifest_info = self.parse_safe_manifest(slc_safe_path)
            safe_path = Path(slc_safe_path)
            
            # Process VV and VH polarizations
            for measurement_file in manifest_info['files']['measurement']:
                file_path = safe_path / measurement_file
                
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Determine polarization from filename
                pol = 'vv' if '-vv-' in str(file_path).lower() else 'vh'
                
                # Find corresponding calibration file
                cal_file = None
                for cal_file_path in manifest_info['files']['calibration']:
                    if pol in cal_file_path.lower():
                        cal_file = safe_path / cal_file_path
                        break
                
                if cal_file and cal_file.exists():
                    # Use the new complex data reading method
                    tile_data = self.read_slc_complex_data(
                        str(file_path), corner_coords, str(cal_file)
                    )
                    
                    if tile_data is not None:
                        # Apply histogram adjustment
                        amplitude_adj = self._adjust_histogram(tile_data.astype(np.uint8))
                        
                        # Save as TIFF
                        output_path = self.output_dirs['slc'] / f'DB_OPENSAR_VD_{tile_id}_SLC_{pol.upper()}.tiff'
                        
                        with rasterio.open(
                            output_path, 'w',
                            driver='GTiff',
                            height=self.tile_size,
                            width=self.tile_size,
                            count=1,
                            dtype=amplitude_adj.dtype,
                            crs='EPSG:4326'  # Default to WGS84
                        ) as dst:
                            dst.write(amplitude_adj, 1)
                        
                        logger.info(f"Saved SLC {pol.upper()} tile: {output_path}")
                else:
                    logger.warning(f"Calibration file not found for {pol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing SLC data: {e}")
            return False
    
    def _adjust_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram adjustment similar to MATLAB's imadjust.
        
        Args:
            image: Input image array
            
        Returns:
            Histogram-adjusted image
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            img_norm = image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if cv2 is not None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img_norm)
        else:
            # Fallback: simple histogram equalization using numpy
            hist, bins = np.histogram(img_norm.flatten(), 256, (0, 256))
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            return cdf[img_norm]
    
    def create_vessel_metadata(self, vessels: Optional[List[VesselData]], tile_id: int,
                             corner_coords: List[Tuple[float, float]],
                             wind_direction: float = 0.0, wind_speed: float = 0.0) -> Dict:
        """
        Create XML metadata for vessels in the tile.
        
        Args:
            vessels: List of vessels in the tile (can be None or empty)
            tile_id: Unique identifier for the tile
            corner_coords: Corner coordinates of the tile
            wind_direction: Wind direction in degrees
            wind_speed: Wind speed in m/s
            
        Returns:
            Dictionary containing metadata
        """
        # Handle None or empty vessel list
        vessel_list = vessels if vessels is not None else []
        
        metadata = {
            'ProcessingData': {
                'Corner_Coord': {
                    'Latitude': [coord[1] for coord in corner_coords],
                    'Longitude': [coord[0] for coord in corner_coords]
                },
                'WindData': {
                    'Direction': wind_direction,
                    'Speed': wind_speed
                },
                'StatisticsReport': {
                    'Number_of_ships': len(vessel_list)
                },
                'List_of_ships': {'Ship': []}
            }
        }
        
        # Add vessel information
        for i, vessel in enumerate(vessel_list):
            ship_data = {
                'Name': f'Ship_{i+1}',
                'Centroid_Position': {
                    'Latitude': vessel.latitude,
                    'Longitude': vessel.longitude,
                    'SARData_Sample': int(vessel.x_pixel),
                    'SARData_Line': int(vessel.y_pixel),
                    'Scene_Sample': int(vessel.x_pixel),  # Simplified
                    'Scene_Line': int(vessel.y_pixel)     # Simplified
                },
                'Size': vessel.length_pixel,
                'BoundingBox': {
                    'Top': int(vessel.top_pixel),
                    'Left': int(vessel.left_pixel),
                    'Bottom': int(vessel.bottom_pixel),
                    'Right': int(vessel.right_pixel)
                }
            }
            metadata['ProcessingData']['List_of_ships']['Ship'].append(ship_data)
        
        return metadata
    
    def save_metadata_xml(self, metadata: Dict, tile_id: int) -> bool:
        """
        Save metadata as XML file.
        
        Args:
            metadata: Metadata dictionary
            tile_id: Unique identifier for the tile
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = self.output_dirs['xml'] / f'DB_OPENSAR_VD_{tile_id}.xml'
            
            # Create XML tree
            root = ET.Element('outxml')
            self._dict_to_xml(metadata, root)
            
            # Write to file
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"Saved metadata: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False
    
    def _dict_to_xml(self, data: Dict, parent: ET.Element) -> None:
        """
        Recursively convert dictionary to XML elements.
        
        Args:
            data: Dictionary to convert
            parent: Parent XML element
        """
        for key, value in data.items():
            if isinstance(value, dict):
                element = ET.SubElement(parent, key)
                self._dict_to_xml(value, element)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        element = ET.SubElement(parent, key)
                        self._dict_to_xml(item, element)
                    else:
                        element = ET.SubElement(parent, key)
                        element.text = str(item)
            else:
                element = ET.SubElement(parent, key)
                element.text = str(value)
    
    def create_tiles_from_coordinates(self, image_data: np.ndarray, 
                                    corner_coords: List[Tuple[float, float]],
                                    image_transform: rasterio.transform.Affine,
                                    image_crs: rasterio.crs.CRS) -> List[Tuple[np.ndarray, List[Tuple[float, float]]]]:
        """
        Create 512x512 tiles from image data using corner coordinates.
        
        Args:
            image_data: Input image data array
            corner_coords: Corner coordinates [(lon, lat), ...]
            image_transform: Rasterio transform for the image
            image_crs: Coordinate reference system
            
        Returns:
            List of tuples containing (tile_data, tile_corner_coords)
        """
        tiles = []
        
        # Transform corner coordinates to pixel coordinates
        pixel_coords = []
        for lon, lat in corner_coords:
            col, row = xy(image_transform, lat, lon, offset='center')
            pixel_coords.append((int(col), int(row)))
        
        # Calculate bounding box in pixel coordinates
        min_col = max(0, min([coord[0] for coord in pixel_coords]))
        max_col = min(image_data.shape[1], max([coord[0] for coord in pixel_coords]))
        min_row = max(0, min([coord[1] for coord in pixel_coords]))
        max_row = min(image_data.shape[0], max([coord[1] for coord in pixel_coords]))
        
        # Extract the region of interest
        roi_width = max_col - min_col
        roi_height = max_row - min_row
        
        if roi_width <= 0 or roi_height <= 0:
            logger.warning("Invalid ROI dimensions")
            return tiles
        
        # Resize ROI to 512x512 if needed
        roi_data = image_data[min_row:max_row, min_col:max_col]
        
        if roi_data.shape != (self.tile_size, self.tile_size):
            if cv2 is not None:
                roi_resized = cv2.resize(
                    roi_data.astype(np.float32), 
                    (self.tile_size, self.tile_size)
                )
            else:
                zoom_factors = (self.tile_size / roi_data.shape[0], 
                              self.tile_size / roi_data.shape[1])
                roi_resized = zoom(roi_data.astype(np.float32), zoom_factors)
        else:
            roi_resized = roi_data
        
        # Create tile with original corner coordinates
        tiles.append((roi_resized, corner_coords))
        
        return tiles
    
    def read_slc_complex_data(self, slc_file_path: str, 
                            corner_coords: List[Tuple[float, float]],
                            calibration_file_path: str) -> Optional[np.ndarray]:
        """
        Read SLC complex data and apply calibration.
        
        Args:
            slc_file_path: Path to SLC measurement file
            corner_coords: Corner coordinates for the tile
            calibration_file_path: Path to calibration XML file
            
        Returns:
            Calibrated complex data array or None if failed
        """
        try:
            # Parse calibration data
            cal_tree = ET.parse(calibration_file_path)
            cal_root = cal_tree.getroot()
            
            # Extract calibration factor (simplified - real implementation needs proper parsing)
            cal_vector = cal_root.find('.//calibrationVector')
            if cal_vector is not None:
                dn_elem = cal_vector.find('.//dn')
                if dn_elem is not None and dn_elem.text is not None:
                    cal_factor = float(dn_elem.text.split()[0])  # Take first value
                else:
                    cal_factor = 1.0
            else:
                cal_factor = 1.0
            
            # Read complex SLC data
            with open(slc_file_path, 'rb') as f:
                # Get file info (this is simplified - real implementation needs proper TIFF parsing)
                # For now, assume we can read with rasterio
                pass
            
            # Try reading as TIFF first
            try:
                with rasterio.open(slc_file_path) as src:
                    # Read the entire image
                    if src.count >= 2:
                        # Complex data stored as separate real/imaginary bands
                        real_part = src.read(1).astype(np.float32)
                        imag_part = src.read(2).astype(np.float32)
                        complex_data = real_part + 1j * imag_part
                    else:
                        # Try to read as single band complex
                        data = src.read(1)
                        if data.dtype == np.complex64 or data.dtype == np.complex128:
                            complex_data = data
                        else:
                            # Assume it's amplitude data
                            complex_data = data.astype(np.complex64)
                    
                    # Apply calibration
                    calibrated_data = complex_data / cal_factor
                    
                    # Create tiles from the data
                    tiles = self.create_tiles_from_coordinates(
                        np.abs(calibrated_data), corner_coords, src.transform, src.crs
                    )
                    
                    if tiles:
                        return tiles[0][0]  # Return first tile data
                    
            except Exception as e:
                logger.warning(f"Could not read as TIFF: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading SLC data: {e}")
            return None
        
        return None
    
    def generate_scene_tiles(self, slc_safe_path: str, 
                           vessel_data: Optional[List[VesselData]] = None,
                           tile_overlap: float = 0.0) -> List[int]:
        """
        Generate multiple tiles from an entire SLC scene.
        
        Args:
            slc_safe_path: Path to SLC SAFE directory
            vessel_data: Optional vessel data for the scene
            tile_overlap: Overlap between tiles (0.0 to 1.0)
            
        Returns:
            List of generated tile IDs
        """
        generated_tiles = []
        
        try:
            manifest_info = self.parse_safe_manifest(slc_safe_path)
            safe_path = Path(slc_safe_path)
            
            if not manifest_info['files']['measurement']:
                logger.error("No measurement files found")
                return generated_tiles
            
            # Use first measurement file to get scene extent
            first_measurement = safe_path / manifest_info['files']['measurement'][0]
            
            with rasterio.open(first_measurement) as src:
                bounds = src.bounds
                transform = src.transform
                
                # Calculate number of tiles needed
                # This is a simplified approach - real implementation would be more sophisticated
                scene_width = bounds.right - bounds.left
                scene_height = bounds.top - bounds.bottom
                
                # Estimate tile size in geographic coordinates
                # (This is approximate and would need proper calculation)
                tile_size_geo = 0.01  # ~1km at equator
                
                tiles_x = max(1, int(scene_width / tile_size_geo))
                tiles_y = max(1, int(scene_height / tile_size_geo))
                
                tile_id_counter = 1
                
                for i in range(tiles_x):
                    for j in range(tiles_y):
                        # Calculate tile bounds
                        tile_left = bounds.left + i * tile_size_geo
                        tile_right = min(bounds.right, tile_left + tile_size_geo)
                        tile_bottom = bounds.bottom + j * tile_size_geo
                        tile_top = min(bounds.top, tile_bottom + tile_size_geo)
                        
                        # Create corner coordinates for this tile
                        corner_coords = [
                            (tile_left, tile_top),     # Top-left
                            (tile_right, tile_top),    # Top-right
                            (tile_right, tile_bottom), # Bottom-right
                            (tile_left, tile_bottom)   # Bottom-left
                        ]
                        
                        # Filter vessels for this tile if vessel data provided
                        tile_vessels = None
                        if vessel_data:
                            tile_vessels = []
                            for vessel in vessel_data:
                                if (tile_left <= vessel.longitude <= tile_right and
                                    tile_bottom <= vessel.latitude <= tile_top):
                                    tile_vessels.append(vessel)
                        
                        # Process this tile
                        success = self.process_scene(
                            "",  # Not using GRD
                            slc_safe_path,
                            tile_vessels,
                            tile_id_counter
                        )
                        
                        if success:
                            generated_tiles.append(tile_id_counter)
                            logger.info(f"Generated tile {tile_id_counter}")
                        
                        tile_id_counter += 1
            
        except Exception as e:
            logger.error(f"Error generating scene tiles: {e}")
        
        return generated_tiles
    
    def process_scene(self, grd_product: str, slc_product: str,
                     vessel_data: Optional[List[VesselData]], tile_id: int) -> bool:
        """
        Process a complete scene with SLC products only.
        
        Args:
            grd_product: Path to GRD SAFE directory (not used in current implementation)
            slc_product: Path to SLC SAFE directory
            vessel_data: List of vessels in the scene (optional, can be None)
            tile_id: Unique identifier for the tile
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Generate corner coordinates for a 512x512 tile
            # This is simplified - in practice, you'd want to tile the entire scene
            # or use specific coordinates based on vessel locations
            corner_coords = [
                (-1.0, 50.0),   # Top-left (lon, lat)
                (1.0, 50.0),    # Top-right
                (1.0, 48.0),    # Bottom-right
                (-1.0, 48.0)    # Bottom-left
            ]
            
            # Process SLC data only
            slc_success = self.process_slc_data(slc_product, tile_id, corner_coords)
            
            # Create and save metadata only if vessel_data is provided
            xml_success = True
            if vessel_data is not None:
                metadata = self.create_vessel_metadata(vessel_data, tile_id, corner_coords)
                xml_success = self.save_metadata_xml(metadata, tile_id)
            else:
                logger.info(f"No vessel data provided for tile {tile_id}, skipping metadata generation")
            
            return slc_success and xml_success
            
        except Exception as e:
            logger.error(f"Error processing scene: {e}")
            return False


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dark Vessel Detection Database Creator')
    parser.add_argument('--base-dir', required=True, help='Base directory containing SAR products')
    parser.add_argument('--vessel-data', required=True, help='Path to vessel validation CSV file')
    parser.add_argument('--grd-product', required=True, help='Path to GRD SAFE directory')
    parser.add_argument('--slc-product', required=True, help='Path to SLC SAFE directory')
    parser.add_argument('--tile-id', type=int, default=1, help='Tile ID number')
    parser.add_argument('--tile-size', type=int, default=512, help='Output tile size')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DarkVesselProcessor(args.base_dir, args.tile_size)
    
    # Load vessel data
    vessels = processor.load_vessel_data(args.vessel_data)
    
    # Process scene
    success = processor.process_scene(
        args.grd_product,
        args.slc_product,
        vessels,
        args.tile_id
    )
    
    if success:
        logger.info("Processing completed successfully")
    else:
        logger.error("Processing failed")
        exit(1)


if __name__ == '__main__':
    main()