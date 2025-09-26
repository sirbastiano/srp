#!/usr/bin/env python3
"""
Dark Vessel Detection Database Creator - Python Implementation.

This module mirrors the MATLAB tiling workflow used for Sentinel-1 dark vessel
processing. It keeps the core ideas of the original pipeline—per-swath SLC
calibration, GRD co-registration, and XML reporting—while adhering to the
following restrictions from the requester:

* Wind measurements from OCN products are intentionally ignored.
* No contrast enhancement or histogram normalisation is applied before saving
  tiles; data are written as calibrated amplitude or radiometrically linear
  values.
* When no external validation catalogue is supplied the code slides a 512×512
  window across the whole SLC scene to produce exhaustive coverage.
* Every generated tile produces both a GeoTIFF (SLC and, when available, GRD)
  and an accompanying XML metadata file.

The implementation favours clarity over micro-optimisation and therefore
contains extensive inline comments to guide future maintenance.
"""

from __future__ import annotations

import argparse
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio import windows
from rasterio.io import DatasetReader
from rasterio.transform import xy
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform, transform_bounds
from scipy.ndimage import zoom
import xml.etree.ElementTree as ET

try:  # Optional dependency used for fast, high quality resizing when available
    import cv2
except ImportError:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------

@dataclass
class VesselData:
    """Container for vessel validation data in SLC pixel coordinates."""

    detect_lat: float
    detect_lon: float
    vessel_length_m: float
    source: str
    detect_scene_row: float
    detect_scene_column: float
    is_vessel: bool
    is_fishing: bool
    distance_from_shore_km: float
    scene_id: str
    confidence: float
    top: float
    left: float
    bottom: float
    right: float
    detect_id: str


@dataclass
class TileDefinition:
    """Definition of a 512×512 tile within the SLC radar grid."""

    window: windows.Window
    corner_coords: List[Tuple[float, float]]
    origin_row: int
    origin_col: int
    centre_lat: float
    centre_lon: float


# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Processor implementation
# ----------------------------------------------------------------------------


class Tiler:
    """Tiling pipeline for Sentinel-1 data."""

    def __init__(self, base_directory: str, tile_size: int = 512, tile_overlap: float = 0.0) -> None:
        """
        Initialize the processor and create output directories on demand.
        
        Args:
            base_directory: Base directory for output files
            tile_size: Size of tiles in pixels
            tile_overlap: Fractional overlap between tiles
        """

        self.base_dir = Path(base_directory)
        self.tile_size = tile_size
        self.tile_overlap = max(0.0, min(tile_overlap, 0.95))  # Prevent full overlap
        self.output_dirs = {
            "grd": self.base_dir / "grdfile",
            "slc": self.base_dir / "slcfile",
            "xml": self.base_dir / "xmlfile",
        }
        self._create_output_directories()

        # Simple caches to avoid reparsing manifest.safe multiple times.
        self._manifest_cache: Dict[str, Dict[str, List[Dict[str, Optional[str]]]]] = {}

    def _window_from_corner_coords(
        self,
        dataset: Optional[DatasetReader],
        corner_coords: List[Tuple[float, float]],
    ) -> Optional[Tuple[windows.Window, int, int]]:
        """Return dataset window and origin derived from geographic corner coords."""

        if dataset is None:
            return None

        dataset_crs = dataset.crs or CRS.from_epsg(4326)

        lon_vals, lat_vals = zip(*corner_coords)
        if dataset_crs.to_string() != "EPSG:4326":
            xs, ys = warp_transform("EPSG:4326", dataset_crs, lon_vals, lat_vals)
        else:
            xs, ys = lon_vals, lat_vals

        rows: List[int] = []
        cols: List[int] = []
        for x_coord, y_coord in zip(xs, ys):
            try:
                row, col = dataset.index(x_coord, y_coord)
            except Exception:
                continue

            if np.isnan(row) or np.isnan(col):
                continue

            rows.append(int(np.clip(row, 0, dataset.height - 1)))
            cols.append(int(np.clip(col, 0, dataset.width - 1)))

        if not rows or not cols:
            return None

        row_start = max(0, min(rows))
        row_stop = min(dataset.height, max(rows) + 1)
        col_start = max(0, min(cols))
        col_stop = min(dataset.width, max(cols) + 1)

        if row_stop <= row_start or col_stop <= col_start:
            return None

        window = windows.Window.from_slices(
            slice(row_start, row_stop),
            slice(col_start, col_stop),
        )

        return window, row_start, col_start

    @staticmethod
    def _lonlat_to_colrow(
        dataset: Optional[DatasetReader],
        lon: float,
        lat: float,
    ) -> Optional[Tuple[int, int]]:
        """Project geographic coordinates onto a dataset grid returning (col, row)."""

        if dataset is None:
            return None

        dataset_crs = dataset.crs or CRS.from_epsg(4326)

        if dataset_crs.to_string() != "EPSG:4326":
            xs, ys = warp_transform("EPSG:4326", dataset_crs, [lon], [lat])
            x_coord, y_coord = xs[0], ys[0]
        else:
            x_coord, y_coord = lon, lat

        try:
            row, col = dataset.index(x_coord, y_coord)
        except Exception:
            return None

        if np.isnan(row) or np.isnan(col):
            return None

        col_idx = int(np.clip(col, 0, dataset.width - 1))
        row_idx = int(np.clip(row, 0, dataset.height - 1))
        return col_idx, row_idx

    def _convert_bbox_grd_to_slc(
        self,
        vessel: VesselData,
        grd_dataset: Optional[DatasetReader],
        slc_dataset: Optional[DatasetReader],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Map a GRD-aligned bounding box onto the SLC grid."""

        if grd_dataset is None or slc_dataset is None:
            return None

        grd_crs = grd_dataset.crs or CRS.from_epsg(4326)
        slc_crs = slc_dataset.crs or CRS.from_epsg(4326)

        if any(
            np.isnan(value)
            for value in (vessel.top, vessel.left, vessel.bottom, vessel.right)
        ):
            return None

        grd_rows = [vessel.top, vessel.top, vessel.bottom, vessel.bottom]
        grd_cols = [vessel.left, vessel.right, vessel.left, vessel.right]

        xs: List[float] = []
        ys: List[float] = []
        for row_value, col_value in zip(grd_rows, grd_cols):
            row_clamped = float(np.clip(row_value, 0, grd_dataset.height - 1))
            col_clamped = float(np.clip(col_value, 0, grd_dataset.width - 1))
            x_coord, y_coord = grd_dataset.xy(row_clamped, col_clamped)
            xs.append(x_coord)
            ys.append(y_coord)

        if grd_crs.to_string() != "EPSG:4326":
            lon_vals, lat_vals = warp_transform(grd_crs, "EPSG:4326", xs, ys)
        else:
            lon_vals, lat_vals = xs, ys

        slc_rows: List[int] = []
        slc_cols: List[int] = []
        for lon_value, lat_value in zip(lon_vals, lat_vals):
            result = self._lonlat_to_colrow(slc_dataset, lon_value, lat_value)
            if result is None:
                return None
            slc_col, slc_row = result
            slc_rows.append(slc_row)
            slc_cols.append(slc_col)

        if not slc_rows or not slc_cols:
            return None

        top = min(slc_rows)
        bottom = max(slc_rows)
        left = min(slc_cols)
        right = max(slc_cols)

        return top, left, bottom, right
    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def _create_output_directories(self) -> None:
        """Ensure that output directories exist before writing any artefacts."""

        for directory in self.output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def load_vessel_data(self, vessel_df: Optional[pd.DataFrame]) -> List[VesselData]:
        """
        Load and validate the vessel catalogue from DataFrame if provided.
        
        Args:
            vessel_df: DataFrame containing vessel detection data
            
        Returns:
            List of VesselData objects with validated and cleaned data
        """
        if vessel_df is None or vessel_df.empty:
            logger.info('No vessel data provided; sliding across full scene.')
            return []

        required_columns = {
            'detect_lat', 'detect_lon', 'vessel_length_m', 'source', 'detect_scene_row',
            'detect_scene_column', 'is_vessel', 'is_fishing', 'distance_from_shore_km',
            'scene_id', 'confidence', 'top', 'left', 'bottom', 'right', 'detect_id'
        }

        missing = required_columns.difference(vessel_df.columns)
        if missing:
            raise ValueError(f'Vessel DataFrame missing required columns: {sorted(missing)}')

        vessels = []
        for _, row in vessel_df.iterrows():
            # Handle NaN values with safe defaults
            vessel_length = row['vessel_length_m'] if not pd.isna(row['vessel_length_m']) else 50.0
            detect_scene_row = row['detect_scene_row'] if not pd.isna(row['detect_scene_row']) else 0.0
            detect_scene_column = row['detect_scene_column'] if not pd.isna(row['detect_scene_column']) else 0.0
            
            vessel = VesselData(
                detect_lat=row['detect_lat'],
                detect_lon=row['detect_lon'],
                vessel_length_m=vessel_length,
                source=row['source'],
                detect_scene_row=detect_scene_row,
                detect_scene_column=detect_scene_column,
                is_vessel=bool(row['is_vessel']),
                is_fishing=bool(row['is_fishing']),
                distance_from_shore_km=row['distance_from_shore_km'],
                scene_id=row['scene_id'],
                confidence=row['confidence'],
                top=row['top'] if not pd.isna(row['top']) else np.nan,
                left=row['left'] if not pd.isna(row['left']) else np.nan,
                bottom=row['bottom'] if not pd.isna(row['bottom']) else np.nan,
                right=row['right'] if not pd.isna(row['right']) else np.nan,
                detect_id=row['detect_id'],
            )
            vessels.append(vessel)

        logger.info(f'Loaded {len(vessels)} vessel records from DataFrame')
        return vessels

    def _parse_pol_and_swath(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract polarization and swath identifiers from a Sentinel-1 filename."""

        lower = filename.lower()
        pol = None
        for candidate in ("vv", "vh", "hh", "hv"):
            token = f"_{candidate}_"
            if token in lower or f"-{candidate}-" in lower:
                pol = candidate.upper()
                break

        swath = None
        for candidate in ("iw1", "iw2", "iw3", "ew1", "ew2", "ew3", "ew4", "ew5"):
            if candidate in lower:
                swath = candidate.upper()
                break

        return pol, swath

    def parse_safe_manifest(self, safe_path: str) -> Dict[str, List[Dict[str, Optional[str]]]]:
        """Parse manifest.safe and return measurement/annotation/calibration entries."""

        if safe_path in self._manifest_cache:
            return self._manifest_cache[safe_path]

        manifest_path = Path(safe_path) / "manifest.safe"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.safe not found in {safe_path}")

        tree = ET.parse(manifest_path)
        root = tree.getroot()

        manifest_info: Dict[str, List[Dict[str, Optional[str]]]] = {
            "measurement": [],
            "annotation": [],
            "calibration": [],
        }

        # Iterate over every <fileLocation> entry and classify its payload.
        for node in root.findall(".//{*}fileLocation"):
            href = node.get("href", "").lstrip("./")
            pol, swath = self._parse_pol_and_swath(href)
            entry = {"href": href, "polarization": pol, "swath": swath}

            lower_href = href.lower()
            if "measurement" in lower_href:
                manifest_info["measurement"].append(entry)
            elif "annotation" in lower_href and "calibration" not in lower_href:
                manifest_info["annotation"].append(entry)
            elif "calibration" in lower_href:
                manifest_info["calibration"].append(entry)

        self._manifest_cache[safe_path] = manifest_info
        return manifest_info

    # ------------------------------------------------------------------
    # Tile generation helpers
    # ------------------------------------------------------------------
    def _build_tiles_from_vessels(self, dataset: DatasetReader, vessels: List[VesselData]) -> List[TileDefinition]:
        """
        Create one tile per vessel using georeferenced positions.
        
        This method now properly accounts for vessel positions in SLC coordinates
        and ensures tiles are centered on actual vessel locations.
        
        Args:
            dataset: Reference dataset for tile positioning
            vessels: List of vessel detections
            
        Returns:
            List of tile definitions centered on vessels
        """
        tiles: List[TileDefinition] = []
        seen_offsets: set[Tuple[int, int]] = set()
        half = self.tile_size // 2
        max_row_off = max(dataset.height - self.tile_size, 0)
        max_col_off = max(dataset.width - self.tile_size, 0)

        for vessel in vessels:
            centre_pixel = self._lonlat_to_colrow(dataset, vessel.detect_lon, vessel.detect_lat)
            if centre_pixel is None:
                logger.warning(
                    "Unable to geolocate vessel %s via lat/lon; skipping tile.",
                    vessel.detect_id,
                )
                continue

            centre_col, centre_row = centre_pixel
            
            # Calculate tile origin ensuring vessel is centered
            col_off = max(0, min(centre_col - half, max_col_off))
            row_off = max(0, min(centre_row - half, max_row_off))
            
            # Adjust if tile would extend beyond image bounds
            if col_off + self.tile_size > dataset.width:
                col_off = dataset.width - self.tile_size
            if row_off + self.tile_size > dataset.height:
                row_off = dataset.height - self.tile_size
            
            key = (row_off, col_off)
            if key in seen_offsets:
                continue  # Avoid duplicating tiles when vessels cluster together
            seen_offsets.add(key)

            window = windows.Window.from_slices(
                slice(int(row_off), int(row_off) + min(self.tile_size, dataset.height - row_off)),
                slice(int(col_off), int(col_off) + min(self.tile_size, dataset.width - col_off))
            )
            corner_coords = self._window_corner_coords(dataset, window)
            tiles.append(
                TileDefinition(
                    window=window,
                    corner_coords=corner_coords,
                    origin_row=int(row_off),
                    origin_col=int(col_off),
                    centre_lat=vessel.detect_lat,
                    centre_lon=vessel.detect_lon,
                )
            )

        return tiles

    def _build_full_scene_tiles(self, dataset: DatasetReader) -> List[TileDefinition]:
        """Slide a fixed-size window across the full scene covering every pixel."""

        tiles: List[TileDefinition] = []
        visited: set[Tuple[int, int]] = set()
        step = int(self.tile_size * (1.0 - self.tile_overlap))
        if step <= 0:
            step = self.tile_size

        row = 0
        while row < dataset.height:
            if row + self.tile_size > dataset.height:
                row = max(dataset.height - self.tile_size, 0)
            col = 0
            while col < dataset.width:
                if col + self.tile_size > dataset.width:
                    col = max(dataset.width - self.tile_size, 0)
                key = (int(row), int(col))
                if key not in visited:
                    window = windows.Window.from_slices(
                        slice(int(row), int(row) + min(self.tile_size, dataset.height)),
                        slice(int(col), int(col) + min(self.tile_size, dataset.width))
                    )
                    corner_coords = self._window_corner_coords(dataset, window)
                    tiles.append(
                        TileDefinition(
                            window=window,
                            corner_coords=corner_coords,
                            origin_row=int(row),
                            origin_col=int(col),
                        )
                    )
                    visited.add(key)
                if col + self.tile_size >= dataset.width:
                    break
                col += step
            if row + self.tile_size >= dataset.height:
                break
            row += step

        return tiles

    def _window_corner_coords(self, dataset: DatasetReader, window: windows.Window) -> List[Tuple[float, float]]:
        """Return the geographic corner coordinates (lon, lat) of a window."""

        transform = dataset.window_transform(window)
        # Calculate coordinates for the four corners (top-left first, clockwise)
        rows = [0, 0, int(window.height) - 1, int(window.height) - 1]
        cols = [0, int(window.width) - 1, int(window.width) - 1, 0]
        coords = [xy(transform, row, col, offset="center") for row, col in zip(rows, cols)]

        dataset_crs = dataset.crs or CRS.from_epsg(4326)
        if dataset_crs.to_string() != "EPSG:4326":
            xs, ys = zip(*coords)
            transform_result = warp_transform(dataset_crs, "EPSG:4326", xs, ys)
            lon, lat = transform_result[0], transform_result[1]
            coords = list(zip(lon, lat))

        return coords

    def _window_from_center(
        self,
        dataset: Optional[DatasetReader],
        centre_lon: float,
        centre_lat: float,
    ) -> Optional[Tuple[windows.Window, int, int]]:
        """Derive a tile window from a lat/lon centre for the given dataset."""

        if dataset is None:
            return None

        centre = self._lonlat_to_colrow(dataset, centre_lon, centre_lat)
        if centre is None:
            return None

        centre_col, centre_row = centre
        half = self.tile_size // 2

        row_start = max(0, centre_row - half)
        col_start = max(0, centre_col - half)

        if row_start + self.tile_size > dataset.height:
            row_start = max(0, dataset.height - self.tile_size)
        if col_start + self.tile_size > dataset.width:
            col_start = max(0, dataset.width - self.tile_size)

        row_end = min(dataset.height, row_start + self.tile_size)
        col_end = min(dataset.width, col_start + self.tile_size)

        if row_end <= row_start or col_end <= col_start:
            return None

        window = windows.Window.from_slices(
            slice(row_start, row_end),
            slice(col_start, col_end),
        )

        return window, row_start, col_start

    def _vessels_in_window(
        self,
        dataset: Optional[DatasetReader],
        vessels: Optional[List[VesselData]],
        tile: TileDefinition,
    ) -> List[VesselData]:
        """
        Filter vessel detections that fall inside the given tile window.
        
        This method now properly handles coordinate system conversions and
        ensures vessels are correctly associated with tiles.
        """
        if not vessels:
            return []

        col_start = tile.origin_col
        col_end = col_start + int(tile.window.width)
        row_start = tile.origin_row
        row_end = row_start + int(tile.window.height)

        selected: List[VesselData] = []
        for vessel in vessels:
            if dataset is None:
                continue

            col_row = self._lonlat_to_colrow(dataset, vessel.detect_lon, vessel.detect_lat)
            if col_row is None:
                logger.debug("Skipping vessel %s due to missing geographic match", vessel.detect_id)
                continue

            vessel_col, vessel_row = col_row

            if col_start <= vessel_col < col_end and row_start <= vessel_row < row_end:
                selected.append(vessel)

        return selected

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------
    def _match_calibration(self, measurement: Dict[str, Optional[str]], manifest: Dict[str, List[Dict[str, Optional[str]]]]) -> Optional[str]:
        """Return the calibration file href best matching a measurement entry."""

        for candidate in manifest["calibration"]:
            if measurement["polarization"] and candidate["polarization"] != measurement["polarization"]:
                continue
            if measurement["swath"] and candidate["swath"] == measurement["swath"]:
                return candidate["href"]

        # Fall back to the first calibration file sharing the same polarisation.
        for candidate in manifest["calibration"]:
            if measurement["polarization"] and candidate["polarization"] == measurement["polarization"]:
                return candidate["href"]

        return manifest["calibration"][0]["href"] if manifest["calibration"] else None

    def _read_calibration_factor(self, calibration_path: Path) -> float:
        """Compute a representative calibration factor from an XML calibration file."""

        try:
            tree = ET.parse(calibration_path)
            root = tree.getroot()
            values: List[float] = []
            for dn_node in root.findall(".//{*}dn"):
                if dn_node.text:
                    for token in dn_node.text.split():
                        try:
                            values.append(float(token))
                        except ValueError:
                            continue
            if values:
                return float(np.mean(values))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to parse calibration at %s: %s", calibration_path, exc)
        return 1.0

    # ------------------------------------------------------------------
    # Tile IO helpers
    # ------------------------------------------------------------------
    def _prepare_tile_array(self, data: np.ndarray) -> np.ndarray:
        """Resize or pad a raster slice so it becomes tile_size×tile_size."""

        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(0)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if data.shape == (self.tile_size, self.tile_size):
            return data.astype(np.float32, copy=False)

        if cv2 is not None:
            resized = cv2.resize(
                data.astype(np.float32),
                (self.tile_size, self.tile_size),
                interpolation=cv2.INTER_LINEAR,
            )
            return resized

        zoom_factors = (
            self.tile_size / max(data.shape[0], 1),
            self.tile_size / max(data.shape[1], 1),
        )
        return zoom(data.astype(np.float32), zoom_factors, order=1)

    def _write_tiff(self, output_path: Path, data: np.ndarray, transform, crs) -> None:
        """Persist a single-band GeoTIFF tile."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data.astype(np.float32), 1)

    def _read_slc_window(
        self,
        dataset: DatasetReader,
        window: windows.Window,
        calibration_factor: float,
    ) -> Optional[np.ndarray]:
        """
        Read, calibrate, and amplitude-compute a SLC tile.
        
        This method now properly handles SLC complex data calibration
        to match the MATLAB sigma0 calibration approach.
        """
        try:
            array = dataset.read(window=window, boundless=False)
        except ValueError as exc:
            logger.warning(f'Failed reading SLC window: {exc}')
            return None

        if array.ndim == 3:
            if array.shape[0] == 2:
                # Handle I/Q data stored in separate bands
                real_part = array[0].astype(np.float32)
                imag_part = array[1].astype(np.float32)
                complex_data = real_part + 1j * imag_part
            else:
                # Handle single-band complex data
                if np.iscomplexobj(array[0]):
                    complex_data = array[0].astype(np.complex64)
                else:
                    # Assume real data represents amplitude
                    complex_data = array[0].astype(np.complex64)
        else:
            # Handle 2D complex data
            if np.iscomplexobj(array):
                complex_data = array.astype(np.complex64)
            else:
                complex_data = array.astype(np.complex64)

        # Apply calibration to convert to sigma0 (similar to MATLAB approach)
        # MATLAB: imcal = sqrt(imgSCLabs/calfactor^2)
        if calibration_factor > 0:
            amplitude_squared = np.abs(complex_data) ** 2
            calibrated_amplitude = np.sqrt(amplitude_squared / (calibration_factor ** 2))
        else:
            calibrated_amplitude = np.abs(complex_data)

        return self._prepare_tile_array(calibrated_amplitude)

    def _write_slc_tiles(
        self,
        tile_id: int,
        tile: TileDefinition,
        measurements: List[Dict[str, Optional[str]]],
        datasets: Dict[str, DatasetReader],
        calibration_factors: Dict[str, float],
    ) -> Tuple[bool, Dict[str, Tuple[int, int]]]:
        """Write SLC tiles for every available polarisation."""

        if not datasets:
            logger.warning("No SLC datasets opened; skipping SLC export.")
            return False, {}

        success = False
        origins: Dict[str, Tuple[int, int]] = {}
        for entry in measurements:
            href = entry["href"]
            if href is None:
                continue
            dataset = datasets.get(href)
            if dataset is None:
                continue

            window_info = self._window_from_center(dataset, tile.centre_lon, tile.centre_lat)
            if window_info is None:
                logger.warning(
                    "Unable to derive SLC window for tile %s using lat/lon; skipping entry %s",
                    tile_id,
                    href,
                )
                continue

            window, row_start, col_start = window_info

            amplitude = self._read_slc_window(
                dataset,
                window,
                calibration_factors.get(href, 1.0),
            )
            if amplitude is None:
                continue

            pol_label = entry["polarization"] or "UNKNOWN"
            output_path = self.output_dirs["slc"] / f"DB_OPENSAR_VD_{tile_id}_SLC_{pol_label}.tiff"
            transform = dataset.window_transform(window)
            self._write_tiff(output_path, amplitude, transform, dataset.crs)
            origins[href] = (row_start, col_start)
            success = True

        return success, origins

    def _write_grd_tiles(
        self,
        tile_id: int,
        tile: TileDefinition,
        measurements: List[Dict[str, Optional[str]]],
        datasets: Dict[str, DatasetReader],
    ) -> bool:
        """
        Write GRD tiles co-registered to the SLC tile footprint.
        
        This method now properly handles GRD data without contrast enhancement
        as specified in the requirements.
        """
        if not datasets:
            logger.debug('No GRD datasets opened; skipping GRD export.')
            return True  # Not an error condition when GRD is optional

        success = False
        for entry in measurements:
            href = entry['href']
            if href is None:
                continue
            dataset = datasets.get(href)
            if dataset is None:
                continue

            window_info = self._window_from_center(dataset, tile.centre_lon, tile.centre_lat)
            if window_info is None:
                logger.warning(
                    'Unable to derive GRD window for tile %s at (%.6f, %.6f)',
                    tile_id,
                    tile.centre_lat,
                    tile.centre_lon,
                )
                continue

            window, _, _ = window_info

            try:
                data = dataset.read(1, window=window, boundless=False)
                if data is None or data.size == 0:
                    logger.warning(f'No data read for tile {tile_id}')
                    continue
            except Exception as exc:
                logger.warning(f'Failed to read data for tile {tile_id}: {exc}')
                continue

            # Convert to linear scale if data appears to be in dB
            if np.median(data[data > 0]) < 1.0:
                # Likely dB data, convert to linear
                data_linear = 10 ** (data / 10.0)
                prepared = self._prepare_tile_array(data_linear)
            else:
                prepared = self._prepare_tile_array(data)

            pol_label = entry['polarization'] or 'UNKNOWN'
            output_path = self.output_dirs['grd'] / f'DB_OPENSAR_VD_{tile_id}_GRD_{pol_label}.tiff'
            window_transform = dataset.window_transform(window)
            self._write_tiff(output_path, prepared, window_transform, dataset.crs)
            success = True

        return success

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def create_vessel_metadata(
        self,
        tile_id: int,
        tile: TileDefinition,
        vessels: List[VesselData],
        reference_dataset: Optional[DatasetReader],
    ) -> Dict:
        """
        Build the XML metadata structure equivalent to MATLAB's outxml.
        
        Args:
            tile_id: Unique identifier for the tile
            tile: Tile definition containing window and coordinates
            vessels: List of vessel detections within the tile
            
        Returns:
            Dictionary containing metadata structure for XML serialization
        """
        metadata = {
            'ProcessingData': {
                'Tile_ID': tile_id,
                'Corner_Coord': {
                    'Latitude': [coord[1] for coord in tile.corner_coords],
                    'Longitude': [coord[0] for coord in tile.corner_coords],
                },
                'StatisticsReport': {
                    'Number_of_ships': len(vessels),
                },
                'List_of_ships': {'Ship': []},
            }
        }

        for idx, vessel in enumerate(vessels, start=1):
            sample_line = self._lonlat_to_colrow(reference_dataset, vessel.detect_lon, vessel.detect_lat) if reference_dataset else None
            if sample_line is None:
                logger.debug(
                    "Reference dataset missing geolocation for vessel %s; falling back to stored pixel coords.",
                    vessel.detect_id,
                )
                sample_col = int(round(vessel.detect_scene_column))
                sample_row = int(round(vessel.detect_scene_row))
            else:
                sample_col, sample_row = sample_line

            scene_sample = sample_col - tile.origin_col + 1
            scene_line = sample_row - tile.origin_row + 1

            scene_sample = max(1, min(self.tile_size, scene_sample))
            scene_line = max(1, min(self.tile_size, scene_line))

            # Handle NaN vessel length with safe default
            vessel_length = vessel.vessel_length_m if not np.isnan(vessel.vessel_length_m) else 50.0
            
            # Calculate bounding box - use vessel-specific bbox if available
            if not (np.isnan(vessel.top) or np.isnan(vessel.left) or 
                   np.isnan(vessel.bottom) or np.isnan(vessel.right)):
                # Convert GRD-based bounding box to SLC tile coordinates
                top = int(round(vessel.top)) - tile.origin_row + 1
                left = int(round(vessel.left)) - tile.origin_col + 1
                bottom = int(round(vessel.bottom)) - tile.origin_row + 1
                right = int(round(vessel.right)) - tile.origin_col + 1
            else:
                # Create default bounding box based on vessel length
                vessel_length_pixels = max(10, int(vessel_length / 10))  # Rough m to pixel conversion
                bbox_size = max(vessel_length_pixels, 10)
                
                top = max(1, scene_line - bbox_size // 2)
                left = max(1, scene_sample - bbox_size // 2)  
                bottom = min(self.tile_size, scene_line + bbox_size // 2)
                right = min(self.tile_size, scene_sample + bbox_size // 2)

            # Ensure bounding box is within tile bounds
            top = max(1, min(self.tile_size, top))
            left = max(1, min(self.tile_size, left))
            bottom = max(1, min(self.tile_size, bottom))
            right = max(1, min(self.tile_size, right))

            ship_entry = {
                'Name': f'Ship_{idx}',
                'Centroid_Position': {
                    'Latitude': vessel.detect_lat,
                    'Longitude': vessel.detect_lon,
                    'SARData_Sample': int(round(vessel.detect_scene_column)),
                    'SARData_Line': int(round(vessel.detect_scene_row)),
                    'Scene_Sample': scene_sample,
                    'Scene_Line': scene_line,
                },
                'Size': vessel_length,
                'BoundingBox': {
                    'Top': top,
                    'Left': left,
                    'Bottom': bottom,
                    'Right': right,
                },
            }
            metadata['ProcessingData']['List_of_ships']['Ship'].append(ship_entry)

        return metadata

    def save_metadata_xml(self, metadata: Dict, tile_id: int) -> bool:
        """Persist metadata dictionary to an XML file."""

        try:
            output_path = self.output_dirs["xml"] / f"DB_OPENSAR_VD_{tile_id}.xml"
            root = ET.Element("outxml")
            self._dict_to_xml(metadata, root)
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            return True
        except Exception as exc:
            logger.error("Failed to write XML for tile %d: %s", tile_id, exc)
            return False

    def _dict_to_xml(self, data: Dict, parent: ET.Element) -> None:
        """Recursively serialise a Python dictionary into XML elements."""

        for key, value in data.items():
            if isinstance(value, dict):
                element = ET.SubElement(parent, key)
                self._dict_to_xml(value, element)
            elif isinstance(value, list):
                for item in value:
                    element = ET.SubElement(parent, key)
                    if isinstance(item, dict):
                        self._dict_to_xml(item, element)
                    else:
                        element.text = str(item)
            else:
                element = ET.SubElement(parent, key)
                element.text = str(value)

    # ------------------------------------------------------------------
    # High level orchestration
    # ------------------------------------------------------------------
    def process_scene(
        self,
        grd_product: Optional[str],
        slc_product: str,
        vessel_data: Optional[List[VesselData]] = None,
    ) -> bool:
        """
        Process one Sentinel-1 scene and export all requested artefacts.
        
        This method now properly handles the coordinate system transformations
        and ensures vessels are correctly positioned within tiles.
        """
        slc_manifest = self.parse_safe_manifest(slc_product)
        slc_measurements = slc_manifest['measurement']
        if not slc_measurements:
            logger.error(f'No SLC measurement files discovered in {slc_product}')
            return False

        reference_href = slc_measurements[0]['href']
        if reference_href is None:
            logger.error('Reference SLC measurement href is None')
            return False
        reference_path = Path(slc_product) / reference_href
        if not reference_path.exists():
            logger.error(f'Reference SLC file missing: {reference_path}')
            return False

        tiles: List[TileDefinition] = []
        overall_success = True
        with ExitStack() as stack:
            slc_datasets: Dict[str, rasterio.DatasetReader] = {}
            slc_calibration: Dict[str, float] = {}
            
            # Open all SLC datasets and load calibration factors
            for entry in slc_measurements:
                href = entry['href']
                if href is None:
                    continue
                measurement_path = Path(slc_product) / href
                if not measurement_path.exists():
                    logger.warning(f'Missing SLC measurement: {measurement_path}')
                    continue
                slc_datasets[href] = stack.enter_context(rasterio.open(measurement_path))
                cal_href = self._match_calibration(entry, slc_manifest)
                cal_factor = 1.0
                if cal_href:
                    cal_path = Path(slc_product) / cal_href
                    if cal_path.exists():
                        cal_factor = self._read_calibration_factor(cal_path)
                slc_calibration[href] = cal_factor

            grd_measurements: List[Dict[str, Optional[str]]] = []
            grd_datasets: Dict[str, rasterio.DatasetReader] = {}
            if grd_product:
                grd_manifest = self.parse_safe_manifest(grd_product)
                grd_measurements = grd_manifest["measurement"]
                for entry in grd_measurements:
                    href = entry["href"]
                    if href is None:
                        continue
                    grd_path = Path(grd_product) / href
                    if grd_path.exists():
                        grd_datasets[href] = stack.enter_context(rasterio.open(grd_path))
                    else:
                        logger.warning("Missing GRD measurement: %s", grd_path)

            # Prefer GRD dataset for tile referencing due to reliable georeferencing
            tile_reference_dataset = next(iter(grd_datasets.values()), None)
            if tile_reference_dataset is None:
                tile_reference_dataset = slc_datasets.get(reference_href) or next(iter(slc_datasets.values()), None)
            
            if tile_reference_dataset is not None:
                if vessel_data:
                    tiles = self._build_tiles_from_vessels(tile_reference_dataset, vessel_data)
                else:
                    tiles = self._build_full_scene_tiles(tile_reference_dataset)
            else:
                logger.error('No reference dataset available for tile generation')
                return False

            if not tiles:
                logger.warning('No tiles generated for the scene; nothing to export.')
                return False

            # Process each tile
            for idx, tile in enumerate(tiles, start=1):
                tile_vessels = self._vessels_in_window(tile_reference_dataset, vessel_data, tile)
                
                slc_written, _ = self._write_slc_tiles(
                    idx,
                    tile,
                    slc_measurements,
                    slc_datasets,
                    slc_calibration,
                )
                grd_written = self._write_grd_tiles(idx, tile, grd_measurements, grd_datasets)
                metadata = self.create_vessel_metadata(idx, tile, tile_vessels, tile_reference_dataset)
                xml_written = self.save_metadata_xml(metadata, idx)
                
                if not (slc_written and grd_written and xml_written):
                    logger.warning(f'Failed to write some outputs for tile {idx}')
                    overall_success = False

        return overall_success


# ----------------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------------


def main() -> None:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description="Dark Vessel Detection Database Creator")
    parser.add_argument("--base-dir", required=True, help="Base output directory for generated artefacts")
    parser.add_argument("--slc-product", required=True, help="Path to Sentinel-1 SLC SAFE directory")
    parser.add_argument("--grd-product", default=None, help="Path to Sentinel-1 GRD SAFE directory (optional)")
    parser.add_argument("--vessel-data", default=None, help="CSV file containing vessel annotations (optional)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels (default: 512)")
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.0,
        help="Fractional overlap between neighbouring tiles when sliding (default: 0.0)",
    )

    args = parser.parse_args()

    processor = Tiler(
        base_directory=args.base_dir,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    # Load DataFrame from CSV if vessel-data path is provided
    vessel_df = None
    if args.vessel_data:
        try:
            vessel_df = pd.read_csv(args.vessel_data)
        except Exception as exc:
            logger.warning("Failed to load vessel data from %s: %s", args.vessel_data, exc)

    vessels = processor.load_vessel_data(vessel_df)
    success = processor.process_scene(args.grd_product, args.slc_product, vessels if vessels else None)

    if success:
        logger.info("Processing completed successfully.")
    else:
        logger.error("Processing completed with errors.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
