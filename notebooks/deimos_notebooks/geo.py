"""Utilities to derive per-pixel geographic coordinates from Sentinel-1 SAFE products."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio import windows
from rasterio.io import DatasetReader
from rasterio.transform import Affine, xy as transform_xy
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifestEntry:
    """Lightweight container for manifest entries of interest."""

    href: str
    polarization: Optional[str] = None
    swath: Optional[str] = None

    @property
    def base_name(self) -> str:
        """Return the filename stem, ignoring directory structure and extension."""

        return Path(self.href).stem.lower()


class Sentinel1Georeferencer:
    """Base functionality shared by Sentinel-1 georeferencers."""

    def __init__(self, safe_path: Union[str, Path]) -> None:
        self.safe_path = Path(safe_path)
        manifest_path = self.safe_path / "manifest.safe"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.safe not found under {self.safe_path}")

        self._manifest_tree = ET.parse(manifest_path)
        self._manifest = self._parse_manifest()
        self._annotation_lookup: Dict[str, ManifestEntry] = {
            entry.base_name: entry for entry in self._manifest["annotation"]
        }
        self._geolocation_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _parse_pol_and_swath(href: str) -> Tuple[Optional[str], Optional[str]]:
        lower = href.lower()

        polarization: Optional[str] = None
        for candidate in ("vv", "vh", "hh", "hv"):
            token = f"_{candidate}_"
            if token in lower or f"-{candidate}-" in lower:
                polarization = candidate.upper()
                break

        swath: Optional[str] = None
        for candidate in ("iw1", "iw2", "iw3", "ew1", "ew2", "ew3", "ew4", "ew5"):
            if candidate in lower:
                swath = candidate.upper()
                break

        return polarization, swath

    def _parse_manifest(self) -> Dict[str, List[ManifestEntry]]:
        root = self._manifest_tree.getroot()
        manifest: Dict[str, List[ManifestEntry]] = {"measurement": [], "annotation": [], "calibration": []}

        for node in root.findall(".//{*}fileLocation"):
            href = node.get("href", "").lstrip("./")
            if not href:
                continue

            pol, swath = self._parse_pol_and_swath(href)
            entry = ManifestEntry(href=href, polarization=pol, swath=swath)

            lower_href = href.lower()
            if "measurement" in lower_href:
                manifest["measurement"].append(entry)
            elif "annotation" in lower_href and "calibration" not in lower_href:
                manifest["annotation"].append(entry)
            elif "calibration" in lower_href:
                manifest["calibration"].append(entry)

        if not manifest["measurement"]:
            raise ValueError("Manifest does not describe any measurement files")

        return manifest

    # ------------------------------------------------------------------
    # Helpers operating on manifest content
    # ------------------------------------------------------------------
    def list_measurements(self) -> List[ManifestEntry]:
        return list(self._manifest["measurement"])

    def list_annotations(self) -> List[ManifestEntry]:
        return list(self._manifest["annotation"])

    def _resolve_measurement_entry(
        self, measurement: Optional[Union[str, ManifestEntry]]
    ) -> ManifestEntry:
        entries = self._manifest["measurement"]
        if not entries:
            raise ValueError("No measurement entries available in manifest")

        if isinstance(measurement, ManifestEntry):
            return measurement

        if measurement is None:
            return entries[0]

        candidate_path = Path(measurement)
        target_str = str(candidate_path).lower()
        target_name = candidate_path.name.lower()
        target_stem = candidate_path.stem.lower()

        for entry in entries:
            entry_path = Path(entry.href)
            if (
                entry.href.lower() == target_str
                or entry_path.name.lower() == target_name
                or entry_path.stem.lower() == target_stem
            ):
                return entry

        raise ValueError(f"Measurement {measurement!r} not present in manifest")

    def _match_annotation_entry(self, measurement_entry: ManifestEntry) -> ManifestEntry:
        annotation_entry = self._annotation_lookup.get(measurement_entry.base_name)
        if annotation_entry is None:
            raise ValueError(
                f"No annotation entry found for measurement {measurement_entry.href}"
            )
        return annotation_entry

    @staticmethod
    def _normalise_window(window: Optional[Window], dataset: DatasetReader) -> Window:
        if window is None:
            return Window(col_off=0, row_off=0, width=dataset.width, height=dataset.height)

        if isinstance(window, Window):
            return window.round_offsets().round_lengths()

        if isinstance(window, tuple) and len(window) == 2:
            row_slice, col_slice = window
            derived = Window.from_slices(row_slice, col_slice, height=dataset.height, width=dataset.width)
            return derived.round_offsets().round_lengths()

        raise TypeError("Window must be None, a rasterio Window, or a pair of slices")

    def _load_geolocation_grid(
        self, annotation_entry: ManifestEntry
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = annotation_entry.href
        cached = self._geolocation_cache.get(cache_key)
        if cached is not None:
            return cached

        annotation_path = self.safe_path / annotation_entry.href
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file missing: {annotation_path}")

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        lines: List[int] = []
        pixels: List[int] = []
        latitudes: List[float] = []
        longitudes: List[float] = []

        for point in root.findall(".//{*}geolocationGridPoint"):
            line_text = point.findtext("./{*}line")
            pixel_text = point.findtext("./{*}pixel")
            lat_text = point.findtext("./{*}latitude")
            lon_text = point.findtext("./{*}longitude")
            if None in (line_text, pixel_text, lat_text, lon_text):
                continue
            lines.append(int(float(line_text)))
            pixels.append(int(float(pixel_text)))
            latitudes.append(float(lat_text))
            longitudes.append(float(lon_text))

        if not lines or not pixels:
            raise ValueError(f"No geolocation grid points found in {annotation_path}")

        unique_lines = np.array(sorted(set(lines)), dtype=np.float64)
        unique_pixels = np.array(sorted(set(pixels)), dtype=np.float64)

        lat_grid = np.full((unique_lines.size, unique_pixels.size), np.nan, dtype=np.float64)
        lon_grid = np.full_like(lat_grid, np.nan)

        line_index = {int(value): idx for idx, value in enumerate(unique_lines)}
        pixel_index = {int(value): idx for idx, value in enumerate(unique_pixels)}

        for line, pixel, lat, lon in zip(lines, pixels, latitudes, longitudes):
            li = line_index[int(line)]
            pj = pixel_index[int(pixel)]
            lat_grid[li, pj] = lat
            lon_grid[li, pj] = lon

        if np.isnan(lat_grid).any() or np.isnan(lon_grid).any():
            raise ValueError(
                "Geolocation grid contains gaps; interpolation would be unreliable"
            )

        self._geolocation_cache[cache_key] = (unique_lines, unique_pixels, lat_grid, lon_grid)
        return self._geolocation_cache[cache_key]

    @staticmethod
    def _bilinear_interpolate(
        line_coords: np.ndarray,
        pixel_coords: np.ndarray,
        values: np.ndarray,
        target_rows: np.ndarray,
        target_cols: np.ndarray,
    ) -> np.ndarray:
        line_coords = np.asarray(line_coords, dtype=np.float64)
        pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
        target_rows = np.asarray(target_rows, dtype=np.float64)
        target_cols = np.asarray(target_cols, dtype=np.float64)

        if line_coords.ndim != 1 or pixel_coords.ndim != 1:
            raise ValueError("Reference coordinates must be one-dimensional")
        if values.shape != (line_coords.size, pixel_coords.size):
            raise ValueError("Geolocation grid shape does not match coordinate vectors")

        target_cols = np.clip(target_cols, pixel_coords[0], pixel_coords[-1])
        target_rows = np.clip(target_rows, line_coords[0], line_coords[-1])

        col_idx = np.searchsorted(pixel_coords, target_cols, side="right")
        col_idx = np.clip(col_idx, 1, pixel_coords.size - 1)
        x0 = pixel_coords[col_idx - 1]
        x1 = pixel_coords[col_idx]
        denom_x = np.where((x1 - x0) == 0.0, 1.0, x1 - x0)
        t = ((target_cols - x0) / denom_x)[np.newaxis, :]

        left = np.take(values, col_idx - 1, axis=1)
        right = np.take(values, col_idx, axis=1)
        interp_cols = left + (right - left) * t

        row_idx = np.searchsorted(line_coords, target_rows, side="right")
        row_idx = np.clip(row_idx, 1, line_coords.size - 1)
        y0 = line_coords[row_idx - 1]
        y1 = line_coords[row_idx]
        denom_y = np.where((y1 - y0) == 0.0, 1.0, y1 - y0)
        s = ((target_rows - y0) / denom_y)[:, np.newaxis]

        low = np.take(interp_cols, row_idx - 1, axis=0)
        high = np.take(interp_cols, row_idx, axis=0)
        return low + (high - low) * s

    def open_measurement(self, measurement: Optional[Union[str, ManifestEntry]] = None) -> DatasetReader:
        entry = self._resolve_measurement_entry(measurement)
        file_path = self.safe_path / entry.href
        if not file_path.exists():
            raise FileNotFoundError(f"Measurement raster missing: {file_path}")
        return rasterio.open(file_path)

    def compute_lat_lon(
        self,
        measurement: Optional[Union[str, ManifestEntry]] = None,
        window: Optional[Window] = None,
        dtype: Optional[np.dtype] = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute latitude/longitude arrays for a measurement window."""

        raise NotImplementedError


class GRDGeoreferencer(Sentinel1Georeferencer):
    """Georeference Sentinel-1 GRD rasters via their affine transform and CRS."""

    def compute_lat_lon(
        self,
        measurement: Optional[Union[str, ManifestEntry]] = None,
        window: Optional[Window] = None,
        dtype: Optional[np.dtype] = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        measurement_entry = self._resolve_measurement_entry(measurement)
        with self.open_measurement(measurement_entry) as dataset:
            resolved_window = self._normalise_window(window, dataset)
            height = int(round(resolved_window.height))
            width = int(round(resolved_window.width))
            if height <= 0 or width <= 0:
                raise ValueError("Requested window has zero area")

            base_row = int(round(resolved_window.row_off))
            base_col = int(round(resolved_window.col_off))

            row_indices = base_row + np.arange(height, dtype=np.float64)
            col_indices = base_col + np.arange(width, dtype=np.float64)

            dataset_transform = dataset.transform
            use_affine = dataset.crs is not None and self._has_meaningful_transform(dataset_transform)

            if use_affine:
                transform = dataset.window_transform(resolved_window)
                row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing="ij")
                x_vals, y_vals = transform_xy(transform, row_grid, col_grid, offset="center")
                x_arr = np.asarray(x_vals, dtype=np.float64)
                y_arr = np.asarray(y_vals, dtype=np.float64)
                lon_flat, lat_flat = warp_transform( # type: ignore
                    dataset.crs, "EPSG:4326", x_arr.ravel(), y_arr.ravel()
                )
                lon = np.asarray(lon_flat, dtype=np.float64).reshape((height, width))
                lat = np.asarray(lat_flat, dtype=np.float64).reshape((height, width))
            else:
                annotation_entry = self._match_annotation_entry(measurement_entry)
                line_coords, pixel_coords, lat_grid, lon_grid = self._load_geolocation_grid(annotation_entry)
                lat = self._bilinear_interpolate(line_coords, pixel_coords, lat_grid, row_indices, col_indices)
                lon = self._bilinear_interpolate(line_coords, pixel_coords, lon_grid, row_indices, col_indices)

        if dtype is not None:
            lon = lon.astype(dtype, copy=False)
            lat = lat.astype(dtype, copy=False)

        return lat, lon

    @staticmethod
    def _has_meaningful_transform(transform: Affine) -> bool:
        return not np.allclose(transform, Affine.identity())


class SLCGeoreferencer(Sentinel1Georeferencer):
    """Georeference Sentinel-1 SLC rasters using the geolocation grid from annotations."""

    def compute_lat_lon(
        self,
        measurement: Optional[Union[str, ManifestEntry]] = None,
        window: Optional[Window] = None,
        dtype: Optional[np.dtype] = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self.open_measurement(measurement) as dataset:
            resolved_window = self._normalise_window(window, dataset)
            height = int(round(resolved_window.height))
            width = int(round(resolved_window.width))
            if height <= 0 or width <= 0:
                raise ValueError("Requested window has zero area")

            base_row = int(round(resolved_window.row_off))
            base_col = int(round(resolved_window.col_off))
            target_rows = base_row + np.arange(height, dtype=np.float64)
            target_cols = base_col + np.arange(width, dtype=np.float64)

            measurement_entry = self._resolve_measurement_entry(measurement)
            annotation_entry = self._match_annotation_entry(measurement_entry)
            line_coords, pixel_coords, lat_grid, lon_grid = self._load_geolocation_grid(annotation_entry)

            lat = self._bilinear_interpolate(line_coords, pixel_coords, lat_grid, target_rows, target_cols)
            lon = self._bilinear_interpolate(line_coords, pixel_coords, lon_grid, target_rows, target_cols)

        if dtype is not None:
            lon = lon.astype(dtype, copy=False)
            lat = lat.astype(dtype, copy=False)

        return lat, lon

