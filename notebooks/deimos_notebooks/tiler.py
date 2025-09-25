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
from rasterio.transform import xy
from rasterio.warp import transform, transform_bounds
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
class TileDefinition:
    """Definition of a 512×512 tile within the SLC radar grid."""

    window: windows.Window
    corner_coords: List[Tuple[float, float]]
    origin_row: int
    origin_col: int


# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Processor implementation
# ----------------------------------------------------------------------------


class DarkVesselProcessor:
    """Replicates the MATLAB tiling pipeline for Sentinel-1 dark vessel data."""

    def __init__(self, base_directory: str, tile_size: int = 512, tile_overlap: float = 0.0) -> None:
        """Initialise the processor and create output directories on demand."""

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

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def _create_output_directories(self) -> None:
        """Ensure that output directories exist before writing any artefacts."""

        for directory in self.output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def load_vessel_data(self, validation_file: Optional[str]) -> List[VesselData]:
        """Load and validate the vessel catalogue (CSV) if provided."""

        if not validation_file:
            logger.info("No validation file provided; sliding across full scene.")
            return []

        path = Path(validation_file)
        if not path.exists():
            logger.warning("Validation file %s not found; proceeding without annotations.", path)
            return []

        required_columns = {
            "latitude",
            "longitude",
            "x_pixel",
            "y_pixel",
            "length_pixel",
            "top_pixel",
            "left_pixel",
            "bottom_pixel",
            "right_pixel",
            "swath",
        }

        df = pd.read_csv(path)
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Validation file missing required columns: {sorted(missing)}")

        vessels = [
            VesselData(
                latitude=row["latitude"],
                longitude=row["longitude"],
                x_pixel=row["x_pixel"],
                y_pixel=row["y_pixel"],
                length_pixel=row["length_pixel"],
                top_pixel=row["top_pixel"],
                left_pixel=row["left_pixel"],
                bottom_pixel=row["bottom_pixel"],
                right_pixel=row["right_pixel"],
                swath=int(row["swath"]),
            )
            for _, row in df.iterrows()
        ]

        logger.info("Loaded %d vessel records from %s", len(vessels), path)
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
    def _build_tiles_from_vessels(self, dataset: rasterio.io.DatasetReader, vessels: List[VesselData]) -> List[TileDefinition]:
        """Create one tile per vessel, centred on its SLC pixel coordinates."""

        tiles: List[TileDefinition] = []
        seen_offsets: set[Tuple[int, int]] = set()
        half = self.tile_size // 2
        max_row_off = max(dataset.height - self.tile_size, 0)
        max_col_off = max(dataset.width - self.tile_size, 0)

        for vessel in vessels:
            centre_col = int(round(vessel.x_pixel))
            centre_row = int(round(vessel.y_pixel))
            col_off = max(0, min(centre_col - half, max_col_off))
            row_off = max(0, min(centre_row - half, max_row_off))
            key = (row_off, col_off)
            if key in seen_offsets:
                continue  # Avoid duplicating tiles when vessels cluster together
            seen_offsets.add(key)

            window = windows.Window(
                col_off=float(col_off),
                row_off=float(row_off),
                width=min(self.tile_size, dataset.width),
                height=min(self.tile_size, dataset.height),
            )
            corner_coords = self._window_corner_coords(dataset, window)
            tiles.append(
                TileDefinition(
                    window=window,
                    corner_coords=corner_coords,
                    origin_row=int(row_off),
                    origin_col=int(col_off),
                )
            )

        return tiles

    def _build_full_scene_tiles(self, dataset: rasterio.io.DatasetReader) -> List[TileDefinition]:
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
                    window = windows.Window(
                        col_off=float(col),
                        row_off=float(row),
                        width=min(self.tile_size, dataset.width),
                        height=min(self.tile_size, dataset.height),
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

    def _window_corner_coords(self, dataset: rasterio.io.DatasetReader, window: windows.Window) -> List[Tuple[float, float]]:
        """Return the geographic corner coordinates (lon, lat) of a window."""

        transform = dataset.window_transform(window)
        # Calculate coordinates for the four corners (top-left first, clockwise)
        rows = [0, 0, int(window.height) - 1, int(window.height) - 1]
        cols = [0, int(window.width) - 1, int(window.width) - 1, 0]
        coords = [xy(transform, row, col, offset="center") for row, col in zip(rows, cols)]

        if dataset.crs and dataset.crs.to_string() != "EPSG:4326":
            xs, ys = zip(*coords)
            lon, lat = transform(dataset.crs, "EPSG:4326", xs, ys)
            coords = list(zip(lon, lat))

        return coords

    def _vessels_in_window(self, vessels: Optional[List[VesselData]], tile: TileDefinition) -> List[VesselData]:
        """Filter vessel detections that fall inside the given tile window."""

        if not vessels:
            return []

        col_start = tile.origin_col
        col_end = col_start + int(tile.window.width)
        row_start = tile.origin_row
        row_end = row_start + int(tile.window.height)

        selected: List[VesselData] = []
        for vessel in vessels:
            if col_start <= vessel.x_pixel < col_end and row_start <= vessel.y_pixel < row_end:
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
        dataset: rasterio.io.DatasetReader,
        window: windows.Window,
        calibration_factor: float,
    ) -> Optional[np.ndarray]:
        """Read, calibrate, and amplitude-compute a SLC tile."""

        try:
            array = dataset.read(window=window, boundless=False)
        except ValueError as exc:
            logger.warning("Failed reading SLC window: %s", exc)
            return None

        if array.ndim == 3:
            if array.shape[0] == 2:
                real_part = array[0].astype(np.float32)
                imag_part = array[1].astype(np.float32)
                amplitude = np.abs(real_part + 1j * imag_part)
            else:
                amplitude = array[0].astype(np.float32)
        else:
            amplitude = array.astype(np.float32)

        if calibration_factor not in (0.0, None):
            amplitude /= calibration_factor

        return self._prepare_tile_array(amplitude)

    def _write_slc_tiles(
        self,
        tile_id: int,
        tile: TileDefinition,
        measurements: List[Dict[str, Optional[str]]],
        datasets: Dict[str, rasterio.io.DatasetReader],
        calibration_factors: Dict[str, float],
    ) -> bool:
        """Write SLC tiles for every available polarisation."""

        if not datasets:
            logger.warning("No SLC datasets opened; skipping SLC export.")
            return False

        success = False
        for entry in measurements:
            href = entry["href"]
            dataset = datasets.get(href)
            if dataset is None:
                continue

            amplitude = self._read_slc_window(
                dataset,
                tile.window,
                calibration_factors.get(href, 1.0),
            )
            if amplitude is None:
                continue

            pol_label = entry["polarization"] or "UNKNOWN"
            output_path = self.output_dirs["slc"] / f"DB_OPENSAR_VD_{tile_id}_SLC_{pol_label}.tiff"
            transform = dataset.window_transform(tile.window)
            self._write_tiff(output_path, amplitude, transform, dataset.crs)
            success = True

        return success

    def _write_grd_tiles(
        self,
        tile_id: int,
        tile: TileDefinition,
        measurements: List[Dict[str, Optional[str]]],
        datasets: Dict[str, rasterio.io.DatasetReader],
    ) -> bool:
        """Write GRD tiles co-registered to the SLC tile footprint."""

        if not datasets:
            logger.debug("No GRD datasets opened; skipping GRD export.")
            return True  # Not an error condition when GRD is optional

        lons, lats = zip(*tile.corner_coords)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        success = False
        for entry in measurements:
            href = entry["href"]
            dataset = datasets.get(href)
            if dataset is None:
                continue

            if dataset.crs and dataset.crs.to_string() != "EPSG:4326":
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326",
                    dataset.crs,
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    densify_pts=21,
                )
            else:
                left, bottom, right, top = min_lon, min_lat, max_lon, max_lat

            window = windows.from_bounds(left, bottom, right, top, dataset.transform, boundless=True)
            window = window.round_offsets().round_lengths()
            window = window.intersection(windows.Window(0, 0, dataset.width, dataset.height))

            data = dataset.read(1, window=window, boundless=True, fill_value=0)
            prepared = self._prepare_tile_array(data)

            pol_label = entry["polarization"] or "UNKNOWN"
            output_path = self.output_dirs["grd"] / f"DB_OPENSAR_VD_{tile_id}_GRD_{pol_label}.tiff"
            transform = dataset.window_transform(window)
            self._write_tiff(output_path, prepared, transform, dataset.crs)
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
    ) -> Dict:
        """Build the XML metadata structure equivalent to MATLAB's outxml."""

        metadata = {
            "ProcessingData": {
                "Tile_ID": tile_id,
                "Corner_Coord": {
                    "Latitude": [coord[1] for coord in tile.corner_coords],
                    "Longitude": [coord[0] for coord in tile.corner_coords],
                },
                "StatisticsReport": {
                    "Number_of_ships": len(vessels),
                },
                "List_of_ships": {"Ship": []},
            }
        }

        for idx, vessel in enumerate(vessels, start=1):
            scene_sample = int(round(vessel.x_pixel)) - tile.origin_col + 1
            scene_line = int(round(vessel.y_pixel)) - tile.origin_row + 1
            scene_sample = max(1, min(self.tile_size, scene_sample))
            scene_line = max(1, min(self.tile_size, scene_line))

            top = int(round(vessel.top_pixel)) - tile.origin_row + 1
            left = int(round(vessel.left_pixel)) - tile.origin_col + 1
            bottom = int(round(vessel.bottom_pixel)) - tile.origin_row + 1
            right = int(round(vessel.right_pixel)) - tile.origin_col + 1

            clamp = lambda value: max(1, min(self.tile_size, value))

            ship_entry = {
                "Name": f"Ship_{idx}",
                "Centroid_Position": {
                    "Latitude": vessel.latitude,
                    "Longitude": vessel.longitude,
                    "SARData_Sample": int(round(vessel.x_pixel)),
                    "SARData_Line": int(round(vessel.y_pixel)),
                    "Scene_Sample": clamp(scene_sample),
                    "Scene_Line": clamp(scene_line),
                },
                "Size": vessel.length_pixel,
                "BoundingBox": {
                    "Top": clamp(top),
                    "Left": clamp(left),
                    "Bottom": clamp(bottom),
                    "Right": clamp(right),
                },
            }
            metadata["ProcessingData"]["List_of_ships"]["Ship"].append(ship_entry)

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
        """Process one Sentinel-1 scene and export all requested artefacts."""

        slc_manifest = self.parse_safe_manifest(slc_product)
        slc_measurements = slc_manifest["measurement"]
        if not slc_measurements:
            logger.error("No SLC measurement files discovered in %s", slc_product)
            return False

        reference_path = Path(slc_product) / slc_measurements[0]["href"]
        if not reference_path.exists():
            logger.error("Reference SLC file missing: %s", reference_path)
            return False

        with rasterio.open(reference_path) as reference_dataset:
            if vessel_data:
                tiles = self._build_tiles_from_vessels(reference_dataset, vessel_data)
            else:
                tiles = self._build_full_scene_tiles(reference_dataset)

        if not tiles:
            logger.warning("No tiles generated for the scene; nothing to export.")
            return False

        overall_success = True
        with ExitStack() as stack:
            slc_datasets: Dict[str, rasterio.io.DatasetReader] = {}
            slc_calibration: Dict[str, float] = {}
            for entry in slc_measurements:
                measurement_path = Path(slc_product) / entry["href"]
                if not measurement_path.exists():
                    logger.warning("Missing SLC measurement: %s", measurement_path)
                    continue
                slc_datasets[entry["href"]] = stack.enter_context(rasterio.open(measurement_path))
                cal_href = self._match_calibration(entry, slc_manifest)
                cal_factor = 1.0
                if cal_href:
                    cal_path = Path(slc_product) / cal_href
                    if cal_path.exists():
                        cal_factor = self._read_calibration_factor(cal_path)
                slc_calibration[entry["href"]] = cal_factor

            grd_measurements: List[Dict[str, Optional[str]]] = []
            grd_datasets: Dict[str, rasterio.io.DatasetReader] = {}
            if grd_product:
                grd_manifest = self.parse_safe_manifest(grd_product)
                grd_measurements = grd_manifest["measurement"]
                for entry in grd_measurements:
                    grd_path = Path(grd_product) / entry["href"]
                    if grd_path.exists():
                        grd_datasets[entry["href"]] = stack.enter_context(rasterio.open(grd_path))
                    else:
                        logger.warning("Missing GRD measurement: %s", grd_path)

            for idx, tile in enumerate(tiles, start=1):
                tile_vessels = self._vessels_in_window(vessel_data, tile)
                slc_written = self._write_slc_tiles(idx, tile, slc_measurements, slc_datasets, slc_calibration)
                grd_written = self._write_grd_tiles(idx, tile, grd_measurements, grd_datasets)
                metadata = self.create_vessel_metadata(idx, tile, tile_vessels)
                xml_written = self.save_metadata_xml(metadata, idx)
                overall_success = overall_success and slc_written and grd_written and xml_written

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

    processor = DarkVesselProcessor(
        base_directory=args.base_dir,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    vessels = processor.load_vessel_data(args.vessel_data)
    success = processor.process_scene(args.grd_product, args.slc_product, vessels if vessels else None)

    if success:
        logger.info("Processing completed successfully.")
    else:
        logger.error("Processing completed with errors.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
