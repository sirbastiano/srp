"""
Copernicus DEM COG tile utilities.

Download and manage Copernicus GLO-30 (30m) and GLO-90 (90m) DEM tiles
from the public AWS S3 bucket, selecting only the tiles that intersect
a given WKT geometry.

Tile naming convention (GLO-30 example)::

    Copernicus_DSM_COG_10_N54_00_W004_00_DEM
    ─────────────────── ── ────── ─────── ───
           prefix       res  lat     lon  suffix

Resolution codes: 10 → GLO-30 (30 m), 30 → GLO-90 (90 m).
Each tile covers a 1×1 degree cell.

Public HTTPS base URLs (no authentication required)::

    GLO-30: https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/
    GLO-90: https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com/
"""

from __future__ import annotations

import logging
import math
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from shapely import wkt as shapely_wkt
from shapely.geometry import box as shapely_box

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

_BASE_URLS = {
    30: "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com",
    90: "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com",
}

_RES_CODES = {
    30: 10,   # GLO-30 → resolution code 10 (arc-seconds)
    90: 30,   # GLO-90 → resolution code 30
}


# ──────────────────────────────────────────────────────────────────────
# Tile name helpers
# ──────────────────────────────────────────────────────────────────────

def _lat_label(lat: int) -> str:
    """Convert an integer latitude to the tile label, e.g. 54 → 'N54_00', -1 → 'S01_00'."""
    hemisphere = "N" if lat >= 0 else "S"
    return f"{hemisphere}{abs(lat):02d}_00"


def _lon_label(lon: int) -> str:
    """Convert an integer longitude to the tile label, e.g. -4 → 'W004_00', 30 → 'E030_00'."""
    hemisphere = "E" if lon >= 0 else "W"
    return f"{hemisphere}{abs(lon):03d}_00"


def tile_name(lat: int, lon: int, resolution_m: int = 30) -> str:
    """Build the canonical Copernicus DEM COG tile name.

    Parameters
    ----------
    lat : int
        Latitude of the **south edge** of the 1×1° cell.
    lon : int
        Longitude of the **west edge** of the 1×1° cell.
    resolution_m : int
        DEM resolution in metres (30 or 90).

    Returns
    -------
    str
        e.g. ``'Copernicus_DSM_COG_10_N54_00_W004_00_DEM'``
    """
    res_code = _RES_CODES[resolution_m]
    return f"Copernicus_DSM_COG_{res_code}_{_lat_label(lat)}_{_lon_label(lon)}_DEM"


def tile_url(lat: int, lon: int, resolution_m: int = 30) -> str:
    """Full HTTPS URL for a Copernicus DEM COG GeoTIFF tile.

    Returns
    -------
    str
        e.g. ``'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/
        Copernicus_DSM_COG_10_N54_00_W004_00_DEM/
        Copernicus_DSM_COG_10_N54_00_W004_00_DEM.tif'``
    """
    name = tile_name(lat, lon, resolution_m)
    base = _BASE_URLS[resolution_m]
    return f"{base}/{name}/{name}.tif"


# ──────────────────────────────────────────────────────────────────────
# Geometry → tile grid
# ──────────────────────────────────────────────────────────────────────

def _bounding_tiles(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> List[Tuple[int, int]]:
    """Return ``(lat, lon)`` integer pairs for every 1×1° cell that
    intersects the bounding box.

    The tile at ``(lat, lon)`` covers ``[lat, lat+1) × [lon, lon+1)``.
    """
    lat_start = math.floor(min_lat)
    lat_end = math.ceil(max_lat)       # exclusive upper bound
    lon_start = math.floor(min_lon)
    lon_end = math.ceil(max_lon)

    # If max_lat/max_lon land exactly on a boundary we don't need the next tile
    if max_lat == lat_end:
        lat_end = lat_end               # keep, the tile at lat_end-1 covers up to lat_end
    if max_lon == lon_end:
        lon_end = lon_end

    tiles = []
    for lat in range(lat_start, lat_end):
        for lon in range(lon_start, lon_end):
            tiles.append((lat, lon))
    return tiles


def tiles_from_wkt(
    wkt_string: str,
    resolution_m: int = 30,
) -> List[Tuple[int, int, str, str]]:
    """Compute the DEM tile grid cells that intersect a WKT geometry.

    Parameters
    ----------
    wkt_string : str
        Any valid WKT geometry (POLYGON, MULTIPOLYGON, POINT, …).
    resolution_m : int
        DEM resolution (30 or 90).

    Returns
    -------
    list of (lat, lon, tile_name, tile_url)
        One entry per intersecting 1×1° tile.
    """
    geom = shapely_wkt.loads(wkt_string)
    min_lon, min_lat, max_lon, max_lat = geom.bounds

    results = []
    for lat, lon in _bounding_tiles(min_lon, min_lat, max_lon, max_lat):
        # Build the 1×1° cell and check intersection with the actual geometry
        cell = shapely_box(lon, lat, lon + 1, lat + 1)
        if geom.intersects(cell):
            name = tile_name(lat, lon, resolution_m)
            url = tile_url(lat, lon, resolution_m)
            results.append((lat, lon, name, url))

    logger.info(
        "WKT bbox [%.2f, %.2f, %.2f, %.2f] → %d tiles (res=%dm)",
        min_lon, min_lat, max_lon, max_lat, len(results), resolution_m,
    )
    return results


# ──────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────

def _download_one(
    url: str,
    dest: Path,
    overwrite: bool = False,
    timeout: int = 300,
) -> Path:
    """Download a single tile from *url* to *dest*.

    Raises
    ------
    urllib.error.HTTPError
        If the tile does not exist on the server (404) or other HTTP error.
    """
    if dest.exists() and not overwrite:
        logger.debug("Skipping (exists): %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tif.part")

    try:
        logger.info("Downloading %s …", dest.name)
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
        logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    return dest


def download_tiles_for_wkt(
    wkt_string: str,
    output_dir: str | Path,
    resolution_m: int = 30,
    overwrite: bool = False,
    max_workers: int = 4,
) -> List[Path]:
    """Download all Copernicus DEM tiles intersecting a WKT geometry.

    Parameters
    ----------
    wkt_string : str
        Any valid WKT geometry string.
    output_dir : str or Path
        Directory to save the downloaded ``.tif`` files into.
        A flat layout is used: ``<output_dir>/<tile_name>.tif``.
    resolution_m : int
        DEM resolution in metres — 30 (GLO-30) or 90 (GLO-90).
    overwrite : bool
        Re-download tiles that already exist locally.
    max_workers : int
        Number of parallel download threads.

    Returns
    -------
    list of Path
        Paths to the downloaded (or already existing) GeoTIFF files.

    Examples
    --------
    >>> from sarpyx.utils.dem_utils import download_tiles_for_wkt
    >>> wkt = "POLYGON ((-3.18 54.28, -3.78 55.89, 0.31 56.30, 0.75 54.69, -3.18 54.28))"
    >>> tiles = download_tiles_for_wkt(wkt, "/tmp/dem_tiles")
    >>> print([t.name for t in tiles])
    ['Copernicus_DSM_COG_10_N54_00_W004_00_DEM.tif', ...]
    """
    if resolution_m not in _RES_CODES:
        raise ValueError(f"resolution_m must be 30 or 90, got {resolution_m}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_info = tiles_from_wkt(wkt_string, resolution_m)
    if not tile_info:
        logger.warning("No tiles found for the given WKT geometry.")
        return []

    logger.info(
        "Will download %d tile(s) to %s (overwrite=%s, workers=%d)",
        len(tile_info), output_dir, overwrite, max_workers,
    )

    # Print tile summary
    for lat, lon, name, url in tile_info:
        logger.info("  • %s  (%+d°, %+d°)", name, lat, lon)

    downloaded: List[Path] = []
    failed: List[str] = []

    def _job(info: Tuple[int, int, str, str]) -> Optional[Path]:
        _, _, name, url = info
        dest = output_dir / f"{name}.tif"
        try:
            return _download_one(url, dest, overwrite=overwrite)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                logger.warning("Tile not available (404): %s — likely ocean", name)
            else:
                logger.error("HTTP %d for %s: %s", exc.code, name, exc.reason)
            return None
        except Exception as exc:
            logger.error("Failed to download %s: %s", name, exc)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_job, info): info for info in tile_info}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                downloaded.append(result)
            else:
                failed.append(futures[future][2])

    downloaded.sort(key=lambda p: p.name)

    logger.info(
        "Download complete: %d succeeded, %d failed/skipped",
        len(downloaded), len(failed),
    )
    if failed:
        logger.warning("Failed tiles: %s", ", ".join(failed))

    return downloaded


def build_vrt(
    tile_paths: Sequence[Path | str],
    output_vrt: str | Path,
) -> Path:
    """Merge multiple DEM GeoTIFFs into a single GDAL VRT (virtual mosaic).

    Requires ``rasterio`` (already a project dependency).

    Parameters
    ----------
    tile_paths : sequence of Path
        Paths to ``.tif`` DEM tiles.
    output_vrt : str or Path
        Path for the output ``.vrt`` file.

    Returns
    -------
    Path
        The output VRT path.
    """
    try:
        from rasterio.merge import merge
        import rasterio
    except ImportError as exc:
        raise ImportError("rasterio is required for VRT building: pip install rasterio") from exc

    output_vrt = Path(output_vrt)
    output_vrt.parent.mkdir(parents=True, exist_ok=True)

    # Build VRT via GDAL command line (more efficient than rasterio merge for VRT)
    import subprocess
    tile_strs = [str(p) for p in tile_paths]
    cmd = ["gdalbuildvrt", str(output_vrt)] + tile_strs
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalbuildvrt failed: {result.stderr}")

    logger.info("Created VRT: %s (%d tiles)", output_vrt, len(tile_paths))
    return output_vrt


# ──────────────────────────────────────────────────────────────────────
# CLI-friendly entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Command-line entry point for downloading Copernicus DEM tiles."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Copernicus DEM COG tiles for a WKT geometry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--wkt", required=True,
        help="WKT geometry string (POLYGON, MULTIPOLYGON, …)",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory to save downloaded .tif tiles",
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=30, choices=[30, 90],
        help="DEM resolution in metres (default: 30)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download tiles that already exist locally",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Number of parallel download threads (default: 4)",
    )
    parser.add_argument(
        "--build-vrt", action="store_true",
        help="Build a merged VRT after downloading",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    tiles = download_tiles_for_wkt(
        wkt_string=args.wkt,
        output_dir=args.output_dir,
        resolution_m=args.resolution,
        overwrite=args.overwrite,
        max_workers=args.workers,
    )

    if args.build_vrt and tiles:
        vrt_path = Path(args.output_dir) / "copernicus_dem_mosaic.vrt"
        build_vrt(tiles, vrt_path)

    print(f"\n{'='*60}")
    print(f"Downloaded {len(tiles)} tile(s) to {args.output_dir}")
    for t in tiles:
        print(f"  {t.name}  ({t.stat().st_size / 1e6:.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
