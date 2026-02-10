# DEM Utilities API

The `sarpyx.utils.dem_utils` module provides tools for downloading and managing Copernicus GLO-30 (30 m) and GLO-90 (90 m) Digital Elevation Model tiles from the public AWS S3 bucket, selecting only tiles that intersect a given WKT geometry.

## Overview

The Copernicus DEM is a global elevation model distributed as Cloud-Optimised GeoTIFF (COG) tiles on a 1×1 degree grid. This module automates:

- **Tile computation** — determine which 1×1° cells intersect an arbitrary WKT geometry
- **Parallel download** — fetch tiles from AWS S3 (no authentication required)
- **VRT mosaic** — merge downloaded tiles into a single GDAL Virtual Raster

## Quick Start

```python
from sarpyx.utils.dem_utils import download_tiles_for_wkt

wkt = "POLYGON ((-3.18 54.28, -3.78 55.89, 0.31 56.30, 0.75 54.69, -3.18 54.28))"

tiles = download_tiles_for_wkt(
    wkt_string=wkt,
    output_dir="/path/to/dem_tiles",
    resolution_m=30,
    max_workers=4,
)

for t in tiles:
    print(f"  {t.name}  ({t.stat().st_size / 1e6:.1f} MB)")
```

## Tile Naming Convention

Each tile covers a 1×1 degree cell. The canonical name encodes resolution, latitude, and longitude:

```
Copernicus_DSM_COG_10_N54_00_W004_00_DEM
─────────────────── ── ────── ─────── ───
       prefix       res  lat     lon  suffix
```

| Field | Description | Examples |
|-------|-------------|----------|
| `res` | `10` = GLO-30 (30 m), `30` = GLO-90 (90 m) | `10`, `30` |
| `lat` | South edge of the cell, `N`/`S` + 2-digit value | `N54_00`, `S01_00` |
| `lon` | West edge of the cell, `E`/`W` + 3-digit value | `W004_00`, `E030_00` |

Tiles are served from the public AWS S3 bucket with no authentication:

```
https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/<tile_name>/<tile_name>.tif
```

## Functions

### `tiles_from_wkt()`

```python
tiles_from_wkt(wkt_string: str, resolution_m: int = 30) -> list[tuple[int, int, str, str]]
```

Compute which DEM tiles intersect a WKT geometry.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wkt_string` | `str` | — | Any valid WKT geometry (`POLYGON`, `MULTIPOLYGON`, `POINT`, …) |
| `resolution_m` | `int` | `30` | DEM resolution: `30` (GLO-30) or `90` (GLO-90) |

**Returns:** `list` of `(lat, lon, tile_name, tile_url)` tuples, one per intersecting 1×1° cell.

**Example:**

```python
from sarpyx.utils.dem_utils import tiles_from_wkt

wkt = "POLYGON ((10.0 44.0, 10.0 46.0, 12.0 46.0, 12.0 44.0, 10.0 44.0))"
tiles = tiles_from_wkt(wkt, resolution_m=30)

for lat, lon, name, url in tiles:
    print(f"  ({lat:+d}, {lon:+d})  {name}")
```

---

### `download_tiles_for_wkt()`

```python
download_tiles_for_wkt(
    wkt_string: str,
    output_dir: str | Path,
    resolution_m: int = 30,
    overwrite: bool = False,
    max_workers: int = 4,
) -> list[Path]
```

Download all Copernicus DEM tiles intersecting a WKT geometry.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wkt_string` | `str` | — | Any valid WKT geometry string |
| `output_dir` | `str \| Path` | — | Directory to save `.tif` files (flat layout) |
| `resolution_m` | `int` | `30` | `30` (GLO-30) or `90` (GLO-90) |
| `overwrite` | `bool` | `False` | Re-download tiles that already exist locally |
| `max_workers` | `int` | `4` | Number of parallel download threads |

**Returns:** `list[Path]` — paths to all downloaded (or already-existing) GeoTIFF files, sorted by name.

**Notes:**
- Tiles over open ocean return HTTP 404 and are logged as warnings, not errors.
- Existing files are skipped unless `overwrite=True`.
- Partial downloads use a `.tif.part` suffix and are cleaned up on failure.

**Example:**

```python
from sarpyx.utils.dem_utils import download_tiles_for_wkt

wkt = "POLYGON ((-3.18 54.28, -3.78 55.89, 0.31 56.30, 0.75 54.69, -3.18 54.28))"

tiles = download_tiles_for_wkt(
    wkt_string=wkt,
    output_dir="./AUX/copernicus-dem-30m",
    resolution_m=30,
    overwrite=False,
    max_workers=4,
)
# Output:
# Will download 14 tile(s) ...
# Saved Copernicus_DSM_COG_10_N54_00_W003_00_DEM.tif (29.8 MB)
# Tile not available (404): ...N54_00_E000_00... — likely ocean
# Download complete: 8 succeeded, 6 failed/skipped
```

---

### `tile_name()`

```python
tile_name(lat: int, lon: int, resolution_m: int = 30) -> str
```

Build the canonical tile name for a given 1×1° cell.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `int` | Latitude of the south edge |
| `lon` | `int` | Longitude of the west edge |
| `resolution_m` | `int` | `30` or `90` |

**Returns:** `str` — e.g. `'Copernicus_DSM_COG_10_N54_00_W004_00_DEM'`

---

### `tile_url()`

```python
tile_url(lat: int, lon: int, resolution_m: int = 30) -> str
```

Full HTTPS download URL for a tile's GeoTIFF.

**Returns:** `str` — the public S3 URL for the tile.

---

### `build_vrt()`

```python
build_vrt(tile_paths: Sequence[Path | str], output_vrt: str | Path) -> Path
```

Merge multiple DEM GeoTIFFs into a single GDAL VRT (Virtual Raster).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile_paths` | `Sequence[Path]` | Paths to `.tif` DEM tiles |
| `output_vrt` | `str \| Path` | Output `.vrt` file path |

**Returns:** `Path` — the output VRT path.

**Requires:** `gdalbuildvrt` command (available with GDAL/rasterio installation).

**Example:**

```python
from sarpyx.utils.dem_utils import download_tiles_for_wkt, build_vrt

tiles = download_tiles_for_wkt(wkt, output_dir="./dem")
vrt = build_vrt(tiles, "./dem/mosaic.vrt")
```

---

## CLI Usage

The module can be run as a standalone script:

```bash
python -m sarpyx.utils.dem_utils \
    --wkt "POLYGON ((-3.18 54.28, -3.78 55.89, 0.31 56.30, 0.75 54.69, -3.18 54.28))" \
    --output-dir ./AUX/copernicus-dem-30m \
    --resolution 30 \
    --workers 4 \
    --build-vrt \
    --verbose
```

| Flag | Description |
|------|-------------|
| `--wkt` | WKT geometry string (required) |
| `-o`, `--output-dir` | Output directory (required) |
| `-r`, `--resolution` | `30` or `90` (default: `30`) |
| `--overwrite` | Re-download existing tiles |
| `-w`, `--workers` | Parallel threads (default: `4`) |
| `--build-vrt` | Create a merged VRT after downloading |
| `-v`, `--verbose` | Enable DEBUG-level logging |

## Integration Examples

### With SNAP Terrain Correction (external DEM)

```python
from sarpyx.snapflow.engine import GPT
from sarpyx.utils.dem_utils import download_tiles_for_wkt, build_vrt

# 1. Download DEM tiles for your area
wkt = "POLYGON ((-3.18 54.28, -3.78 55.89, 0.31 56.30, 0.75 54.69, -3.18 54.28))"
tiles = download_tiles_for_wkt(wkt, output_dir="./dem", resolution_m=30)
vrt = build_vrt(tiles, "./dem/copernicus_30m.vrt")

# 2. Use as external DEM in SNAP processing
gpt = GPT(product="S1A_IW_SLC.SAFE", outdir="./output")
gpt.terrain_correction(
    dem_name="External DEM",
    external_dem_file=str(vrt),
    external_dem_no_data_value=0.0,
)
```

### With WorldSAR Pipeline

```python
from sarpyx.utils.dem_utils import download_tiles_for_wkt

# Download DEM for the product footprint before processing
product_wkt = "POLYGON ((-3.18 54.28, ...))"
download_tiles_for_wkt(
    wkt_string=product_wkt,
    output_dir="./AUX/copernicus-dem-30m",
)
```

## See Also

- [Utils API](README.md) — overview of all utility modules
- [SNAP Integration](../../user_guide/snap_integration.md) — using external DEMs with SNAP
- [Geometry Utilities](README.md#geometry-utilities) — WKT and geometry helpers
