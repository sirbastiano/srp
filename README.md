<div align="center">

<img src="src/sarpyx_logo.png" width="1400px" alt="sarpyx">

<br />

<a href="docs/user_guide/README.md">
  <img alt="User Manual" src="https://img.shields.io/badge/Read-User%20Manual-111827?style=for-the-badge" />
</a>
<a href="docs/user_guide/getting_started.md">
  <img alt="Quick Start" src="https://img.shields.io/badge/Start-Quick%20Start-0f766e?style=for-the-badge" />
</a>
<a href="LICENSE">
  <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-374151?style=for-the-badge" />
</a>
</div>

##

**sarpyx** is a specialized Python toolkit for **Synthetic Aperture Radar (SAR)** processing with tight integration to ESA **SNAP**. It focuses on reproducible pipelines, fast tiling workflows, and advanced research features like **sub-aperture decomposition**.

## Highlights

- SNAP GPT integration with configurable graphs and operator chaining.
- Sub-aperture decomposition for squint-angle diversity and motion sensitivity.
- Parallel tiling and batch processing for large product volumes.
- Geocoded outputs ready for GIS and downstream ML.
- Extensible architecture compatible with `rasterio`, `geopandas`, and `pyproj`.

## Install

For container workflows, use the Docker Compose CLI plugin (`docker compose`) with full commands:

```bash
docker compose version
make recreate
```

<details open>
<summary><strong>Using uv (recommended)</strong></summary>

```bash
uv sync
```

For development, testing, and optional Copernicus tooling:

```bash
uv sync --group dev
uv sync --group dev --extra copernicus
uv run pytest -q
uv build
```
</details>

<details>
<summary><strong>Using pip (editable)</strong></summary>

```bash
python -m pip install -e .
```
</details>


## Docs

See [docs/user_guide/README.md](docs/user_guide/README.md) for usage and workflows, and [docs/developer_guide/contributing.md](docs/developer_guide/contributing.md) for contributor commands.

## Container grid configuration

At startup the container checks for grid files in this order:

1. `GRID_PATH` (or `grid_path`) if it points to an existing in-container `*.geojson`
2. First `*.geojson` found in `/workspace/grid`

If neither exists, the container exits with an error. Automatic grid generation
on startup has been removed.

To use a mounted grid:

```bash
mkdir -p ./grid
# put any grid GeoJSON here, e.g. ./grid/my_region.geojson
docker compose up
```

For direct `docker run`, pass an explicit in-container path when needed:

```bash
docker run --rm \
  -v "$PWD/grid:/workspace/grid:ro" \
  -e GRID_PATH=/workspace/grid/my_region.geojson \
  sirbastiano94/sarpyx:latest \
  /usr/local/bin/start-jupyter.sh
```

You can also pass `--grid-path` to the `worldsar` CLI command.

##
<div align="center">

**With Love By:** Roberto Del Prete

</div>
