<div align="center">

<img src="src/sarpyx_logo.png" width="1400px" alt="sarpyx">

<br />

<a href="docs/user_manual.md">
  <img alt="User Manual" src="https://img.shields.io/badge/Read-User%20Manual-111827?style=for-the-badge" />
</a>
<a href="docs/user_manual.md#quick-start">
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

<details open>
<summary><strong>Using uv (recommended)</strong></summary>

```bash
uv sync --extra copernicus
```

For development installation with extras:

```bash
uv sync --extra copernicus --extra dev --extra test --extra docs
```
</details>

<details>
<summary><strong>Using pip (editable)</strong></summary>

```bash
python -m pip install -e .
```
</details>


## Docs

See `docs/user_manual.md` for full CLI usage and end-to-end workflows.

##
<div align="center">

**With Love By:** Roberto Del Prete

</div>
