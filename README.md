![GitHub stars](https://img.shields.io/github/stars/sirbastiano/srp.svg)
![GitHub forks](https://img.shields.io/github/forks/sirbastiano/srp.svg)
![GitHub issues](https://img.shields.io/github/issues/sirbastiano/srp.svg)
![GitHub pull requests](https://img.shields.io/github/issues-pr/sirbastiano/srp.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/sirbastiano/srp.svg)
![GitHub code size](https://img.shields.io/github/languages/code-size/sirbastiano/srp.svg)
![GitHub top language](https://img.shields.io/github/languages/top/sirbastiano/srp.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/sirbastiano/srp.svg)
![GitHub contributors](https://img.shields.io/github/contributors/sirbastiano/srp.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)



<p align="center">
    <img src="src/sarpyx_logo.png" alt="sarpyx logo" width="1200"/>
</p>


**sarpyx** is a specialized Python package for advanced Synthetic Aperture Radar (SAR) data processing, **sub-aperture decomposition** and full integration with ESA's **SNAP (Sentinel Application Platform)** engine. It is tailored for researchers and developers.

## Key Features

- 🛰️ **Sub-Aperture Decomposition**  
  Perform azimuthal sub-band partitioning for enhanced resolution, motion sensitivity, and squint-angle diversity exploitation.

- ⚙️ **SNAP Engine Integration**  
  Interface directly with the SNAP Graph Processing Tool (GPT) to automate calibration, coregistration, interferometry, and other Level-1/2 workflows.

- 🗂️ **Modular Processing Pipeline**  
  Chain together preprocessing, sub-aperture slicing, interferogram generation, and differential phase analysis with customizable steps.

- 📦 **Data Compatibility**  
  - Native support for Sentinel-1 SLC (SAFE format)
  - Efficient in-memory SAR matrix manipulation via `xarray` and `numpy`
  - Output geocoded products for GIS integration

- 🔌 **Extensible Architecture**  
  Designed for interoperability with geospatial libraries like `rasterio`, `geopandas`, and `pyproj`.

## Installation

### Using PDM (recommended)

1. Make sure you have [PDM](https://pdm.fming.dev/latest/#installation) installed:
   ```bash
   pip install pdm
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/sirbastiano/sarpyx.git
   cd sarpyx
   ```

3. Install with PDM:
   ```bash
   pdm install
   ```

4. For development installation with extras:
   ```bash
   pdm install -G dev -G test -G docs
   ```

### Using pip

Coming soon: pip install sarpyx

---
