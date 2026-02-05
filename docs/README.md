# SARPyX Documentation

This documentation provides a comprehensive overview of the SARPyX project, focusing on the models, dataset, and configuration files. It is intended to help users understand the structure, configuration, and usage of the SARPyX codebase for Synthetic Aperture Radar (SAR) data processing and deep learning.

## Table of Contents
- [Overview](#overview)
- [Configuration Files](#configuration-files)
    - [YAML Structure](#yaml-structure)
    - [Model Configuration](#model-configuration)
    - [Dataloader Configuration](#dataloader-configuration)
    - [Training Configuration](#training-configuration)
- [Models](#models)
    - [Model Utilities](#model-utilities)
    - [Transformer Models](#transformer-models)
    - [SSM Models](#ssm-models)
- [Dataset](#dataset)
    - [SARZarrDataset](#sarzarrdataset)
    - [SampleFilter](#samplefilter)
    - [SARDataloader](#sardataloader)
- [Training and Inference](#training-and-inference)
- [Visualization](#visualization)

---

## Overview
SARPyX is a modular framework for deep learning on SAR data, supporting flexible model architectures, efficient data loading from Zarr archives, and robust training/inference pipelines. The project is organized into modules for models, data loading, training, and utilities.

## Configuration Files
SARPyX uses YAML configuration files to define experiments. These files specify model parameters, dataloader settings, training hyperparameters, and more. Example configuration files are found in `training/` (e.g., `rv_transformer_autoregressive.yaml`).

### YAML Structure
A typical configuration file contains the following sections:
- `model`: Defines the model architecture and parameters.
- `dataloader`: Specifies data loading options.
- `training`: Contains training hyperparameters and settings.
- (Optional) `optimizer`, `scheduler`, etc.

#### Example
```yaml
model:
    name: rv_transformer_autoregressive
    input_channels: 2
    output_channels: 2
    hidden_dim: 128
    num_layers: 6
    ...
dataloader:
    data_dir: /path/to/data
    patch_size: [1000, 1]
    level_from: rcmc
    level_to: az
    batch_size: 16
    ...
training:
    save_dir: ./results
    num_epochs: 50
    learning_rate: 1e-4
    ...
```

### Model Configuration
- `name`: Model type (e.g., `rv_transformer_autoregressive`, `cv_transformer`, `s4_ssm_complex`).
- `input_channels`, `output_channels`: Number of input/output channels.
- `hidden_dim`, `num_layers`, etc.: Model-specific hyperparameters.
- Additional parameters depend on the model (see model docstrings).

### Dataloader Configuration
- `data_dir`: Path to Zarr data directory.
- `patch_size`: Size of data patches (height, width).
- `level_from`, `level_to`: SAR processing levels (e.g., `rcmc`, `az`).
- `batch_size`: Batch size for loading data.
- `patch_mode`: Patch extraction mode (`rectangular`, `parabolic`).
- `buffer`, `stride`: Patch extraction parameters.
- `online`: Whether to use remote Hugging Face Zarr stores.
- `max_products`, `samples_per_prod`: Limits for data loading.
- See `SARZarrDataset` for all options.

### Training Configuration
- `save_dir`: Directory to save results and checkpoints.
- `num_epochs`: Number of training epochs.
- `learning_rate`: Learning rate for optimizer.
- `device`: Device to use (`cuda`, `cpu`).
- Additional options: optimizer, scheduler, loss function, etc.

## Models

### Model Utilities (`model/model_utils.py`)
- `get_model_from_configs(config)`: Instantiates a model from configuration.
- `create_model_with_pretrained(config, pretrained_path, device)`: Loads a model with pretrained weights.

### Transformer Models (`model/transformers/`)
- `rv_transformer.py`, `cv_transformer.py`: Implement row/column and cross-variant transformer architectures for SAR data.
- Configurable via YAML (see above).
- Key parameters: number of layers, hidden dimensions, attention heads, etc.

### SSM Models (`model/SSMs/`)
- `SSM.py`: Implements State Space Models for SAR data.
- Configurable for real/complex-valued data.

## Dataset

### SARZarrDataset (`dataloader/dataloader.py`)
A PyTorch `Dataset` for loading SAR data patches from Zarr archives.

#### Attributes
- `data_dir`: Directory containing Zarr files.
- `filters`: Optional `SampleFilter` for selecting data.
- `patch_size`: Patch size (height, width).
- `level_from`, `level_to`: Input/target SAR processing levels.
- `patch_mode`: Patch extraction mode.
- `buffer`, `stride`: Patch extraction parameters.
- `complex_valued`: Whether to return complex-valued tensors.
- `transform`: Optional transform module.
- `max_products`, `samples_per_prod`: Data loading limits.
- `cache_size`: LRU cache size for chunk loading.
- `positional_encoding`: Whether to add positional encoding.
- `dataset_length`: Optional override for dataset length.

#### Methods
- `__init__`: Initializes the dataset with all configuration options.
- `get_patch_size(zfile)`: Returns patch size for a file.
- `get_metadata(zfile, rows)`: Returns metadata for a file.
- `get_samples_by_file(zfile)`: Returns patch coordinates for a file.
- `get_files()`: Returns list of Zarr files.
- `open_archive(zfile)`: Opens a Zarr archive.
- `get_store_at_level(zfile, level)`: Returns Zarr array at a given level.
- `get_whole_sample_shape(zfile)`: Returns shape of the whole sample.
- `calculate_patches_from_store(zfile, patch_order, window)`: Calculates patch coordinates.
- `reorder_samples(zfile, patch_order)`: Reorders patch samples.
- `__len__`: Returns dataset length.
- `__getitem__(idx)`: Returns a data sample (input, target).
- (Many more utility methods for efficient patch extraction and caching.)

### SampleFilter
- Used to filter dataset samples by part, year, month, polarization, stripmap mode, etc.
- Methods: `get_filter_dict()`, `matches(record)`, `_filter_products(df)`.

### SARDataloader
- Subclass of PyTorch `DataLoader` for SAR data.
- Use `get_sar_dataloader()` to instantiate with all dataset options.

## Training and Inference
- Training and inference scripts are in `training/`.
- Use `training_script.py` for training, `inference_script.py` for inference.
- Visualizations and metrics are handled by `visualize.py` and `inference_visualization.ipynb`.

## Visualization
- Use `display_inference_results` to visualize input, ground truth, and predictions.
- Visualizations are saved to the directory specified in the config (`save_dir`).

---

For more details, see the docstrings in each module and the example configuration files in `training/`.
# SARPyX Documentation

Welcome to the SARPyX documentation! SARPyX is a specialized Python package for advanced Synthetic Aperture Radar (SAR) data processing, sub-aperture decomposition, and full integration with ESA's SNAP (Sentinel Application Platform) engine.

<p align="center">
    <img src="../assets/sarpyx_logo.png" alt="sarpyx logo" width="1200"/>
</p>

## Table of Contents

### 📚 [User Guide](user_guide/README.md)
Complete guide for getting started with SARPyX, including installation, basic concepts, and common workflows.

### 🎯 [Tutorials](tutorials/README.md)
Step-by-step tutorials covering various SAR processing techniques and real-world applications.

### 💻 [Examples](examples/README.md)
Ready-to-run code examples demonstrating key features and processing workflows.

### 🔧 [API Reference](api/README.md)
Comprehensive API documentation for all modules, classes, and functions.

### 👩‍💻 [Developer Guide](developer_guide/README.md)
Information for developers contributing to SARPyX, including architecture, coding standards, and contribution guidelines.

### 📖 [Reference](reference/README.md)
Technical references, mathematical background, and external resources.

## Quick Start

```python
import sarpyx

# Example: Calculate vegetation indices from Sentinel-1 data
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll

# Load your SAR data (VV and VH polarizations)
sigma_vv = your_vv_data  # Linear scale backscatter coefficients
sigma_vh = your_vh_data  # Linear scale backscatter coefficients

# Calculate Radar Vegetation Index
rvi = calculate_rvi(sigma_vv, sigma_vh)

# Calculate Normalized Difference Polarization Index
ndpoll = calculate_ndpoll(sigma_vv, sigma_vh)
```

## Key Features

- 🛰️ **Sub-Aperture Decomposition**: Azimuthal sub-band partitioning for enhanced resolution
- ⚙️ **SNAP Engine Integration**: Direct interface with SNAP Graph Processing Tool (GPT)
- 🗂️ **Modular Processing Pipeline**: Customizable preprocessing and analysis workflows
- 📦 **Data Compatibility**: Support for Sentinel-1, COSMO-SkyMed, and other SAR missions
- 🔌 **Extensible Architecture**: Designed for interoperability with geospatial libraries

## Installation

```bash
pip install sarpyx
```

For development installation:
```bash
git clone https://github.com/ESA-PhiLab/sarpyx.git
cd sarpyx
pip install -e .
```

## Support and Community

- 📧 **Issues**: [GitHub Issues](https://github.com/ESA-PhiLab/sarpyx/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ESA-PhiLab/sarpyx/discussions)
- 📖 **Documentation**: [Full Documentation](https://sarpyx.readthedocs.io)

## License

SARPyX is released under the Apache 2.0 License. See the [LICENSE](../LICENSE) file for details.
