# Installation Guide

This guide will help you install sarpyx and its dependencies on your system.

## Prerequisites

Before installing sarpyx, ensure you have:

- Python 3.11 or higher
- pip package manager
- Git (for development installation)

### System-specific Requirements

#### For SNAP Integration (Optional)
- ESA SNAP Desktop (version 9.0 or higher)
- Java Runtime Environment (JRE) 11 or higher
- At least 8GB of RAM recommended

#### For Scientific Computing
- NumPy compatible system
- GDAL libraries (for geospatial data handling)

## Installation Methods

### 1. Using pip (Recommended for Users)

Once sarpyx is published to PyPI, you can install it using:

```bash
pip install sarpyx
```

For specific versions:
```bash
pip install sarpyx==0.1.5
```

### 2. Using PDM (Recommended for Development)

PDM (Python Dependency Management) is the preferred tool for development:

1. **Install PDM:**
   ```bash
   pip install pdm
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/ESA-PhiLab/sarpyx.git
   cd sarpyx
   ```

3. **Install dependencies:**
   ```bash
   pdm install
   ```

4. **For development with extras:**
   ```bash
   pdm install -G dev -G test -G docs
   ```

### 3. Development Installation

For contributors or advanced users who want the latest features:

```bash
git clone https://github.com/ESA-PhiLab/sarpyx.git
cd sarpyx
pip install -e .
```

This creates an editable installation that reflects code changes immediately.

## Verifying Installation

To verify your installation works correctly:

```python
import sarpyx
print(f"sarpyx version: {sarpyx.__version__}")

# Test basic functionality
from sarpyx.utils import show_image
from sarpyx.sla import SubLookAnalysis
print("Installation successful!")
```

## Optional Dependencies

### SNAP Integration
For full SNAP functionality, install ESA SNAP:

1. Download from [ESA SNAP website](https://step.esa.int/main/download/snap-download/)
2. Follow platform-specific installation instructions
3. Ensure `gpt` command is available in your PATH

### Jupyter Support
For interactive notebooks:
```bash
pip install jupyter matplotlib ipywidgets
```

### Visualization
For enhanced plotting capabilities:
```bash
pip install matplotlib seaborn plotly
```

## Troubleshooting

### Common Issues

#### GDAL Installation Problems
On some systems, GDAL can be challenging to install:

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install gdal
```

**macOS (using Homebrew):**
```bash
brew install gdal
pip install gdal
```

**Windows:**
Consider using conda:
```bash
conda install -c conda-forge gdal
```

#### Memory Issues
For large SAR datasets, ensure sufficient memory:
- Minimum 8GB RAM
- 16GB+ recommended for large-scale processing

#### Permission Errors
On Unix systems, you might need:
```bash
pip install --user sarpyx
```

### Getting Help

If you encounter installation issues:

1. Check [GitHub Issues](https://github.com/ESA-PhiLab/sarpyx/issues)
2. Create a new issue with:
   - Your operating system
   - Python version
   - Complete error message
   - Installation method used

## Next Steps

After successful installation:

1. Read [Getting Started](getting_started.md)
2. Explore [Basic Concepts](basic_concepts.md)
3. Try the [Examples](../examples/README.md)

## Environment Setup

### Virtual Environment (Recommended)

Create an isolated environment for sarpyx:

```bash
# Using venv
python -m venv sarpyx-env
source sarpyx-env/bin/activate  # Linux/macOS
# or
sarpyx-env\Scripts\activate     # Windows

# Using conda
conda create -n sarpyx python=3.9
conda activate sarpyx
```

### Development Environment

For development work, also install development tools:

```bash
pip install pytest black flake8 mypy sphinx
```

This ensures you have all tools needed for testing, linting, and documentation generation.
