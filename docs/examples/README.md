# SARPYX Examples

This directory contains ready-to-run code examples demonstrating SARPYX features and real-world applications. Examples are organized by complexity and application domain.

## 📁 Directory Structure

```
examples/
├── README.md                          # This file
├── basic/                            # Basic usage examples
│   ├── basic_sublook_analysis.py     # Simple SLA workflow
│   ├── snap_integration.py           # SNAP automation
│   ├── visualization_gallery.py      # Plotting and display
│   └── data_io_examples.py           # Input/output operations
├── intermediate/                     # Intermediate applications
│   ├── vegetation_monitoring.py      # Multi-temporal vegetation analysis
│   ├── polarimetric_analysis.py      # Dual-pol decomposition
│   ├── ship_detection_cfar.py        # Maritime target detection
│   └── quality_assessment.py         # Processing validation
├── advanced/                        # Advanced workflows
│   ├── insar_time_series.py          # Interferometric analysis
│   ├── custom_processing_chains.py   # Pipeline development
│   ├── batch_processing.py           # Large-scale automation
│   └── performance_optimization.py   # Computational efficiency
├── notebooks/                       # Jupyter notebooks
│   ├── tutorial_01_getting_started.ipynb
│   ├── tutorial_02_snap_workflows.ipynb
│   ├── tutorial_03_vegetation_indices.ipynb
│   └── tutorial_04_ship_detection.ipynb
└── data/                            # Sample datasets
    ├── README.md                    # Data descriptions
    ├── sample_s1_slc/               # Sample Sentinel-1 SLC
    ├── sample_s1_grd/               # Sample Sentinel-1 GRD
    └── reference_outputs/            # Expected results
```

## 🚀 Quick Start Examples

### Basic Sub-Look Analysis
```python
from sarpyx.sla import SubLookAnalysis

# Simple sub-look decomposition
sla = SubLookAnalysis("data/sample_s1_slc/S1A_IW_SLC__1SDV.zip")
sla.numberOfLooks = 3
sla.frequencyComputation()
sla.SpectrumComputation()
sla.Generation()
```

### SNAP Workflow Automation
```python
from sarpyx.snap import GPT

# Automated SNAP processing
gpt = GPT(product="data/sample_s1_grd/S1A_IW_GRDH_1SDV.zip", outdir="output")
calibrated = gpt.Calibration(outputSigmaBand=True)
terrain_corrected = gpt.TerrainCorrection()
```

### Vegetation Monitoring
```python
from sarpyx.science import vegetation_indices

# Calculate RVI time series
rvi_values = vegetation_indices.calculate_rvi_timeseries(
    vh_data=vh_stack, 
    vv_data=vv_stack, 
    dates=acquisition_dates
)

# Enhanced SAR visualization
show_image(sar_data, title="SAR Amplitude", 
          cmap='gray', enhance=True)
show_histogram(sar_data, bins=50)
```

## Processing Examples

### [Sentinel-1 Processing Chain](processing/sentinel1_chain.py)
Complete processing workflow for Sentinel-1 data including calibration, filtering, and analysis.

### [Multi-temporal Analysis](processing/multitemporal.py)
Time series analysis for change detection and monitoring applications.

### [Polarimetric Analysis](processing/polarimetric.py)
Dual-pol data processing for vegetation and land cover classification.

### [Ship Detection](processing/ship_detection.py)
CFAR-based ship detection in maritime SAR imagery.

## Scientific Applications

### [Vegetation Indices](science/vegetation_indices.py)
```python
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll

# Calculate radar vegetation indices
rvi = calculate_rvi(sigma_vv, sigma_vh)
ndpoll = calculate_ndpoll(sigma_vv, sigma_vh)
```

### [Forest Monitoring](science/forest_monitoring.py)
Forest change detection using multi-temporal SAR data and vegetation indices.

### [Coastal Change Detection](science/coastal_monitoring.py)
Shoreline change analysis using sub-look coherence techniques.

### [Urban Area Analysis](science/urban_analysis.py)
Built-up area mapping and urban growth monitoring.

## Advanced Examples

### [Custom Processing Pipeline](advanced/custom_pipeline.py)
```python
def custom_sar_pipeline(input_path, output_dir):
    """Custom SAR processing pipeline."""
    
    # Step 1: SNAP preprocessing
    gpt = GPT(product=input_path, outdir=output_dir)
    calibrated = gpt.Calibration()
    
    # Step 2: Sub-look analysis
    sla = SubLookAnalysis(calibrated)
    sla.numberOfLooks = 5
    sla.frequencyComputation()
    sla.Generation()
    
    # Step 3: Quality assessment
    quality = assess_quality(sla.Looks)
    
    return sla.Looks, quality
```

### [Batch Processing](advanced/batch_processing.py)
Process multiple SAR products automatically with error handling and quality control.

### [Performance Optimization](advanced/optimization.py)
Memory-efficient processing for large datasets and production workflows.

### [Integration Examples](advanced/integration.py)
Integrating SARPyX with other geospatial libraries like GDAL, rasterio, and geopandas.

## Utility Examples

### [Data Format Conversion](utils/format_conversion.py)
Convert between different SAR data formats and coordinate systems.

### [Quality Assessment](utils/quality_assessment.py)
Comprehensive quality metrics for SAR processing results.

### [Visualization Techniques](utils/visualization.py)
Advanced plotting and visualization techniques for SAR data.

### [File Management](utils/file_management.py)
Utilities for organizing and managing SAR datasets.

## Jupyter Notebooks

Interactive notebooks for learning and experimentation:

### [Interactive Sub-Look Analysis](notebooks/sublook_analysis.ipynb)
Step-by-step notebook with interactive widgets and real-time visualization.

### [SNAP Integration Playground](notebooks/snap_playground.ipynb)
Experiment with different SNAP processing parameters and workflows.

### [Scientific Analysis Workshop](notebooks/science_workshop.ipynb)
Comprehensive notebook covering various scientific applications.

### [Performance Benchmarking](notebooks/performance_benchmark.ipynb)
Compare processing times and memory usage across different approaches.

## Real-World Case Studies

### [Case Study 1: Agricultural Monitoring](case_studies/agriculture.py)
Monitor crop growth and harvest timing using multi-temporal Sentinel-1 data.

### [Case Study 2: Flood Mapping](case_studies/flood_mapping.py)
Rapid flood extent mapping using emergency SAR acquisitions.

### [Case Study 3: Ice Monitoring](case_studies/ice_monitoring.py)
Sea ice classification and monitoring in polar regions.

### [Case Study 4: Infrastructure Monitoring](case_studies/infrastructure.py)
Monitor large infrastructure projects using interferometric techniques.

## Example Data

### Sample Datasets

We provide sample datasets for all examples:

```bash
# Download example datasets
wget https://github.com/ESA-PhiLab/sarpyx-examples/raw/main/data/samples.zip
unzip samples.zip -d examples/data/
```

### Data Structure

```
examples/
├── data/
│   ├── sentinel1/
│   │   ├── S1A_IW_SLC_sample.zip
│   │   └── S1A_IW_GRD_sample.zip
│   ├── cosmo/
│   │   └── CSK_sample.h5
│   ├── multitemporal/
│   │   ├── t1_S1A_*.zip
│   │   └── t2_S1A_*.zip
│   └── auxiliary/
│       ├── landmask.shp
│       └── dem.tif
├── scripts/
└── notebooks/
```

## Running Examples

### Python Scripts

```bash
# Run individual examples
cd examples
python basic_sublook_analysis.py

# Run with custom data
python basic_sublook_analysis.py --input your_data.zip --output results/
```

### Jupyter Notebooks

```bash
# Start Jupyter server
jupyter notebook examples/notebooks/

# Or use JupyterLab
jupyter lab examples/notebooks/
```

### Google Colab

Many examples are available as Google Colab notebooks:
- No local installation required
- Pre-configured environment
- Free GPU access for computationally intensive examples

## 📋 Example Categories

### Basic Examples (`basic/`)
Perfect for getting started with SARPYX:
- **basic_sublook_analysis.py**: Core SLA functionality demonstration
- **snap_integration.py**: SNAP GPT automation basics  
- **visualization_gallery.py**: Plotting and display utilities
- **data_io_examples.py**: Reading and writing SAR data

### Intermediate Examples (`intermediate/`)
Real-world applications and workflows:
- **vegetation_monitoring.py**: Multi-temporal vegetation analysis
- **polarimetric_analysis.py**: Dual-polarization decomposition
- **ship_detection_cfar.py**: CFAR-based target detection
- **quality_assessment.py**: Processing validation and metrics

### Advanced Examples (`advanced/`)
Complex processing chains and optimization:
- **insar_time_series.py**: Interferometric deformation analysis
- **custom_processing_chains.py**: Building custom workflows
- **batch_processing.py**: Large-scale data processing
- **performance_optimization.py**: Computational efficiency techniques

### Interactive Notebooks (`notebooks/`)
Jupyter notebooks with step-by-step explanations:
- Complete tutorials with explanations and visualizations
- Interactive parameter exploration
- Educational content for learning SAR processing concepts

## 🔧 Running Examples

### Prerequisites
```bash
# Install SARPYX with all dependencies
pip install sarpyx[full]

# Or for development
pip install -e .[dev]
```

### Basic Usage
```bash
# Navigate to examples directory
cd docs/examples

# Run a basic example
python basic/basic_sublook_analysis.py

# Run with custom data
python basic/basic_sublook_analysis.py --input /path/to/your/S1_SLC.zip
```

### With Sample Data
```bash
# Download sample data (if available)
python download_sample_data.py

# Run examples with sample data
python basic/basic_sublook_analysis.py --input data/sample_s1_slc/
```

### Jupyter Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Open any notebook in notebooks/ directory
# Follow step-by-step instructions
```

## 📊 Expected Outputs

Each example produces specific outputs:

### Basic Examples
- **Sub-look images**: Forward, backward, and center look decomposition
- **Processed SAR data**: Calibrated and terrain-corrected products
- **Visualization plots**: RGB composites, histograms, and statistics

### Intermediate Examples
- **Vegetation indices**: Time series of RVI, NDPoll, DPDD
- **Polarimetric parameters**: Alpha angle, entropy, anisotropy
- **Detection results**: Ship detection maps with performance metrics

### Advanced Examples
- **Displacement maps**: Interferometric deformation measurements
- **Processing reports**: Comprehensive analysis summaries
- **Optimized workflows**: Performance-tuned processing chains

## 🎯 Application Domains

### Agriculture & Forestry
```python
# Crop monitoring workflow
from sarpyx.applications import agriculture
monitor = agriculture.CropMonitor()
growth_metrics = monitor.analyze_growth_cycle(data_stack)
```

### Maritime Surveillance  
```python
# Ship detection pipeline
from sarpyx.applications import maritime
detector = maritime.ShipDetector()
detections = detector.process_scene(sar_image)
```

### Disaster Monitoring
```python
# Flood mapping
from sarpyx.applications import disaster
flood_map = disaster.map_flooding(before_image, after_image)
```

### Infrastructure Monitoring
```python
# Deformation analysis
from sarpyx.applications import infrastructure
deformation = infrastructure.monitor_subsidence(interferogram_stack)
```

## 🔍 Code Structure

All examples follow consistent patterns:

```python
#!/usr/bin/env python3
"""
Example: [Description]
Application: [Domain]
Complexity: [Basic|Intermediate|Advanced]
"""

import numpy as np
import matplotlib.pyplot as plt
from sarpyx import SLA, SNAPProcessor
from sarpyx.utils import visualization

def main():
    """Main processing function"""
    # 1. Configuration
    config = {...}
    
    # 2. Data loading
    data = load_data(input_file)
    
    # 3. Processing
    results = process_data(data, config)
    
    # 4. Visualization
    visualization.display_results(results)
    
    # 5. Output
    save_results(results, output_dir)

if __name__ == "__main__":
    main()
```

## 📈 Performance Benchmarks

Example processing times on reference hardware:

| Example | Input Size | Processing Time | Memory Usage |
|---------|------------|----------------|--------------|
| Basic SLA | 100MB SLC | 2-5 minutes | 2-4 GB |
| Vegetation Monitoring | 10x 50MB GRD | 10-15 minutes | 4-8 GB |
| Ship Detection | 200MB GRD | 5-10 minutes | 2-4 GB |
| InSAR Time Series | 20x 100MB SLC | 1-2 hours | 8-16 GB |

*Times measured on Intel i7-8700K with 32GB RAM*

## 🛠 Customization

### Adapting Examples
Each example can be customized for your specific needs:

```python
# Modify processing parameters
config = {
    'number_of_looks': 5,  # Instead of default 3
    'window_size': (7, 7),  # Custom filtering
    'output_format': 'GeoTIFF'  # Specify output
}

# Add custom processing steps
def custom_preprocessing(data):
    # Your custom logic here
    return processed_data

# Integration with external tools
from external_tool import CustomProcessor
processor = CustomProcessor()
enhanced_results = processor.enhance(sarpyx_results)
```

### Creating New Examples
Template for new examples:

```python
# Use the template in templates/example_template.py
# 1. Copy template
# 2. Modify for your application
# 3. Add documentation
# 4. Include in appropriate category
```

## 📚 Learning Path

### Beginner (Start Here)
1. `basic/basic_sublook_analysis.py` - Learn core concepts
2. `basic/visualization_gallery.py` - Understand data visualization
3. `notebooks/tutorial_01_getting_started.ipynb` - Interactive learning

### Intermediate 
1. `intermediate/vegetation_monitoring.py` - Real application
2. `intermediate/polarimetric_analysis.py` - Advanced SAR concepts
3. `notebooks/tutorial_03_vegetation_indices.ipynb` - Detailed analysis

### Advanced
1. `advanced/insar_time_series.py` - Complex workflows
2. `advanced/custom_processing_chains.py` - Pipeline development
3. `advanced/performance_optimization.py` - Production systems

## 🤝 Contributing Examples

We welcome contributions! To add a new example:

1. **Fork the repository**
2. **Create your example** following the template
3. **Add documentation** and expected outputs
4. **Test thoroughly** with sample data
5. **Submit a pull request**

### Example Contribution Checklist
- [ ] Code follows SARPYX style guidelines
- [ ] Includes comprehensive docstrings
- [ ] Has corresponding test data
- [ ] Generates expected outputs
- [ ] Includes performance benchmarks
- [ ] Documentation is complete

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install missing dependencies
pip install sarpyx[full]
```

**Memory Issues**
```python
# Process data in chunks
for chunk in data_chunks:
    result = process_chunk(chunk)
```

**SNAP Integration Problems**
```bash
# Verify SNAP installation
snap --version
```

**Path Issues**
```python
# Use absolute paths
input_file = os.path.abspath("data/sample.zip")
```

### Getting Help
- Check the [troubleshooting guide](../user_guide/troubleshooting.md)
- Review [API documentation](../api/)
- Post issues on [GitHub](https://github.com/your-repo/SARPYX/issues)

## 📄 License

All examples are provided under the same license as SARPYX. See LICENSE file for details.

---

*Start with the basic examples and gradually progress to more complex applications. Each example builds upon previous concepts and introduces new SARPYX capabilities.*
