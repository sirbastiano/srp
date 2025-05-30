# Tutorials

Step-by-step tutorials for learning SARPyX capabilities and SAR processing techniques.

## Getting Started Tutorials

### [Tutorial 1: Your First Sub-Look Analysis](01_first_sublook_analysis.md)
Learn the basics of sub-look decomposition with a simple Sentinel-1 example.
- **Duration**: 15 minutes
- **Prerequisites**: Basic Python knowledge
- **Data**: Sample Sentinel-1 SLC product

### [Tutorial 2: SNAP Integration Basics](02_snap_integration_basics.md)
Use SARPyX to automate SNAP processing workflows.
- **Duration**: 20 minutes  
- **Prerequisites**: SNAP installation, Tutorial 1
- **Data**: Sentinel-1 GRD product

### [Tutorial 3: Visualization and Quality Assessment](03_visualization_quality.md)
Learn to visualize results and assess processing quality.
- **Duration**: 25 minutes
- **Prerequisites**: Tutorial 1
- **Data**: Results from previous tutorials

## Intermediate Tutorials

### [Tutorial 4: Multi-temporal Analysis](04_multitemporal_analysis.md)
Process time series of SAR data for change detection.
- **Duration**: 30 minutes
- **Prerequisites**: Tutorials 1-3
- **Data**: Multi-temporal Sentinel-1 dataset

### [Tutorial 5: Polarimetric Analysis](05_polarimetric_analysis.md)
Work with dual-pol data for vegetation monitoring.
- **Duration**: 35 minutes
- **Prerequisites**: Tutorial 2
- **Data**: Dual-pol Sentinel-1 data

### [Tutorial 6: Custom Processing Workflows](06_custom_workflows.md)
Build custom processing pipelines for specific applications.
- **Duration**: 40 minutes
- **Prerequisites**: Tutorials 1-5
- **Data**: Various SAR products

## Advanced Tutorials

### [Tutorial 7: Ship Detection with CFAR](07_ship_detection_cfar.md)
Implement ship detection using Constant False Alarm Rate algorithms.
- **Duration**: 45 minutes
- **Prerequisites**: Tutorial 2
- **Data**: Maritime Sentinel-1 data

### [Tutorial 8: Interferometric Processing](08_interferometric_processing.md)
Create interferograms and analyze surface deformation.
- **Duration**: 50 minutes
- **Prerequisites**: Tutorials 4, 6
- **Data**: Interferometric pair

### [Tutorial 9: Performance Optimization](09_performance_optimization.md)
Optimize processing for large datasets and production workflows.
- **Duration**: 30 minutes
- **Prerequisites**: All previous tutorials
- **Data**: Large-scale dataset

## Specialized Applications

### [Tutorial 10: Forest Monitoring](10_forest_monitoring.md)
Use radar vegetation indices for forest change detection.
- **Duration**: 35 minutes
- **Prerequisites**: Tutorial 5
- **Data**: Forest area time series

### [Tutorial 11: Urban Area Analysis](11_urban_analysis.md)
Analyze urban environments using multi-look techniques.
- **Duration**: 40 minutes
- **Prerequisites**: Tutorials 1, 4
- **Data**: Urban area SAR data

### [Tutorial 12: Coastal Monitoring](12_coastal_monitoring.md)
Monitor coastal changes using sub-look analysis.
- **Duration**: 45 minutes
- **Prerequisites**: Tutorials 4, 7
- **Data**: Coastal area time series

## Tutorial Structure

Each tutorial follows a consistent structure:

1. **Overview**: What you'll learn and accomplish
2. **Prerequisites**: Required knowledge and previous tutorials
3. **Setup**: Data preparation and environment setup
4. **Step-by-step Instructions**: Detailed walkthrough with code
5. **Results and Analysis**: Interpretation of outputs
6. **Exercises**: Additional practice problems
7. **Next Steps**: Connections to related tutorials

## Required Data

### Sample Datasets

We provide sample datasets for all tutorials:

- **Sentinel-1 SLC**: Single Look Complex data for sub-look analysis
- **Sentinel-1 GRD**: Ground Range Detected data for SNAP workflows  
- **Multi-temporal**: Time series for change detection
- **Dual-pol**: VV+VH data for polarimetric analysis
- **Maritime**: Ocean scenes for ship detection
- **Forest**: Vegetation areas for monitoring
- **Urban**: City areas for built-up analysis
- **Coastal**: Shoreline areas for change detection

### Data Download

```bash
# Download tutorial datasets (example)
wget https://github.com/ESA-PhiLab/sarpyx-tutorials/raw/main/data/tutorial_datasets.zip
unzip tutorial_datasets.zip
```

### Data Organization

```
tutorials/
├── data/
│   ├── sentinel1_slc/
│   ├── sentinel1_grd/
│   ├── multitemporal/
│   ├── dualpol/
│   ├── maritime/
│   ├── forest/
│   ├── urban/
│   └── coastal/
├── notebooks/
└── scripts/
```

## Learning Path Recommendations

### For SAR Beginners
1. Tutorial 1: First Sub-Look Analysis
2. Tutorial 3: Visualization and Quality Assessment
3. Tutorial 2: SNAP Integration Basics
4. Tutorial 5: Polarimetric Analysis

### For SNAP Users
1. Tutorial 2: SNAP Integration Basics
2. Tutorial 7: Ship Detection with CFAR
3. Tutorial 6: Custom Processing Workflows
4. Tutorial 9: Performance Optimization

### For Research Applications
1. Tutorial 4: Multi-temporal Analysis
2. Tutorial 5: Polarimetric Analysis
3. Tutorial 10: Forest Monitoring
4. Tutorial 8: Interferometric Processing

### For Production Systems
1. Tutorial 6: Custom Processing Workflows
2. Tutorial 9: Performance Optimization
3. Tutorial 7: Ship Detection with CFAR
4. Tutorial 12: Coastal Monitoring

## Interactive Formats

### Jupyter Notebooks

All tutorials are available as Jupyter notebooks with:
- Interactive code cells
- Embedded visualizations
- Progressive difficulty
- Built-in exercises

```python
# Run tutorials interactively
jupyter notebook tutorials/01_first_sublook_analysis.ipynb
```

### Python Scripts

Standalone scripts for automation:
```python
# Run complete tutorial as script
python tutorials/scripts/01_first_sublook_analysis.py
```

### Google Colab

Cloud-based execution for users without local setup:
- Pre-configured environment
- Sample data included
- GPU acceleration available

## Community Contributions

### Contributing New Tutorials

We welcome community contributions! Guidelines:

1. **Topic Relevance**: Should demonstrate SARPyX capabilities
2. **Clear Structure**: Follow established tutorial format
3. **Tested Code**: All code must run successfully
4. **Documentation**: Include clear explanations
5. **Data**: Provide or link to required datasets

### Tutorial Templates

```python
# Tutorial template structure
"""
Tutorial: [Title]

Overview: [What users will learn]
Prerequisites: [Required knowledge/tutorials]
Duration: [Estimated time]
Data: [Required datasets]
"""

# Setup section
import sarpyx
import numpy as np
import matplotlib.pyplot as plt

# Main tutorial content
# ... step-by-step instructions ...

# Exercises section
# ... practice problems ...
```

## Troubleshooting

### Common Issues

1. **Data Access**: Ensure tutorial datasets are downloaded
2. **Dependencies**: Check all required packages are installed
3. **Memory**: Some tutorials require substantial RAM
4. **SNAP**: Verify SNAP installation for integration tutorials

### Getting Help

- **Tutorial Issues**: [GitHub Issues](https://github.com/ESA-PhiLab/sarpyx/issues)
- **General Questions**: [GitHub Discussions](https://github.com/ESA-PhiLab/sarpyx/discussions)
- **Community Forum**: [SARPyX Community](https://community.sarpyx.org)

## Additional Resources

- **[Examples](../examples/README.md)**: Ready-to-run code examples
- **[API Reference](../api/README.md)**: Detailed function documentation
- **[User Guide](../user_guide/README.md)**: Comprehensive usage guide
- **[Developer Guide](../developer_guide/README.md)**: Contributing information

Start your learning journey with [Tutorial 1: Your First Sub-Look Analysis](01_first_sublook_analysis.md)!
