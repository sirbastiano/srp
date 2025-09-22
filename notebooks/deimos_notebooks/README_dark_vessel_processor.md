# Dark Vessel Detection Database Creator - Python Implementation

## Overview

This Python script (`dark_vessel_processor.py`) is a modern implementation of the MATLAB `bbdd_darkvessel.m` script, designed to process Sentinel-1 SAR data for dark vessel detection database creation. It focuses on GRD and SLC product processing as requested.

## Key Features

- **Object-oriented design** with clean separation of concerns
- **Robust error handling** and logging
- **Flexible input/output** with configurable directories
- **Cross-platform compatibility** (Linux, Windows, macOS)
- **Optional dependencies** with fallback implementations
- **Command-line interface** for easy automation

## Dependencies

### Required Dependencies
```bash
pip install numpy pandas rasterio pathlib
```

### Optional Dependencies (with fallbacks)
```bash
pip install opencv-python scipy  # For enhanced image processing
```

## Installation

1. **Clone or download** the script to your working directory
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare your data structure** (see Data Structure section)

## Usage

### Command Line Interface

```bash
python dark_vessel_processor.py \
    --base-dir /path/to/sar/data \
    --vessel-data /path/to/vessel_validation.csv \
    --grd-product /path/to/grd_product.SAFE \
    --slc-product /path/to/slc_product.SAFE \
    --tile-id 1 \
    --tile-size 512
```

### Python API

```python
from dark_vessel_processor import DarkVesselProcessor

# Initialize processor
processor = DarkVesselProcessor(
    base_directory='/path/to/sar/data',
    tile_size=512
)

# Load vessel validation data
vessels = processor.load_vessel_data('/path/to/vessel_validation.csv')

# Process scene
success = processor.process_scene(
    grd_product='/path/to/grd_product.SAFE',
    slc_product='/path/to/slc_product.SAFE',
    vessel_data=vessels,
    tile_id=1
)
```

## Input Requirements

### 1. Directory Structure
```
base_directory/
├── grdfile/          # Output GRD tiles (created automatically)
├── slcfile/          # Output SLC tiles (created automatically)
├── xmlfile/          # Output metadata (created automatically)
└── [product_dirs]/   # Input SAR products
```

### 2. Vessel Validation Data (CSV)
**Required columns:**
- `latitude`: Vessel latitude (decimal degrees)
- `longitude`: Vessel longitude (decimal degrees)
- `x_pixel`: X pixel coordinate in SAR image
- `y_pixel`: Y pixel coordinate in SAR image
- `length_pixel`: Vessel length in pixels
- `top_pixel`: Top boundary of vessel bounding box
- `left_pixel`: Left boundary of vessel bounding box
- `bottom_pixel`: Bottom boundary of vessel bounding box
- `right_pixel`: Right boundary of vessel bounding box
- `swath`: SAR swath number (1, 2, or 3)

**Example CSV format:**
```csv
latitude,longitude,x_pixel,y_pixel,length_pixel,top_pixel,left_pixel,bottom_pixel,right_pixel,swath
45.123,-123.456,100.5,200.3,15.2,195,95,205,105,1
46.789,-124.012,150.1,300.7,20.8,290,140,310,160,2
```

### 3. SAR Products (Sentinel-1 SAFE format)
- **GRD Product**: Ground Range Detected data
- **SLC Product**: Single Look Complex data

Both products should be in standard Sentinel-1 SAFE format with:
- `manifest.safe` file
- `measurement/` directory with data files
- `annotation/` directory with metadata
- `calibration/` directory with calibration data

## Outputs

### 1. GRD TIFF Files
- **Location**: `base_directory/grdfile/`
- **Format**: `DB_OPENSAR_VD_{tile_id}_GRD_{polarization}.tiff`
- **Content**: Histogram-adjusted GRD amplitude data
- **Polarizations**: VV and VH
- **Georeferencing**: Preserved from source

**Example files:**
- `DB_OPENSAR_VD_1_GRD_VV.tiff`
- `DB_OPENSAR_VD_1_GRD_VH.tiff`

### 2. SLC TIFF Files
- **Location**: `base_directory/slcfile/`
- **Format**: `DB_OPENSAR_VD_{tile_id}_SLC_{polarization}.tiff`
- **Content**: Calibrated amplitude data from complex SLC
- **Size**: 512×512 pixels (configurable)
- **Polarizations**: VV and VH

**Example files:**
- `DB_OPENSAR_VD_1_SLC_VV.tiff`
- `DB_OPENSAR_VD_1_SLC_VH.tiff`

### 3. XML Metadata Files
- **Location**: `base_directory/xmlfile/`
- **Format**: `DB_OPENSAR_VD_{tile_id}.xml`
- **Content**: Comprehensive vessel and scene metadata

**XML Structure:**
```xml
<?xml version='1.0' encoding='utf-8'?>
<outxml>
    <ProcessingData>
        <Corner_Coord>
            <Latitude>45.123</Latitude>
            <Longitude>-123.456</Longitude>
            <!-- ... more coordinates ... -->
        </Corner_Coord>
        <WindData>
            <Direction>45.0</Direction>
            <Speed>10.5</Speed>
        </WindData>
        <StatisticsReport>
            <Number_of_ships>2</Number_of_ships>
        </StatisticsReport>
        <List_of_ships>
            <Ship>
                <Name>Ship_1</Name>
                <Centroid_Position>
                    <Latitude>45.123</Latitude>
                    <Longitude>-123.456</Longitude>
                    <SARData_Sample>100</SARData_Sample>
                    <SARData_Line>200</SARData_Line>
                </Centroid_Position>
                <Size>15.2</Size>
                <BoundingBox>
                    <Top>195</Top>
                    <Left>95</Left>
                    <Bottom>205</Bottom>
                    <Right>105</Right>
                </BoundingBox>
            </Ship>
        </List_of_ships>
    </ProcessingData>
</outxml>
```

## Key Differences from MATLAB Version

### Improvements
1. **Modern Python architecture** with classes and type hints
2. **Better error handling** with try-catch blocks and logging
3. **Flexible I/O** with configurable paths
4. **Optional dependencies** with graceful fallbacks
5. **Command-line interface** for automation
6. **Cross-platform compatibility**

### Simplifications
1. **Focused on GRD/SLC processing** as requested
2. **Simplified georeferencing** (can be enhanced for specific needs)
3. **Basic wind data handling** (placeholder implementation)
4. **Streamlined burst processing** for SLC data

## Configuration Options

### Command Line Arguments
- `--base-dir`: Base directory containing SAR products (required)
- `--vessel-data`: Path to vessel validation CSV file (required)
- `--grd-product`: Path to GRD SAFE directory (required)
- `--slc-product`: Path to SLC SAFE directory (required)
- `--tile-id`: Unique identifier for output tiles (default: 1)
- `--tile-size`: Size of output SLC tiles (default: 512)

### Class Parameters
```python
processor = DarkVesselProcessor(
    base_directory='/path/to/data',
    tile_size=512  # Output tile size for SLC data
)
```

## Performance Considerations

- **Memory usage**: Proportional to SAR image size
- **Processing time**: Depends on tile size and number of vessels
- **Disk space**: Each tile generates ~4 files (2 GRD + 2 SLC TIFF files + XML)
- **I/O optimization**: Uses rasterio for efficient geospatial data handling

## Error Handling

The script includes comprehensive error handling for:
- **Missing input files**
- **Corrupted SAR data**
- **Invalid vessel coordinates**
- **Disk space issues**
- **Permission errors**

All errors are logged with informative messages to help with debugging.

## Extending the Script

### Adding New Polarizations
Modify the polarization detection logic in `process_grd_data()` and `process_slc_data()` methods.

### Custom Calibration
Override the calibration logic in the SLC processing section to implement specific calibration algorithms.

### Additional Metadata
Extend the `create_vessel_metadata()` method to include more fields in the XML output.

## Troubleshooting

### Common Issues

1. **Import errors**: Install missing dependencies
   ```bash
   pip install rasterio numpy pandas
   ```

2. **File not found**: Check SAR product paths and SAFE directory structure

3. **Memory errors**: Reduce tile size or process smaller areas

4. **Permission errors**: Ensure write access to output directories

### Debug Mode
Enable detailed logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Example Workflow

```bash
# 1. Prepare directory structure
mkdir -p /data/sar_processing/{grdfile,slcfile,xmlfile}

# 2. Prepare vessel validation CSV
# Create vessel_validation.csv with required columns

# 3. Run processing
python dark_vessel_processor.py \
    --base-dir /data/sar_processing \
    --vessel-data /data/vessel_validation.csv \
    --grd-product /data/S1A_IW_GRDH_*.SAFE \
    --slc-product /data/S1A_IW_SLC_*.SAFE \
    --tile-id 1

# 4. Verify outputs
ls /data/sar_processing/grdfile/  # GRD TIFF files
ls /data/sar_processing/slcfile/  # SLC TIFF files  
ls /data/sar_processing/xmlfile/  # Metadata XML files
```

## License

This implementation maintains compatibility with the original MATLAB script's intended use for SAR-based vessel detection research and development.