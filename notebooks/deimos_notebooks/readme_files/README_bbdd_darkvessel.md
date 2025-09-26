# Dark Vessel Detection Database Creator

## Overview

This MATLAB script (`bbdd_darkvessel.m`) processes Sentinel-1 SAR (Synthetic Aperture Radar) data to create a database for dark vessel detection. It combines vessel validation data with SAR imagery to generate georeferenced tiles and metadata for machine learning training.

## What the Script Does

The script performs the following main operations:

1. **Loads vessel validation data** from `SLCvalidation.mat` containing ground truth vessel positions
2. **Processes three types of Sentinel-1 products**:
   - **GRD (Ground Range Detected)**: Preprocessed SAR images
   - **SLC (Single Look Complex)**: Raw complex SAR data
   - **OCN (Ocean)**: Ocean wind information
3. **Extracts metadata** from SAR product XML files including calibration and annotation data
4. **Generates georeferenced tiles** (512x512 pixels) from SAR data
5. **Creates vessel annotations** with bounding boxes and metadata
6. **Outputs processed data** for training dark vessel detection models

## Key Inputs

### Required Files
- `SLCvalidation.mat`: MATLAB file containing vessel validation data with columns:
  - Scene names
  - Swath numbers
  - Latitude/longitude coordinates
  - Pixel coordinates (x, y)
  - Vessel dimensions and bounding boxes

### SAR Products (Sentinel-1 SAFE format)
- **GRD Product**: `S1A_IW_GRDH_1SDV_20200928T171800_20200928T171831_034562_0405EA_F164.SAFE`
- **SLC Product**: `S1A_IW_SLC__1SDV_20200928T171759_20200928T171831_034562_0405EA_2228.SAFE`
- **OCN Product**: `S1A_IW_OCN__2SDV_20200928T171800_20200928T171831_034562_0405EA_A13B.SAFE`

### Directory Structure Expected
```
G:\repositorio\OpenSAR\Databases\Dark Vessels Detection\
├── [PRODUCT_NAME].SAFE/
│   ├── manifest.safe
│   ├── measurement/
│   ├── annotation/
│   └── calibration/
├── xmlfile/          (output directory)
├── grdfile/          (output directory)
└── slcfile/          (output directory)
```

## Key Outputs

### 1. GRD TIFF Files
- **Format**: Georeferenced TIFF images
- **Polarizations**: VV and VH
- **Size**: Variable (cropped to area of interest)
- **Naming**: `DB_OPENSAR_VD_[NUMBER]_GRD_[POL].tiff`

### 2. SLC TIFF Files
- **Format**: Complex SAR data converted to amplitude TIFF
- **Polarizations**: VV and VH
- **Size**: 512x512 pixels
- **Naming**: `DB_OPENSAR_VD_[NUMBER]_SLC_[POL].tiff`

### 3. XML Metadata Files
- **Content**: Vessel detection metadata including:
  - Corner coordinates (lat/lon)
  - Wind direction and speed
  - Number of vessels detected
  - Vessel positions and bounding boxes
- **Naming**: `DB_OPENSAR_VD_[NUMBER].xml`

## Processing Workflow

1. **Data Loading**: Load vessel validation data and SAR product metadata
2. **Swath Processing**: Process each of the 3 IW swaths separately
3. **Georeferencing**: Create coordinate transformations between pixel and geographic coordinates
4. **Vessel Matching**: Find vessels within each tile using polygon intersection
5. **Wind Data Extraction**: Extract wind information from OCN product
6. **Tile Generation**: Create 512x512 pixel tiles from SAR data
7. **Calibration**: Apply radiometric calibration to SLC data
8. **Output Generation**: Save TIFF images and XML metadata

## Technical Details

### Coordinate Systems
- **Geographic**: WGS84 latitude/longitude
- **Projected**: SAR sensor geometry (range/azimuth)
- **Pixel**: Image coordinates (samples/lines)

### Calibration
- **SLC Data**: Calibrated using calibration vectors from annotation files
- **GRD Data**: Already calibrated, applied with histogram adjustment

### Tile Size
- **Standard**: 512x512 pixels
- **Overlap**: Determined by burst boundaries in SLC data

## Dependencies

### MATLAB Toolboxes Required
- Image Processing Toolbox
- Mapping Toolbox
- Signal Processing Toolbox

### External Functions
- `xml2struct()`: XML parsing
- `struct2xml()`: XML writing
- `geotiffinfo()`, `geotiffread()`: GeoTIFF handling
- `ncread()`: NetCDF reading for ocean data

## Usage Notes

- Modify the `ruta_general` variable to point to your data directory
- Ensure all SAR products are in SAFE format
- The script processes one scene at a time
- Output directories must exist before running
- Processing time depends on data size and number of vessels

## Limitations

- Hardcoded for specific Sentinel-1 product format
- Requires MATLAB with specific toolboxes
- Memory intensive for large SAR scenes
- Limited error handling for missing files