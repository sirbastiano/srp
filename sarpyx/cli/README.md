# SARPyX CLI Tools

This directory contains command-line interface tools for SARPyX, providing easy access to SAR processing workflows from the terminal.

## Available Commands

### Main CLI Entry Point

```bash
sarpyx --help
```

The main `sarpyx` command provides access to all CLI tools through subcommands.

### Ship Detection (`shipdet`)

Detect ships in SAR data using SNAP's adaptive thresholding and object discrimination algorithms.

#### Basic Usage

```bash
# Basic ship detection
sarpyx shipdet --product-path /path/to/S1A_*.SAFE --outdir /path/to/output

# Direct command (also available)
sarpyx-shipdet --product-path /path/to/S1A_*.SAFE --outdir /path/to/output
```

#### Advanced Usage

```bash
# Custom parameters with verbose output
sarpyx shipdet \
  --product-path /path/to/S1A_IW_GRDH_1SDV_20250531T165718_20250531T165743_059441_0760E1_588E.SAFE \
  --outdir /path/to/output \
  --format GeoTIFF \
  --polarizations VV VH \
  --pfa 8.0 \
  --background-window-m 1000 \
  --guard-window-m 600 \
  --target-window-m 50 \
  --min-target-m 25 \
  --max-target-m 1000 \
  --verbose
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--product-path` | str | Required | Path to SAR product (.SAFE directory or zip) |
| `--outdir` | str | Required | Output directory for processed data |
| `--format` | str | BEAM-DIMAP | Output format (BEAM-DIMAP, GeoTIFF, NetCDF4-CF, ENVI, HDF5) |
| `--gpt-path` | str | None | Path to SNAP GPT executable |
| `--polarizations` | list | [VV] | Polarizations to process (VV, VH, HH, HV) |
| `--output-complex` | bool | False | Output complex values for calibration |
| `--pfa` | float | 6.5 | Probability of false alarm for adaptive thresholding |
| `--background-window-m` | float | 800.0 | Background window size in meters |
| `--guard-window-m` | float | 500.0 | Guard window size in meters |
| `--target-window-m` | float | 50.0 | Target window size in meters |
| `--min-target-m` | float | 50.0 | Minimum target size in meters |
| `--max-target-m` | float | 600.0 | Maximum target size in meters |
| `--skip-calibration` | bool | False | Skip calibration step |
| `--skip-discrimination` | bool | False | Skip object discrimination step |
| `--verbose` | bool | False | Enable verbose output |

#### Processing Pipeline

The ship detection pipeline consists of the following steps:

1. **Calibration** (optional): Convert digital numbers to backscatter coefficients
   - Applies radiometric calibration
   - Supports multiple polarizations
   - Can output complex or intensity values

2. **Adaptive Thresholding**: CFAR-based ship detection
   - Uses background, guard, and target windows
   - Configurable probability of false alarm (PFA)
   - Detects potential ship targets

3. **Object Discrimination** (optional): Filter detections by size
   - Removes objects outside specified size range
   - Reduces false alarms from small clutter or large land features

#### Examples

```bash
# Process Sentinel-1 data with default settings
sarpyx shipdet \
  --product-path ./data/S1A_IW_GRDH_1SDV_20250531T165718_20250531T165743_059441_0760E1_588E.SAFE \
  --outdir ./output/shipdet_default

# High sensitivity detection (lower PFA)
sarpyx shipdet \
  --product-path ./data/S1A_IW_GRDH_1SDV_20250531T165718_20250531T165743_059441_0760E1_588E.SAFE \
  --outdir ./output/shipdet_sensitive \
  --pfa 4.0 \
  --min-target-m 15 \
  --verbose

# Process dual-pol data with custom windows
sarpyx shipdet \
  --product-path ./data/S1A_IW_GRDH_1SDV_20250531T165718_20250531T165743_059441_0760E1_588E.SAFE \
  --outdir ./output/shipdet_dualpol \
  --polarizations VV VH \
  --background-window-m 1200 \
  --guard-window-m 700 \
  --target-window-m 40 \
  --format GeoTIFF

# Skip calibration (if data is already calibrated)
sarpyx shipdet \
  --product-path ./data/preprocessed_product.dim \
  --outdir ./output/shipdet_precalibrated \
  --skip-calibration \
  --format GeoTIFF
```

## Installation and Setup

### Prerequisites

1. **SNAP Installation**: The CLI tools require ESA SNAP to be installed with GPT available in your system PATH or specify the path explicitly.

2. **SARPyX Installation**: Install SARPyX with CLI tools:
   ```bash
   # From source (development)
   cd /path/to/sarpyx
   pip install -e .
   
   # From PyPI (when available)
   pip install sarpyx
   ```

### SNAP Configuration

The tools will automatically detect SNAP GPT in common locations:
- **Linux**: `/home/*/ESA-STEP/snap/bin/gpt`
- **macOS**: `/Applications/snap/bin/gpt`
- **Windows**: `gpt.exe` in PATH

You can also specify the GPT path explicitly:
```bash
sarpyx shipdet --gpt-path /custom/path/to/gpt --product-path ... --outdir ...
```

## Error Handling

The CLI tools include comprehensive error handling:

- **Input validation**: Checks for valid paths, parameters, and SNAP installation
- **Processing errors**: Captures and reports SNAP GPT errors
- **Resource monitoring**: Provides warnings for large files or unusual parameters
- **Verbose output**: Detailed progress information when `--verbose` is used

## Performance Considerations

- **Memory usage**: Large products may require significant memory
- **Processing time**: Depends on product size and complexity
- **Disk space**: Ensure adequate space for intermediate and output files
- **Parallelism**: SNAP GPT automatically uses multiple CPU cores

## Output Files

Ship detection outputs include:
- **Detection products**: Processed SAR data with detected targets
- **Metadata**: Processing parameters and statistics
- **Log files**: Processing logs and error messages (when verbose)

## Troubleshooting

### Common Issues

1. **SNAP not found**:
   ```bash
   Error: SNAP GPT not found. Please check your SNAP installation.
   ```
   Solution: Install SNAP or specify GPT path with `--gpt-path`

2. **Memory errors**:
   ```bash
   Error: Insufficient memory for processing
   ```
   Solution: Process smaller subsets or increase available memory

3. **Invalid product format**:
   ```bash
   Error: Product path does not exist or is not readable
   ```
   Solution: Check product path and ensure SNAP can read the format

### Getting Help

```bash
# General help
sarpyx --help

# Command-specific help
sarpyx shipdet --help

# Version information
sarpyx --version
```

## Future Commands

Additional CLI tools planned for future releases:
- `sarpyx calibrate`: Standalone calibration tool
- `sarpyx subset`: Extract spatial/temporal subsets
- `sarpyx coregister`: Image coregistration
- `sarpyx interferometry`: Interferometric processing
- `sarpyx timeseries`: Multi-temporal analysis
