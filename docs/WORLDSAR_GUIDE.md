# WorldSAR CLI User Guide

The WorldSAR CLI tool processes SAR (Synthetic Aperture Radar) data from multiple satellite missions using SNAP GPT pipelines and creates tiled outputs.

## Quick Start

### Basic Usage

Process a Sentinel-1 TOPS product with minimal arguments:

```bash
sarpyx worldsar --input S1A_IW_SLC__1SDV_20240101T120000.SAFE \
                --mode S1TOPS \
                --wkt "POLYGON((10.0 45.0, 11.0 45.0, 11.0 46.0, 10.0 46.0, 10.0 45.0))"
```

This will:
- Process the input product with default settings
- Store intermediate outputs in `./worldsar_output/`
- Store tiles in `./worldsar_tiles/`

### Specify Custom Output Directories

```bash
sarpyx worldsar --input product.SAFE \
                --mode S1TOPS \
                --wkt "POLYGON(...)" \
                --output ./my_processed \
                --tiles ./my_tiles
```

### Use a Configuration File

For complex workflows, use a YAML configuration file:

```bash
sarpyx worldsar --config my_config.yaml
```

See `worldsar_config.example.yaml` for a complete configuration template.

## Supported Modes

| Mode | Description | Satellite/Mission |
|------|-------------|-------------------|
| `S1TOPS` | Sentinel-1 TOPS mode | Sentinel-1 IW/EW |
| `S1STRIP` | Sentinel-1 Stripmap mode | Sentinel-1 SM |
| `BM` | BIOMASS mission data | BIOMASS |
| `NISAR` | NISAR mission data | NISAR |
| `TSX` | TerraSAR-X data | TerraSAR-X |
| `CSG` | COSMO-SkyMed data | COSMO-SkyMed |

## Essential Arguments

### --input, -i (required)
Path to the input SAR product. Format depends on the mission:
- Sentinel-1: `.SAFE` directory
- NISAR: `.h5` file
- Others: Check mission-specific format

### --mode, -m (required)
Processing mode based on satellite type. Choose from: `S1TOPS`, `S1STRIP`, `BM`, `NISAR`, `TSX`, `CSG`

### --wkt, -w (required)
WKT (Well-Known Text) string defining the region of interest. Example:
```
"POLYGON((lon1 lat1, lon2 lat2, lon3 lat3, lon4 lat4, lon1 lat1))"
```

## Output Configuration

### --output, -o
Directory for intermediate processing outputs (default: `./worldsar_output`)

### --tiles, -t
Directory for extracted tiles (default: `./worldsar_tiles`)

### --create-db
Create a metadata database from tiles (disabled by default)

### --db-dir
Database output directory (default: from `DB_DIR` environment variable)

## Processing Options

### --skip-preprocessing
Skip the preprocessing step. Use if your product is already preprocessed.

### --skip-tiling
Skip the tiling step. Only preprocess the product without creating tiles.

### --use-graph
Use GPT graph XML pipeline instead of sequential operations. Can be faster for batch processing.

## SNAP/GPT Configuration

### --gpt-path
Path to the GPT (Graph Processing Tool) executable. Default: from `gpt_path` env var or system PATH.

### --gpt-memory
Java heap size for GPT processing (e.g., `"8G"`, `"16G"`, `"24G"`). Increase for large products.

### --gpt-parallelism
Number of tiles to process in parallel. Default: auto-detected based on system.

### --snap-userdir
SNAP user directory. Default: from `SNAP_USERDIR` environment variable or `.snap/` in project root.

## Grid Configuration

### --grid-path
Path to the grid GeoJSON file that defines the tiling system. Default: from `grid_path` environment variable.

## Orbit Configuration (Sentinel-1 Only)

### --orbit-type
Orbit file type for SNAP's Apply-Orbit-File operator. Default: `"Sentinel Precise (Auto Download)"`

### --orbit-continue-on-fail
Continue processing even if orbit file cannot be applied.

### --prefetch-orbits
Download orbit files in advance before processing. Useful for batch processing multiple products.

### --orbit-years
Years to prefetch orbit files for (e.g., `"2024"` or `"2020-2026"`). Required when using `--prefetch-orbits`.

### --orbit-satellites
Comma-separated list of satellites for orbit prefetch (default: `"S1A,S1B,S1C"`).

### --orbit-type-download
Type of orbit file to download: `POEORB` (precise) or `RESORB` (restituted). Default: `POEORB`.

### --orbit-base-url
Base URL for orbit downloads. Default: `https://step.esa.int/auxdata/orbits/Sentinel-1`

### --orbit-outdir
Directory to store downloaded orbit files. Default: `SNAP_USERDIR/auxdata/Orbits/Sentinel-1`

## Configuration File

Using a YAML configuration file is recommended for complex setups. Create a file like this:

```yaml
# Essential settings
input: "/path/to/product.SAFE"
mode: "S1TOPS"
wkt: "POLYGON((10.0 45.0, 11.0 45.0, 11.0 46.0, 10.0 46.0, 10.0 45.0))"

# Output settings
output: "./processed"
tiles: "./tiles"

# Processing options
gpt_memory: "16G"
gpt_parallelism: 4
use_graph: true

# Orbit settings (for Sentinel-1)
orbit_continue_on_fail: true
```

Then run:
```bash
sarpyx worldsar --config my_config.yaml
```

Command-line arguments override configuration file settings.

## Environment Variables

WorldSAR uses these environment variables for default paths:

- `gpt_path`: Path to GPT executable
- `grid_path`: Path to grid GeoJSON file
- `db_dir`: Database output directory
- `SNAP_USERDIR` or `snap_userdir`: SNAP user directory
- `orbit_base_url` or `ORBIT_BASE_URL`: Base URL for orbit downloads

You can set these in a `.env` file in your project root.

## Examples

### Example 1: Simple Processing

Process a Sentinel-1 product with default settings:

```bash
sarpyx worldsar \
  --input S1A_IW_SLC__1SDV_20240101T120000.SAFE \
  --mode S1TOPS \
  --wkt "POLYGON((10.0 45.0, 11.0 45.0, 11.0 46.0, 10.0 46.0, 10.0 45.0))"
```

### Example 2: High-Performance Processing

Process with increased memory and parallelism:

```bash
sarpyx worldsar \
  --input product.SAFE \
  --mode S1TOPS \
  --wkt "POLYGON(...)" \
  --gpt-memory 32G \
  --gpt-parallelism 8 \
  --use-graph
```

### Example 3: Batch Processing Setup

Prefetch orbits for multiple products:

```bash
sarpyx worldsar \
  --input first_product.SAFE \
  --mode S1TOPS \
  --wkt "POLYGON(...)" \
  --prefetch-orbits \
  --orbit-years "2024" \
  --orbit-satellites "S1A,S1B"
```

### Example 4: Create Database

Process and create a metadata database:

```bash
sarpyx worldsar \
  --input product.SAFE \
  --mode S1TOPS \
  --wkt "POLYGON(...)" \
  --create-db \
  --db-dir ./database
```

### Example 5: NISAR Processing

Process NISAR data:

```bash
sarpyx worldsar \
  --input NISAR_L1_PR_RRSD_001_001_A_128M_20240101T120000_20240101T120030.h5 \
  --mode NISAR \
  --wkt "POLYGON(...)"
```

### Example 6: Using Configuration File

Create `my_project.yaml`:
```yaml
input: "S1A_IW_SLC__1SDV_20240101T120000.SAFE"
mode: "S1TOPS"
wkt: "POLYGON((10.0 45.0, 11.0 45.0, 11.0 46.0, 10.0 46.0, 10.0 45.0))"
output: "./project_output"
tiles: "./project_tiles"
gpt_memory: "16G"
gpt_parallelism: 4
create_database: true
db_dir: "./project_db"
```

Run:
```bash
sarpyx worldsar --config my_project.yaml
```

## Troubleshooting

### "No grid points contained within the provided WKT"
- Check that your WKT polygon is valid and uses the correct coordinate system
- Ensure the grid GeoJSON file covers your area of interest
- Verify the WKT coordinates are in the correct order: longitude, latitude

### "Intermediate product does not exist"
- Check that SNAP/GPT is properly installed and configured
- Verify the `gpt_path` is correct
- Check system resources (memory, disk space)

### Memory Issues
- Increase `--gpt-memory` (e.g., from 8G to 16G or 24G)
- Reduce `--gpt-parallelism` to process fewer tiles at once
- Check available system RAM

### Orbit File Errors
- Use `--orbit-continue-on-fail` to continue despite orbit file issues
- Manually download orbit files using `--prefetch-orbits`
- Check internet connectivity for auto-download

## Getting Help

For more help on any command:
```bash
sarpyx worldsar --help
```

For general sarpyx help:
```bash
sarpyx --help
```
