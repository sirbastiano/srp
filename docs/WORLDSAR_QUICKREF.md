# WorldSAR CLI Quick Reference

## Simple Usage (3 Required Arguments)

```bash
sarpyx worldsar --input <product> --mode <mode> --wkt "<polygon>"
```

## Supported Modes

| Mode | Satellite | Description |
|------|-----------|-------------|
| `S1TOPS` | Sentinel-1 | TOPS mode (IW/EW) |
| `S1STRIP` | Sentinel-1 | Stripmap mode |
| `BM` | BIOMASS | BIOMASS mission |
| `NISAR` | NISAR | NISAR mission |
| `TSX` | TerraSAR-X | TerraSAR-X |
| `CSG` | COSMO-SkyMed | COSMO-SkyMed |

## Common Options

```bash
# Custom output directories
sarpyx worldsar -i product.SAFE -m S1TOPS -w "POLYGON(...)" \
                -o ./processed -t ./tiles

# Increase memory for large products
sarpyx worldsar -i product.SAFE -m S1TOPS -w "POLYGON(...)" \
                --gpt-memory 24G --gpt-parallelism 8

# Use configuration file
sarpyx worldsar --config my_config.yaml

# Skip preprocessing (if already done)
sarpyx worldsar -i product.SAFE -m S1TOPS -w "POLYGON(...)" \
                --skip-preprocessing

# Create metadata database
sarpyx worldsar -i product.SAFE -m S1TOPS -w "POLYGON(...)" \
                --create-db
```

## Configuration File Example

Create `config.yaml`:
```yaml
input: "product.SAFE"
mode: "S1TOPS"
wkt: "POLYGON((10 45, 11 45, 11 46, 10 46, 10 45))"
output: "./processed"
tiles: "./tiles"
gpt_memory: "16G"
create_database: true
```

Run:
```bash
sarpyx worldsar --config config.yaml
```

## Before (Complex - 23 Arguments)

```bash
# OLD WAY - Too many required arguments!
sarpyx worldsar \
  --input product.SAFE \
  --output ./output \
  --cuts-outdir ./tiles \
  --product-wkt "POLYGON(...)" \
  --prod-mode S1TOPS \
  --gpt-path /usr/bin/gpt \
  --grid-path ./grid.geojson \
  --db-dir ./db \
  --gpt-memory 24G \
  --gpt-parallelism 4 \
  --snap-userdir ~/.snap \
  --orbit-type "Sentinel Precise (Auto Download)" \
  --orbit-continue-on-fail \
  --orbit-download-type POEORB \
  --orbit-years 2024 \
  --orbit-satellites S1A,S1B,S1C \
  --orbit-base-url https://step.esa.int/auxdata/orbits/Sentinel-1 \
  --orbit-outdir ./orbits \
  --prefetch-orbits \
  --use-graph
```

## After (Simple - 3 Required, Smart Defaults)

```bash
# NEW WAY - Only 3 required arguments!
sarpyx worldsar \
  --input product.SAFE \
  --mode S1TOPS \
  --wkt "POLYGON(...)"
```

With optional customization:
```bash
sarpyx worldsar \
  --input product.SAFE \
  --mode S1TOPS \
  --wkt "POLYGON(...)" \
  --gpt-memory 24G \
  --gpt-parallelism 4
```

Or use a config file for complex setups:
```bash
sarpyx worldsar --config production.yaml
```

## Full Documentation

See [WORLDSAR_GUIDE.md](WORLDSAR_GUIDE.md) for complete documentation.
