# SSM4SAR Quick Start Guide

## Overview
SSM4SAR is ready to use with the existing environment at `/Data_large/marine/PythonProjects/SAR/sarpyx/.venv`. All dependencies are installed and configured.

## Quick Setup Verification

Run the dependency check:
```bash
cd /Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR
./check_dependencies.sh
```

## Training Options

### 1. Single Training Run

Basic training with default parameters:
```bash
./run_train.sh
```

Training with custom parameters:
```bash
./run_train.sh --epochs 20 --batch-size 16 --learning-rate 0.001
```

Training with Weights & Biases monitoring:
```bash
./run_train.sh --epochs 15 --wandb --wandb-project ssm4sar
```

### 2. Parameter Sweeps with Nextflow

Run automated hyperparameter sweeps:
```bash
# Basic sweep
nextflow run sweep.nf --outdir results/sweep_$(date +%Y%m%d)

# Sweep with W&B logging
nextflow run sweep.nf --outdir results/sweep_$(date +%Y%m%d) \
  --use_wandb true --wandb_project ssm4sar_sweep
```

### 3. Available Parameters

| Parameter | Short | Default | Description |
|-----------|--------|---------|-------------|
| `--epochs` | `-e` | 15 | Number of training epochs |
| `--batch-size` | `-b` | 10 | Training batch size |
| `--learning-rate` | `-lr` | 0.005 | Learning rate |
| `--layers` | `-l` | 4 | Number of SSM layers |
| `--hidden-size` | `-hs` | 8 | Hidden state dimension |
| `--activation` | `-a` | leakyrelu | Activation function |
| `--weight-decay` | `-wd` | 0.01 | Weight decay |
| `--ssim-prop` | `-sp` | 0.5 | SSIM loss proportion |
| `--gpu` | `-g` | 1 | Number of GPUs |
| `--wandb` | - | false | Enable W&B logging |

## Weights & Biases Setup

1. Login to W&B (first time only):
```bash
source /Data_large/marine/PythonProjects/SAR/sarpyx/.venv/bin/activate
wandb login
```

2. Run training with W&B:
```bash
./run_train.sh --wandb --wandb-project your_project_name
```

## Data Structure

The system expects data in this format:
```
maya4_data/
├── training/           # ✓ Found (3 zarr files)
│   └── *.zarr/
│       ├── rc/        # Range compressed (input)
│       └── az/        # Azimuth compressed (target)
└── validation/        # ✓ Found (3 zarr files)
    └── *.zarr/
        ├── rc/
        └── az/
```

## Environment Status

✅ Virtual environment: `/Data_large/marine/PythonProjects/SAR/sarpyx/.venv`
✅ Python packages: All required packages installed
✅ Nextflow: v25.04.7 installed and working
✅ Training data: 3 zarr files available
✅ Validation data: 3 zarr files available
⚠️  W&B: Available but needs login (`wandb login`)

## Quick Test

Run a short training test:
```bash
./run_train.sh --epochs 1 --batch-size 5
```

## Output Structure

Training results are saved to:
```
experiment_1/
├── model_1/
│   ├── version_0/
│   │   ├── checkpoints/
│   │   ├── events.out.tfevents.*
│   │   └── hparams.yaml
│   ├── model              # Full model
│   ├── model_weights       # State dict
│   └── original_script.py  # Script backup
```

## Troubleshooting

1. **Permission denied**: Ensure scripts are executable
   ```bash
   chmod +x run_train.sh check_dependencies.sh
   ```

2. **GPU not available**: The system reports 1 GPU but CUDA unavailable
   - Training will fall back to CPU
   - For GPU training, ensure CUDA drivers are properly installed

3. **W&B login**: If you see login warnings
   ```bash
   source /Data_large/marine/PythonProjects/SAR/sarpyx/.venv/bin/activate
   wandb login
   ```

4. **Memory issues**: Reduce batch size
   ```bash
   ./run_train.sh --batch-size 5 --epochs 10
   ```
