# SSM4SAR: State Space Models for SAR Processing

A PyTorch Lightning-based framework for training State Space Models (SSMs) on Synthetic Aperture Radar (SAR) data, specifically designed for range compression to azimuth compression transformation.

## Overview

This project implements a State Space Model architecture based on S4D (Structured State Space for Deep Sequence Modeling) for processing SAR data. The model learns to transform range-compressed SAR data to azimuth-compressed format, effectively performing azimuth focusing.

### Key Features

- **S4D Architecture**: Implements diagonal structured state space models optimized for long sequences
- **SAR Data Processing**: Custom data loaders for zarr-based SAR data with complex-valued support
- **Lightning Integration**: Full PyTorch Lightning support for distributed training and logging
- **Parameter Sweeps**: Nextflow integration for automated hyperparameter optimization
- **Experiment Tracking**: Weights & Biases integration for comprehensive monitoring

## Architecture

### Model Components

1. **Position Embedding**: Linear layer to mix 3-channel input (I, Q, position) to 2 channels
2. **SSM Layers**: Configurable number of S4D layers for sequence modeling
3. **Output Layer**: Final linear transformation to produce azimuth-focused output

### Loss Functions

The model uses a weighted combination of:
- Mean Absolute Error (MAE)
- Structural Similarity Index (SSIM)
- Mean Squared Error (MSE)
- Huber Loss
- Edge Loss

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Anaconda/Miniconda

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SSM4SAR
```

2. Create and activate environment:
```bash
conda create -n ssm4sar python=3.11
conda activate ssm4sar
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install lightning tensorboard wandb nextflow
pip install zarr xarray numpy scipy scikit-image einops opt_einsum
pip install kornia pykeops
```

## Data Format

The model expects SAR data in zarr format with the following structure:
```
data_directory/
├── training/
│   └── s1a-*.zarr/
│       ├── rc/  # Range compressed data
│       └── az/  # Azimuth compressed data (ground truth)
└── validation/
    └── s1a-*.zarr/
        ├── rc/
        └── az/
```

## Usage

### Basic Training

Run a single training experiment:

```bash
python main.py --epochs 15 --batch_size 10 --learning_rate 0.005
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--directory` | `-dn` | `experiment_1` | Experiment directory name |
| `--model_name` | `-mn` | `model_1` | Model name for logging |
| `--gpu_no` | `-gpu` | `1` | Number of GPUs to use |
| `--num_layers` | `-nl` | `4` | Number of SSM layers |
| `--hidden_state_size` | `-hs` | `8` | Hidden state dimension |
| `--epochs` | `-ep` | `3` | Number of training epochs |
| `--batch_size` | `-bs` | `10` | Training batch size |
| `--valid_batch_size` | `-vb` | `10` | Validation batch size |
| `--learning_rate` | `-lr` | `0.005` | Learning rate |
| `--weight_decay` | `-wd` | `0.01` | Weight decay |
| `--ssim` | `-sp` | `0.5` | SSIM loss proportion |
| `--act_fun` | `-af` | `leakyrelu` | Activation function |

### Using the Training Script

```bash
# Make script executable
chmod +x run_train.sh

# Run training
./run_train.sh
```

## Parameter Sweeps with Nextflow

### Installing Nextflow

```bash
# Install Nextflow
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/

# Verify installation
nextflow -version
```

### Running Parameter Sweeps

The project includes Nextflow workflows for systematic parameter exploration:

```bash
# Run hyperparameter sweep
nextflow run sweep.nf --outdir results/sweep_$(date +%Y%m%d)
```

### Sweep Configuration

Edit `nextflow.config` to customize parameter ranges:

```groovy
params {
    learning_rates = [0.001, 0.005, 0.01]
    batch_sizes = [8, 16, 32]
    num_layers = [2, 4, 8]
    hidden_sizes = [8, 16, 32]
}
```

## Weights & Biases Integration

### Setup

1. Install wandb:
```bash
pip install wandb
```

2. Login:
```bash
wandb login
```

3. Initialize in your training script:
```python
import wandb
from lightning.pytorch.loggers import WandbLogger

# In main.py
wandb_logger = WandbLogger(
    project="ssm4sar",
    name=f"{model_name}_{exp_dir}",
    config=arguments
)

trainer = pl.Trainer(
    max_epochs=num_epochs,
    logger=wandb_logger,
    # ... other args
)
```

### Monitoring

Track the following metrics:
- Training/Validation Loss (MAE, MSE, SSIM)
- Learning rates for different parameter groups
- Model weights and gradients
- Sample predictions vs ground truth

## Model Details

### S4D Implementation

The model uses a diagonal parameterization of structured state space models:

```
x'(t) = Ax(t) + Bu(t)
y(t) = Cx(t) + Du(t)
```

Where A is diagonal, enabling efficient computation via FFT convolutions.

### Hyperparameters

**Architecture:**
- `num_layers`: Number of S4D blocks (2-8 recommended)
- `d_state`: Hidden state dimension (8-64)
- `activation`: Activation function between layers

**Training:**
- `lr`: General learning rate (0.001-0.01)
- `weight_decay`: L2 regularization (0.01-0.1)
- `ssim_proportion`: Balance between MAE and SSIM loss

**Data:**
- `patch_size`: Sequence length and width (10000, 1)
- `stride`: Overlap between patches
- `batch_size`: Sequences per batch

## Output Structure

Training produces:
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

### Common Issues

1. **CUDA Memory Errors**:
   - Reduce batch size or sequence length
   - Use gradient checkpointing
   - Enable mixed precision training

2. **NaN Losses**:
   - Check learning rate (try lower values)
   - Verify data preprocessing
   - Monitor gradient norms

3. **Slow Training**:
   - Increase batch size if memory allows
   - Use more GPUs with DDP
   - Optimize data loading (increase num_workers)

### Performance Optimization

```python
# Enable optimizations
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

# Use mixed precision
trainer = pl.Trainer(precision="16-mixed")

# Data loading optimization
data_module = SARDataModule(
    num_workers=min(16, os.cpu_count()),
    pin_memory=True,
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Add tests if applicable
5. Submit a pull request

## License

[Add license information]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ssm4sar2025,
  title={SSM4SAR: State Space Models for SAR Processing},
  author={[Author names]},
  journal={[Journal]},
  year={2025}
}
```

## References

- [S4: Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Weights & Biases](https://docs.wandb.ai/)
- [Nextflow Documentation](https://www.nextflow.io/docs/latest/)
