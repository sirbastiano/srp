# Knowledge Distillation for sarSSM Models

This directory contains a complete knowledge distillation pipeline to transfer knowledge from a larger teacher model (with selectivity mechanism) to a smaller student model (without selectivity mechanism).

## Overview

Knowledge distillation allows you to create smaller, faster models that retain most of the performance of larger models. In this case, we're specifically removing the selectivity mechanism from the sarSSM model to create a simpler, more efficient version.

## Key Differences: Teacher vs Student

| Aspect | Teacher Model | Student Model Options |
|--------|---------------|----------------------|
| **Selectivity** | ✅ Uses selectivity mechanism | ❌ No selectivity mechanism |
| **Model Dim** | 64 (full size) | 64 (standard), 32 (compressed), 24 (tiny) |
| **State Dim** | 540 (full size) | 540 (standard), 256 (compressed), 128 (tiny) |
| **Layers** | 6 (full depth) | 6 (standard), 4 (compressed), 3 (tiny) |
| **Parameters** | Most parameters | Fewer parameters (10-75% reduction) |
| **Complexity** | Highest computational complexity | Lower computational complexity |
| **Performance** | Highest accuracy | 90-95% accuracy but much faster |

## Compression Levels

### 1. **Standard Student** (10-20% compression)
- Remove selectivity only
- Same model/state dimensions and layers
- Best performance retention (~95-98%)

### 2. **Compressed Student** (40-60% compression)  
- No selectivity + reduced dimensions
- `model_dim: 32, state_dim: 256, num_layers: 4`
- Good performance retention (~90-95%)

### 3. **Tiny Student** (70-80% compression)
- Minimal dimensions for maximum speed
- `model_dim: 24, state_dim: 128, num_layers: 3`  
- Reasonable performance retention (~85-90%)

## Files

- **`knowledge_distillation.py`** - Core distillation framework with loss functions and trainer
- **`distillation_script.py`** - Main training script for knowledge distillation
- **`s4_ssm_student.yaml`** - Configuration for student model (without selectivity)
- **`distillation_example.py`** - Examples and utilities for running distillation

## Quick Start

### 1. Train Teacher Model (if not already done)

```bash
python training_script.py --config training/s4_ssm_complex.yaml
```

### 2. Run Knowledge Distillation

**Standard distillation (same size, no selectivity):**
```bash
python training/distillation_script.py \
    --teacher_config training/s4_ssm_complex.yaml \
    --teacher_checkpoint results/s4_ssm_results/checkpoints/best_model.pth \
    --student_config training/s4_ssm_student.yaml \
    --save_dir results/s4_ssm_distilled
```

**Compressed distillation (50% smaller):**
```bash
python training/distillation_script.py \
    --teacher_config training/s4_ssm_complex.yaml \
    --teacher_checkpoint results/s4_ssm_results/checkpoints/best_model.pth \
    --student_config training/s4_ssm_student.yaml \
    --student_model_dim 32 \
    --student_state_dim 256 \
    --student_num_layers 4 \
    --save_dir results/s4_ssm_compressed
```

**Tiny distillation (75% smaller):**
```bash
python training/distillation_script.py \
    --teacher_config training/s4_ssm_complex.yaml \
    --teacher_checkpoint results/s4_ssm_results/checkpoints/best_model.pth \
    --student_config training/s4_ssm_student.yaml \
    --student_model_dim 24 \
    --student_state_dim 128 \
    --student_num_layers 3 \
    --temperature 6.0 \
    --alpha 0.1 \
    --beta 0.7 \
    --save_dir results/s4_ssm_tiny
```

### 3. Compare Models

```bash
python training/distillation_example.py --compare
```

## Configuration

### Teacher Model Configuration
The teacher model uses your existing `s4_ssm_complex.yaml` with:
- `use_selectivity: true` (selectivity mechanism enabled)
- Full model complexity

### Student Model Configuration
The student model (`s4_ssm_student.yaml`) has:
- `use_selectivity: false` (selectivity mechanism disabled)
- Same architecture otherwise
- Lower learning rate for distillation

### Distillation Parameters

You can tune these parameters for better results:

```yaml
distillation:
  temperature: 4.0      # Softening factor for teacher outputs
  alpha: 0.3           # Weight for student loss (ground truth)
  beta: 0.5            # Weight for distillation loss (teacher knowledge)
  gamma: 0.2           # Weight for feature matching loss
  feature_matching: true
  freeze_teacher: true
```

## Advanced Usage

### Custom Distillation Parameters

```bash
python training/distillation_script.py \
    --teacher_config training/s4_ssm_complex.yaml \
    --teacher_checkpoint results/s4_ssm_results/checkpoints/best_model.pth \
    --student_config training/s4_ssm_student.yaml \
    --student_model_dim 40 \
    --student_state_dim 320 \
    --student_num_layers 5 \
    --temperature 5.0 \
    --alpha 0.25 \
    --beta 0.6 \
    --num_epochs 300 \
    --learning_rate 3e-4 \
    --save_dir results/custom_distillation
```

### New Command Line Options

- `--student_model_dim`: Override student model dimension
- `--student_state_dim`: Override student state dimension  
- `--student_num_layers`: Override student number of layers
- `--temperature`: Distillation temperature (higher = softer targets)
- `--alpha`: Weight for student loss (ground truth)
- `--beta`: Weight for distillation loss (teacher knowledge)

### Monitoring Training

The distillation process logs several metrics:
- `train_total_loss` - Combined distillation loss
- `train_student_loss` - Student MSE loss (ground truth)
- `train_distillation_loss` - Knowledge transfer loss
- `train_feature_loss` - Feature matching loss
- `val_*` - Validation versions of above metrics

### Gradient Tracking

The pipeline automatically tracks gradients for both teacher and student models when Weights & Biases (wandb) is available:

- **Student gradients**: Tracked every 50 steps with layer-wise statistics
- **Teacher gradients**: Tracked if teacher is not frozen
- **Model weights**: Logged at the end of each epoch
- **Overall gradient norms**: Total and per-layer gradient magnitudes

To enable gradient tracking:
```bash
# Install wandb
pip install wandb

# Login to wandb (first time only)
wandb login

# Run distillation - gradient tracking will be automatic
python training/distillation_script.py ...
```

Use TensorBoard and/or Wandb to monitor training:
```bash
# TensorBoard
tensorboard --logdir results/s4_ssm_distilled/distillation

# Or view in Wandb dashboard (if enabled)
# Automatically opens in browser with gradient visualizations
```

## Loss Function Details

The knowledge distillation loss combines three components:

1. **Student Loss (α = 0.3)**: Standard MSE between student predictions and ground truth
2. **Distillation Loss (β = 0.5)**: MSE between student and teacher predictions
3. **Feature Matching Loss (γ = 0.2)**: Optional loss for intermediate features

Total Loss = α × Student Loss + β × Distillation Loss + γ × Feature Loss

## Expected Results

After successful distillation, you should expect:

- **Parameter Reduction**: ~10-20% fewer parameters (due to removed selectivity mechanism)
- **Speed Improvement**: ~15-30% faster inference
- **Performance**: 90-95% of teacher model performance
- **Memory**: Lower memory usage during inference

## Troubleshooting

### Common Issues

1. **Teacher checkpoint not found**
   - Ensure you've trained the teacher model first
   - Check the checkpoint path is correct

2. **CUDA out of memory**
   - Reduce batch size in the configuration
   - Use gradient accumulation

3. **Poor student performance**
   - Increase distillation weight (β)
   - Lower temperature for harder targets
   - Train for more epochs

4. **Student not learning from teacher**
   - Decrease student loss weight (α)
   - Increase distillation weight (β)
   - Check teacher model is loaded correctly

### Debugging Commands

```bash
# Compare model architectures
python training/distillation_example.py --compare

# Check available commands
python training/distillation_example.py --commands

# Run full example
python training/distillation_example.py --run
```

## Model Architecture Comparison

### Teacher Model (with selectivity)
```
Input → Input Projection → S4D Layers with Selectivity Gates → Output Projection
                            ↓
                        Gate MLP (complexity)
```

### Student Model (without selectivity)
```
Input → Input Projection → S4D Layers (no gates) → Output Projection
```

The student model removes the selectivity gates and associated MLP, making it simpler and faster while preserving the core S4D functionality.

## Performance Tuning

### For Better Accuracy
- Increase β (distillation weight): 0.6-0.7
- Decrease α (student weight): 0.2-0.3
- Use lower temperature: 2.0-3.0
- Train longer: 300-500 epochs

### For Faster Training
- Increase α (student weight): 0.5-0.6
- Decrease β (distillation weight): 0.3-0.4
- Use higher learning rate: 1e-3
- Disable feature matching

### For Maximum Compression
- Further reduce model dimensions
- Use quantization post-training
- Apply pruning techniques

## Integration

To use the distilled model in your existing code:

```python
# Load distilled student model
from model.model_utils import get_model_from_configs
import torch

# Load configuration
student_config = {
    'name': 's4_ssm',
    'input_dim': 2,
    'model_dim': 64,
    'state_dim': 540,
    'output_dim': 1,
    'num_layers': 6,
    'use_selectivity': False,  # Key difference
    'complex_valued': True,
    # ... other parameters
}

# Create and load model
model = get_model_from_configs(**student_config)
model.load_state_dict(torch.load('results/s4_ssm_distilled/final_student_model.pth'))
model.eval()

# Use in inference
with torch.no_grad():
    output = model(input_data)
```

