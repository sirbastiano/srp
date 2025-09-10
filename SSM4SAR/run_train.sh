#!/bin/bash
# Enhanced training script with W&B integration and parameter sweep support

set -e  # Exit immediately if a command exits with a non-zero status

# ==============================================================================
# Configuration
# ==============================================================================

# Environment and paths
CONDA_ENV_PATH='/Data_large/marine/anaconda3/bin/activate'
PROJECT_DIR='/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR'
SCRIPT_NAME='main.py'

# Default training parameters
EPOCHS=15
BATCH_SIZE=10
LEARNING_RATE=0.005
NUM_LAYERS=4
HIDDEN_SIZE=8
ACTIVATION='leakyrelu'
WEIGHT_DECAY=0.01
SSIM_PROPORTION=0.5

# Experiment configuration
EXPERIMENT_NAME="$(date +%Y%m%d_%H%M%S)_experiment"
MODEL_NAME="ssm4sar_model"
USE_WANDB=false
WANDB_PROJECT="ssm4sar"

# GPU configuration
GPU_COUNT=1

# ==============================================================================
# Functions
# ==============================================================================

print_header() {
    echo "=============================================="
    echo "SSM4SAR Training Script"
    echo "=============================================="
    echo "Timestamp: $(date)"
    echo "Project Dir: $PROJECT_DIR"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "=============================================="
}

print_config() {
    echo "Training Configuration:"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  Layers: $NUM_LAYERS"
    echo "  Hidden Size: $HIDDEN_SIZE"
    echo "  Activation: $ACTIVATION"
    echo "  Weight Decay: $WEIGHT_DECAY"
    echo "  SSIM Proportion: $SSIM_PROPORTION"
    echo "  GPU Count: $GPU_COUNT"
    echo "  Use W&B: $USE_WANDB"
    if [ "$USE_WANDB" = true ]; then
        echo "  W&B Project: $WANDB_PROJECT"
    fi
    echo "=============================================="
}

install_dependencies() {
    echo "Installing/checking dependencies..."
    
    # Install nextflow if not present
    if ! command -v nextflow &> /dev/null; then
        echo "Installing Nextflow..."
        curl -s https://get.nextflow.io | bash
        sudo mv nextflow /usr/local/bin/ 2>/dev/null || {
            echo "Could not move nextflow to /usr/local/bin, installing locally..."
            mkdir -p ~/bin
            mv nextflow ~/bin/
            export PATH=$PATH:~/bin
        }
    fi
    
    # Install Python dependencies
    pip install -q wandb einops opt_einsum kornia pykeops || {
        echo "Warning: Some dependencies failed to install"
    }
    
    echo "Dependencies check complete."
}

setup_wandb() {
    if [ "$USE_WANDB" = true ]; then
        echo "Setting up Weights & Biases..."
        
        # Check if logged in
        if ! wandb status | grep -q "Logged in"; then
            echo "Please login to Weights & Biases:"
            wandb login
        fi
        
        echo "W&B setup complete."
    fi
}

run_training() {
    local additional_args="$1"
    
    echo "Starting training..."
    echo "Command: python $SCRIPT_NAME $additional_args"
    
    # Build command arguments
    local cmd_args=(
        "--epochs" "$EPOCHS"
        "--batch_size" "$BATCH_SIZE"
        "--learning_rate" "$LEARNING_RATE"
        "--num_layers" "$NUM_LAYERS"
        "--hidden_state_size" "$HIDDEN_SIZE"
        "--act_fun" "$ACTIVATION"
        "--weight_decay" "$WEIGHT_DECAY"
        "--ssim" "$SSIM_PROPORTION"
        "--gpu_no" "$GPU_COUNT"
        "--directory" "$EXPERIMENT_NAME"
        "--model_name" "$MODEL_NAME"
    )
    
    # Add W&B arguments if enabled
    if [ "$USE_WANDB" = true ]; then
        cmd_args+=("--use_wandb" "true")
        cmd_args+=("--wandb_project" "$WANDB_PROJECT")
        cmd_args+=("--wandb_tags" "training" "$(date +%Y%m%d)")
    fi
    
    # Add any additional arguments
    if [ -n "$additional_args" ]; then
        cmd_args+=($additional_args)
    fi
    
    # Execute training
    python "$SCRIPT_NAME" "${cmd_args[@]}"
}

run_parameter_sweep() {
    echo "Running parameter sweep with Nextflow..."
    
    # Check if nextflow.config exists
    if [ ! -f "nextflow.config" ]; then
        echo "Warning: nextflow.config not found, using default parameters"
    fi
    
    # Create output directory
    local sweep_dir="results/sweep_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$sweep_dir"
    
    # Run nextflow sweep
    nextflow run sweep.nf --outdir "$sweep_dir" --use_wandb true --wandb_project "$WANDB_PROJECT"
    
    echo "Parameter sweep completed. Results in: $sweep_dir"
}

cleanup() {
    echo "Cleaning up..."
    # Add any cleanup tasks here
    echo "Cleanup complete."
}

show_help() {
    cat << EOF
SSM4SAR Training Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -e, --epochs N          Number of epochs (default: $EPOCHS)
    -b, --batch-size N      Batch size (default: $BATCH_SIZE)
    -lr, --learning-rate F  Learning rate (default: $LEARNING_RATE)
    -l, --layers N          Number of layers (default: $NUM_LAYERS)
    -hs, --hidden-size N    Hidden state size (default: $HIDDEN_SIZE)
    -a, --activation STR    Activation function (default: $ACTIVATION)
    -wd, --weight-decay F   Weight decay (default: $WEIGHT_DECAY)
    -sp, --ssim-prop F      SSIM proportion (default: $SSIM_PROPORTION)
    -g, --gpu N             Number of GPUs (default: $GPU_COUNT)
    -n, --name STR          Experiment name (default: auto-generated)
    -m, --model STR         Model name (default: $MODEL_NAME)
    --wandb                 Enable Weights & Biases logging
    --wandb-project STR     W&B project name (default: $WANDB_PROJECT)
    --sweep                 Run parameter sweep instead of single training
    --install-deps          Install missing dependencies
    
EXAMPLES:
    # Basic training
    $0 --epochs 20 --batch-size 16
    
    # Training with W&B
    $0 --epochs 50 --wandb --wandb-project my_project
    
    # Parameter sweep
    $0 --sweep --wandb --wandb-project sweep_project
    
    # Install dependencies
    $0 --install-deps
EOF
}

# ==============================================================================
# Argument parsing
# ==============================================================================

SWEEP_MODE=false
INSTALL_DEPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -lr|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -l|--layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        -hs|--hidden-size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        -a|--activation)
            ACTIVATION="$2"
            shift 2
            ;;
        -wd|--weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        -sp|--ssim-prop)
            SSIM_PROPORTION="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_COUNT="$2"
            shift 2
            ;;
        -n|--name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --sweep)
            SWEEP_MODE=true
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Main execution
# ==============================================================================

# Set up trap for cleanup
trap cleanup EXIT

print_header

# Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
    install_dependencies
fi

# Activate environment and navigate to project directory
echo "Activating environment and setting up workspace..."
source "$CONDA_ENV_PATH"
cd "$PROJECT_DIR"

# Set environment variables for optimization
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

print_config

# Setup W&B if enabled
setup_wandb

# Run training or sweep
if [ "$SWEEP_MODE" = true ]; then
    run_parameter_sweep
else
    run_training
fi

echo "=============================================="
echo "Training completed successfully!"
echo "Experiment: $EXPERIMENT_NAME"
echo "Timestamp: $(date)"
echo "=============================================="