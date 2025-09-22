#!/bin/bash
# Enhanced training script with W&B integration and parameter sweep support

set -e  # Exit immediately if a command exits with a non-zero status

# ==============================================================================
# Configuration
# ==============================================================================

# Environment and paths
VENV_PATH='/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR/.venv'
PROJECT_DIR='/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR'
SCRIPT_NAME='main_updated.py'

# Default training parameters
EPOCHS=500
MAX_STEPS=''  # If set, overrides epochs for iteration-based training
VAL_CHECK_INTERVAL=50  # Validate every N steps
MAX_STEPS_VALIDATION=''  # If set, limits validation to N steps
BATCH_SIZE=64
LEARNING_RATE=0.0001
NUM_LAYERS=4
HIDDEN_SIZE=8
ACTIVATION='gelu'
WEIGHT_DECAY=0.01
SSIM_PROPORTION=0.1

# Default normalization parameters
NORMALIZATION_SCHEME='minmax'
INPUT_NORM_SCHEME=''
OUTPUT_NORM_SCHEME=''
INPUT_MEAN=0.0
INPUT_STD=1000.0
OUTPUT_MEAN=0.0
OUTPUT_STD=5000.0
INPUT_MEDIAN=0.0
INPUT_IQR=2000.0
OUTPUT_MEDIAN=0.0
OUTPUT_IQR=8000.0
LOG_OFFSET=1e-8
LOG_SCALE=1.0
ADAPTIVE_PERCENTILE_LOW=1.0
ADAPTIVE_PERCENTILE_HIGH=99.0
SEPARATE_REAL_IMAG=false
CUSTOM_INPUT_MIN=''
CUSTOM_INPUT_MAX=''
CUSTOM_OUTPUT_MIN=''
CUSTOM_OUTPUT_MAX=''

# Experiment configuration
EXPERIMENT_NAME="$(date +%Y%m%d_%H%M%S)_experiment"
MODEL_NAME="model_bs_${BATCH_SIZE}_lr_${LEARNING_RATE}_act_${ACTIVATION}_wd_${WEIGHT_DECAY}_ssim_${SSIM_PROPORTION}"
USE_WANDB=true
WANDB_PROJECT="ssm4sar"

# GPU configuration
GPU_COUNT=0

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
    if [ -n "$MAX_STEPS" ]; then
        echo "  Mode: Iteration-based"
        echo "  Max Steps: $MAX_STEPS"
        echo "  Validation Interval: every $VAL_CHECK_INTERVAL steps"
    else
        echo "  Mode: Epoch-based"
        echo "  Epochs: $EPOCHS"
    fi
    if [ -n "$MAX_STEPS_VALIDATION" ]; then
        echo "  Validation Steps Limit: $MAX_STEPS_VALIDATION"
    fi
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
    echo ""
    echo "Normalization Configuration:"
    echo "  Default Scheme: $NORMALIZATION_SCHEME"
    echo "  Input Scheme: ${INPUT_NORM_SCHEME:-$NORMALIZATION_SCHEME}"
    echo "  Output Scheme: ${OUTPUT_NORM_SCHEME:-$NORMALIZATION_SCHEME}"
    if [ "$NORMALIZATION_SCHEME" = "standard" ] || [ "$INPUT_NORM_SCHEME" = "standard" ] || [ "$OUTPUT_NORM_SCHEME" = "standard" ]; then
        echo "  Input Mean/Std: $INPUT_MEAN / $INPUT_STD"
        echo "  Output Mean/Std: $OUTPUT_MEAN / $OUTPUT_STD"
    fi
    if [ "$NORMALIZATION_SCHEME" = "robust" ] || [ "$INPUT_NORM_SCHEME" = "robust" ] || [ "$OUTPUT_NORM_SCHEME" = "robust" ]; then
        echo "  Input Median/IQR: $INPUT_MEDIAN / $INPUT_IQR"
        echo "  Output Median/IQR: $OUTPUT_MEDIAN / $OUTPUT_IQR"
    fi
    if [ "$NORMALIZATION_SCHEME" = "log" ] || [ "$INPUT_NORM_SCHEME" = "log" ] || [ "$OUTPUT_NORM_SCHEME" = "log" ]; then
        echo "  Log Offset/Scale: $LOG_OFFSET / $LOG_SCALE"
    fi
    if [ "$NORMALIZATION_SCHEME" = "adaptive" ] || [ "$INPUT_NORM_SCHEME" = "adaptive" ] || [ "$OUTPUT_NORM_SCHEME" = "adaptive" ]; then
        echo "  Adaptive Percentiles: $ADAPTIVE_PERCENTILE_LOW - $ADAPTIVE_PERCENTILE_HIGH"
    fi
    if [ "$SEPARATE_REAL_IMAG" = true ]; then
        echo "  Separate Real/Imag: Yes"
    fi
    echo "=============================================="
}

install_dependencies() {
    echo "Installing/checking dependencies..."
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
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
    
    # Install Python dependencies (if not already installed)
    pip install -q wandb einops opt_einsum kornia pykeops || {
        echo "Warning: Some dependencies failed to install (may already be installed)"
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
        "--model_type" "sequential" 
        "--sequential_mode"
        "--epochs" "$EPOCHS"
        "--batch_size" "$BATCH_SIZE"
        "--valid_batch_size" "$BATCH_SIZE"
        "--learning_rate" "$LEARNING_RATE"
        "--num_layers" "$NUM_LAYERS"
        "--hidden_state_size" "$HIDDEN_SIZE"
        "--act_fun" "$ACTIVATION"
        "--weight_decay" "$WEIGHT_DECAY"
        "--ssim" "$SSIM_PROPORTION"
        "--gpu_no" "$GPU_COUNT"
        "--directory" "$EXPERIMENT_NAME"
        "--model_name" "$MODEL_NAME"
        "--val_check_interval" "$VAL_CHECK_INTERVAL"
    )
    
    # Add max_steps if specified (for iteration-based training)
    if [ -n "$MAX_STEPS" ]; then
        cmd_args+=("--max_steps" "$MAX_STEPS")
    fi
    
    # Add max_steps_validation if specified
    if [ -n "$MAX_STEPS_VALIDATION" ]; then
        cmd_args+=("--max_steps_validation" "$MAX_STEPS_VALIDATION")
    fi
    
    # Add normalization arguments
    cmd_args+=("--normalization_scheme" "$NORMALIZATION_SCHEME")
    
    if [ -n "$INPUT_NORM_SCHEME" ]; then
        cmd_args+=("--input_norm_scheme" "$INPUT_NORM_SCHEME")
    fi
    
    if [ -n "$OUTPUT_NORM_SCHEME" ]; then
        cmd_args+=("--output_norm_scheme" "$OUTPUT_NORM_SCHEME")
    fi
    
    # Add normalization parameters based on schemes
    if [ "$NORMALIZATION_SCHEME" = "standard" ] || [ "$INPUT_NORM_SCHEME" = "standard" ] || [ "$OUTPUT_NORM_SCHEME" = "standard" ]; then
        cmd_args+=("--input_mean" "$INPUT_MEAN")
        cmd_args+=("--input_std" "$INPUT_STD")
        cmd_args+=("--output_mean" "$OUTPUT_MEAN")
        cmd_args+=("--output_std" "$OUTPUT_STD")
    fi
    
    if [ "$NORMALIZATION_SCHEME" = "robust" ] || [ "$INPUT_NORM_SCHEME" = "robust" ] || [ "$OUTPUT_NORM_SCHEME" = "robust" ]; then
        cmd_args+=("--input_median" "$INPUT_MEDIAN")
        cmd_args+=("--input_iqr" "$INPUT_IQR")
        cmd_args+=("--output_median" "$OUTPUT_MEDIAN")
        cmd_args+=("--output_iqr" "$OUTPUT_IQR")
    fi
    
    if [ "$NORMALIZATION_SCHEME" = "log" ] || [ "$INPUT_NORM_SCHEME" = "log" ] || [ "$OUTPUT_NORM_SCHEME" = "log" ]; then
        cmd_args+=("--log_offset" "$LOG_OFFSET")
        cmd_args+=("--log_scale" "$LOG_SCALE")
    fi
    
    if [ "$NORMALIZATION_SCHEME" = "adaptive" ] || [ "$INPUT_NORM_SCHEME" = "adaptive" ] || [ "$OUTPUT_NORM_SCHEME" = "adaptive" ]; then
        cmd_args+=("--adaptive_percentile_low" "$ADAPTIVE_PERCENTILE_LOW")
        cmd_args+=("--adaptive_percentile_high" "$ADAPTIVE_PERCENTILE_HIGH")
    fi
    
    if [ "$SEPARATE_REAL_IMAG" = true ]; then
        cmd_args+=("--separate_real_imag")
    fi
    
    # Add custom min-max ranges if specified
    if [ -n "$CUSTOM_INPUT_MIN" ]; then
        cmd_args+=("--custom_input_min" "$CUSTOM_INPUT_MIN")
    fi
    
    if [ -n "$CUSTOM_INPUT_MAX" ]; then
        cmd_args+=("--custom_input_max" "$CUSTOM_INPUT_MAX")
    fi
    
    if [ -n "$CUSTOM_OUTPUT_MIN" ]; then
        cmd_args+=("--custom_output_min" "$CUSTOM_OUTPUT_MIN")
    fi
    
    if [ -n "$CUSTOM_OUTPUT_MAX" ]; then
        cmd_args+=("--custom_output_max" "$CUSTOM_OUTPUT_MAX")
    fi
    
    # Add W&B arguments if enabled
    if [ "$USE_WANDB" = true ]; then
        cmd_args+=("--use_wandb")
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

BASIC OPTIONS:
    -h, --help              Show this help message
    -e, --epochs N          Number of epochs (default: $EPOCHS)
    --max-steps N           Maximum training steps (overrides epochs for iteration-based training)
    --val-check-interval N  Validate every N steps (default: $VAL_CHECK_INTERVAL)
    --max-steps-validation N Maximum validation steps per validation run
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

NORMALIZATION OPTIONS:
    --norm-scheme STR       Default normalization scheme: minmax|standard|robust|log|adaptive (default: $NORMALIZATION_SCHEME)
    --input-norm STR        Input normalization scheme (overrides default)
    --output-norm STR       Output normalization scheme (overrides default)
    --input-mean F          Input mean for standard normalization (default: $INPUT_MEAN)
    --input-std F           Input std for standard normalization (default: $INPUT_STD)
    --output-mean F         Output mean for standard normalization (default: $OUTPUT_MEAN)
    --output-std F          Output std for standard normalization (default: $OUTPUT_STD)
    --input-median F        Input median for robust normalization (default: $INPUT_MEDIAN)
    --input-iqr F           Input IQR for robust normalization (default: $INPUT_IQR)
    --output-median F       Output median for robust normalization (default: $OUTPUT_MEDIAN)
    --output-iqr F          Output IQR for robust normalization (default: $OUTPUT_IQR)
    --log-offset F          Log offset for log normalization (default: $LOG_OFFSET)
    --log-scale F           Log scale for log normalization (default: $LOG_SCALE)
    --adapt-low F           Adaptive low percentile (default: $ADAPTIVE_PERCENTILE_LOW)
    --adapt-high F          Adaptive high percentile (default: $ADAPTIVE_PERCENTILE_HIGH)
    --separate-real-imag    Normalize real and imaginary parts separately
    --custom-input-min F    Custom input minimum for minmax normalization
    --custom-input-max F    Custom input maximum for minmax normalization
    --custom-output-min F   Custom output minimum for minmax normalization
    --custom-output-max F   Custom output maximum for minmax normalization

W&B AND MISC OPTIONS:
    --wandb                 Enable Weights & Biases logging
    --wandb-project STR     W&B project name (default: $WANDB_PROJECT)
    --sweep                 Run parameter sweep instead of single training
    --install-deps          Install missing dependencies
    
EXAMPLES:
    # Basic training
    $0 --epochs 20 --batch-size 16
    
    # Training with adaptive normalization
    $0 --norm-scheme adaptive --adapt-low 2.0 --adapt-high 98.0
    
    # Different schemes for input and output
    $0 --input-norm minmax --output-norm log --log-scale 2.0
    
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
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --val-check-interval)
            VAL_CHECK_INTERVAL="$2"
            shift 2
            ;;
        --max-steps-validation)
            MAX_STEPS_VALIDATION="$2"
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
        --norm-scheme)
            NORMALIZATION_SCHEME="$2"
            shift 2
            ;;
        --input-norm)
            INPUT_NORM_SCHEME="$2"
            shift 2
            ;;
        --output-norm)
            OUTPUT_NORM_SCHEME="$2"
            shift 2
            ;;
        --input-mean)
            INPUT_MEAN="$2"
            shift 2
            ;;
        --input-std)
            INPUT_STD="$2"
            shift 2
            ;;
        --output-mean)
            OUTPUT_MEAN="$2"
            shift 2
            ;;
        --output-std)
            OUTPUT_STD="$2"
            shift 2
            ;;
        --input-median)
            INPUT_MEDIAN="$2"
            shift 2
            ;;
        --input-iqr)
            INPUT_IQR="$2"
            shift 2
            ;;
        --output-median)
            OUTPUT_MEDIAN="$2"
            shift 2
            ;;
        --output-iqr)
            OUTPUT_IQR="$2"
            shift 2
            ;;
        --log-offset)
            LOG_OFFSET="$2"
            shift 2
            ;;
        --log-scale)
            LOG_SCALE="$2"
            shift 2
            ;;
        --adapt-low)
            ADAPTIVE_PERCENTILE_LOW="$2"
            shift 2
            ;;
        --adapt-high)
            ADAPTIVE_PERCENTILE_HIGH="$2"
            shift 2
            ;;
        --separate-real-imag)
            SEPARATE_REAL_IMAG=true
            shift
            ;;
        --custom-input-min)
            CUSTOM_INPUT_MIN="$2"
            shift 2
            ;;
        --custom-input-max)
            CUSTOM_INPUT_MAX="$2"
            shift 2
            ;;
        --custom-output-min)
            CUSTOM_OUTPUT_MIN="$2"
            shift 2
            ;;
        --custom-output-max)
            CUSTOM_OUTPUT_MAX="$2"
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
source "$VENV_PATH/bin/activate"
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


