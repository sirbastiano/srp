#!/bin/bash
# Example: Launch parallel training with tmux sessions
# Usage: ./launch_tmux_training.sh

set -e

# Configuration
CONFIG_FILE="training/training_configs/s4_ssm_complex_sweep.yaml"
SAVE_DIR="./results/tmux_sweep_$(date +%Y%m%d_%H%M%S)"
TMUX_PREFIX="sar_sweep"

echo "=================================================="
echo "Launching SAR Training Sweep with Tmux"
echo "=================================================="
echo "Config:       $CONFIG_FILE"
echo "Save Dir:     $SAVE_DIR"
echo "Tmux Prefix:  $TMUX_PREFIX"
echo "=================================================="
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux is not installed"
    echo "Install with: sudo apt-get install tmux"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Launch training
python training/training_script.py \
    --config "$CONFIG_FILE" \
    --sweep \
    --parallel \
    --use_tmux \
    --tmux_prefix "$TMUX_PREFIX" \
    --save_dir "$SAVE_DIR"

echo ""
echo "=================================================="
echo "Training sessions launched!"
echo "=================================================="
echo ""
echo "Monitor sessions with:"
echo "  tmux ls"
echo ""
echo "Attach to a session:"
echo "  tmux attach -t ${TMUX_PREFIX}_0000"
echo ""
echo "Kill all training sessions:"
echo "  for session in \$(tmux ls | grep ${TMUX_PREFIX} | cut -d: -f1); do"
echo "      tmux kill-session -t \"\$session\""
echo "  done"
echo ""
echo "=================================================="
