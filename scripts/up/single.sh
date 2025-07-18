#!/bin/bash
# This script uploads the dataset to Hugging Face using the CLI.
# Ensure you have the Hugging Face CLI installed and authenticated

set -euo pipefail  # Exit on error, undefined vars, pipe failures

FOLDER=$1
# Assert that the folder argument is provided
if [[ -z "$FOLDER" ]]; then
    echo "[ERROR] No dataset folder specified. Usage: $0 <dataset_folder>"
    exit 1
fi


USER="sirbastiano94"
DATASET="Maya4"
LOG_DIR="/Data_large/marine/PythonProjects/SAR/sarpyx/logs/up"
LOG_FILE="$LOG_DIR/upload_$(date '+%Y%m%d_%H%M%S').log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to print timestamped messages
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message"
    echo "$message" >> "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check if huggingface-cli is installed
    if ! command -v huggingface-cli &> /dev/null; then
        log "❌ huggingface-cli not found. Please install it first: pip install huggingface_hub[cli]"
        exit 1
    fi
    
    # Check if user is authenticated
    if ! huggingface-cli whoami &> /dev/null; then
        log "❌ Not authenticated with Hugging Face. Please run: huggingface-cli login"
        exit 1
    fi
    
    # Check if folder exists and is not empty
    if [[ ! -d "$FOLDER" ]]; then
        log "❌ Dataset folder does not exist: $FOLDER"
        exit 1
    fi
    
    if [[ -z "$(ls -A "$FOLDER" 2>/dev/null)" ]]; then
        log "❌ Dataset folder is empty: $FOLDER"
        exit 1
    fi
    
    log "✅ All prerequisites met"
}

# Function to get folder size
get_folder_size() {
    du -sh "$FOLDER" 2>/dev/null | cut -f1 || echo "unknown"
}

log "🚀 Starting dataset upload to Hugging Face..."
log "📂 Dataset folder: $FOLDER"
log "👤 User: $USER"
log "📊 Dataset: $DATASET"
log "📏 Folder size: $(get_folder_size)"
log "📝 Log file: $LOG_FILE"

check_prerequisites

log "🔑 Authenticated as: $(huggingface-cli whoami)"

log "⬆️  Starting upload..."
if huggingface-cli upload-large-folder "$USER/$DATASET" "$FOLDER" --repo-type=dataset; then
    log "✅ Upload completed successfully!"
    log "🌐 Dataset available at: https://huggingface.co/datasets/$USER/$DATASET"
    exit 0
else
    log "❌ Upload failed. Please check the error messages above."
    exit 1
fi