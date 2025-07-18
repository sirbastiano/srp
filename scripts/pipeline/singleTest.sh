#!/bin/bash
cd /Data_large/marine/PythonProjects/SAR/sarpyx/data/maya4ps

set -euo pipefail
set -x  # Enable debug tracing

LOGFILE="/Data_large/marine/PythonProjects/SAR/sarpyx/logs/decoder/full_log_$1.log"
BASE_PATH="/Data_large/marine/PythonProjects/SAR/sarpyx/scripts"

# Validate required argument
if [[ -z "${1:-}" ]]; then
    echo "Error: Missing required argument" >&2
    exit 1
fi

TARGET_ID="$1"

echo "=== PIPELINE TEST MODE ==="
echo "Input TARGET_ID: $TARGET_ID"
echo "LOGFILE: $LOGFILE"
echo "BASE_PATH: $BASE_PATH"
echo "Current directory: $(pwd)"

load_zarr_paths() {
    local search_dir="$1"
    echo "load_zarr_paths() input: $search_dir"
    [[ -d "$search_dir" ]] || { echo "Directory not found: $search_dir" >&2; exit 1; }
    
    local zarr_files
    zarr_files=$(find "$search_dir" -maxdepth 1 -type d -name '*.zarr')
    [[ -n "$zarr_files" ]] || { echo "No .zarr files found in: $search_dir" >&2; exit 1; }
    
    echo "load_zarr_paths() output:"
    echo "$zarr_files"
}

# 1. Download
echo "=== STEP 1: DOWNLOAD ==="
download_input="$TARGET_ID"
echo "Download input: $download_input"
echo "Would execute: source '$BASE_PATH/down/single.sh' '$download_input'"

# 2. Decode
echo "=== STEP 2: DECODE ==="
decode_input="/Data_large/marine/PythonProjects/SAR/sarpyx/data/1_downloaded/$TARGET_ID"
echo "Decode input: $decode_input"
echo "Would execute: source '$BASE_PATH/decode/single.sh' '$decode_input'"

# 3. Focus
echo "=== STEP 3: FOCUS ==="
decoded_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data/2_decoded/$TARGET_ID"
echo "Focus input directory: $decoded_dir"

# Test the load_zarr_paths function
if [[ -d "$decoded_dir" ]]; then
    zarr_files=$(load_zarr_paths "$decoded_dir")
    echo "Found zarr files:"
    while IFS= read -r zarr_file; do
        echo "  - $zarr_file"
        zarr_basename=$(basename "$zarr_file")
        echo "    basename: $zarr_basename"
        echo "    Would execute: source '$BASE_PATH/focus/single.sh' '$zarr_file'"
    done <<< "$zarr_files"
else
    echo "Decoded directory does not exist: $decoded_dir"
    echo "Creating mock zarr files for testing..."
    echo "Mock zarr file 1: $decoded_dir/mock1.zarr"
    echo "Mock zarr file 2: $decoded_dir/mock2.zarr"
    echo "Would execute: source '$BASE_PATH/focus/single.sh' '$decoded_dir/mock1.zarr'"
    echo "Would execute: source '$BASE_PATH/focus/single.sh' '$decoded_dir/mock2.zarr'"
fi

echo "=== PIPELINE TEST COMPLETED ==="
echo "Pipeline would have completed successfully for: $TARGET_ID"
echo "=== PIPELINE TEST COMPLETED ==="
echo "Pipeline would have completed successfully for: $TARGET_ID"

