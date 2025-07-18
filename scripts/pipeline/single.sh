#!/bin/bash

set -euo pipefail

LOGFILE="/Data_large/marine/PythonProjects/SAR/sarpyx/logs/decoder/full_log_$1.log"
BASE_PATH="/Data_large/marine/PythonProjects/SAR/sarpyx/scripts"

# Validate required argument
if [[ -z "${1:-}" ]]; then
    echo "Error: Missing required argument" >&2
    exit 1
fi

TARGET_ID="$1"

load_zarr_paths() {
    local search_dir="$1"
    [[ -d "$search_dir" ]] || { echo "Directory not found: $search_dir" >&2; exit 1; }
    
    local zarr_files
    zarr_files=$(find "$search_dir" -maxdepth 1 -type d -name '*.zarr')
    [[ -n "$zarr_files" ]] || { echo "No .zarr files found in: $search_dir" >&2; exit 1; }
    
    echo "$zarr_files"
}

#------------- 1. Download
echo "Starting download for: $TARGET_ID"
source "$BASE_PATH/down/single.sh" "$TARGET_ID"


#------------- 2. Decode
echo "Starting decode for: $TARGET_ID"
source "$BASE_PATH/decode/single.sh" "/Data_large/marine/PythonProjects/SAR/sarpyx/data/1_downloaded/$TARGET_ID"
rm -rf /Data_large/marine/PythonProjects/SAR/sarpyx/data/1_downloaded/$TARGET_ID # Cleanup Down

# Fix: Use correct path and add error handling
decoded_json_path="/Data_large/marine/PythonProjects/SAR/sarpyx/data/2_decoded/$TARGET_ID/*.json"
if ls $decoded_json_path 1> /dev/null 2>&1; then
    cp $decoded_json_path "/Data_large/marine/PythonProjects/SAR/sarpyx/data/map/"
else
    echo "Warning: No JSON files found in decoded directory: $decoded_json_path" >&2
fi


#------------- 3. Focus
echo "Starting focus processing for: $TARGET_ID"
decoded_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data/2_decoded/$TARGET_ID"
zarr_files=$(load_zarr_paths "$decoded_dir")

# Processing
while IFS= read -r zarr_file; do
    echo "Processing zarr file: $zarr_file"
    zarr_basename=$(basename "$zarr_file")
    source "$BASE_PATH/focus/single.sh" "$zarr_file"
done <<< "$zarr_files"

# Cleanup Decoded:
if [[ -d "/Data_large/marine/PythonProjects/SAR/sarpyx/data/2_decoded/$TARGET_ID" ]]; then
    rm -rf "/Data_large/marine/PythonProjects/SAR/sarpyx/data/2_decoded/$TARGET_ID"
    echo "Cleaned up decoded directory for: $TARGET_ID"
else
    echo "Warning: Decoded directory not found for cleanup: $TARGET_ID" >&2
fi

#------------- 4. Upload
echo "Starting upload for: $TARGET_ID"
upload_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data/3_parsed/$TARGET_ID"
source "$BASE_PATH/up/single.sh" "$upload_dir"

# Cleanup with error handling
if [[ -d "$upload_dir" ]]; then
    rm -rf "$upload_dir"
    echo "Cleaned up parsed directory for: $TARGET_ID"
else
    echo "Warning: Parsed directory not found for cleanup: $upload_dir" >&2
fi


#------------- 4. Finish
echo "Pipeline completed successfully for: $TARGET_ID"
exit 0