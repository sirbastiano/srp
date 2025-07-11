#!/bin/bash
clear
# Usage: focus_single.sh <zarr_file_path>
# Processes a single .zarr file using the focusing script and logs errors.

if [ $# -ne 1 ]; then
    echo "Usage: $0 <zarr_file_path>"
    exit 2
fi

zarr_file="$1"
EXECUTABLE=/Data_large/marine/PythonProjects/SAR/sarpyx/.venv/bin/python
FOCUS_SCRIPT='/Data_large/marine/PythonProjects/SAR/sarpyx/pyscripts/focusing.py'
LOG_FILE="/Data_large/marine/PythonProjects/SAR/sarpyx/logs/focus_errors_single.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Initialize log file if not exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Focus processing errors - $(date)" > "$LOG_FILE"
    echo "=================================================" >> "$LOG_FILE"
fi

echo "Processing: $zarr_file"
echo "Error log: $LOG_FILE"

# Run the focusing script and capture exit code
if $EXECUTABLE "$FOCUS_SCRIPT" "$zarr_file" --output-dir "/Data_large/marine/PythonProjects/SAR/sarpyx/data/focused_data"; then
    echo "✓ Success: $zarr_file"
    exit 0
else
    exit_code=$?
    echo "✗ Failed: $zarr_file (exit code: $exit_code)"
    {
        echo ""
        echo "ERROR - $(date)"
        echo "File: $zarr_file"
        echo "Exit code: $exit_code"
        echo "Command: pdm run $FOCUS_SCRIPT $zarr_file"
        echo "---"
    } >> "$LOG_FILE"
    echo "⚠️  Error occurred. Check log file: $LOG_FILE"
    exit 1
fi
