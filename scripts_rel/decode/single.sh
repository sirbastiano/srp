#!/bin/bash
set -euo pipefail

source /workspace/.venv/bin/activate

SCRIPT="/workspace/pyscripts/decode_single.py"
LOGFILE="/workspace/logs/decode.log"
OUTPUT_DIR="/workspace/data/2_decoded"

# 1) make dir if not exists
mkdir -p "$OUTPUT_DIR"  
mkdir -p "$(dirname "$LOGFILE")"


# ============================================================================
# Validate required argument
if [[ -z "${1:-}" ]]; then
    echo "Error: Missing required argument" >&2
    echo "Usage: $0 <input_file>" >&2
    exit 1
fi

input_file="$1"
filename="$(basename "$input_file")"

# Assert input file exists
if [[ ! -f "$input_file" ]]; then
    echo "Error: Input file '$input_file' does not exist." >&2
    exit 2
fi

echo "[$(date)] Starting decode" | tee -a "$LOGFILE"
pdm run "$SCRIPT" --input "$input_file" --output "$OUTPUT_DIR/$filename" >> "$LOGFILE" 2>&1
EXIT_CODE=$?
echo "[$(date)] Finished with exit code $EXIT_CODE" | tee -a "$LOGFILE"