#!/bin/bash
clear
source /workspace/.venv/bin/activate

SCRIPT="/workspace/pyscripts/decode_single.py"
LOGFILE="/workspace/logs/decode.log"
OUTPUT_DIR="/workspace/data/2_decoded"
set -euo pipefail


input_file="$1"
filename="$(basename "$input_file")"


echo "[$(date)] Starting decode" | tee -a "$LOGFILE"
pdm run "$SCRIPT" --input $1 --output "$OUTPUT_DIR/$filename" >> "$LOGFILE" 2>&1
EXIT_CODE=$?
echo "[$(date)] Finished with exit code $EXIT_CODE" | tee -a "$LOGFILE"