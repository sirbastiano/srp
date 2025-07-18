#!/bin/bash
clear
source /Data_large/marine/PythonProjects/SAR/sarpyx/.venv/bin/activate

SCRIPT="/Data_large/marine/PythonProjects/SAR/sarpyx/pyscripts/decode_single.py"
LOGFILE="/Data_large/marine/PythonProjects/SAR/sarpyx/logs/decode.log"
OUTPUT_DIR="/Data_large/marine/PythonProjects/SAR/sarpyx/data/2_decoded"
set -euo pipefail


input_file="$1"
filename="$(basename "$input_file")"


echo "[$(date)] Starting decode" | tee -a "$LOGFILE"
pdm run "$SCRIPT" --input $1 --output "$OUTPUT_DIR/$filename" >> "$LOGFILE" 2>&1
EXIT_CODE=$?
echo "[$(date)] Finished with exit code $EXIT_CODE" | tee -a "$LOGFILE"
exit $EXIT_CODE