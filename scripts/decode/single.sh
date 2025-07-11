#!/bin/bash
clear
# decode_batch.sh
PYTHON_BIN="/Data_large/marine/PythonProjects/SAR/sarpyx/.venv/bin/python"
SCRIPT="/Data_large/marine/PythonProjects/SAR/sarpyx/pyscripts/decode_single.py"
LOGFILE="/Data_large/marine/PythonProjects/SAR/sarpyx/logs/decode_batch.log"

set -euo pipefail

echo "[$(date)] Starting decode.py" | tee -a "$LOGFILE"
"$PYTHON_BIN" "$SCRIPT" --input $1 >> "$LOGFILE" 2>&1
EXIT_CODE=$?
echo "[$(date)] Finished with exit code $EXIT_CODE" | tee -a "$LOGFILE"
exit $EXIT_CODE