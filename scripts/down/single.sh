#!/bin/bash
# =======================================================================================

source /Data_large/marine/PythonProjects/SAR/sarpyx/.venv/bin/activate
# STEP1: Download product
echo "================ Downloading: $1 ====================="

SCRIPT="/Data_large/marine/PythonProjects/SAR/sarpyx/pyscripts/down.py"
PRD_PATH="/Data_large/marine/PythonProjects/SAR/sarpyx/data/1_downloaded/$1"

python "$SCRIPT" --filename "$1" --output_dir $PRD_PATH

EXIT_CODE=$?
echo "================ Completed: $1 ====================="
echo ""
echo ""
echo "----------------- EXIT CODE: $EXIT_CODE -----------------"
# =======================================================================================