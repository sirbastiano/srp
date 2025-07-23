#!/bin/bash
# =======================================================================================

source ../../.venv/bin/activate
# STEP1: Download product
echo "================ Downloading: $1 ====================="

SCRIPT="../../pyscripts/down.py"
PRD_PATH="../../data/1_downloaded/"

python "$SCRIPT" --filename "$1" --output_dir $PRD_PATH

EXIT_CODE=$?
echo "================ Completed: $1 ====================="
echo ""
echo ""
echo "----------------- EXIT CODE: $EXIT_CODE -----------------"
# =======================================================================================