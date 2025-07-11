#!/bin/bash

# Check if section number is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <section_number>"
    echo "Example: $0 2"
    exit 1
fi

# Configuration
TOTAL_SECTIONS=10
CURRENT_SECTION=$1
FOCUS_SCRIPT="/Data_large/marine/PythonProjects/SAR/sarpyx/pyscripts/focusing.py"

# Validate section number
if ! [[ "$CURRENT_SECTION" =~ ^[1-9][0-9]*$ ]] || [ "$CURRENT_SECTION" -gt "$TOTAL_SECTIONS" ]; then
    echo "Error: Section number must be between 1 and $TOTAL_SECTIONS"
    exit 1
fi

# Find all .zarr files and convert to array
mapfile -t zarr_files < <(find /Data_large/marine/PythonProjects/SAR/sarpyx/decoded_data -name "*.zarr" -type d)

# Calculate total files and section size
total_files=${#zarr_files[@]}
section_size=$((total_files / TOTAL_SECTIONS))
remainder=$((total_files % TOTAL_SECTIONS))

# Calculate start and end indices for current section
start_idx=$(((CURRENT_SECTION - 1) * section_size))
if [ $CURRENT_SECTION -le $remainder ]; then
    section_size=$((section_size + 1))
    start_idx=$((start_idx + CURRENT_SECTION - 1))
else
    start_idx=$((start_idx + remainder))
fi
end_idx=$((start_idx + section_size - 1))

echo "Processing section $CURRENT_SECTION/$TOTAL_SECTIONS"
echo "Files $((start_idx + 1)) to $((end_idx + 1)) of $total_files total files"

# Process files in current section
for ((i=start_idx; i<=end_idx && i<total_files; i++)); do
    zarr_file="${zarr_files[i]}"
    current_file_num=$((i + 1))
    echo "Processing ${current_file_num}/${total_files}: $zarr_file"
    pdm run  "$FOCUS_SCRIPT" "$zarr_file"
done

echo "Section $CURRENT_SECTION completed"