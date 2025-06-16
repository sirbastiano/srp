#!/bin/bash

# Configuration
CURRENT_SECTION=1
TOTAL_SECTIONS=3
FOCUS_SCRIPT="/Data_large/marine/PythonProjects/SAR/sarpyx/pyscripts/focusing.py"
LOG_FILE="/Data_large/marine/PythonProjects/SAR/sarpyx/logs/focus_errors_section_${CURRENT_SECTION}.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Initialize log file
echo "Focus processing errors for section $CURRENT_SECTION - $(date)" > "$LOG_FILE"
echo "=================================================" >> "$LOG_FILE"

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
echo "Error log: $LOG_FILE"

# Counters for statistics
success_count=0
error_count=0

# Process files in current section
for ((i=start_idx; i<=end_idx && i<total_files; i++)); do
    zarr_file="${zarr_files[i]}"
    echo "Processing ($((i + 1))/$total_files): $zarr_file"
    
    # Run the focusing script and capture exit code
    if pdm run "$FOCUS_SCRIPT" "$zarr_file"; then
        ((success_count++))
        echo "✓ Success: $zarr_file"
    else
        exit_code=$?
        ((error_count++))
        echo "✗ Failed: $zarr_file (exit code: $exit_code)"
        
        # Log the error
        {
            echo ""
            echo "ERROR - $(date)"
            echo "File: $zarr_file"
            echo "Exit code: $exit_code"
            echo "Command: pdm run $FOCUS_SCRIPT $zarr_file"
            echo "---"
        } >> "$LOG_FILE"
    fi
done

# Final statistics
echo ""
echo "Section $CURRENT_SECTION completed"
echo "Success: $success_count files"
echo "Errors: $error_count files"

# Add summary to log file
{
    echo ""
    echo "SUMMARY - $(date)"
    echo "Total processed: $((success_count + error_count))"
    echo "Successful: $success_count"
    echo "Failed: $error_count"
    echo "================================================="
} >> "$LOG_FILE"

if [ $error_count -gt 0 ]; then
    echo "⚠️  $error_count errors occurred. Check log file: $LOG_FILE"
    exit 1
else
    echo "✅ All files processed successfully"
    exit 0
fi