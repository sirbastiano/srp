#!/bin/bash

# Function to check if required files exist
check_prerequisites() {
    local csv_file="$1"
    local single_script="$2"
    
    if [ ! -f "$csv_file" ]; then
        echo "Error: CSV file not found: $csv_file"
        exit 1
    fi
    
    if [ ! -f "$single_script" ]; then
        echo "Error: single.sh not found: $single_script"
        exit 1
    fi
    
    # Make sure single.sh is executable
    chmod +x "$single_script"
}

# Function to process a single file
process_file() {
    local filename="$1"
    local single_script="$2"
    local flags_dir="$3"
    
    echo "Processing file: $filename"
    
    # Execute the single.sh script with the filename
    bash "$single_script" "$filename" 2>&1
    
    # Capture exit status
    local exit_code=$?
    
    # Save exit code to filename.txt in flags folder
    echo "$exit_code" > "$flags_dir/${filename}.txt"
    
    # Check exit status
    if [ $exit_code -eq 0 ]; then
        echo "Successfully processed: $filename"
    else
        echo "Error processing: $filename (exit code: $exit_code)"
    fi
    
    echo "---"
    
    # Small delay to prevent resource contention
    sleep 10
}

# Function to process CSV file
process_csv() {
    local csv_file="$1"
    local single_script="$2"
    local flags_dir="$3"
    
    echo "Starting processing of files from $csv_file"
    
    # Skip header and extract Name column (3rd column)
    tail -n +2 "$csv_file" | while IFS=',' read -r col1 col2 filename rest; do
        # Remove quotes if present
        filename=$(echo "$filename" | sed 's/^"//;s/"$//')
        
        # Skip empty lines
        if [ -z "$filename" ]; then
            continue
        fi
        
        process_file "$filename" "$single_script" "$flags_dir"
    done
}

# Main execution
main() {
    clear
    
    # Set script directory
    local script_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/scripts/pipeline"
    local csv_file="$script_dir/Maya4strip.csv"
    local single_script="$script_dir/single.sh"
    local flags_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data/map/flags"
    
    # Create flags directory if it doesn't exist
    mkdir -p "$flags_dir"
    
    # Check prerequisites
    check_prerequisites "$csv_file" "$single_script"
    
    # Process CSV file
    process_csv "$csv_file" "$single_script" "$flags_dir"
    
    echo "Processing complete"
}

# Run main function
main "$@"
