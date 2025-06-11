#!/bin/bash

# Function to run a command in a Docker container with a local file
run_docker_command() {
    local image="$1"
    local local_file="$2"
    local container_path="${3:-/app/input}"
    local working_dir="${4:-/app}"
    shift 4
    local command=("$@")
    
    # Validate inputs
    if [[ -z "$image" ]]; then
        echo "Error: Docker image name is required" >&2
        return 1
    fi
    
    if [[ -z "$local_file" ]]; then
        echo "Error: Local file path is required" >&2
        return 1
    fi
    
    if [[ ! -f "$local_file" ]]; then
        echo "Error: Local file does not exist: $local_file" >&2
        return 1
    fi
    
    if [[ ${#command[@]} -eq 0 ]]; then
        echo "Error: Command to run is required" >&2
        return 1
    fi
    
    # Get absolute path
    local abs_path
    abs_path=$(readlink -f "$local_file")
    
    # Run Docker command
    echo "Running: docker run --rm -v $abs_path:$container_path -w $working_dir $image ${command[*]}"
    docker run --rm \
        -v "$abs_path:$container_path" \
        -w "$working_dir" \
        "$image" \
        "${command[@]}"
}

# Function to run a Python script in Docker with additional files
run_python_in_docker() {
    local script_path="$1"
    local python_image="${2:-python:3.9-slim}"
    local additional_files=("${@:3}")
    
    # Validate script file
    if [[ ! -f "$script_path" ]]; then
        echo "Error: Python script does not exist: $script_path" >&2
        return 1
    fi
    
    if [[ ! "$script_path" =~ \.py$ ]]; then
        echo "Error: File must be a Python script (.py): $script_path" >&2
        return 1
    fi
    
    # Build volume mounts
    local volume_args=()
    local abs_script_path
    abs_script_path=$(readlink -f "$script_path")
    volume_args+=(-v "$abs_script_path:/app/script.py")
    
    # Add additional files
    local counter=0
    for file in "${additional_files[@]}"; do
        if [[ -f "$file" ]]; then
            local abs_file_path
            abs_file_path=$(readlink -f "$file")
            volume_args+=(-v "$abs_file_path:/app/file_$counter")
            ((counter++))
        else
            echo "Warning: Additional file does not exist: $file" >&2
        fi
    done
    
    # Run Python script
    echo "Running Python script in Docker..."
    docker run --rm \
        "${volume_args[@]}" \
        -w /app \
        "$python_image" \
        python script.py
}

# Function to process text file with various tools
process_text_file() {
    local file_path="$1"
    local tool="${2:-wc}"
    
    if [[ ! -f "$file_path" ]]; then
        echo "Error: File does not exist: $file_path" >&2
        return 1
    fi
    
    case "$tool" in
        "wc")
            echo "Counting lines, words, and characters..."
            run_docker_command "ubuntu:20.04" "$file_path" "/app/input" "/app" wc "/app/input"
            ;;
        "awk")
            echo "Processing with AWK (adding line numbers)..."
            run_docker_command "ubuntu:20.04" "$file_path" "/app/input" "/app" awk '{print NR, $0}' "/app/input"
            ;;
        "sort")
            echo "Sorting file contents..."
            run_docker_command "ubuntu:20.04" "$file_path" "/app/input" "/app" sort "/app/input"
            ;;
        "grep")
            local pattern="${3:-the}"
            echo "Searching for pattern: $pattern"
            run_docker_command "ubuntu:20.04" "$file_path" "/app/input" "/app" grep -i "$pattern" "/app/input"
            ;;
        *)
            echo "Error: Unknown tool: $tool" >&2
            echo "Available tools: wc, awk, sort, grep" >&2
            return 1
            ;;
    esac
}

# Function to analyze CSV file with Python pandas
analyze_csv_file() {
    local csv_file="$1"
    
    if [[ ! -f "$csv_file" ]]; then
        echo "Error: CSV file does not exist: $csv_file" >&2
        return 1
    fi
    
    # Create temporary analysis script
    local temp_script="/tmp/analyze_csv_$$.py"
    cat > "$temp_script" << 'EOF'
import pandas as pd
import sys
import os

try:
    # Look for CSV file in mounted locations
    csv_paths = ['/app/input', '/app/file_0']
    csv_file = None
    
    for path in csv_paths:
        if os.path.exists(path):
            csv_file = path
            break
    
    if not csv_file:
        print('Error: No CSV file found in container')
        sys.exit(1)
    
    print(f'Analyzing file: {csv_file}')
    df = pd.read_csv(csv_file)
    
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print('\nFirst 5 rows:')
    print(df.head())
    print('\nBasic statistics:')
    print(df.describe())
    print('\nData types:')
    print(df.dtypes)
    
except Exception as e:
    print(f'Error processing CSV: {e}')
    sys.exit(1)
EOF
    
    # Run analysis with pandas
    echo "Analyzing CSV file with pandas..."
    docker run --rm \
        -v "$(readlink -f "$csv_file"):/app/input" \
        -v "$temp_script:/app/analyze.py" \
        -w /app \
        python:3.9 \
        bash -c "pip install pandas > /dev/null 2>&1 && python analyze.py"
    
    # Clean up
    rm -f "$temp_script"
}

# Function to run multiple commands on a file
batch_process_file() {
    local file_path="$1"
    shift
    local commands=("$@")
    
    if [[ ! -f "$file_path" ]]; then
        echo "Error: File does not exist: $file_path" >&2
        return 1
    fi
    
    if [[ ${#commands[@]} -eq 0 ]]; then
        echo "Error: No commands provided" >&2
        return 1
    fi
    
    echo "Batch processing file: $file_path"
    echo "Commands: ${commands[*]}"
    
    for cmd in "${commands[@]}"; do
        echo ""
        echo "=== Running: $cmd ==="
        run_docker_command "ubuntu:20.04" "$file_path" "/app/input" "/app" bash -c "$cmd /app/input"
    done
}

# Function to create a test file for demonstrations
create_test_file() {
    local filename="${1:-test_data.txt}"
    
    cat > "$filename" << 'EOF'
Hello World
This is a test file
For Docker processing
Line number four
Another line here
Final line of test data
EOF
    
    echo "Created test file: $filename"
}

# Function to create a test CSV file
create_test_csv() {
    local filename="${1:-test_data.csv}"
    
    cat > "$filename" << 'EOF'
name,age,city,salary
John,25,New York,50000
Jane,30,Los Angeles,60000
Bob,35,Chicago,55000
Alice,28,Houston,52000
Charlie,32,Phoenix,58000
EOF
    
    echo "Created test CSV file: $filename"
}

# Main execution function
main() {
    echo "Docker File Processing Script"
    echo "============================"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed or not in PATH" >&2
        exit 1
    fi
    
    # Create test files if they don't exist
    if [[ ! -f "test_data.txt" ]]; then
        create_test_file "test_data.txt"
    fi
    
    if [[ ! -f "test_data.csv" ]]; then
        create_test_csv "test_data.csv"
    fi
    
    # Example usage
    echo ""
    echo "Example 1: Word count"
    process_text_file "test_data.txt" "wc"
    
    echo ""
    echo "Example 2: Add line numbers with AWK"
    process_text_file "test_data.txt" "awk"
    
    echo ""
    echo "Example 3: Sort file contents"
    process_text_file "test_data.txt" "sort"
    
    echo ""
    echo "Example 4: Search for pattern"
    process_text_file "test_data.txt" "grep" "line"
    
    echo ""
    echo "Example 5: Batch processing"
    batch_process_file "test_data.txt" "wc -l" "wc -w" "head -3"
    
    echo ""
    echo "Example 6: CSV analysis with pandas"
    analyze_csv_file "test_data.csv"
}

# Usage function
usage() {
    cat << 'EOF'
Usage: ./docker_file_processor.sh [options]

Functions available:
  run_docker_command <image> <file> [container_path] [working_dir] <command...>
  process_text_file <file> [tool] [pattern]
  analyze_csv_file <csv_file>
  batch_process_file <file> <command1> [command2] [...]
  create_test_file [filename]
  create_test_csv [filename]

Examples:
  ./docker_file_processor.sh
  run_docker_command ubuntu:20.04 data.txt /app/input /app wc -l /app/input
  process_text_file data.txt awk
  analyze_csv_file data.csv

Tools for process_text_file: wc, awk, sort, grep
EOF
}

# Handle command line arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-main}" in
        "help"|"-h"|"--help")
            usage
            ;;
        "main"|"")
            main
            ;;
        *)
            echo "Error: Unknown command: $1" >&2
            usage
            exit 1
            ;;
    esac
fi
EOF