#!/bin/bash

# Bash script to focus Sentinel-1 TOPS mode DAT files using the C++ sentinel1_decode library.
#
# Usage: ./focus_tops_dat.sh [OPTIONS] <input_dat_file> <output_dir>
#
# This script processes Sentinel-1 Level-0 raw data (.dat files) in TOPS mode
# and generates focused SAR images using the high-performance C++ decoder.
# TOPS mode includes IW (Interferometric Wide) and EW (Extra Wide) swaths.
#
# # Focus all TOPS swaths
# ./focus_tops_dat.sh /path/to/data.dat /path/to/output
# # Focus specific IW1 swath
# ./focus_tops_dat.sh -s IW1 /path/to/data.dat /path/to/output
# # Focus specific burst 3 of IW2 swath
# ./focus_tops_dat.sh -s IW2 -b 3 /path/to/data.dat /path/to/output
# # Focus with merged bursts per swath
# ./focus_tops_dat.sh --merged-bursts /path/to/data.dat /path/to/output
# Author: RDP
# Date: $(date 2025-06-20)

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Path to the sentinel1_decode C++ library
SENTINEL1_DECODE_DIR="/Data_large/marine/PythonProjects/SAR/sarpyx/sentinel1_decode"
BUILD_DIR="${SENTINEL1_DECODE_DIR}/build"
S1_WRITE_BIN="${BUILD_DIR}/bin/s1_write"
S1_PRINT_BIN="${BUILD_DIR}/bin/s1_print"

# TOPS swath identifiers
TOPS_IW_SWATHS=("IW1" "IW2" "IW3")
TOPS_EW_SWATHS=("EW1" "EW2" "EW3" "EW4" "EW5")
ALL_TOPS_SWATHS=("${TOPS_IW_SWATHS[@]}" "${TOPS_EW_SWATHS[@]}")

# Default processing options
DEFAULT_SCALING="--norm"
DEFAULT_FORMAT="tif"
PROCESS_ALL_SWATHS=true
GENERATE_INTERMEDIATE=true
PROCESS_ALL_BURSTS=true
VERBOSE=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <input_dat_file> <output_dir>

Focus Sentinel-1 TOPS mode DAT files using C++ sentinel1_decode library.
Supports both IW (Interferometric Wide) and EW (Extra Wide) swaths.

ARGUMENTS:
    input_dat_file      Path to the input .dat file (Sentinel-1 Level-0 raw data)
    output_dir          Directory where output files will be saved

OPTIONS:
    -s, --swath SWATH           Process specific swath only (IW1, IW2, IW3, EW1-EW5)
    -b, --burst BURST           Process specific burst number (1-based index)
    -f, --format FORMAT         Output format: tif (default), cf32
    -c, --scaling SCALE         Scaling option: --norm (default), --norm_log, --mag, --real, --imag
    -i, --intermediate          Generate intermediate products (range compressed, etc.)
    --no-intermediate           Skip intermediate products generation
    --all-bursts                Process all bursts individually (default)
    --merged-bursts             Process merged bursts per swath
    -v, --verbose               Enable verbose output
    -h, --help                  Show this help message

EXAMPLES:
    # Focus all TOPS swaths with default settings
    $0 /path/to/data.dat /path/to/output

    # Focus specific IW1 swath only
    $0 -s IW1 /path/to/data.dat /path/to/output

    # Focus specific burst 3 of IW2 swath
    $0 -s IW2 -b 3 /path/to/data.dat /path/to/output

    # Focus with logarithmic scaling and verbose output
    $0 -c --norm_log -v /path/to/data.dat /path/to/output

    # Focus with merged bursts per swath
    $0 --merged-bursts /path/to/data.dat /path/to/output

    # Focus without intermediate products
    $0 --no-intermediate /path/to/data.dat /path/to/output

PROCESSING STEPS:
    1. Validate input DAT file and check available TOPS swaths
    2. Create output directory structure
    3. Extract metadata, state vectors, and burst information
    4. Generate range-compressed images (if --intermediate)
    5. Generate range-doppler images (if --intermediate)
    6. Generate focused burst images (individual or merged)
    7. Save complex data as CF32 format (if requested)

TOPS MODE SPECIFICS:
    - TOPS mode uses burst-based acquisition
    - Each swath contains multiple bursts with timing gaps
    - Burst synchronization is critical for interferometry
    - Memory usage is generally lower than stripmap mode

REQUIREMENTS:
    - sentinel1_decode C++ library must be compiled
    - Input file must be Sentinel-1 Level-0 raw data (.dat)
    - FFTW3, OpenMP, and libtiff libraries
    - Sufficient memory (typically 8-16GB for TOPS)

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if sentinel1_decode is built
    if [[ ! -f "$S1_WRITE_BIN" ]]; then
        log_error "sentinel1_decode not built. Please compile it first:"
        echo "  cd $SENTINEL1_DECODE_DIR/build"
        echo "  cmake .."
        echo "  cmake --build ."
        exit 1
    fi
    
    if [[ ! -f "$S1_PRINT_BIN" ]]; then
        log_error "s1_print binary not found at $S1_PRINT_BIN"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

validate_input_file() {
    local input_file="$1"
    
    if [[ ! -f "$input_file" ]]; then
        log_error "Input file does not exist: $input_file"
        exit 1
    fi
    
    if [[ ! "$input_file" =~ \.dat$ ]]; then
        log_warning "Input file does not have .dat extension: $input_file"
    fi
    
    # Check file size (should be substantial for SAR data)
    local file_size=$(stat -c%s "$input_file" 2>/dev/null || stat -f%z "$input_file" 2>/dev/null)
    if [[ $file_size -lt 1048576 ]]; then  # Less than 1MB
        log_warning "Input file seems small for SAR data: $(( file_size / 1024 ))KB"
    fi
    
    log_success "Input file validation passed"
}

get_available_swaths() {
    local input_file="$1"
    
    log_info "Detecting available TOPS swaths in $input_file..."
    
    # Get swath names from the file
    local swath_output
    if ! swath_output=$("$S1_PRINT_BIN" swath_names "$input_file" 2>&1); then
        log_error "Failed to get swath information from file"
        log_error "Output: $swath_output"
        exit 1
    fi
    
    log_verbose "Swath detection output: $swath_output"
    
    # Parse swath names and counts
    local available_swaths=()
    while IFS= read -r line; do
        if [[ "$line" =~ ^([^:]+):[[:space:]]*([0-9]+)$ ]]; then
            local swath_name="${BASH_REMATCH[1]}"
            local packet_count="${BASH_REMATCH[2]}"
            
            # Check if it's a TOPS swath
            for tops_swath in "${ALL_TOPS_SWATHS[@]}"; do
                if [[ "$swath_name" == "$tops_swath" ]]; then
                    if [[ $packet_count -gt 0 ]]; then
                        available_swaths+=("$swath_name")
                        log_info "Found TOPS swath: $swath_name ($packet_count packets)"
                    fi
                    break
                fi
            done
        fi
    done <<< "$swath_output"
    
    if [[ ${#available_swaths[@]} -eq 0 ]]; then
        log_error "No TOPS swaths found in the input file"
        log_info "Available swaths in file:"
        echo "$swath_output"
        exit 1
    fi
    
    printf '%s\n' "${available_swaths[@]}"
}

get_burst_info() {
    local input_file="$1"
    local swath="$2"
    
    log_verbose "Getting burst information for swath $swath..."
    
    # Get burst count for the swath
    local burst_output
    if ! burst_output=$("$S1_PRINT_BIN" burst_count "$swath" "$input_file" 2>&1); then
        log_warning "Failed to get burst information for swath $swath"
        echo "0"
        return
    fi
    
    # Extract burst count from output
    local burst_count=0
    if [[ "$burst_output" =~ ([0-9]+) ]]; then
        burst_count="${BASH_REMATCH[1]}"
    fi
    
    log_verbose "Swath $swath has $burst_count bursts"
    echo "$burst_count"
}

create_output_structure() {
    local output_dir="$1"
    
    log_info "Creating output directory structure..."
    
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/metadata"
    mkdir -p "$output_dir/focused"
    mkdir -p "$output_dir/bursts"
    
    if [[ "$GENERATE_INTERMEDIATE" == true ]]; then
        mkdir -p "$output_dir/range_compressed"
        mkdir -p "$output_dir/range_doppler"
    fi
    
    log_success "Output directory structure created at $output_dir"
}

extract_metadata() {
    local input_file="$1"
    local output_dir="$2"
    
    log_info "Extracting metadata and state vectors..."
    
    # Extract state vectors
    local state_vectors_file="$output_dir/metadata/state_vectors.txt"
    if "$S1_PRINT_BIN" state_vectors "$input_file" > "$state_vectors_file" 2>&1; then
        log_success "State vectors saved to $state_vectors_file"
    else
        log_warning "Failed to extract state vectors"
    fi
    
    # Extract packet info for first packet
    local packet_info_file="$output_dir/metadata/packet_info.txt"
    if "$S1_PRINT_BIN" packet_info 0 "$input_file" > "$packet_info_file" 2>&1; then
        log_success "Packet information saved to $packet_info_file"
    else
        log_warning "Failed to extract packet information"
    fi
    
    # Extract index records
    local index_records_file="$output_dir/metadata/index_records.txt"
    if "$S1_PRINT_BIN" index_records "$input_file" > "$index_records_file" 2>&1; then
        log_success "Index records saved to $index_records_file"
    else
        log_warning "Failed to extract index records"
    fi
    
    # Extract burst timing information for TOPS
    local burst_timing_file="$output_dir/metadata/burst_timing.txt"
    if "$S1_PRINT_BIN" burst_timing "$input_file" > "$burst_timing_file" 2>&1; then
        log_success "Burst timing information saved to $burst_timing_file"
    else
        log_warning "Failed to extract burst timing information"
    fi
}

process_burst() {
    local input_file="$1"
    local output_dir="$2"
    local swath="$3"
    local burst_num="$4"
    local scaling="$5"
    local format="$6"
    
    log_info "Processing burst $burst_num of swath $swath..."
    
    local base_name="$(basename "$input_file" .dat)_${swath}_burst${burst_num}"
    
    # Generate intermediate products if requested
    if [[ "$GENERATE_INTERMEDIATE" == true ]]; then
        log_verbose "Generating range-compressed burst image for $swath burst $burst_num..."
        local rc_output="$output_dir/range_compressed/${base_name}_RC.${format}"
        
        if "$S1_WRITE_BIN" range_compressed_burst "$swath" "$burst_num" "$input_file" "$rc_output" "$scaling" 2>/dev/null; then
            log_verbose "Range-compressed burst image saved: $rc_output"
        else
            log_warning "Failed to generate range-compressed image for $swath burst $burst_num"
        fi
        
        log_verbose "Generating range-doppler burst image for $swath burst $burst_num..."
        local rd_output="$output_dir/range_doppler/${base_name}_RD.${format}"
        
        if "$S1_WRITE_BIN" range_doppler_burst "$swath" "$burst_num" "$input_file" "$rd_output" "$scaling" 2>/dev/null; then
            log_verbose "Range-doppler burst image saved: $rd_output"
        else
            log_warning "Failed to generate range-doppler image for $swath burst $burst_num"
        fi
    fi
    
    # Generate focused burst image
    log_verbose "Generating focused burst image for $swath burst $burst_num..."
    local focused_output="$output_dir/bursts/${base_name}_FOCUSED.${format}"
    
    if "$S1_WRITE_BIN" azimuth_compressed_burst "$swath" "$burst_num" "$input_file" "$focused_output" "$scaling" 2>/dev/null; then
        log_verbose "Focused burst image saved: $focused_output"
        return 0
    else
        log_warning "Failed to generate focused image for $swath burst $burst_num"
        return 1
    fi
}

process_swath() {
    local input_file="$1"
    local output_dir="$2"
    local swath="$3"
    local scaling="$4"
    local format="$5"
    local specific_burst="$6"  # Optional: specific burst number or empty
    
    log_info "Processing TOPS swath: $swath"
    
    local base_name="$(basename "$input_file" .dat)_${swath}"
    
    # Get burst count for this swath
    local burst_count
    burst_count=$(get_burst_info "$input_file" "$swath")
    
    if [[ $burst_count -eq 0 ]]; then
        log_warning "No bursts found for swath $swath"
        return 1
    fi
    
    log_info "Found $burst_count bursts in swath $swath"
    
    # Process individual bursts or specific burst
    local successful_bursts=0
    local failed_bursts=0
    
    if [[ -n "$specific_burst" ]]; then
        # Process only the specific burst
        if [[ $specific_burst -gt $burst_count ]]; then
            log_error "Burst $specific_burst does not exist in swath $swath (max: $burst_count)"
            return 1
        fi
        
        if process_burst "$input_file" "$output_dir" "$swath" "$specific_burst" "$scaling" "$format"; then
            ((successful_bursts++))
        else
            ((failed_bursts++))
        fi
    else
        # Process all bursts if PROCESS_ALL_BURSTS is true
        if [[ "$PROCESS_ALL_BURSTS" == true ]]; then
            for ((burst=1; burst<=burst_count; burst++)); do
                if process_burst "$input_file" "$output_dir" "$swath" "$burst" "$scaling" "$format"; then
                    ((successful_bursts++))
                else
                    ((failed_bursts++))
                fi
            done
        fi
    fi
    
    # Generate merged swath image if not processing specific burst
    if [[ -z "$specific_burst" ]] && [[ "$PROCESS_ALL_BURSTS" == false || successful_bursts -gt 1 ]]; then
        log_info "Generating merged swath image for $swath..."
        local merged_output="$output_dir/focused/${base_name}_MERGED.${format}"
        
        if "$S1_WRITE_BIN" azimuth_compressed_swath "$swath" "$input_file" "$merged_output" "$scaling" 2>/dev/null; then
            log_success "Merged swath image saved: $merged_output"
        else
            log_warning "Failed to generate merged swath image for $swath"
        fi
    fi
    
    # Save complex data as CF32 if requested
    if [[ "$format" == "cf32" ]] || [[ "$GENERATE_INTERMEDIATE" == true ]]; then
        log_info "Saving complex data as CF32 for $swath..."
        local cf32_output="$output_dir/focused/${base_name}_COMPLEX.cf32"
        
        if "$S1_WRITE_BIN" save_swath_as_cf32 "$swath" "$input_file" "$cf32_output" 2>/dev/null; then
            log_success "Complex data saved: $cf32_output"
        else
            log_warning "Failed to save complex data for $swath"
        fi
    fi
    
    log_info "Swath $swath processing completed: $successful_bursts successful, $failed_bursts failed"
    
    # Return success if at least one burst was processed successfully
    return $((successful_bursts > 0 ? 0 : 1))
}

estimate_memory_usage() {
    local swath_count="$1"
    local burst_count="$2"
    
    # Rough estimate: ~2-4GB per TOPS swath (much less than stripmap)
    local memory_per_swath=3
    local total_memory=$((swath_count * memory_per_swath))
    
    log_info "Memory usage estimate: ~${total_memory}GB for $swath_count swath(s) with $burst_count total bursts"
    
    # Check available memory (Linux)
    if command -v free >/dev/null 2>&1; then
        local available_gb=$(free -g | awk '/^Mem:/{print $7}')
        if [[ $available_gb -lt $total_memory ]]; then
            log_warning "Available memory (~${available_gb}GB) may be insufficient"
            log_warning "Consider processing swaths individually"
        else
            log_success "Available memory (~${available_gb}GB) should be sufficient"
        fi
    fi
}

generate_summary() {
    local output_dir="$1"
    local swaths_processed="$2"
    local processing_time="$3"
    local burst_mode="$4"
    
    local summary_file="$output_dir/processing_summary.txt"
    
    cat > "$summary_file" << EOF
Sentinel-1 TOPS Processing Summary
==================================

Processing Date: $(date)
Processing Time: ${processing_time} seconds
Script Version: Generated for SAR processing workflow

Input File: $INPUT_FILE
Output Directory: $output_dir

Swaths Processed: $swaths_processed
Scaling Option: $SCALING_OPTION
Output Format: $OUTPUT_FORMAT
Intermediate Products: $GENERATE_INTERMEDIATE
Burst Processing Mode: $burst_mode

Directory Structure:
- metadata/          : Metadata, state vectors, and burst timing
- focused/           : Fully focused SAR images (merged swaths)
- bursts/            : Individual focused burst images
- range_compressed/  : Range-compressed images (if generated)
- range_doppler/     : Range-doppler images (if generated)

TOPS Mode Information:
- TOPS uses burst-based acquisition with timing gaps
- Each swath contains multiple bursts (typically 9-13 for IW, 5-9 for EW)
- Burst synchronization enables interferometric processing
- Processing individual bursts preserves timing information

Processing completed successfully.
EOF

    log_success "Processing summary saved to $summary_file"
}

# =============================================================================
# MAIN SCRIPT LOGIC
# =============================================================================

main() {
    local input_file=""
    local output_dir=""
    local specific_swath=""
    local specific_burst=""
    local scaling="$DEFAULT_SCALING"
    local format="$DEFAULT_FORMAT"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--swath)
                specific_swath="$2"
                shift 2
                ;;
            -b|--burst)
                specific_burst="$2"
                shift 2
                ;;
            -f|--format)
                format="$2"
                shift 2
                ;;
            -c|--scaling)
                scaling="$2"
                shift 2
                ;;
            -i|--intermediate)
                GENERATE_INTERMEDIATE=true
                shift
                ;;
            --no-intermediate)
                GENERATE_INTERMEDIATE=false
                shift
                ;;
            --all-bursts)
                PROCESS_ALL_BURSTS=true
                shift
                ;;
            --merged-bursts)
                PROCESS_ALL_BURSTS=false
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
            *)
                if [[ -z "$input_file" ]]; then
                    input_file="$1"
                elif [[ -z "$output_dir" ]]; then
                    output_dir="$1"
                else
                    log_error "Too many arguments"
                    print_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$input_file" ]] || [[ -z "$output_dir" ]]; then
        log_error "Missing required arguments"
        print_usage
        exit 1
    fi
    
    # Validate specific swath if provided
    if [[ -n "$specific_swath" ]]; then
        local valid_swath=false
        for tops_swath in "${ALL_TOPS_SWATHS[@]}"; do
            if [[ "$specific_swath" == "$tops_swath" ]]; then
                valid_swath=true
                break
            fi
        done
        if [[ "$valid_swath" == false ]]; then
            log_error "Invalid TOPS swath: $specific_swath"
            log_error "Valid TOPS swaths: ${ALL_TOPS_SWATHS[*]}"
            exit 1
        fi
    fi
    
    # Validate burst number if provided
    if [[ -n "$specific_burst" ]]; then
        if [[ -z "$specific_swath" ]]; then
            log_error "Burst number requires a specific swath (-s option)"
            exit 1
        fi
        if ! [[ "$specific_burst" =~ ^[1-9][0-9]*$ ]]; then
            log_error "Invalid burst number: $specific_burst (must be positive integer)"
            exit 1
        fi
    fi
    
    # Store for summary
    INPUT_FILE="$input_file"
    SCALING_OPTION="$scaling"
    OUTPUT_FORMAT="$format"
    
    # Determine burst processing mode for summary
    local burst_mode="All bursts individually"
    if [[ "$PROCESS_ALL_BURSTS" == false ]]; then
        burst_mode="Merged bursts per swath"
    fi
    if [[ -n "$specific_burst" ]]; then
        burst_mode="Specific burst $specific_burst"
    fi
    
    # Start processing
    local start_time=$(date +%s)
    
    log_info "Starting Sentinel-1 TOPS focusing process..."
    log_info "Input file: $input_file"
    log_info "Output directory: $output_dir"
    log_info "Scaling: $scaling"
    log_info "Format: $format"
    log_info "Generate intermediate: $GENERATE_INTERMEDIATE"
    log_info "Burst processing: $burst_mode"
    
    # Check dependencies
    check_dependencies
    
    # Validate input file
    validate_input_file "$input_file"
    
    # Get available swaths
    local available_swaths_array=()
    mapfile -t available_swaths_array < <(get_available_swaths "$input_file")
    
    # Determine swaths to process
    local swaths_to_process=()
    if [[ -n "$specific_swath" ]]; then
        # Check if specific swath is available
        local swath_found=false
        for swath in "${available_swaths_array[@]}"; do
            if [[ "$swath" == "$specific_swath" ]]; then
                swaths_to_process=("$specific_swath")
                swath_found=true
                break
            fi
        done
        if [[ "$swath_found" == false ]]; then
            log_error "Requested swath $specific_swath not found in file"
            log_error "Available swaths: ${available_swaths_array[*]}"
            exit 1
        fi
    else
        swaths_to_process=("${available_swaths_array[@]}")
    fi
    
    log_info "Swaths to process: ${swaths_to_process[*]}"
    
    # Count total bursts for memory estimation
    local total_bursts=0
    for swath in "${swaths_to_process[@]}"; do
        local burst_count
        burst_count=$(get_burst_info "$input_file" "$swath")
        total_bursts=$((total_bursts + burst_count))
    done
    
    # Estimate memory usage
    estimate_memory_usage "${#swaths_to_process[@]}" "$total_bursts"
    
    # Create output directory structure
    create_output_structure "$output_dir"
    
    # Extract metadata
    extract_metadata "$input_file" "$output_dir"
    
    # Process each swath
    local successful_swaths=()
    local failed_swaths=()
    
    for swath in "${swaths_to_process[@]}"; do
        log_info "=== Processing TOPS swath $swath ==="
        
        if process_swath "$input_file" "$output_dir" "$swath" "$scaling" "$format" "$specific_burst"; then
            successful_swaths+=("$swath")
            log_success "Successfully processed swath: $swath"
        else
            failed_swaths+=("$swath")
            log_error "Failed to process swath: $swath"
        fi
        
        log_info "=== Completed swath $swath ==="
    done
    
    # Calculate processing time
    local end_time=$(date +%s)
    local processing_time=$((end_time - start_time))
    
    # Generate summary
    generate_summary "$output_dir" "${successful_swaths[*]}" "$processing_time" "$burst_mode"
    
    # Final report
    log_info "=== PROCESSING COMPLETED ==="
    log_success "Successfully processed ${#successful_swaths[@]} swath(s): ${successful_swaths[*]}"
    
    if [[ ${#failed_swaths[@]} -gt 0 ]]; then
        log_error "Failed to process ${#failed_swaths[@]} swath(s): ${failed_swaths[*]}"
        exit 1
    fi
    
    log_success "Total processing time: ${processing_time} seconds"
    log_success "Output saved to: $output_dir"
    log_info "TOPS processing completed with $total_bursts total bursts processed"
    
    return 0
}

# Run main function with all arguments
main "$@"
