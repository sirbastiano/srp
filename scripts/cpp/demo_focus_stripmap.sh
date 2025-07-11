#!/bin/bash

# Test script to demonstrate the focus_stripmap_dat.sh usage.
# This script shows how to use the focusing script with different options.

set -euo pipefail

# Path to the focusing script
FOCUS_SCRIPT="/Data_large/marine/PythonProjects/SAR/sarpyx/scripts/focus_stripmap_dat.sh"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Sentinel-1 Stripmap Focusing Script Demo ===${NC}\n"

echo -e "${GREEN}1. Show help information:${NC}"
echo "$FOCUS_SCRIPT --help"
echo

echo -e "${GREEN}2. Basic usage - Focus all stripmap swaths:${NC}"
echo "$FOCUS_SCRIPT /path/to/your/data.dat /path/to/output"
echo

echo -e "${GREEN}3. Focus specific swath (S1):${NC}"
echo "$FOCUS_SCRIPT -s S1 /path/to/your/data.dat /path/to/output"
echo

echo -e "${GREEN}4. Focus with logarithmic scaling:${NC}"
echo "$FOCUS_SCRIPT -c --norm_log /path/to/your/data.dat /path/to/output"
echo

echo -e "${GREEN}5. Focus without intermediate products:${NC}"
echo "$FOCUS_SCRIPT --no-intermediate /path/to/your/data.dat /path/to/output"
echo

echo -e "${GREEN}6. Focus with verbose output:${NC}"
echo "$FOCUS_SCRIPT -v /path/to/your/data.dat /path/to/output"
echo

echo -e "${GREEN}7. Focus specific swath with custom options:${NC}"
echo "$FOCUS_SCRIPT -s S2 -c --mag -f tif -v /path/to/your/data.dat /path/to/output"
echo

echo -e "${GREEN}8. Focus and save as complex float32:${NC}"
echo "$FOCUS_SCRIPT -f cf32 /path/to/your/data.dat /path/to/output"
echo

echo -e "${YELLOW}Note: Replace /path/to/your/data.dat with your actual Sentinel-1 DAT file path${NC}"
echo -e "${YELLOW}      Replace /path/to/output with your desired output directory${NC}"

echo -e "\n${BLUE}Processing Pipeline:${NC}"
echo "1. Input validation and swath detection"
echo "2. Metadata extraction (state vectors, packet info)"
echo "3. Range compression (if intermediate products enabled)"
echo "4. Range-Doppler processing (if intermediate products enabled)"
echo "5. Azimuth compression (focusing)"
echo "6. Output generation in specified format"

echo -e "\n${BLUE}Output Structure:${NC}"
echo "output_dir/"
echo "├── metadata/"
echo "│   ├── state_vectors.txt"
echo "│   ├── packet_info.txt"
echo "│   └── index_records.txt"
echo "├── focused/"
echo "│   └── *_FOCUSED.tif (or .cf32)"
echo "├── range_compressed/ (if enabled)"
echo "│   └── *_RC.tif"
echo "├── range_doppler/ (if enabled)"
echo "│   └── *_RD.tif"
echo "└── processing_summary.txt"

echo -e "\n${BLUE}Memory Requirements:${NC}"
echo "- Stripmap processing can require up to ~52GB RAM per swath"
echo "- Consider processing swaths individually if memory is limited"
echo "- Use --no-intermediate to reduce memory usage"
