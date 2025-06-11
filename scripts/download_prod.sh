#!/bin/bash

set -euo pipefail

# Download product
echo "Downloading: $1"
python -m phidown.downloader -eo_product_name "$1"
