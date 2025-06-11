#!/bin/bash

set -euo pipefail

# Download product
echo "Downloading: $1"
/home/vessel/anaconda3/envs/py39/bin/python -m phidown.downloader -eo_product_name "$1"