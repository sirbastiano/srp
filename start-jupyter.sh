#!/bin/bash
set -euo pipefail

PORT="${JUPYTER_PORT:-8888}"
TOKEN="${JUPYTER_TOKEN:-}"
ROOT_DIR="${JUPYTER_ROOT_DIR:-/workspace}"

exec jupyter lab \
  --ip=0.0.0.0 \
  --port="${PORT}" \
  --no-browser \
  --allow-root \
  --ServerApp.token="${TOKEN}" \
  --ServerApp.allow_origin="*" \
  --ServerApp.root_dir="${ROOT_DIR}"
