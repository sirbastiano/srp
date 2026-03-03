#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="/shared/home/rdelprete/PythonProjects/srp/grid"
REMOTE_HOST="SpaceHPC"
REMOTE_DIR="/lustre/projects/1001/rdelprete/WORLDSAR"

rsync -avh --progress "${LOCAL_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}/"
