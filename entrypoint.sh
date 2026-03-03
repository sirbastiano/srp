#!/bin/bash
set -e

echo "=== Container Startup Script ==="

# Update SNAP
echo "Updating SNAP..."
if [ -d "${SNAP_HOME}/bin" ]; then
    if [ "${SNAP_SKIP_UPDATES:-1}" = "1" ]; then
        echo "SNAP updates skipped (SNAP_SKIP_UPDATES=1)."
    else
        ${SNAP_HOME}/bin/snap --nosplash --nogui --modules --update-all || echo "Warning: SNAP update failed"
        echo "SNAP update completed."
    fi
else
    echo "Warning: SNAP_HOME not found at ${SNAP_HOME}"
fi

GRID_DIR="/workspace/grid"
GRID_PATH_CANDIDATE="${GRID_PATH:-${grid_path:-}}"

echo "Preparing grid assets..."
mkdir -p "${GRID_DIR}"
shopt -s nullglob
GRID_GEOJSON_FILES=("${GRID_DIR}"/*.geojson)
shopt -u nullglob

if [ -n "${GRID_PATH_CANDIDATE}" ] && [ -f "${GRID_PATH_CANDIDATE}" ] && [[ "${GRID_PATH_CANDIDATE}" == *.geojson ]]; then
    export GRID_PATH="${GRID_PATH_CANDIDATE}"
    echo "Using mounted/configured grid file: ${GRID_PATH}"
elif [ ${#GRID_GEOJSON_FILES[@]} -gt 0 ]; then
    export GRID_PATH="${GRID_GEOJSON_FILES[0]}"
    echo "Using existing grid file: ${GRID_PATH}"
else
    echo "No .geojson grid found in ${GRID_DIR}. Generating grid on startup..."
    (
        cd "${GRID_DIR}"
        python -m sarpyx.utils.grid
    )
    shopt -s nullglob
    GRID_GEOJSON_FILES=("${GRID_DIR}"/*.geojson)
    shopt -u nullglob
    if [ ${#GRID_GEOJSON_FILES[@]} -gt 0 ]; then
        export GRID_PATH="${GRID_GEOJSON_FILES[0]}"
        echo "Grid generation completed: ${GRID_PATH}"
    else
        echo "ERROR: Grid generation did not create any .geojson in ${GRID_DIR}" >&2
        exit 1
    fi
fi

# Execute the CMD passed to the container
echo "Starting main process..."
exec "$@"
