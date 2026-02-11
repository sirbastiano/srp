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

# Generate the grid
echo "Generating grid..."
mkdir -p /workspace/grid
cd /workspace/grid
if python3.11 -m sarpyx.utils.grid; then
    echo "Grid generation completed."
    cd /workspace
else
    echo "Error: Grid generation failed"
    exit 1
fi

# Execute the CMD passed to the container
echo "Starting main process..."
exec "$@"
