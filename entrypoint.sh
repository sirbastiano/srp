#!/bin/bash
set -e

echo "=== Container Startup Script ==="

# Update SNAP
echo "Updating SNAP..."
if [ -d "${SNAP_HOME}/bin" ]; then
    ${SNAP_HOME}/bin/snap --nosplash --nogui --modules --list --refresh || echo "Warning: SNAP refresh failed"
    ${SNAP_HOME}/bin/snap --nosplash --nogui --modules --update-all || echo "Warning: SNAP update failed"
    echo "SNAP update completed."
else
    echo "Warning: SNAP_HOME not found at ${SNAP_HOME}"
fi

# Generate the grid
echo "Generating grid..."
mkdir -p /workspace/grid
cd /workspace/grid
python3.11 -m sarpyx.utils.grid
echo "Grid generation completed."

# Execute the CMD passed to the container
echo "Starting main process..."
exec "$@"
