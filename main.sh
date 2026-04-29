#!/bin/bash
#PBS -N worldsar
#PBS -S /bin/bash
#PBS -q cpu_std
#PBS -l walltime=23:30:00
#PBS -l select=1:ncpus=192:mem=128g

set -euo pipefail

# ---- Runtime mode defaults (edit here for your preferred defaults) ----
# Set this to "vm" for local default, or "hpc" for cluster default.
WORLDSAR_MODE_DEFAULT="${WORLDSAR_MODE_DEFAULT:-vm}"
WORLDSAR_MODE_HPC="${WORLDSAR_MODE_HPC:-hpc}"
WORLDSAR_MODE_VM="${WORLDSAR_MODE_VM:-vm}"

# ---- Runtime mode ----
WORLDSAR_MODE="${WORLDSAR_MODE:-${RUN_MODE:-${WORLDSAR_MODE_DEFAULT}}}"
RUN_MODE="${WORLDSAR_MODE}"

# ---- Paths (override via Makefile/env; hpc mode keeps repo-specific hardcoded defaults) ----
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"

# ---- HPC hardcoded defaults ----
HPC_BASE_DIR="${HPC_BASE_DIR:-/lustre/projects/1001/rdelprete/WORLDSAR}"
HPC_WORKSPACE_PREFIX="${HPC_WORKSPACE_PREFIX:-/work}"
HPC_DATA_DIR="${HPC_BASE_DIR}/phidown_data"
HPC_PY_SCRIPT_DIR="${HPC_BASE_DIR}/pyscripts"
HPC_SIF_IMAGE="${HPC_BASE_DIR}/sarpyx.sif"
HPC_SNAP_USER_DIR="${HPC_BASE_DIR}/.snap"
HPC_OUTPUT_DIR="${HPC_BASE_DIR}/OUT/worldsar_output"
HPC_TILES_DIR="${HPC_BASE_DIR}/OUT/tiles"
HPC_DB_DIR="${HPC_BASE_DIR}/OUT/DB"
HPC_GRID_FILE="${HPC_GRID_FILE:-${HPC_BASE_DIR}/grid/grid_10km.geojson}"
HPC_GRID_PATH="${HPC_WORKSPACE_PREFIX}/grid/grid_10km.geojson"

# ---- VM defaults (relative/portable) ----
VM_BASE_DIR="${VM_BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
VM_WORKSPACE_PREFIX="${VM_WORKSPACE_PREFIX:-/work}"
VM_DATA_DIR="${VM_BASE_DIR}/phidown_data"
VM_PY_SCRIPT_DIR="${VM_BASE_DIR}/pyscripts"
VM_SIF_IMAGE="${VM_BASE_DIR}/sarpyx.sif"
VM_SNAP_USER_DIR="${VM_BASE_DIR}/.snap"
VM_OUTPUT_DIR="${VM_BASE_DIR}/OUT/worldsar_output"
VM_TILES_DIR="${VM_BASE_DIR}/OUT/tiles"
VM_DB_DIR="${VM_BASE_DIR}/OUT/DB"
VM_GRID_FILE="${VM_GRID_FILE:-${VM_BASE_DIR}/grid/grid_10km.geojson}"
VM_GRID_PATH="${VM_WORKSPACE_PREFIX}/grid/grid_10km.geojson"

if [[ "${WORLDSAR_MODE}" == "${WORLDSAR_MODE_HPC}" ]]; then
  BASE_DIR="${BASE_DIR:-${HPC_BASE_DIR}}"
  DATA_DIR="${DATA_DIR:-${HPC_DATA_DIR}}"
  PY_SCRIPT_DIR="${PY_SCRIPT_DIR:-${HPC_PY_SCRIPT_DIR}}"
  SIF_IMAGE="${SIF_IMAGE:-${HPC_SIF_IMAGE}}"
  SNAP_USER_DIR="${SNAP_USER_DIR:-${HPC_SNAP_USER_DIR}}"
  OUTPUT_PATH="${OUTPUT_PATH:-${HPC_OUTPUT_DIR}}"
  CUTS_OUTDIR="${CUTS_OUTDIR:-${HPC_TILES_DIR}}"
  DB_DIR="${DB_DIR:-${HPC_DB_DIR}}"
  WORKSPACE_PREFIX="${WORKSPACE_PREFIX:-${HPC_WORKSPACE_PREFIX}}"
  GRID_FILE="${GRID_FILE:-${HOST_GRID_FILE:-${HPC_GRID_FILE}}}"
  GRID_PATH="${GRID_PATH:-${HPC_GRID_PATH}}"
  GPT_PARALLELISM="${GPT_PARALLELISM:-164}"
elif [[ "${WORLDSAR_MODE}" == "${WORLDSAR_MODE_VM}" ]]; then
  BASE_DIR="${BASE_DIR:-${VM_BASE_DIR}}"
  DATA_DIR="${DATA_DIR:-${VM_DATA_DIR}}"
  PY_SCRIPT_DIR="${PY_SCRIPT_DIR:-${VM_PY_SCRIPT_DIR}}"
  SIF_IMAGE="${SIF_IMAGE:-${VM_SIF_IMAGE}}"
  SNAP_USER_DIR="${SNAP_USER_DIR:-${VM_SNAP_USER_DIR}}"
  OUTPUT_PATH="${OUTPUT_PATH:-${VM_OUTPUT_DIR}}"
  CUTS_OUTDIR="${CUTS_OUTDIR:-${VM_TILES_DIR}}"
  DB_DIR="${DB_DIR:-${VM_DB_DIR}}"
  WORKSPACE_PREFIX="${WORKSPACE_PREFIX:-${VM_WORKSPACE_PREFIX}}"
  GRID_FILE="${GRID_FILE:-${HOST_GRID_FILE:-${VM_GRID_FILE}}}"
  GRID_PATH="${GRID_PATH:-${VM_GRID_PATH}}"
  GPT_PARALLELISM="${GPT_PARALLELISM:-16}"
else
  echo "ERROR: unknown WORLDSAR_MODE='${WORLDSAR_MODE}'." >&2
  echo "Set WORLDSAR_MODE to either '${WORLDSAR_MODE_VM}' or '${WORLDSAR_MODE_HPC}'." >&2
  exit 2
fi

# ---- Product ----
# Input resolution order: positional arg -> PRODUCT -> WORLDSAR_PRODUCT
PRODUCT_INPUT="${1:-${PRODUCT:-${WORLDSAR_PRODUCT:-}}}"
if [[ -z "${PRODUCT_INPUT}" ]]; then
  echo "ERROR: Product name is required." >&2
  echo "Usage: ${0##*/} <product_name>" >&2
  echo "Or set PRODUCT=<product_name>" >&2
  exit 2
fi

# If PRODUCT contains a slash, treat it as a path (absolute or relative).
# Otherwise treat it as a SAFE directory name under DATA_DIR.
if [[ "${PRODUCT_INPUT}" == */* ]]; then
  PROD_PATH="${PRODUCT_INPUT}"
else
  PROD_PATH="${DATA_DIR}/${PRODUCT_INPUT}"
fi

echo "PRODUCT_INPUT=<${PRODUCT_INPUT}>"
echo "DATA_DIR=<${DATA_DIR}>"
echo "PROD_PATH=<${PROD_PATH}>"

if [[ ! -e "${PROD_PATH}" ]]; then
  echo "ERROR: Product not found: ${PROD_PATH}" >&2
  echo "Set PRODUCT to a SAFE directory name under ${DATA_DIR}, or pass an existing path (directory or file)." >&2
  ls -ld "${DATA_DIR}" "${PROD_PATH}" 2>&1 || true
  exit 2
fi

PRODUCT_NAME="$(basename "${PROD_PATH}")"

# ---- Parameters ----
GPT_MEMORY="${GPT_MEMORY:-64G}"
GPT_TIMEOUT="${GPT_TIMEOUT:-3600}"
# SNAP userdir stores cache/config; GPT binary location is independent.
GPT_PATH="${GPT_PATH:-gpt}"
GRID_PATH="${GRID_PATH:-${WORKSPACE_PREFIX}/grid/grid_10km.geojson}"
# Optional exact host grid file to mount read-only into the container.
# Example:
#   GRID_FILE=/lustre/projects/1001/rdelprete/WORLDSAR/grid/grid_10km.geojson
#   GRID_PATH=/work/grid/grid_10km.geojson
GRID_FILE="${GRID_FILE:-${HOST_GRID_FILE:-}}"
GRID_HOST_DIR="${GRID_HOST_DIR:-}"

# ---- Basic validation ----
[[ -d "${DATA_DIR}" ]]     || { echo "ERROR: DATA_DIR not found: ${DATA_DIR}" >&2; exit 2; }
[[ -d "${PY_SCRIPT_DIR}" ]] || { echo "ERROR: PY_SCRIPT_DIR not found: ${PY_SCRIPT_DIR}" >&2; exit 2; }
[[ -f "${SIF_IMAGE}" ]]    || { echo "ERROR: SIF_IMAGE not found: ${SIF_IMAGE}" >&2; exit 2; }
[[ -e "${PROD_PATH}" ]]    || { echo "ERROR: PROD_PATH not found: ${PROD_PATH}" >&2; exit 2; }
[[ -d "${SNAP_USER_DIR}" ]] || { echo "ERROR: SNAP_USER_DIR not found: ${SNAP_USER_DIR}" >&2; exit 2; }
[[ "${GRID_PATH}" == /* ]] || { echo "ERROR: GRID_PATH must be an absolute in-container path: ${GRID_PATH}" >&2; exit 2; }
[[ "${GRID_PATH}" == *.geojson ]] || { echo "ERROR: GRID_PATH must end with .geojson: ${GRID_PATH}" >&2; exit 2; }

FALLBACK_OUTPUT_ROOT="${BASE_DIR}/outputs"
use_fallback_outputs() {
  OUTPUT_PATH="${FALLBACK_OUTPUT_ROOT}/worldsar_output"
  CUTS_OUTDIR="${FALLBACK_OUTPUT_ROOT}/tiles"
  DB_DIR="${FALLBACK_OUTPUT_ROOT}/DB"
}

configured_outputs_resolved=1
for out_path in "${OUTPUT_PATH}" "${CUTS_OUTDIR}" "${DB_DIR}"; do
  if [[ -L "${out_path}" && ! -e "${out_path}" ]]; then
    echo "WARN: unresolved output symlink: ${out_path} -> $(readlink "${out_path}" || echo "<unknown>")" >&2
    configured_outputs_resolved=0
  fi
done
if [[ "${configured_outputs_resolved}" -eq 0 ]]; then
  echo "WARN: using fallback output root: ${FALLBACK_OUTPUT_ROOT}" >&2
  use_fallback_outputs
fi

if ! mkdir -p "${OUTPUT_PATH}" "${CUTS_OUTDIR}" "${DB_DIR}"; then
  if [[ "${OUTPUT_PATH}" != "${FALLBACK_OUTPUT_ROOT}/worldsar_output" || \
        "${CUTS_OUTDIR}" != "${FALLBACK_OUTPUT_ROOT}/tiles" || \
        "${DB_DIR}" != "${FALLBACK_OUTPUT_ROOT}/DB" ]]; then
    echo "WARN: configured output paths are not writable/valid. Falling back to ${FALLBACK_OUTPUT_ROOT}" >&2
    use_fallback_outputs
    mkdir -p "${OUTPUT_PATH}" "${CUTS_OUTDIR}" "${DB_DIR}"
  else
    echo "ERROR: failed to create fallback output directories under ${FALLBACK_OUTPUT_ROOT}" >&2
    exit 2
  fi
fi

# ---- Grid binding ----
# Preferred mode: bind one exact host grid file to GRID_PATH in the container.
# Compatibility mode: bind GRID_HOST_DIR to ${WORKSPACE_PREFIX}/grid.
if [[ -z "${GRID_FILE}" && -n "${GRID_HOST_DIR}" && -f "${GRID_HOST_DIR}" ]]; then
  GRID_FILE="${GRID_HOST_DIR}"
  GRID_HOST_DIR=""
fi

if [[ -n "${GRID_FILE}" ]]; then
  [[ -f "${GRID_FILE}" ]] || { echo "ERROR: GRID_FILE not found: ${GRID_FILE}" >&2; exit 2; }
  [[ "${GRID_FILE}" == *.geojson ]] || { echo "ERROR: GRID_FILE must end with .geojson: ${GRID_FILE}" >&2; exit 2; }
  GRID_BIND_SOURCE="${GRID_FILE}"
  GRID_BIND_TARGET="${GRID_PATH}"
  LEGACY_GRID_BIND_SOURCE="$(cd "$(dirname "${GRID_FILE}")" && pwd -P)"
else
  if [[ -z "${GRID_HOST_DIR}" ]]; then
    GRID_HOST_DIR="${OUTPUT_PATH}/grid"
  fi
  mkdir -p "${GRID_HOST_DIR}"
  GRID_BIND_SOURCE="${GRID_HOST_DIR}"
  GRID_BIND_TARGET="${WORKSPACE_PREFIX}/grid"
  LEGACY_GRID_BIND_SOURCE="${GRID_HOST_DIR}"
  if [[ ! -f "${GRID_HOST_DIR}/$(basename "${GRID_PATH}")" ]]; then
    echo "WARN: no host grid file found at ${GRID_HOST_DIR}/$(basename "${GRID_PATH}")" >&2
    echo "WARN: set GRID_FILE=/host/path/grid.geojson to bind a specific grid file." >&2
  fi
fi

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
if ! command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1; then
  if [[ "${CONTAINER_RUNTIME}" == "apptainer" ]] && command -v singularity >/dev/null 2>&1; then
    echo "WARN: apptainer not found. Falling back to singularity." >&2
    CONTAINER_RUNTIME="singularity"
  else
    echo "ERROR: container runtime not found: ${CONTAINER_RUNTIME}" >&2
    if [[ "${CONTAINER_RUNTIME}" == "apptainer" ]]; then
      echo "ERROR: singularity also not found." >&2
    fi
    exit 2
  fi
fi

if ! "${CONTAINER_RUNTIME}" exec "${SIF_IMAGE}" bash -lc "[ -x \"${GPT_PATH}\" ] || command -v \"${GPT_PATH}\" >/dev/null 2>&1"; then
  if "${CONTAINER_RUNTIME}" exec "${SIF_IMAGE}" bash -lc "command -v gpt >/dev/null 2>&1"; then
    echo "WARN: GPT_PATH not found in container (${GPT_PATH}). Falling back to 'gpt' from PATH." >&2
    GPT_PATH="gpt"
  else
    echo "ERROR: GPT executable not found in container. Requested GPT_PATH=${GPT_PATH}" >&2
    exit 2
  fi
fi

echo "WORLDSAR_MODE=${WORLDSAR_MODE}"
echo "RUN_MODE=${RUN_MODE}"
echo "BASE_DIR=${BASE_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "SIF_IMAGE=${SIF_IMAGE}"
echo "PROD_PATH=${PROD_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "CUTS_OUTDIR=${CUTS_OUTDIR}"
echo "DB_DIR=${DB_DIR}"
echo "SNAP_USER_DIR=${SNAP_USER_DIR}"
echo "GRID_FILE=${GRID_FILE}"
echo "GRID_HOST_DIR=${GRID_HOST_DIR}"
echo "GRID_BIND=${GRID_BIND_SOURCE}:${GRID_BIND_TARGET}:ro"
echo "LEGACY_GRID_BIND=${LEGACY_GRID_BIND_SOURCE}:/workspace/grid:ro"
echo "GRID_PATH=${GRID_PATH}"
echo "CONTAINER_RUNTIME=${CONTAINER_RUNTIME}"

# ---- Run ----
"${CONTAINER_RUNTIME}" run \
  --env "GRID_PATH=${GRID_PATH}" \
  --env "SNAP_USERDIR=${WORKSPACE_PREFIX}/.snap" \
  -B "${PY_SCRIPT_DIR}:${WORKSPACE_PREFIX}/scripts:ro" \
  -B "${DATA_DIR}:${WORKSPACE_PREFIX}/data:ro" \
  -B "${OUTPUT_PATH}:${WORKSPACE_PREFIX}/output" \
  -B "${CUTS_OUTDIR}:${WORKSPACE_PREFIX}/cuts" \
  -B "${DB_DIR}:${WORKSPACE_PREFIX}/db" \
  -B "${GRID_BIND_SOURCE}:${GRID_BIND_TARGET}:ro" \
  -B "${LEGACY_GRID_BIND_SOURCE}:/workspace/grid:ro" \
  -B "${SNAP_USER_DIR}:${WORKSPACE_PREFIX}/.snap" \
  "${SIF_IMAGE}" \
  python "${WORKSPACE_PREFIX}/scripts/worldsar.py" \
    --input "${WORKSPACE_PREFIX}/data/${PRODUCT_NAME}" \
    --output "${WORKSPACE_PREFIX}/output" \
    --cuts-outdir "${WORKSPACE_PREFIX}/cuts" \
    --gpt-path "${GPT_PATH}" \
    --grid-path "${GRID_PATH}" \
    --db-dir "${WORKSPACE_PREFIX}/db" \
    --gpt-memory "${GPT_MEMORY}" \
    --gpt-parallelism "${GPT_PARALLELISM}" \
    --gpt-timeout "${GPT_TIMEOUT}"
