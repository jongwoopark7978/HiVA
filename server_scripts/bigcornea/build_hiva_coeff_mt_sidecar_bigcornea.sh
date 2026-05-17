#!/usr/bin/env bash
set -euo pipefail

# Build the canonical max-target (MT) coefficient sidecar used by MT HiVA runs on bigcornea.
# MT fits the full Dmax future target when available, with terminal hold only near episode ends.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

HIVA_K="${HIVA_K:-10}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
OVERWRITE="${OVERWRITE:-false}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DURATION_SIDECAR="${DURATION_SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_duration_sidecar_all_episodes.parquet}"
OUTPUT="${OUTPUT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.parquet}"
SUMMARY_JSON="${SUMMARY_JSON:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.summary.json}"
BIGFLOW_ROOT="${BIGFLOW_ROOT:-/nfs/bigflow.cs.stonybrook.edu/add_disk0/jongwoopark}"
BIGFLOW_OUTPUT="${BIGFLOW_OUTPUT:-${BIGFLOW_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.parquet}"
BIGFLOW_SUMMARY_JSON="${BIGFLOW_SUMMARY_JSON:-${BIGFLOW_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.summary.json}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${OVERWRITE}" != "true" && -f "${OUTPUT}" && -f "${SUMMARY_JSON}" ]]; then
  echo "Using existing MT sidecar: ${OUTPUT}"
  echo "Using existing MT summary: ${SUMMARY_JSON}"
  exit 0
fi

if [[ "${OVERWRITE}" != "true" && -f "${BIGFLOW_OUTPUT}" && -f "${BIGFLOW_SUMMARY_JSON}" ]]; then
  echo "Syncing MT sidecar from bigflow:"
  echo "  ${BIGFLOW_OUTPUT}"
  echo "  ${BIGFLOW_SUMMARY_JSON}"
  rsync -av "${BIGFLOW_OUTPUT}" "${BIGFLOW_SUMMARY_JSON}" "$(dirname "${OUTPUT}")/"
  exit 0
fi

echo "DATA_ROOT=${DATA_ROOT}"
echo "DURATION_SIDECAR=${DURATION_SIDECAR}"
echo "OUTPUT=${OUTPUT}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "BIGFLOW_OUTPUT=${BIGFLOW_OUTPUT}"
echo "BIGFLOW_SUMMARY_JSON=${BIGFLOW_SUMMARY_JSON}"
echo "HIVA_K=${HIVA_K}"
echo "HIVA_DMAX=${HIVA_DMAX}"
echo "HIVA_DEGREE=${HIVA_DEGREE}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_hp.py" \
  --target-mode max_target \
  --data-root "${DATA_ROOT}" \
  --duration-sidecar "${DURATION_SIDECAR}" \
  --output "${OUTPUT}" \
  --summary-json "${SUMMARY_JSON}" \
  --duration-classes 6 10 15 \
  --duration-map 1:6 3:10 8:15 \
  --dmax "${HIVA_DMAX}" \
  --n-ctrl "${HIVA_K}" \
  --degree "${HIVA_DEGREE}" \
  --rot-scale-eta 0.5
