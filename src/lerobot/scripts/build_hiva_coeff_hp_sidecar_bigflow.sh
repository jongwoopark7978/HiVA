#!/usr/bin/env bash
set -euo pipefail

# Build the canonical hold-pad coefficient sidecar used by HP HiVA runs on bigflow.
# The residual-FFN duration-head variant uses the same HP sidecar as the linear-head HP model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

HIVA_K="${HIVA_K:-10}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
OVERWRITE="${OVERWRITE:-false}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DURATION_SIDECAR="${DURATION_SIDECAR:-/nfs/bigflow/add_disk0/jongwoopark/libero_duration_sidecar_all_episodes.parquet}"
OUTPUT="${OUTPUT:-/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_hp.parquet}"
SUMMARY_JSON="${SUMMARY_JSON:-/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_hp.summary.json}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${OVERWRITE}" != "true" && -e "${OUTPUT}" ]]; then
  echo "Output already exists: ${OUTPUT}"
  echo "Set OVERWRITE=true to rebuild it."
  exit 0
fi

echo "DATA_ROOT=${DATA_ROOT}"
echo "DURATION_SIDECAR=${DURATION_SIDECAR}"
echo "OUTPUT=${OUTPUT}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "HIVA_K=${HIVA_K}"
echo "HIVA_DMAX=${HIVA_DMAX}"
echo "HIVA_DEGREE=${HIVA_DEGREE}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_hp.py" \
  --data-root "${DATA_ROOT}" \
  --duration-sidecar "${DURATION_SIDECAR}" \
  --output "${OUTPUT}" \
  --summary-json "${SUMMARY_JSON}" \
  --duration-classes 6 10 15 \
  --dmax "${HIVA_DMAX}" \
  --n-ctrl "${HIVA_K}" \
  --degree "${HIVA_DEGREE}"
