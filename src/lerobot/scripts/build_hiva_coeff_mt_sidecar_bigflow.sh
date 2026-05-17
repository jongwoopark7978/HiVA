#!/usr/bin/env bash
set -euo pipefail

# Build a canonical max-target (MT) coefficient sidecar on bigflow from any
# frame-aligned duration sidecar.
#
# Example for the balanced {4,15}, W1=10, W3=0 labels:
#   DURATION_CLASSES="4 15" \
#   DURATION_SIDECAR=/nfs/bigflow/add_disk0/jongwoopark/libero_duration_sidecar_d4_15_w1_10_w3_0_all_episodes.parquet \
#   OUTPUT=/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d4_15_w1_10_w3_0_k10_canonical_mt.parquet \
#   SUMMARY_JSON=/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d4_15_w1_10_w3_0_k10_canonical_mt.summary.json \
#   bash server_scripts/bigflow/build_hiva_coeff_mt_sidecar_bigflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

HIVA_K="${HIVA_K:-10}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
ROT_SCALE_ETA="${ROT_SCALE_ETA:-0.5}"
OVERWRITE="${OVERWRITE:-false}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DURATION_CLASSES="${DURATION_CLASSES:-4 15}"
DURATION_SIDECAR="${DURATION_SIDECAR:-/nfs/bigflow/add_disk0/jongwoopark/libero_duration_sidecar_d4_15_w1_10_w3_0_all_episodes.parquet}"
OUTPUT="${OUTPUT:-/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d4_15_w1_10_w3_0_k${HIVA_K}_canonical_mt.parquet}"
SUMMARY_JSON="${SUMMARY_JSON:-/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d4_15_w1_10_w3_0_k${HIVA_K}_canonical_mt.summary.json}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${OVERWRITE}" != "true" && ( -e "${OUTPUT}" || -e "${SUMMARY_JSON}" ) ]]; then
  echo "Output already exists:"
  [[ -e "${OUTPUT}" ]] && echo "  ${OUTPUT}"
  [[ -e "${SUMMARY_JSON}" ]] && echo "  ${SUMMARY_JSON}"
  echo "Set OVERWRITE=true to rebuild."
  exit 0
fi

read -r -a DURATION_CLASS_ARGS <<< "${DURATION_CLASSES}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "DURATION_SIDECAR=${DURATION_SIDECAR}"
echo "OUTPUT=${OUTPUT}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "DURATION_CLASSES=${DURATION_CLASSES}"
echo "HIVA_K=${HIVA_K}"
echo "HIVA_DMAX=${HIVA_DMAX}"
echo "HIVA_DEGREE=${HIVA_DEGREE}"
echo "ROT_SCALE_ETA=${ROT_SCALE_ETA}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_mt.py" \
  --data-root "${DATA_ROOT}" \
  --duration-sidecar "${DURATION_SIDECAR}" \
  --output "${OUTPUT}" \
  --summary-json "${SUMMARY_JSON}" \
  --duration-classes "${DURATION_CLASS_ARGS[@]}" \
  --dmax "${HIVA_DMAX}" \
  --n-ctrl "${HIVA_K}" \
  --degree "${HIVA_DEGREE}" \
  --rot-scale-eta "${ROT_SCALE_ETA}"
