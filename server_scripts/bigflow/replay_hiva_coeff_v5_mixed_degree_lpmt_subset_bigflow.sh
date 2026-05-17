#!/usr/bin/env bash
set -euo pipefail

# Generate decoded-action LIBERO replays for the v5 LP-MT mixed-degree coefficient sidecar.
# Default sidecar uses K=10, F=15, degree_tr=3, degree_rot=3, degree_grip=5.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-1}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"
DATASET_ROOT="${DATASET_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
LABEL="${LABEL:-v5_d4_6_10_wide_commit6_k10_ptr3_prot3_pgrip5_f15_canonical_lp_mt}"
SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_${LABEL}.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_${LABEL}.summary.json}"
OUT="${OUT:-${SIDECAR_ROOT}/duration_replay/hiva_coeff_${LABEL}_ep2_8_14_15_108_115_614_decoded}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin/python}"
SCRIPT="${REPO_ROOT}/src/lerobot/scripts/replay_hiva_duration_in_libero.py"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_replay() {
  local suite="$1"
  local task_id="$2"
  local dataset_task_index="$3"
  local episodes="$4"

  "${PYTHON}" "${SCRIPT}" \
    --dataset.root "${DATASET_ROOT}" \
    --sidecar "${SIDECAR}" \
    --suite "${suite}" \
    --task-id "${task_id}" \
    --dataset-task-index "${dataset_task_index}" \
    --episode-indices "${episodes}" \
    --output-dir "${OUT}" \
    --control-mode relative \
    --action-source hiva-decoded \
    --save-decoded-actions
}

require_file "${SCRIPT}"
require_file "${SIDECAR}"
require_file "${SIDECAR_SUMMARY}"
mkdir -p "${OUT}"

echo "===== Replay ${LABEL} at $(date) ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
echo "OUT=${OUT}"

run_replay libero_10 9 2 "2"
run_replay libero_10 0 5 "8"
run_replay libero_10 3 8 "14,15"
run_replay libero_10 6 1 "108"
run_replay libero_10 8 6 "115"
run_replay libero_goal 3 12 "614"

echo "===== Finished replay ${LABEL} at $(date) ====="
