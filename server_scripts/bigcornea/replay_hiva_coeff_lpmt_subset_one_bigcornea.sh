#!/usr/bin/env bash
set -euo pipefail

# Replay selected LIBERO episodes for one canonical LP-MT coefficient sidecar.
# Set LABEL and SIDECAR_NAME. Set EPISODE_MODE=ep614 to run only episode 614.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
LABEL="${LABEL:?Set LABEL, e.g. v4_d2_4_6_10_prenear2_commit4_k10_f15_canonical_lp_mt}"
SIDECAR_NAME="${SIDECAR_NAME:?Set SIDECAR_NAME without .parquet suffix}"
EPISODE_MODE="${EPISODE_MODE:-full}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin/python}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
DATASET_ROOT="${DATASET_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
SIDECAR="${SIDECAR_ROOT}/${SIDECAR_NAME}.parquet"
SUMMARY="${SIDECAR_ROOT}/${SIDECAR_NAME}.summary.json"
OUT="${OUT:-${SIDECAR_ROOT}/duration_replay/hiva_coeff_${LABEL}_ep2_8_14_15_108_115_614_decoded}"
SCRIPT="${REPO_ROOT}/src/lerobot/scripts/replay_hiva_duration_in_libero.py"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
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

require_dir "${DATASET_ROOT}"
require_file "${SCRIPT}"
require_file "${SIDECAR}"
require_file "${SUMMARY}"
mkdir -p "${OUT}"

echo "===== ${LABEL} ${EPISODE_MODE} at $(date) ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "SUMMARY=${SUMMARY}"
echo "OUT=${OUT}"

if [[ "${EPISODE_MODE}" == "ep614" ]]; then
  run_replay libero_goal 3 12 "614"
else
  run_replay libero_10 9 2 "2"
  run_replay libero_10 0 5 "8"
  run_replay libero_10 3 8 "14,15"
  run_replay libero_10 6 1 "108"
  run_replay libero_10 8 6 "115"
  run_replay libero_goal 3 12 "614"
fi

echo "===== Finished ${LABEL} ${EPISODE_MODE}; wrote decoded replays to ${OUT} at $(date) ====="
