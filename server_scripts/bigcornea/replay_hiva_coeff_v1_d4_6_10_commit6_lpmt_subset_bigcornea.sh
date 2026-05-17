#!/usr/bin/env bash
set -euo pipefail

# Replay selected LIBERO episodes with GT actions decoded from the v1
# d4_6_10 commit6 canonical LP-MT coefficient sidecar.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-4}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin/python}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
DATASET_ROOT="${DATASET_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v1_d4_6_10_commit6_k10_f15_canonical_lp_mt.parquet}"
OUT="${OUT:-${SIDECAR_ROOT}/duration_replay/hiva_coeff_v1_d4_6_10_commit6_k10_f15_canonical_lp_mt_ep2_8_14_15_108_115_614_decoded}"
SCRIPT="${REPO_ROOT}/src/lerobot/scripts/replay_hiva_duration_in_libero.py"

mkdir -p "${OUT}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "OUT=${OUT}"

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
require_file "${SIDECAR}"

run_replay libero_10 9 2 "2"
run_replay libero_10 0 5 "8"
run_replay libero_10 3 8 "14,15"
run_replay libero_10 6 1 "108"
run_replay libero_10 8 6 "115"
run_replay libero_goal 3 12 "614"

echo "Wrote v1 d4_6_10 commit6 LP-MT decoded replays to ${OUT}"
