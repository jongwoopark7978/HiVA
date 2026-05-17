#!/usr/bin/env bash
set -euo pipefail

# Replay selected LIBERO episodes with GT actions decoded from v2-v5
# canonical LP-MT coefficient sidecars.

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
  local sidecar="$1"
  local out="$2"
  local suite="$3"
  local task_id="$4"
  local dataset_task_index="$5"
  local episodes="$6"

  "${PYTHON}" "${SCRIPT}" \
    --dataset.root "${DATASET_ROOT}" \
    --sidecar "${sidecar}" \
    --suite "${suite}" \
    --task-id "${task_id}" \
    --dataset-task-index "${dataset_task_index}" \
    --episode-indices "${episodes}" \
    --output-dir "${out}" \
    --control-mode relative \
    --action-source hiva-decoded \
    --save-decoded-actions
}

run_version() {
  local label="$1"
  local sidecar_name="$2"
  local sidecar="${SIDECAR_ROOT}/${sidecar_name}.parquet"
  local summary="${SIDECAR_ROOT}/${sidecar_name}.summary.json"
  local out="${SIDECAR_ROOT}/duration_replay/hiva_coeff_${label}_ep2_8_14_15_108_115_614_decoded"

  require_file "${sidecar}"
  require_file "${summary}"
  mkdir -p "${out}"

  echo "===== ${label} at $(date) ====="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
  echo "DATASET_ROOT=${DATASET_ROOT}"
  echo "SIDECAR=${sidecar}"
  echo "SUMMARY=${summary}"
  echo "OUT=${out}"

  run_replay "${sidecar}" "${out}" libero_10 9 2 "2"
  run_replay "${sidecar}" "${out}" libero_10 0 5 "8"
  run_replay "${sidecar}" "${out}" libero_10 3 8 "14,15"
  run_replay "${sidecar}" "${out}" libero_10 6 1 "108"
  run_replay "${sidecar}" "${out}" libero_10 8 6 "115"
  run_replay "${sidecar}" "${out}" libero_goal 3 12 "614"

  echo "===== Finished ${label}; wrote decoded replays to ${out} at $(date) ====="
}

require_dir "${DATASET_ROOT}"
require_file "${SCRIPT}"

run_version \
  "v2_d4_6_10_commit4_k10_f15_canonical_lp_mt" \
  "libero_hiva_coeff_sidecar_v2_d4_6_10_commit4_k10_f15_canonical_lp_mt"

run_version \
  "v3_d2_4_6_10_prenear2_commit6_k10_f15_canonical_lp_mt" \
  "libero_hiva_coeff_sidecar_v3_d2_4_6_10_prenear2_commit6_k10_f15_canonical_lp_mt"

run_version \
  "v4_d2_4_6_10_prenear2_commit4_k10_f15_canonical_lp_mt" \
  "libero_hiva_coeff_sidecar_v4_d2_4_6_10_prenear2_commit4_k10_f15_canonical_lp_mt"

run_version \
  "v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt" \
  "libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt"

echo "All v2-v5 decoded replays finished at $(date)."
