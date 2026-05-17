#!/usr/bin/env bash
set -euo pipefail

# Generate decoded-action LIBERO replays for v5 LP-MT coefficient sidecars at degrees P=4..7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-6}"
DEGREES="${DEGREES:-4 5 6 7}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"
DATASET_ROOT="${DATASET_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_K="${HIVA_K:-10}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-15}"

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

require_file "${SCRIPT}"

for degree in ${DEGREES}; do
  label="v5_d4_6_10_wide_commit6_k${HIVA_K}_p${degree}_f${HIVA_FIT_HORIZON}_canonical_lp_mt"
  sidecar="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_${label}.parquet"
  summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_${label}.summary.json"
  out="${SIDECAR_ROOT}/duration_replay/hiva_coeff_${label}_ep2_8_14_15_108_115_614_decoded"

  require_file "${sidecar}"
  require_file "${summary}"
  mkdir -p "${out}"

  echo "===== Replay ${label} at $(date) ====="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "DATASET_ROOT=${DATASET_ROOT}"
  echo "SIDECAR=${sidecar}"
  echo "OUT=${out}"

  run_replay "${sidecar}" "${out}" libero_10 9 2 "2"
  run_replay "${sidecar}" "${out}" libero_10 0 5 "8"
  run_replay "${sidecar}" "${out}" libero_10 3 8 "14,15"
  run_replay "${sidecar}" "${out}" libero_10 6 1 "108"
  run_replay "${sidecar}" "${out}" libero_10 8 6 "115"
  run_replay "${sidecar}" "${out}" libero_goal 3 12 "614"

  echo "===== Finished replay ${label} at $(date) ====="
done
