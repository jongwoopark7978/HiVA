#!/usr/bin/env bash
set -euo pipefail

# Replay selected episodes with GT coefficients decoded from the canonical HP HiVA sidecar.
#
# The HP sidecar uses one canonical Dmax x K B-spline basis for all durations. The replay
# script decodes the full canonical chunk at each greedy segment start and executes the
# first duration-labeled actions.
# Default K is 10, giving sidecar coefficient shape [10,3], [10,3], [10,1].

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
HIVA_K="${HIVA_K:-10}"
SIDECAR="${SIDECAR:-/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_hp.parquet}"
OUT="${OUT:-/nfs/bigflow/add_disk0/jongwoopark/duration_replay/hiva_coeff_canonical_hp_d6_10_15_k${HIVA_K}_ep2_8_14_15_108_115_614_decoded}"
SCRIPT="${REPO_ROOT}/src/lerobot/scripts/replay_hiva_duration_in_libero.py"

mkdir -p "${OUT}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "HIVA_K=${HIVA_K}"
echo "OUT=${OUT}"

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

run_replay libero_10 9 2 "2"
run_replay libero_10 0 5 "8"
run_replay libero_10 3 8 "14,15"
run_replay libero_10 6 1 "108"
run_replay libero_10 8 6 "115"
run_replay libero_goal 3 12 "614"

echo "Wrote HP GT coefficient decoded replays to ${OUT}"
