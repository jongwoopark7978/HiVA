#!/usr/bin/env bash
set -euo pipefail

# Fan out all numeric trrot5 checkpoint evals immediately across GPUs 6 and 7.
# "last" is intentionally skipped when it aliases the final numeric checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

TRAIN_DIR="${TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot5_grip0p5_b768_g4_s2_20260514_040613}"
MODEL_NAME="$(basename "${TRAIN_DIR}")"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_trrot5_all_ckpts_parallel_bigbrain_gpu6_7_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}.queue.log}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval/bigbrain_parallel_${MODEL_NAME}_10eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

collect_numeric_ckpts() {
  find "${TRAIN_DIR}/checkpoints" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | sort -n
}

launch_ckpt() {
  local ckpt="$1"
  local gpu_id="$2"
  local outer_log="${LOG_DIR}/eval_bigbrain_parallel_${MODEL_NAME}_ckpt_${ckpt}_gpu${gpu_id}_${TIMESTAMP}.outer.log"

  require_dir "${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  mkdir -p "${EVAL_DIR}"

  echo "===== $(date) launching ${MODEL_NAME} ckpt_${ckpt} on GPU ${gpu_id} ====="
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "EVAL_DIR=${EVAL_DIR}"
  echo "OUTER_LOG=${outer_log}"

  TRAIN_DIR="${TRAIN_DIR}" \
  MODEL_TAG="${MODEL_NAME}" \
  CKPTS_OVERRIDE="${ckpt}" \
  GPU_IDS="${gpu_id}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  EVAL_CHECKPOINTS_IN_PARALLEL=1 \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  SWEEP_OUTPUT_DIR="${EVAL_DIR}" \
  TIMESTAMP="${TIMESTAMP}" \
  bash "${RUNNER}" > "${outer_log}" 2>&1 &
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_file "${RUNNER}"
  require_dir "${TRAIN_DIR}/checkpoints"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  mapfile -t ckpts < <(collect_numeric_ckpts)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No numeric checkpoints found under ${TRAIN_DIR}/checkpoints" >&2
    exit 1
  fi

  echo "===== trrot5 all-checkpoint parallel BigBrain eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "N_JOBS=${#ckpts[@]}"
  echo "CKPTS=${ckpts[*]}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
  echo "QUEUE_LOG=${QUEUE_LOG}"
  echo "EVAL_DIR=${EVAL_DIR}"

  local pids=()
  local idx ckpt gpu_id status=0
  for idx in "${!ckpts[@]}"; do
    ckpt="${ckpts[$idx]}"
    gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
    launch_ckpt "${ckpt}" "${gpu_id}"
    pids+=("$!")
  done

  echo "LAUNCHED_PIDS=${pids[*]}"
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "One or more trrot5 checkpoint evals failed." >&2
    exit "${status}"
  fi
  echo "===== trrot5 all-checkpoint parallel BigBrain eval finished at $(date) ====="
}

main "$@"
