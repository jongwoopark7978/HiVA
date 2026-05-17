#!/usr/bin/env bash
set -euo pipefail

# Staggered fanout full 50-episode LIBERO evals for selected b256 S=0.5
# stage0 LP-MT HiVA checkpoints on BigBrain GPU 5. Each checkpoint gets an
# independent runner process; launches are spaced by INTERVAL_SECONDS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-5}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

TRAIN_DIR="${TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203}"
MODEL_NAME="$(basename "${TRAIN_DIR}")"
CKPTS=(003625 003750 003875 004000)

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigbrain_gpu${GPU_ID}_${MODEL_NAME}_ckpts_003625_003750_003875_004000_50eps_bs50_${TIMESTAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_full_staggered_fanout_gpu${GPU_ID}_${MODEL_NAME}_ckpts_003625_004000_50eps_bs50_${TIMESTAMP}.queue.log}"

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

launch_ckpt() {
  local ckpt="$1"
  local outer_log="${LOG_DIR}/eval_full_staggered_fanout_gpu${GPU_ID}_${MODEL_NAME}_ckpt_${ckpt}_50eps_bs50_${TIMESTAMP}.outer.log"
  local runner_queue_log="${LOG_DIR}/eval_full_staggered_fanout_gpu${GPU_ID}_${MODEL_NAME}_ckpt_${ckpt}_50eps_bs50_${TIMESTAMP}.queue.log"

  require_dir "${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"

  echo "===== $(date) launching ckpt_${ckpt} full eval on GPU ${GPU_ID} ====="
  echo "OUTER_LOG=${outer_log}"
  echo "RUNNER_QUEUE_LOG=${runner_queue_log}"

  TRAIN_DIR="${TRAIN_DIR}" \
  MODEL_TAG="${MODEL_NAME}" \
  CKPTS_OVERRIDE="${ckpt}" \
  GPU_IDS="${GPU_ID}" \
  EVAL_CHECKPOINTS_IN_PARALLEL=1 \
  N_EPISODES=50 \
  EVAL_BATCH_SIZE=50 \
  MAX_EPISODES_RENDERED=1 \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR}" \
  QUEUE_LOG="${runner_queue_log}" \
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
  mkdir -p "${SWEEP_OUTPUT_DIR}"

  echo "===== b256 S=0.5 full staggered fanout eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "GPU_ID=${GPU_ID}"
  echo "INTERVAL_SECONDS=${INTERVAL_SECONDS}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "CKPTS=${CKPTS[*]}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  local pids=()
  local idx
  local ckpt
  for idx in "${!CKPTS[@]}"; do
    ckpt="${CKPTS[$idx]}"
    launch_ckpt "${ckpt}"
    pids+=("$!")
    if [[ "${idx}" -lt "$((${#CKPTS[@]} - 1))" ]]; then
      echo "===== $(date) sleeping ${INTERVAL_SECONDS}s before next launch ====="
      sleep "${INTERVAL_SECONDS}"
    fi
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "One or more staggered fanout full eval jobs failed." >&2
    exit "${status}"
  fi

  echo "===== b256 S=0.5 full staggered fanout eval finished at $(date) ====="
}

main "$@"
