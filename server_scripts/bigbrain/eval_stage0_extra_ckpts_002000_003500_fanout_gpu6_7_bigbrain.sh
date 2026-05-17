#!/usr/bin/env bash
set -euo pipefail

# Fan out additional stage0 LP-MT checkpoint evals across GPUs 6 and 7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

TRAIN_DIR="${TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p5_20260514_182207}"
MODEL_NAME="$(basename "${TRAIN_DIR}")"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval/lpmt_stage0_v5_all_ckpts_partial_bigbrain_bs10_20260515_000849}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_stage0_extra_ckpts_002000_003500_fanout_gpu6_7_bigbrain_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}.queue.log}"

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
  local gpu_id="$2"
  local outer_log="${LOG_DIR}/eval_stage0_extra_${MODEL_NAME}_ckpt_${ckpt}_gpu${gpu_id}_${TIMESTAMP}.outer.log"

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

  local ckpts=(002000 002500 003000 003125 003500)
  local gpus=(6 7 6 7 6)

  echo "===== stage0 extra checkpoint fanout eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "EVAL_DIR=${EVAL_DIR}"
  echo "CKPTS=${ckpts[*]}"
  echo "GPUS=${gpus[*]}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  local pids=()
  local idx
  for idx in "${!ckpts[@]}"; do
    launch_ckpt "${ckpts[$idx]}" "${gpus[$idx]}"
    pids+=("$!")
  done

  echo "LAUNCHED_PIDS=${pids[*]}"

  local pid
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "One or more stage0 extra checkpoint evals failed." >&2
    exit "${status}"
  fi

  echo "===== stage0 extra checkpoint fanout eval finished at $(date) ====="
}

main "$@"
