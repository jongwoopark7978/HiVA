#!/usr/bin/env bash
set -euo pipefail

# Add selected P5/P7 stage0 LP-MT checkpoint evals to the existing GPU 5
# P5/P7 eval roots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
GPU_ID="${GPU_ID:-5}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
P5_SIDECAR="${P5_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.parquet}"
P5_SUMMARY="${P5_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.summary.json}"
P7_SIDECAR="${P7_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.parquet}"
P7_SUMMARY="${P7_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.summary.json}"

P5_TRAIN_DIR="${P5_TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p5_f15_bigbrain_b160_g2_s0p5_steps5000_20260515_013436}"
P7_TRAIN_DIR="${P7_TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459}"

P5_MODEL_NAME="$(basename "${P5_TRAIN_DIR}")"
P7_MODEL_NAME="$(basename "${P7_TRAIN_DIR}")"
P5_EVAL_DIR="${P5_EVAL_DIR:-${REPO_ROOT}/outputs/eval/bigbrain_gpu5_${P5_MODEL_NAME}_10eps_bs10_20260515_171921}"
P7_EVAL_DIR="${P7_EVAL_DIR:-${REPO_ROOT}/outputs/eval/bigbrain_gpu5_${P7_MODEL_NAME}_10eps_bs10_20260515_171921}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_stage0_p5_p7_more_ckpts_fanout_gpu${GPU_ID}_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}.queue.log}"

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
  local train_dir="$1"
  local model_name="$2"
  local eval_dir="$3"
  local sidecar="$4"
  local summary="$5"
  local ckpt="$6"
  local outer_log="${LOG_DIR}/eval_gpu${GPU_ID}_${model_name}_ckpt_${ckpt}_${TIMESTAMP}.outer.log"
  local runner_queue_log="${LOG_DIR}/eval_gpu${GPU_ID}_${model_name}_ckpt_${ckpt}_${TIMESTAMP}.queue.log"

  require_dir "${train_dir}/checkpoints/${ckpt}/pretrained_model"
  mkdir -p "${eval_dir}"

  echo "===== $(date) launching ${model_name} ckpt_${ckpt} on GPU ${GPU_ID} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "EVAL_DIR=${eval_dir}"
  echo "SIDECAR=${sidecar}"
  echo "OUTER_LOG=${outer_log}"

  TRAIN_DIR="${train_dir}" \
  MODEL_TAG="${model_name}" \
  CKPTS_OVERRIDE="${ckpt}" \
  GPU_IDS="${GPU_ID}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  EVAL_CHECKPOINTS_IN_PARALLEL=1 \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${sidecar}" \
  HIVA_COEFF_SUMMARY="${summary}" \
  SWEEP_OUTPUT_DIR="${eval_dir}" \
  QUEUE_LOG="${runner_queue_log}" \
  TIMESTAMP="${TIMESTAMP}" \
  bash "${RUNNER}" > "${outer_log}" 2>&1 &
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_file "${RUNNER}"
  require_dir "${DATA_ROOT}"
  require_file "${P5_SIDECAR}"
  require_file "${P5_SUMMARY}"
  require_file "${P7_SIDECAR}"
  require_file "${P7_SUMMARY}"

  local p5_ckpts=(003750 004375)
  local p7_ckpts=(003500 003750)

  echo "===== stage0 P5/P7 additional checkpoint fanout eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "GPU_ID=${GPU_ID}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "P5_EVAL_DIR=${P5_EVAL_DIR}"
  echo "P5_CKPTS=${p5_ckpts[*]}"
  echo "P7_EVAL_DIR=${P7_EVAL_DIR}"
  echo "P7_CKPTS=${p7_ckpts[*]}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  local pids=()
  local ckpt
  for ckpt in "${p5_ckpts[@]}"; do
    launch_ckpt "${P5_TRAIN_DIR}" "${P5_MODEL_NAME}" "${P5_EVAL_DIR}" "${P5_SIDECAR}" "${P5_SUMMARY}" "${ckpt}"
    pids+=("$!")
  done
  for ckpt in "${p7_ckpts[@]}"; do
    launch_ckpt "${P7_TRAIN_DIR}" "${P7_MODEL_NAME}" "${P7_EVAL_DIR}" "${P7_SIDECAR}" "${P7_SUMMARY}" "${ckpt}"
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
    echo "One or more additional GPU ${GPU_ID} fanout evals failed." >&2
    exit "${status}"
  fi

  echo "===== stage0 P5/P7 additional checkpoint fanout eval finished at $(date) ====="
}

main "$@"
