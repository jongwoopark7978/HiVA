#!/usr/bin/env bash
set -euo pipefail

# Staggered fanout full 50-episode LIBERO evals for mixed b256/P7 S=0.5
# stage0 LP-MT HiVA checkpoints on BigBrain GPU 3. Launches are spaced by
# INTERVAL_SECONDS to reduce simultaneous model/env initialization pressure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-3}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

B256_TRAIN_DIR="${B256_TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203}"
P7_TRAIN_DIR="${P7_TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459}"
B256_MODEL_NAME="$(basename "${B256_TRAIN_DIR}")"
P7_MODEL_NAME="$(basename "${P7_TRAIN_DIR}")"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
B256_SIDECAR="${B256_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
B256_SUMMARY="${B256_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"
P7_SIDECAR="${P7_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.parquet}"
P7_SUMMARY="${P7_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.summary.json}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigbrain_gpu${GPU_ID}_mixed_b256_p7_s0p5_ckpts_004375_005000_50eps_bs50_${TIMESTAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_full_staggered_fanout_gpu${GPU_ID}_mixed_b256_p7_s0p5_ckpts_004375_005000_50eps_bs50_${TIMESTAMP}.queue.log}"

JOBS=(
  "b256|${B256_TRAIN_DIR}|${B256_MODEL_NAME}|004375|${B256_SIDECAR}|${B256_SUMMARY}"
  "b256|${B256_TRAIN_DIR}|${B256_MODEL_NAME}|005000|${B256_SIDECAR}|${B256_SUMMARY}"
  "p7|${P7_TRAIN_DIR}|${P7_MODEL_NAME}|004375|${P7_SIDECAR}|${P7_SUMMARY}"
  "p7|${P7_TRAIN_DIR}|${P7_MODEL_NAME}|005000|${P7_SIDECAR}|${P7_SUMMARY}"
)

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

launch_job() {
  local tag="$1"
  local train_dir="$2"
  local model_name="$3"
  local ckpt="$4"
  local sidecar="$5"
  local summary="$6"
  local outer_log="${LOG_DIR}/eval_full_staggered_fanout_gpu${GPU_ID}_${tag}_${model_name}_ckpt_${ckpt}_50eps_bs50_${TIMESTAMP}.outer.log"
  local runner_queue_log="${LOG_DIR}/eval_full_staggered_fanout_gpu${GPU_ID}_${tag}_${model_name}_ckpt_${ckpt}_50eps_bs50_${TIMESTAMP}.queue.log"

  require_dir "${train_dir}/checkpoints/${ckpt}/pretrained_model"
  require_file "${sidecar}"
  require_file "${summary}"

  echo "===== $(date) launching ${tag} ckpt_${ckpt} full eval on GPU ${GPU_ID} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "SIDECAR=${sidecar}"
  echo "OUTER_LOG=${outer_log}"
  echo "RUNNER_QUEUE_LOG=${runner_queue_log}"

  TRAIN_DIR="${train_dir}" \
  MODEL_TAG="${model_name}" \
  CKPTS_OVERRIDE="${ckpt}" \
  GPU_IDS="${GPU_ID}" \
  EVAL_CHECKPOINTS_IN_PARALLEL=1 \
  N_EPISODES=50 \
  EVAL_BATCH_SIZE=50 \
  MAX_EPISODES_RENDERED=1 \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${sidecar}" \
  HIVA_COEFF_SUMMARY="${summary}" \
  SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR}/${tag}_${model_name}" \
  QUEUE_LOG="${runner_queue_log}" \
  TIMESTAMP="${TIMESTAMP}" \
  bash "${RUNNER}" > "${outer_log}" 2>&1 &
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_file "${RUNNER}"
  require_dir "${B256_TRAIN_DIR}/checkpoints"
  require_dir "${P7_TRAIN_DIR}/checkpoints"
  require_dir "${DATA_ROOT}"
  mkdir -p "${SWEEP_OUTPUT_DIR}"

  echo "===== mixed b256/P7 full staggered fanout eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "GPU_ID=${GPU_ID}"
  echo "INTERVAL_SECONDS=${INTERVAL_SECONDS}"
  echo "B256_TRAIN_DIR=${B256_TRAIN_DIR}"
  echo "P7_TRAIN_DIR=${P7_TRAIN_DIR}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  local pids=()
  local idx
  local job
  local tag train_dir model_name ckpt sidecar summary
  for idx in "${!JOBS[@]}"; do
    job="${JOBS[$idx]}"
    IFS='|' read -r tag train_dir model_name ckpt sidecar summary <<< "${job}"
    launch_job "${tag}" "${train_dir}" "${model_name}" "${ckpt}" "${sidecar}" "${summary}"
    pids+=("$!")
    if [[ "${idx}" -lt "$((${#JOBS[@]} - 1))" ]]; then
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
    echo "One or more mixed staggered fanout full eval jobs failed." >&2
    exit "${status}"
  fi

  echo "===== mixed b256/P7 full staggered fanout eval finished at $(date) ====="
}

main "$@"
