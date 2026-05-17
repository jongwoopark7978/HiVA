#!/usr/bin/env bash
set -euo pipefail

# Fan out selected stage0 LP-MT HiVA coefficient evaluations on GPU 5.
# - Partial evals: 10 episodes/task, eval.batch_size=10
# - Full evals: 50 episodes/task, eval.batch_size=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-5}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
DEFAULT_SIDECAR="${DEFAULT_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
DEFAULT_SUMMARY="${DEFAULT_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"
P7_SIDECAR="${P7_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.parquet}"
P7_SUMMARY="${P7_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.summary.json}"

BEST_S0P5_TRAIN_DIR="${BEST_S0P5_TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/Best_Models/BestS0p5_78.75_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p5_20260514_182207}"
S0P25_TRAIN_DIR="${S0P25_TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520}"
P7_TRAIN_DIR="${P7_TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459}"

BEST_S0P5_MODEL_NAME="$(basename "${BEST_S0P5_TRAIN_DIR}")"
S0P25_MODEL_NAME="$(basename "${S0P25_TRAIN_DIR}")"
P7_MODEL_NAME="$(basename "${P7_TRAIN_DIR}")"

BEST_S0P5_EVAL_DIR="${BEST_S0P5_EVAL_DIR:-${REPO_ROOT}/outputs/eval/bigbrain_gpu5_${BEST_S0P5_MODEL_NAME}_10eps_bs10_${TIMESTAMP}}"
S0P25_EVAL_DIR="${S0P25_EVAL_DIR:-${REPO_ROOT}/outputs/eval/bigbrain_gpu5_${S0P25_MODEL_NAME}_10eps_bs10_${TIMESTAMP}}"
P7_FULL_EVAL_DIR="${P7_FULL_EVAL_DIR:-${REPO_ROOT}/outputs/eval/full_bigbrain_gpu5_${P7_MODEL_NAME}_50eps_bs50_${TIMESTAMP}}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_stage0_s0p5_s0p25_partial_and_p7_full_gpu${GPU_ID}_${TIMESTAMP}.queue.log}"

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
  local mode="$1"
  local train_dir="$2"
  local model_name="$3"
  local eval_dir="$4"
  local sidecar="$5"
  local summary="$6"
  local ckpt="$7"
  local n_episodes="$8"
  local batch_size="$9"

  local outer_log="${LOG_DIR}/eval_gpu${GPU_ID}_${mode}_${model_name}_ckpt_${ckpt}_${n_episodes}eps_bs${batch_size}_${TIMESTAMP}.outer.log"
  local runner_queue_log="${LOG_DIR}/eval_gpu${GPU_ID}_${mode}_${model_name}_ckpt_${ckpt}_${n_episodes}eps_bs${batch_size}_${TIMESTAMP}.queue.log"

  require_dir "${train_dir}/checkpoints/${ckpt}/pretrained_model"
  mkdir -p "${eval_dir}"

  echo "===== $(date) launching ${mode} ${model_name} ckpt_${ckpt} on GPU ${GPU_ID} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "EVAL_DIR=${eval_dir}"
  echo "N_EPISODES=${n_episodes}"
  echo "EVAL_BATCH_SIZE=${batch_size}"
  echo "SIDECAR=${sidecar}"
  echo "OUTER_LOG=${outer_log}"

  TRAIN_DIR="${train_dir}" \
  MODEL_TAG="${model_name}" \
  CKPTS_OVERRIDE="${ckpt}" \
  GPU_IDS="${GPU_ID}" \
  EVAL_BATCH_SIZE="${batch_size}" \
  N_EPISODES="${n_episodes}" \
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
  require_file "${DEFAULT_SIDECAR}"
  require_file "${DEFAULT_SUMMARY}"
  require_file "${P7_SIDECAR}"
  require_file "${P7_SUMMARY}"

  local best_s0p5_partial_ckpts=(003750)
  local s0p25_partial_ckpts=(002500 003000 003500 003750 004000)
  local p7_full_ckpts=(003125 003500 003750)

  echo "===== stage0 GPU${GPU_ID} mixed fanout eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "GPU_ID=${GPU_ID}"
  echo "BEST_S0P5_EVAL_DIR=${BEST_S0P5_EVAL_DIR}"
  echo "BEST_S0P5_PARTIAL_CKPTS=${best_s0p5_partial_ckpts[*]}"
  echo "S0P25_EVAL_DIR=${S0P25_EVAL_DIR}"
  echo "S0P25_PARTIAL_CKPTS=${s0p25_partial_ckpts[*]}"
  echo "P7_FULL_EVAL_DIR=${P7_FULL_EVAL_DIR}"
  echo "P7_FULL_CKPTS=${p7_full_ckpts[*]}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  local pids=()
  local ckpt
  for ckpt in "${best_s0p5_partial_ckpts[@]}"; do
    launch_ckpt partial "${BEST_S0P5_TRAIN_DIR}" "${BEST_S0P5_MODEL_NAME}" "${BEST_S0P5_EVAL_DIR}" "${DEFAULT_SIDECAR}" "${DEFAULT_SUMMARY}" "${ckpt}" 10 10
    pids+=("$!")
  done
  for ckpt in "${s0p25_partial_ckpts[@]}"; do
    launch_ckpt partial "${S0P25_TRAIN_DIR}" "${S0P25_MODEL_NAME}" "${S0P25_EVAL_DIR}" "${DEFAULT_SIDECAR}" "${DEFAULT_SUMMARY}" "${ckpt}" 10 10
    pids+=("$!")
  done
  for ckpt in "${p7_full_ckpts[@]}"; do
    launch_ckpt full "${P7_TRAIN_DIR}" "${P7_MODEL_NAME}" "${P7_FULL_EVAL_DIR}" "${P7_SIDECAR}" "${P7_SUMMARY}" "${ckpt}" 50 50
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
    echo "One or more GPU ${GPU_ID} fanout evals failed." >&2
    exit "${status}"
  fi

  echo "===== stage0 GPU${GPU_ID} mixed fanout eval finished at $(date) ====="
}

main "$@"
