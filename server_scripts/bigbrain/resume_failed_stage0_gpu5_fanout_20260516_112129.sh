#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RESUME_TS="${RESUME_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

RUNNER="${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh"
SIDECAR_ROOT="/nfs/bigbrain/add_disk0/jongwoopark"

DEFAULT_SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet"
DEFAULT_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json"
P7_SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.parquet"
P7_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.summary.json"

BEST_S0P5_TRAIN="/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/Best_Models/BestS0p5_78.75_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p5_20260514_182207"
BEST_S0P5_OUT="${REPO_ROOT}/outputs/eval/bigbrain_gpu5_BestS0p5_78.75_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p5_20260514_182207_10eps_bs10_20260516_112129"

S0P25_TRAIN="${REPO_ROOT}/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520"
S0P25_OUT="${REPO_ROOT}/outputs/eval/bigbrain_gpu5_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520_10eps_bs10_20260516_112129"

P7_TRAIN="${REPO_ROOT}/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459"
P7_OUT="${REPO_ROOT}/outputs/eval/full_bigbrain_gpu5_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459_50eps_bs50_20260516_112129"

run_resume_group() {
  local name="$1"
  shift
  echo "===== $(date) launching ${name} ====="
  env "$@" bash "${RUNNER}"
  echo "===== $(date) finished ${name} ====="
}

pids=()

run_resume_group "BestS0p5 partial ckpt_003750" \
  TIMESTAMP="${RESUME_TS}" \
  TRAIN_DIR="${BEST_S0P5_TRAIN}" \
  MODEL_TAG="$(basename "${BEST_S0P5_TRAIN}")" \
  CKPTS_OVERRIDE="003750" \
  GPU_IDS="5" \
  EVAL_CHECKPOINTS_IN_PARALLEL="1" \
  EVAL_BATCH_SIZE="10" \
  N_EPISODES="10" \
  SWEEP_OUTPUT_DIR="${BEST_S0P5_OUT}" \
  QUEUE_LOG="${LOG_DIR}/resume_failed_best_s0p5_partial_gpu5_${RESUME_TS}.queue.log" \
  HIVA_COEFF_SIDECAR="${DEFAULT_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${DEFAULT_SUMMARY}" &
pids+=("$!")

run_resume_group "S0p25 partial ckpt_003500 ckpt_003750" \
  TIMESTAMP="${RESUME_TS}" \
  TRAIN_DIR="${S0P25_TRAIN}" \
  MODEL_TAG="$(basename "${S0P25_TRAIN}")" \
  CKPTS_OVERRIDE="003500 003750" \
  GPU_IDS="5,5" \
  EVAL_CHECKPOINTS_IN_PARALLEL="1" \
  EVAL_BATCH_SIZE="10" \
  N_EPISODES="10" \
  SWEEP_OUTPUT_DIR="${S0P25_OUT}" \
  QUEUE_LOG="${LOG_DIR}/resume_failed_s0p25_partial_gpu5_${RESUME_TS}.queue.log" \
  HIVA_COEFF_SIDECAR="${DEFAULT_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${DEFAULT_SUMMARY}" &
pids+=("$!")

run_resume_group "P7 full ckpt_003125 ckpt_003500 ckpt_003750" \
  TIMESTAMP="${RESUME_TS}" \
  TRAIN_DIR="${P7_TRAIN}" \
  MODEL_TAG="$(basename "${P7_TRAIN}")" \
  CKPTS_OVERRIDE="003125 003500 003750" \
  GPU_IDS="5,5,5" \
  EVAL_CHECKPOINTS_IN_PARALLEL="1" \
  EVAL_BATCH_SIZE="50" \
  N_EPISODES="50" \
  SWEEP_OUTPUT_DIR="${P7_OUT}" \
  QUEUE_LOG="${LOG_DIR}/resume_failed_p7_full_gpu5_${RESUME_TS}.queue.log" \
  HIVA_COEFF_SIDECAR="${P7_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${P7_SUMMARY}" &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "One or more GPU5 resume groups failed." >&2
  exit "${status}"
fi

echo "All GPU5 failed eval resume groups completed."
