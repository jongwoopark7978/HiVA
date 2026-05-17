#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO evals for coeff-pool LP-MT follow-ups on bigtoken.
# Each job evaluates 10 episodes x 10 tasks x 4 suites on GPUs 4-7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
SIDECAR_LPMT_V5_D4_6_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet"
SUMMARY_LPMT_V5_D4_6_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json"
SIDECAR_LPMT_V3_D2_4_6_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v3_d2_4_6_10_prenear2_commit6_k10_f15_canonical_lp_mt.parquet"
SUMMARY_LPMT_V3_D2_4_6_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v3_d2_4_6_10_prenear2_commit6_k10_f15_canonical_lp_mt.summary.json"

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

run_eval() {
  local job_id="$1"
  local policy_path="$2"
  local checkpoint_label="$3"
  local sidecar="$4"
  local summary="$5"

  require_dir "${policy_path}"
  require_file "${sidecar}"
  require_file "${summary}"

  echo "===== ${job_id}: ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${sidecar}"
  echo "HIVA_COEFF_SUMMARY=${summary}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${job_id}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${sidecar}" \
  HIVA_COEFF_SUMMARY="${summary}" \
  CHUNK_SIZE=15 \
  N_ACTION_STEPS=10 \
  HIVA_DURATION_EXECUTION_MAP="" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

JOB1="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s2_daw0p5_20260511_021834/checkpoints/last/pretrained_model"
JOB2="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s2_daw1p0_20260511_021834/checkpoints/last/pretrained_model"
JOB3="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_dur2,4,6,10_coeffpool_full_ce_mean_k10_f15_bigflow_b160_s2_20260510_220807/checkpoints/last/pretrained_model"

echo "===== Coeff-pool 3-job evals started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

run_eval "job1" "${JOB1}" "job1_lpmt_v5_d4_6_10_residual_coeffpool_daw0p5_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_V5_D4_6_10_K10_F15}" "${SUMMARY_LPMT_V5_D4_6_10_K10_F15}"
run_eval "job2" "${JOB2}" "job2_lpmt_v5_d4_6_10_residual_coeffpool_daw1p0_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_V5_D4_6_10_K10_F15}" "${SUMMARY_LPMT_V5_D4_6_10_K10_F15}"
run_eval "job3" "${JOB3}" "job3_lpmt_v3_d2_4_6_10_coeffpool_full_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_V3_D2_4_6_10_K10_F15}" "${SUMMARY_LPMT_V3_D2_4_6_10_K10_F15}"

echo "===== Coeff-pool 3-job evals finished at $(date) ====="
