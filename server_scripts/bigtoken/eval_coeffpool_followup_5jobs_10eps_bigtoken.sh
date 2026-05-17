#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO evals for coeff-pool LP-MT and decoded-loss MT follow-ups on
# bigtoken. Each job evaluates 10 episodes x 10 tasks x 4 suites on GPUs 4-7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
SIDECAR_LPMT_D2_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_10_w1_10_w3_0_k10_f15_canonical_lp_mt.parquet"
SUMMARY_LPMT_D2_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_10_w1_10_w3_0_k10_f15_canonical_lp_mt.summary.json"
SIDECAR_LPMT_V5_D4_6_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet"
SUMMARY_LPMT_V5_D4_6_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json"
SIDECAR_LPMT_V1_D4_6_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v1_d4_6_10_commit6_k10_f15_canonical_lp_mt.parquet"
SUMMARY_LPMT_V1_D4_6_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v1_d4_6_10_commit6_k10_f15_canonical_lp_mt.summary.json"
SIDECAR_LPMT_V6_D4_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v6_d4_10_wide_commit4_k10_f15_canonical_lp_mt.parquet"
SUMMARY_LPMT_V6_D4_10_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v6_d4_10_wide_commit4_k10_f15_canonical_lp_mt.summary.json"
SIDECAR_MT_D2_15_K10="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet"
SUMMARY_MT_D2_15_K10="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json"

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
  local chunk_size="$6"
  local n_action_steps="$7"

  require_dir "${policy_path}"
  require_file "${sidecar}"
  require_file "${summary}"

  echo "===== ${job_id}: ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${sidecar}"
  echo "HIVA_COEFF_SUMMARY=${summary}"
  echo "CHUNK_SIZE=${chunk_size}"
  echo "N_ACTION_STEPS=${n_action_steps}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${job_id}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${sidecar}" \
  HIVA_COEFF_SUMMARY="${summary}" \
  CHUNK_SIZE="${chunk_size}" \
  N_ACTION_STEPS="${n_action_steps}" \
  HIVA_DURATION_EXECUTION_MAP="" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

JOB1="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_10_coeffpool_full_ce_mean_k10_f15_bigflow_b160_s2_20260510_024619/checkpoints/last/pretrained_model"
JOB2="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_coeffpool_job1_v5_d4_6_10_full_ce_mean_k10_f15_bigcornea_b64_s2_20260510_041334/checkpoints/last/pretrained_model"
JOB3="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_dur4,6,10_coeffpool_full_ce_mean_k10_f15_bigflow_b160_s2_20260510_091435/checkpoints/last/pretrained_model"
JOB4="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_coeffpool_job2_v6_d4_10_full_ce_mean_k10_f15_bigcornea_b64_s2_20260510_092601/checkpoints/last/pretrained_model"
JOB5="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b128_s2_daw2_rot10_20260509_002956/checkpoints/last/pretrained_model"

echo "===== Coeff-pool follow-up evals started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

run_eval "job1" "${JOB1}" "job1_lpmt_d2_10_coeffpool_full_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_D2_10_K10_F15}" "${SUMMARY_LPMT_D2_10_K10_F15}" 15 10
run_eval "job2" "${JOB2}" "job2_lpmt_v5_d4_6_10_coeffpool_full_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_V5_D4_6_10_K10_F15}" "${SUMMARY_LPMT_V5_D4_6_10_K10_F15}" 15 10
run_eval "job3" "${JOB3}" "job3_lpmt_v1_d4_6_10_coeffpool_full_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_V1_D4_6_10_K10_F15}" "${SUMMARY_LPMT_V1_D4_6_10_K10_F15}" 15 10
run_eval "job4" "${JOB4}" "job4_lpmt_v6_d4_10_coeffpool_full_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_V6_D4_10_K10_F15}" "${SUMMARY_LPMT_V6_D4_10_K10_F15}" 15 10
run_eval "job5" "${JOB5}" "job5_mt_decodedloss_d2_15_daw2_rot10_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 15

echo "===== Coeff-pool follow-up evals finished at $(date) ====="
