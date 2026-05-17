#!/usr/bin/env bash
set -euo pipefail

# Sequential partial LIBERO evals for LP-MT residual xattn scale/beta/grip sweeps.
# Each job evaluates 10 episodes x 10 tasks x 4 suites on GPUs 4-7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

SIDECAR_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet"
SUMMARY_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json"

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

  require_dir "${policy_path}"
  require_file "${SIDECAR_K10_F15}"
  require_file "${SUMMARY_K10_F15}"

  echo "===== ${job_id}: ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${SIDECAR_K10_F15}"
  echo "HIVA_COEFF_SUMMARY=${SUMMARY_K10_F15}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${job_id}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${SIDECAR_K10_F15}" \
  HIVA_COEFF_SUMMARY="${SUMMARY_K10_F15}" \
  CHUNK_SIZE=15 \
  N_ACTION_STEPS=10 \
  HIVA_DURATION_EXECUTION_MAP="" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

echo "===== LP-MT xattn scale/beta/grip sweep evals started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

run_eval "job1" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale6p0_daw1p0_trb0p1_rotb0p1_gripb0p1_b128_s2_20260512_112407_job1_scale6p0_scale6p0_daw1p0_trb0p1_rotb0p1_gripb0p1/checkpoints/last/pretrained_model" \
  "job1_lpmt_stage1_xattn_b9_v5_d4_6_10_scale6p0_daw1p0_betas0p1_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

run_eval "job2" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale5p0_daw1p0_trb0p1_rotb0p1_gripb0p1_b128_s2_20260512_112407_job1_scale5p0_scale5p0_daw1p0_trb0p1_rotb0p1_gripb0p1/checkpoints/last/pretrained_model" \
  "job2_lpmt_stage1_xattn_b9_v5_d4_6_10_scale5p0_daw1p0_betas0p1_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

run_eval "job3" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale4p0_daw1p0_trb0p1_rotb0p1_gripb0p1_b128_s2_20260512_112407_job1_scale4p0_scale4p0_daw1p0_trb0p1_rotb0p1_gripb0p1/checkpoints/last/pretrained_model" \
  "job3_lpmt_stage1_xattn_b9_v5_d4_6_10_scale4p0_daw1p0_betas0p1_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

run_eval "job4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage1_xattn_job1_gripsweep_v5_d4_6_10_tr3_rot3_grip0p3_daw1_betas_0p1_0p05_0p1_b1024_g3_s2_20260512_174744/checkpoints/last/pretrained_model" \
  "job4_lpmt_stage1_xattn_v5_d4_6_10_tr3_rot3_grip0p3_daw1_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

run_eval "job5" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage1_xattn_job1_gripsweep_v5_d4_6_10_tr3_rot3_grip0p0_daw1_betas_0p1_0p05_0p1_b1024_g3_s2_20260512_150540/checkpoints/last/pretrained_model" \
  "job5_lpmt_stage1_xattn_v5_d4_6_10_tr3_rot3_grip0p0_daw1_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

run_eval "job6" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale3p0_daw1p0_trb0p1_rotb0p07_gripb0p1_b128_s2_20260512_112407_job2_rotbeta0p07_scale3p0_daw1p0_trb0p1_rotb0p07_gripb0p1/checkpoints/last/pretrained_model" \
  "job6_lpmt_stage1_xattn_b9_v5_d4_6_10_scale3p0_rotb0p07_daw1p0_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

run_eval "job7" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale3p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b128_s2_20260512_112407_job2_rotbeta0p05_scale3p0_daw1p0_trb0p1_rotb0p05_gripb0p1/checkpoints/last/pretrained_model" \
  "job7_lpmt_stage1_xattn_b9_v5_d4_6_10_scale3p0_rotb0p05_daw1p0_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

run_eval "job8" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale3p0_daw1p0_trb0p1_rotb0p03_gripb0p1_b128_s2_20260512_112407_job2_rotbeta0p03_scale3p0_daw1p0_trb0p1_rotb0p03_gripb0p1/checkpoints/last/pretrained_model" \
  "job8_lpmt_stage1_xattn_b9_v5_d4_6_10_scale3p0_rotb0p03_daw1p0_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

echo "===== LP-MT xattn scale/beta/grip sweep evals finished at $(date) ====="
