#!/usr/bin/env bash
set -euo pipefail

# Queue LP-MT partial LIBERO evals after the existing GPU4-7 queue.
# Each job evaluates 10 episodes x 10 tasks x 4 suites.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WAIT_PATTERN="${WAIT_PATTERN:-eval_lpmt_stage1_xattn_b9_scale1p0_daw1p5_after_sweep_10eps_bigtoken.sh}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

SIDECAR_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet"
SUMMARY_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json"
SIDECAR_K10_P7_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.parquet"
SUMMARY_K10_P7_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.summary.json"

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

echo "===== LP-MT p7 + xattn scale3/4/5 eval queue started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "WAIT_PATTERN=${WAIT_PATTERN}"

while pgrep -f "${WAIT_PATTERN}" >/dev/null; do
  echo "[$(date)] Waiting for existing queued eval jobs matching: ${WAIT_PATTERN}"
  pgrep -af "${WAIT_PATTERN}" || true
  sleep 300
done

run_eval "job1" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_s2_nores_nodl_20260511_184643/checkpoints/last/pretrained_model" \
  "job1_lpmt_v5_d4_6_10_coeffpool_nores_nodl_p7_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" \
  "${SIDECAR_K10_P7_F15}" \
  "${SUMMARY_K10_P7_F15}"

run_eval "job2" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale3p0_daw1p0_b128_s2_20260511_214458_scale3p0_daw1p0/checkpoints/last/pretrained_model" \
  "job2_lpmt_stage1_xattn_b9_v5_d4_6_10_scale3p0_daw1p0_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" \
  "${SIDECAR_K10_F15}" \
  "${SUMMARY_K10_F15}"

run_eval "job3" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale4p0_daw1p0_trb0p1_rotb0p1_gripb0p1_b128_s2_20260512_112407_job1_scale4p0_scale4p0_daw1p0_trb0p1_rotb0p1_gripb0p1/checkpoints/last/pretrained_model" \
  "job3_lpmt_stage1_xattn_b9_v5_d4_6_10_scale4p0_daw1p0_beta0p1_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" \
  "${SIDECAR_K10_F15}" \
  "${SUMMARY_K10_F15}"

run_eval "job4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale5p0_daw1p0_trb0p1_rotb0p1_gripb0p1_b128_s2_20260512_112407_job1_scale5p0_scale5p0_daw1p0_trb0p1_rotb0p1_gripb0p1/checkpoints/last/pretrained_model" \
  "job4_lpmt_stage1_xattn_b9_v5_d4_6_10_scale5p0_daw1p0_beta0p1_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" \
  "${SIDECAR_K10_F15}" \
  "${SUMMARY_K10_F15}"

echo "===== LP-MT p7 + xattn scale3/4/5 eval queue finished at $(date) ====="
