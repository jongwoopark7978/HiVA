#!/usr/bin/env bash
set -euo pipefail

# Queue two stage1-only v5 d4,6,10 LP-MT partial LIBERO evals after the
# currently running GPU4-7 eval. Each job evaluates 10 episodes x 10 tasks x 4 suites.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WAIT_PATTERN="${WAIT_PATTERN:-eval_lpmt_v5_p4_residual_daw0p5_10eps_bigtoken.sh|job1_lpmt_v5_d4_6_10_residual_coeffpool_p4_daw0p5}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

JOB1="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_only_v5_d4_6_10_k10_f15_daw1p0_b128_s2_20260511_171905_daw1p0/checkpoints/last/pretrained_model"
JOB2="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_only_v5_d4_6_10_k10_f15_daw0p5_b64_s2_20260511_171905_daw0p5/checkpoints/last/pretrained_model"

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
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  echo "===== ${job_id}: ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${job_id}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  CHUNK_SIZE=15 \
  N_ACTION_STEPS=10 \
  HIVA_DURATION_EXECUTION_MAP="" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

echo "===== queued stage1-only v5 d4,6,10 evals at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "WAIT_PATTERN=${WAIT_PATTERN}"

while pgrep -f "${WAIT_PATTERN}" >/dev/null; do
  echo "Waiting for current GPU4-7 eval to finish at $(date)"
  pgrep -af "${WAIT_PATTERN}" || true
  sleep 300
done

echo "Current GPU4-7 eval is clear at $(date); starting stage1-only evals"

run_eval "job1" "${JOB1}" "job1_lpmt_stage1_only_v5_d4_6_10_daw1p0_b128_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"
run_eval "job2" "${JOB2}" "job2_lpmt_stage1_only_v5_d4_6_10_daw0p5_b64_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"

echo "===== queued stage1-only v5 d4,6,10 evals finished at $(date) ====="
