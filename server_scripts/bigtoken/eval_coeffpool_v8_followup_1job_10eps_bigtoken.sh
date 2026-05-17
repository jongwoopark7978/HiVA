#!/usr/bin/env bash
set -euo pipefail

# Queue a v8 coeff-pool LP-MT partial LIBERO eval behind the current GPU4-7
# coeff-pool sweep. Evaluates 10 episodes x 10 tasks x 4 suites.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WAIT_PATTERN="${WAIT_PATTERN:-eval_coeffpool_followup_2jobs_10eps_bigtoken.sh}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_coeffpool_job4_v8_d2_4_10_full_ce_mean_k10_f15_bigcornea_b64_s2_20260510_194903/checkpoints/last/pretrained_model}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v8_d2_4_10_wide_prenear2_commit4_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v8_d2_4_10_wide_prenear2_commit4_k10_f15_canonical_lp_mt.summary.json}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-job1_lpmt_v8_d2_4_10_coeffpool_full_k10_f15_10eps_bs${EVAL_BATCH_SIZE}}"

wait_for_existing_queue() {
  echo "Waiting for existing GPU4-7 queue pattern: ${WAIT_PATTERN}"
  while pgrep -f "${WAIT_PATTERN}" >/dev/null; do
    date
    pgrep -af "${WAIT_PATTERN}|lerobot-eval" || true
    sleep 300
  done
  echo "Existing queue pattern is clear at $(date)."
}

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

echo "===== v8 coeff-pool follow-up eval queued at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"

require_dir "${POLICY_PATH}"
require_file "${HIVA_COEFF_SIDECAR}"
require_file "${HIVA_COEFF_SUMMARY}"

wait_for_existing_queue

POLICY_PATH="${POLICY_PATH}" \
CHECKPOINT_LABEL="${CHECKPOINT_LABEL}" \
TIMESTAMP="${TIMESTAMP}_job1" \
GPU_IDS="${GPU_IDS}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
CHUNK_SIZE=15 \
N_ACTION_STEPS=10 \
HIVA_DURATION_EXECUTION_MAP="" \
bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"

echo "===== v8 coeff-pool follow-up eval finished at $(date) ====="
