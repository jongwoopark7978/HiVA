#!/usr/bin/env bash
set -euo pipefail

# Follow-up BestS2 D={2,15} partial LIBERO eval sweep on bigtoken.
# Runs 10 episodes x 10 tasks x 4 suites for each execution mapping:
#   {2,15}->{6,6}, {2,15}->{6,10}, {2,15}->{6,15}
# The sweep waits for the existing GPU4-7 follow-up queue before launching.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-5}"
WAIT_PATTERN="${WAIT_PATTERN:-eval_queued_hiva_coeff_followup_13jobs_10eps_bigtoken.sh}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/BestS2_2-15_smolvla_hiva_coeff_mt_d2_15_w1_10_w3_0_residual_ffn_full_ce_mean_k10_bigflow_b160_s2_20260506_232204/checkpoints/last/pretrained_model}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json}"

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

run_map_eval() {
  local job_id="$1"
  local duration_map="$2"
  local label="job1_BestS2_d2_15_perenv_map2to6_15to${job_id}_10eps_bs${EVAL_BATCH_SIZE}"

  echo "===== ${label} at $(date) ====="
  echo "POLICY_PATH=${POLICY_PATH}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
  echo "HIVA_DURATION_EXECUTION_MAP=${duration_map}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

  POLICY_PATH="${POLICY_PATH}" \
  CHECKPOINT_LABEL="${label}" \
  TIMESTAMP="${TIMESTAMP}_map2to6_15to${job_id}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  CHUNK_SIZE=15 \
  N_ACTION_STEPS=15 \
  HIVA_DURATION_EXECUTION_MAP="${duration_map}" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

echo "===== BestS2 D={2,15} map2to6 sweep queued at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

require_dir "${POLICY_PATH}"
require_file "${HIVA_COEFF_SIDECAR}"
require_file "${HIVA_COEFF_SUMMARY}"

wait_for_existing_queue

run_map_eval "6" "2:6,15:6"
run_map_eval "10" "2:6,15:10"
run_map_eval "15" "2:6,15:15"

echo "===== BestS2 D={2,15} map2to6 sweep finished at $(date) ====="
